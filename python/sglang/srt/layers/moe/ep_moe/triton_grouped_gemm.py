"""Triton-based grouped GEMM for BF16 MoE expert parallelism.

High-performance fused grouped matmul that processes all experts in a single
kernel launch using the persistent-CTA pattern. Fully on-device with zero
host-device synchronization. Targets SM90 (Hopper) and SM100 (Blackwell).

Reference: https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
"""

from typing import List, Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=5,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptrs_ptr,
    c_ptr,
    seg_indptr_ptr,
    num_experts,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent-CTA grouped GEMM: C_g = A_g @ B_g^T for each expert g.

    All experts share contiguous input/output buffers partitioned by seg_indptr.
    Weight matrices are accessed via a per-expert pointer table (supports
    non-contiguous storage for UltraEP redundant experts).

    A_g = a[seg_indptr[g]:seg_indptr[g+1], :K]   shape [M_g, K]
    B_g = *(b_ptrs[g])                            shape [N, K] row-major
    C_g = c[seg_indptr[g]:seg_indptr[g+1], :N]   shape [M_g, N]

    Type discipline: all tile-counting variables (tile_idx, last_problem_end,
    num_tiles, …) are kept as int32 — the Triton compiler requires loop-carried
    variables to have a stable type across iterations.  Pointer-offset arithmetic
    uses int64 only at the final address-computation step.
    """
    pid = tl.program_id(0)          # int32
    num_sms = tl.num_programs(0)    # int32
    tile_idx = pid                  # int32
    last_problem_end = 0            # int32

    for g in range(num_experts):
        # Load segment boundaries as int32 (token counts fit comfortably)
        start_g = tl.load(seg_indptr_ptr + g).to(tl.int32)
        end_g = tl.load(seg_indptr_ptr + g + 1).to(tl.int32)
        gm = end_g - start_g               # int32: tokens for expert g

        num_m_tiles = tl.cdiv(gm, BLOCK_M)  # int32
        num_n_tiles = tl.cdiv(N, BLOCK_N)   # int32
        num_tiles = num_m_tiles * num_n_tiles  # int32

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            b_ptr = tl.load(b_ptrs_ptr + g).to(tl.pointer_type(tl.bfloat16))

            tile_in_gemm = tile_idx - last_problem_end  # int32
            tile_m = tile_in_gemm // num_n_tiles        # int32
            tile_n = tile_in_gemm % num_n_tiles         # int32

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)  # int32
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)  # int32
            offs_k = tl.arange(0, BLOCK_K)                      # int32

            # Pointer arithmetic: promote to int64 only here to avoid overflow
            # A row-major [M, K]: element (m, k) at a_ptr + (start_g + m) * K + k
            a_row_offs = (start_g + offs_m[:, None]).to(tl.int64) * K
            a_tile = a_ptr + a_row_offs + offs_k[None, :]

            # B row-major [N, K]: element (n, k) at b_ptr + n * K + k
            b_tile = b_ptr + offs_n[:, None].to(tl.int64) * K + offs_k[None, :]

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for kk in range(0, tl.cdiv(K, BLOCK_K)):
                k_remaining = K - kk * BLOCK_K
                a_mask = (offs_m[:, None] < gm) & (offs_k[None, :] < k_remaining)
                b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)

                a = tl.load(a_tile, mask=a_mask, other=0.0)
                b = tl.load(b_tile, mask=b_mask, other=0.0)
                # Compute A @ B^T: [BLOCK_M, BLOCK_K] dot [BLOCK_N, BLOCK_K]^T
                acc += tl.dot(a, tl.trans(b))

                a_tile += BLOCK_K
                b_tile += BLOCK_K

            # Store C row-major [M, N]: element (m, n) at c_ptr + (start_g + m) * N + n
            c_row_offs = (start_g + offs_m[:, None]).to(tl.int64) * N
            c_tile = c_ptr + c_row_offs + offs_n[None, :]
            c_mask = (offs_m[:, None] < gm) & (offs_n[None, :] < N)
            tl.store(c_tile, acc.to(tl.bfloat16), mask=c_mask)

            tile_idx += num_sms   # int32 + int32 = int32

        last_problem_end += num_tiles  # int32 + int32 = int32, type stable


def grouped_gemm_bf16(
    a: torch.Tensor,
    b_ptrs: torch.Tensor,
    c: torch.Tensor,
    seg_indptr: torch.Tensor,
    num_experts: int,
    N: int,
    K: int,
):
    """Launch persistent-CTA grouped GEMM for all experts.

    For each expert g, computes: C_g = A_g @ B_g^T
    where A_g and C_g are contiguous slices of a/c determined by seg_indptr,
    and B_g is accessed via b_ptrs pointer table.

    Args:
        a: [total_tokens, K] bf16 contiguous input.
        b_ptrs: [num_experts] int64, data_ptr() values for each [N, K] weight.
        c: [total_tokens, N] bf16 contiguous output.
        seg_indptr: [num_experts + 1] int64, cumulative token counts.
        num_experts: number of experts.
        N: weight row dimension (output features).
        K: reduction dimension (input features).
    """
    NUM_SM = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (NUM_SM,)

    _grouped_gemm_kernel[grid](
        a,
        b_ptrs,
        c,
        seg_indptr,
        num_experts,
        N,
        K,
    )


def build_weight_ptr_table(
    master_weight: torch.Tensor,
    redundant_weights: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Build a device int64 tensor of per-expert weight data_ptr() values.

    One-time setup (called lazily on the first forward pass). The returned
    tensor is reused across all subsequent forward calls since weight addresses
    are stable after model loading.

    Args:
        master_weight: [num_masters, N, K] weight parameter.
        redundant_weights: optional list of [N, K] UltraEP redundant weight views.

    Returns:
        [num_physical_experts] int64 tensor on the same device.
    """
    num_masters = master_weight.shape[0]
    ptrs = [master_weight[i].data_ptr() for i in range(num_masters)]
    if redundant_weights:
        ptrs.extend(w.data_ptr() for w in redundant_weights)
    return torch.tensor(ptrs, dtype=torch.int64, device=master_weight.device)
