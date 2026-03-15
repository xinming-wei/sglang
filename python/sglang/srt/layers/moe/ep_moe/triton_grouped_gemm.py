"""Triton-based grouped GEMM for BF16 MoE expert parallelism.

This kernel keeps the UltraEP-friendly `b_ptrs` pointer-table interface, but
replaces the original naive persistent scheduler with:

1. grouped launch ordering to improve reuse of expert weight tiles,
2. M-bucketed autotune so long-prefill does not inherit decode configs,
3. virtual-SM persistent launch choices, and
4. TensorDescriptor/TMA loads and stores for SM90+.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import triton
import triton.language as tl


def _build_autotune_configs():
    base_configs = [
        (128, 256, 64, 8, 4, 8),
        (128, 128, 64, 8, 4, 8),
        (128, 256, 128, 8, 3, 8),
        (128, 128, 128, 8, 3, 8),
        (64, 256, 64, 8, 4, 4),
        (64, 128, 64, 8, 4, 4),
        (64, 64, 128, 8, 3, 4),
    ]
    virtual_sm_mults = {
        128: (1, 2),
        64: (1, 2, 4),
    }

    configs = []
    for block_m, block_n, block_k, group_size_m, num_stages, num_warps in base_configs:
        for virtual_sm_mult in virtual_sm_mults[block_m]:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "GROUP_SIZE_M": group_size_m,
                        "VIRTUAL_SM_MULT": virtual_sm_mult,
                    },
                    num_stages=num_stages,
                    num_warps=num_warps,
                )
            )
    return configs


def _m_bucket(m: int) -> int:
    if m <= 128:
        return 128
    if m <= 512:
        return 512
    if m <= 2048:
        return 2048
    if m <= 8192:
        return 8192
    if m <= 32768:
        return 32768
    return 131072


def _expert_bucket(num_experts: int) -> int:
    if num_experts <= 2:
        return 2
    if num_experts <= 4:
        return 4
    if num_experts <= 8:
        return 8
    return 16


_TMA_ALLOCATOR_SET = False


def _set_triton_tma_allocator():
    """TensorDescriptor kernels need a Triton scratch allocator."""
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    _TMA_ALLOCATOR_SET = True


@triton.autotune(
    configs=_build_autotune_configs(),
    key=["N", "K", "M_BUCKET", "EXPERT_BUCKET"],
)
@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptrs_ptr,
    c_ptr,
    seg_indptr_ptr,
    num_experts,
    M_ALLOC,
    N,
    K,
    M_BUCKET,
    EXPERT_BUCKET,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    VIRTUAL_SM_MULT: tl.constexpr,
):
    """Persistent grouped BF16 GEMM with grouped tile ordering.

    All experts share contiguous input/output buffers partitioned by
    `seg_indptr`. Each expert's weight is addressed through `b_ptrs`, so the
    kernel remains compatible with UltraEP redundant weights.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_idx = pid
    expert_tile_begin = 0

    for g in range(num_experts):
        start_g = tl.load(seg_indptr_ptr + g).to(tl.int32)
        end_g = tl.load(seg_indptr_ptr + g + 1).to(tl.int32)
        gm = end_g - start_g
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        if num_tiles > 0:
            a_base = a_ptr + start_g.to(tl.int64) * K
            c_base = c_ptr + start_g.to(tl.int64) * N
            b_ptr = tl.load(b_ptrs_ptr + g).to(tl.pointer_type(tl.bfloat16))

            a_desc = tl.make_tensor_descriptor(
                a_base,
                shape=[gm, K],
                strides=[K, 1],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[N, K],
                strides=[K, 1],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_base,
                shape=[gm, N],
                strides=[N, 1],
                block_shape=[BLOCK_M, BLOCK_N],
            )

            while tile_idx >= expert_tile_begin and tile_idx < expert_tile_begin + num_tiles:
                tile_in_expert = tile_idx - expert_tile_begin
                num_pid_in_group = GROUP_SIZE_M * num_n_tiles
                group_id = tile_in_expert // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
                group_offset = tile_in_expert % num_pid_in_group
                tile_m = first_pid_m + (group_offset % group_size_m)
                tile_n = group_offset // group_size_m

                m_start = tile_m * BLOCK_M
                n_start = tile_n * BLOCK_N
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k_start in range(0, K, BLOCK_K):
                    a = a_desc.load([m_start, k_start])
                    b = b_desc.load([n_start, k_start])
                    acc += tl.dot(a, tl.trans(b))

                c_desc.store([m_start, n_start], acc.to(tl.bfloat16))
                tile_idx += num_programs

        expert_tile_begin += num_tiles


def grouped_gemm_bf16(
    a: torch.Tensor,
    b_ptrs: torch.Tensor,
    c: torch.Tensor,
    seg_indptr: torch.Tensor,
    num_experts: int,
    N: int,
    K: int,
):
    """Launch grouped BF16 GEMM for expert-segmented inputs.

    Args:
        a: `[token_capacity, K]` BF16 contiguous input buffer.
        b_ptrs: `[num_experts]` int64 table of weight `data_ptr()` values.
        c: `[token_capacity, N]` BF16 contiguous output buffer.
        seg_indptr: `[num_experts + 1]` int64 cumulative token counts.
        num_experts: number of local physical experts to compute.
        N: output feature dimension.
        K: reduction dimension.
    """
    if a.shape[0] == 0 or num_experts == 0:
        return
    if a.device.type != "cuda":
        raise ValueError("grouped_gemm_bf16 requires CUDA tensors")
    if a.dtype != torch.bfloat16 or c.dtype != torch.bfloat16:
        raise ValueError("grouped_gemm_bf16 only supports BF16 input/output")
    if b_ptrs.dtype != torch.int64 or seg_indptr.dtype != torch.int64:
        raise ValueError("b_ptrs and seg_indptr must be int64 tensors")
    if not a.is_contiguous() or not c.is_contiguous():
        raise ValueError("grouped_gemm_bf16 expects contiguous A/C tensors")
    if not b_ptrs.is_contiguous() or not seg_indptr.is_contiguous():
        raise ValueError("grouped_gemm_bf16 expects contiguous metadata tensors")
    if a.device != b_ptrs.device or a.device != c.device or a.device != seg_indptr.device:
        raise ValueError("A, b_ptrs, C, and seg_indptr must live on the same device")
    if a.shape[1] != K or c.shape[1] != N:
        raise ValueError("A/C shapes do not match the supplied N/K dimensions")
    if b_ptrs.numel() < num_experts or seg_indptr.numel() < num_experts + 1:
        raise ValueError("metadata tensors are smaller than num_experts requires")

    bf16_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
    if (K * bf16_bytes) % 16 != 0 or (N * bf16_bytes) % 16 != 0:
        raise ValueError(
            "BF16 grouped GEMM requires K and N to keep row strides 16-byte aligned for TMA"
        )

    major, _ = torch.cuda.get_device_capability(a.device)
    if major < 9:
        raise RuntimeError("This grouped GEMM path requires SM90 or newer GPUs")

    _set_triton_tma_allocator()

    physical_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    m_bucket = _m_bucket(a.shape[0])
    expert_bucket = _expert_bucket(num_experts)
    grid = lambda META: (physical_sms * META["VIRTUAL_SM_MULT"],)

    _grouped_gemm_kernel[grid](
        a,
        b_ptrs,
        c,
        seg_indptr,
        num_experts,
        a.shape[0],
        N,
        K,
        m_bucket,
        expert_bucket,
    )


def build_weight_ptr_table(
    master_weight: torch.Tensor,
    redundant_weights: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Build a device int64 tensor of per-expert weight data_ptr() values.

    The public interface stays pointer-table based so UltraEP can still attach
    redundant experts backed by independent storage.
    """
    num_masters = master_weight.shape[0]
    ptrs = [master_weight[i].data_ptr() for i in range(num_masters)]
    if redundant_weights:
        ptrs.extend(w.data_ptr() for w in redundant_weights)
    return torch.tensor(ptrs, dtype=torch.int64, device=master_weight.device)
