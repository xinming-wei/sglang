from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

try:
    from deep_ep import HybridEPBuffer as HybridEPBufferRuntime

    use_hybridep = True
except (ImportError, AttributeError):
    use_hybridep = False


def _build_dense_routing_map_torch(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = topk_ids >= 0
    safe_topk_ids = topk_ids.clamp_min(0)
    topk_weights = topk_weights.to(torch.float32)

    probs = torch.zeros(
        (topk_ids.shape[0], num_experts),
        dtype=torch.float32,
        device=topk_weights.device,
    )
    probs.scatter_add_(
        1,
        safe_topk_ids,
        topk_weights * valid_mask.to(topk_weights.dtype),
    )

    routing_counts = torch.zeros(
        (topk_ids.shape[0], num_experts),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    routing_counts.scatter_add_(1, safe_topk_ids, valid_mask.to(torch.int32))
    return routing_counts > 0, probs


def _should_use_fused_dense_routing_map(device: torch.device) -> bool:
    # if not _TRITON_AVAILABLE or device.type != "cuda":
    #     return False
    # major, _ = torch.cuda.get_device_capability(device)
    # return major >= 9
    return True


def _dense_routing_map_launch_config(
    num_experts: int,
) -> tuple[int, int, int]:
    if num_experts <= 64:
        return 64, 64, 4
    if num_experts <= 128:
        return 32, 128, 4
    return 16, 128, 8


if _TRITON_AVAILABLE:

    @triton.jit
    def _dense_routing_map_kernel(
        topk_ids_ptr,
        topk_weights_ptr,
        routing_map_ptr,
        probs_ptr,
        num_tokens,
        num_experts,
        topk_stride,
        routing_stride,
        probs_stride,
        TOPK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_e = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
        token_mask = offs_m < num_tokens
        expert_mask = offs_e < num_experts

        experts = offs_e[None, :]
        probs = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
        routed = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.int1)

        for k in range(TOPK):
            ids_k = tl.load(
                topk_ids_ptr + offs_m * topk_stride + k,
                mask=token_mask,
                other=-1,
            ).to(tl.int32)
            weights_k = tl.load(
                topk_weights_ptr + offs_m * topk_stride + k,
                mask=token_mask,
                other=0,
            ).to(tl.float32)
            match = (ids_k[:, None] >= 0) & (ids_k[:, None] == experts)
            probs += match.to(tl.float32) * weights_k[:, None]
            routed = routed | match

        out_mask = token_mask[:, None] & expert_mask[None, :]
        tl.store(
            probs_ptr + offs_m[:, None] * probs_stride + experts,
            probs,
            mask=out_mask,
        )
        tl.store(
            routing_map_ptr + offs_m[:, None] * routing_stride + experts,
            routed,
            mask=out_mask,
        )


def _build_dense_routing_map_fused(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("topk_ids and topk_weights must be rank-2 tensors")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have identical shapes")
    if topk_ids.device != topk_weights.device:
        raise ValueError("topk_ids and topk_weights must live on the same device")
    if topk_ids.numel() == 0 or num_experts == 0:
        shape = (topk_ids.shape[0], num_experts)
        return (
            torch.empty(shape, device=topk_ids.device, dtype=torch.bool),
            torch.empty(shape, device=topk_weights.device, dtype=torch.float32),
        )
    if not _should_use_fused_dense_routing_map(topk_ids.device):
        return _build_dense_routing_map_torch(topk_ids, topk_weights, num_experts)

    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    num_tokens, topk = topk_ids.shape
    routing_map = torch.empty(
        (num_tokens, num_experts),
        dtype=torch.bool,
        device=topk_ids.device,
    )
    probs = torch.empty(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device=topk_weights.device,
    )

    block_m, block_e, num_warps = _dense_routing_map_launch_config(num_experts)
    grid = (
        triton.cdiv(num_tokens, block_m),
        triton.cdiv(num_experts, block_e),
    )
    _dense_routing_map_kernel[grid](
        topk_ids,
        topk_weights,
        routing_map,
        probs,
        num_tokens,
        num_experts,
        topk_ids.stride(0),
        routing_map.stride(0),
        probs.stride(0),
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_E=block_e,
        num_warps=num_warps,
        num_stages=2,
    )
    return routing_map, probs


class _HybridEPSharedBuffer:
    """Process-local HybridEP buffer cache shared across MoE layers.

    HybridEP keeps its compiled kernels in the runtime object's in-memory
    `kernel_cache`. Creating one buffer per layer causes the exact same kernels
    to be re-JITed repeatedly during warmup. Reuse a single buffer per
    (group, hidden_size, num_local_experts) tuple so one process only compiles
    each signature once.
    """

    _buffers: dict[tuple[int, int, int], HybridEPBufferRuntime] = {}

    @classmethod
    def get_or_create(
        cls,
        group: torch.distributed.ProcessGroup,
        hidden_size: int,
        num_local_experts: int,
        max_num_tokens_per_rank: int,
    ) -> HybridEPBufferRuntime:
        key = (id(group), hidden_size, num_local_experts)
        buffer = cls._buffers.get(key)
        if buffer is None:
            buffer = HybridEPBufferRuntime(
                group=group,
                hidden_dim=hidden_size,
                max_num_of_tokens_per_rank=max_num_tokens_per_rank,
                num_local_experts=num_local_experts,
                use_fp8=False,
                load_cached_kernels=True,
            )
            cls._buffers[key] = buffer
            logger.info(
                "Created HybridEP buffer ("
                "hidden size: %s, "
                "num local experts: %s, "
                "max num tokens per rank: %s)",
                hidden_size,
                num_local_experts,
                max_num_tokens_per_rank,
            )
        return buffer


class HybridEPDispatchOutput(NamedTuple):
    """HybridEP dispatch output for BF16 grouped GEMM."""

    hidden_states: torch.Tensor
    routing_weights: Optional[torch.Tensor]
    tokens_per_expert: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.HYBRIDEP


assert isinstance(HybridEPDispatchOutput, DispatchOutput)


class HybridEPCombineInput(NamedTuple):
    """HybridEP combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.HYBRIDEP


assert isinstance(HybridEPCombineInput, CombineInput)


class HybridEPDispatcher(BaseDispatcher):
    """Dispatcher for HybridEP normal-mode fused permute A2A."""

    # HybridEP metadata preprocessing and fused dispatch kernels are fragile when
    # ranks enter with extremely small batches during decode. Pad every rank to
    # a minimum size. Padded rows stay unrouted so they satisfy the collective
    # shape requirements without turning into extra dispatched expert tokens.
    _MIN_SAFE_RUNTIME_MAX_TOKENS_PER_RANK = 64

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype = None,  # Unused
    ):
        super().__init__()
        if not use_hybridep:
            raise ImportError(
                "HybridEP is not installed. Please install the HybridEP package."
            )

        self.group = group
        self.ep_rank = group.rank()
        self.ep_size = group.size()
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size

        self._buffer: Optional[HybridEPBufferRuntime] = None
        self._handle = None
        self._original_num_tokens = 0
        self._configured_num_max_dispatch_tokens_per_rank = (
            self._get_configured_num_max_dispatch_tokens_per_rank()
        )

    def _get_configured_num_max_dispatch_tokens_per_rank(self) -> int:
        server_args = get_global_server_args()
        default_max_tokens = (
            getattr(server_args, "chunked_prefill_size", 0) if server_args else 0
        )

        env_value = os.getenv("SGLANG_HYBRID_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK")
        if env_value is not None:
            default_max_tokens = int(env_value)

        # Use the local per-rank chunk size. HybridEP internally accounts for the
        # number of ranks in the NVLink domain / cluster when sizing buffers.
        return max(default_max_tokens, self._MIN_SAFE_RUNTIME_MAX_TOKENS_PER_RANK)

    def _select_template_capacity(self, num_tokens_per_rank: int) -> int:
        if num_tokens_per_rank <= self._configured_num_max_dispatch_tokens_per_rank:
            return self._configured_num_max_dispatch_tokens_per_rank

        # Overflow is expected to be rare in this serving setup. Bucketize to a
        # power-of-two capacity so decode overshoots do not trigger a fresh JIT
        # for every distinct batch size.
        return 1 << (num_tokens_per_rank - 1).bit_length()

    def _ensure_buffer(self, num_tokens_per_rank: int) -> HybridEPBufferRuntime:
        target_capacity = self._select_template_capacity(num_tokens_per_rank)
        if target_capacity > self._configured_num_max_dispatch_tokens_per_rank:
            logger.warning(
                "HybridEP token capacity overflow: growing template capacity "
                "from %d to %d tokens per rank. This triggers a one-time larger "
                "JIT specialization.",
                self._configured_num_max_dispatch_tokens_per_rank,
                target_capacity,
            )
        self._buffer = _HybridEPSharedBuffer.get_or_create(
            group=self.group,
            hidden_size=self.hidden_size,
            num_local_experts=self.num_local_experts,
            max_num_tokens_per_rank=target_capacity,
        )
        if self._buffer.config.max_num_of_tokens_per_rank < target_capacity:
            self._buffer.update_template_config(
                hidden_dim=self.hidden_size,
                num_of_tokens_per_rank=target_capacity,
                num_local_experts=self.num_local_experts,
                use_fp8=False,
            )
        return self._buffer

    def _get_group_max_num_tokens(
        self, local_num_tokens: int, device: torch.device
    ) -> int:
        """Get a rank-consistent token count for HybridEP collectives.

        HybridEP's metadata preprocessing all-gathers the dense routing map, so
        every rank in the EP group must enter with the same number of rows.
        Unlike DeepEP, variable per-rank token counts are not supported natively.
        """
        num_tokens_tensor = torch.tensor(
            [local_num_tokens], device=device, dtype=torch.int32
        )
        torch.distributed.all_reduce(
            num_tokens_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=self.group,
        )
        return int(num_tokens_tensor.item())

    def _get_target_num_tokens_per_rank(
        self, local_num_tokens: int, device: torch.device
    ) -> int:
        """Choose the runtime token shape for the current batch.

        Prefill in DP-attention serving is already max-padded to the configured
        chunk size, so avoid an extra all-reduce and directly use that fixed
        template. Decode retains the dynamic group-wide max because token counts
        genuinely diverge across ranks there.
        """
        if (
            get_is_extend_in_batch()
            and local_num_tokens <= self._configured_num_max_dispatch_tokens_per_rank
        ):
            return self._configured_num_max_dispatch_tokens_per_rank

        return max(
            self._MIN_SAFE_RUNTIME_MAX_TOKENS_PER_RANK,
            self._get_group_max_num_tokens(local_num_tokens, device),
        )

    def _pad_dispatch_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        target_num_tokens_per_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._original_num_tokens = hidden_states.shape[0]
        if hidden_states.shape[0] >= target_num_tokens_per_rank:
            return hidden_states, topk_ids, topk_weights

        pad = target_num_tokens_per_rank - hidden_states.shape[0]
        hidden_states = torch.cat(
            [
                hidden_states,
                torch.zeros(
                    (pad, self.hidden_size),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                ),
            ],
            dim=0,
        )
        topk_ids = torch.cat(
            [
                topk_ids,
                torch.full(
                    (pad, self.router_topk),
                    -1,
                    dtype=topk_ids.dtype,
                    device=topk_ids.device,
                ),
            ],
            dim=0,
        )
        topk_weights = torch.cat(
            [
                topk_weights,
                torch.zeros(
                    (pad, self.router_topk),
                    dtype=topk_weights.dtype,
                    device=topk_weights.device,
                ),
            ],
            dim=0,
        )
        return hidden_states, topk_ids, topk_weights

    def _build_dense_routing_map(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _should_use_fused_dense_routing_map(topk_ids.device):
            return self._build_dense_routing_map_fused(topk_ids, topk_weights)
        return self._build_dense_routing_map_torch(topk_ids, topk_weights)

    def _build_dense_routing_map_torch(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _build_dense_routing_map_torch(topk_ids, topk_weights, self.num_experts)

    def _build_dense_routing_map_fused(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _build_dense_routing_map_fused(topk_ids, topk_weights, self.num_experts)

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> HybridEPDispatchOutput:
        target_num_tokens_per_rank = self._get_target_num_tokens_per_rank(
            hidden_states.shape[0],
            hidden_states.device,
        )
        hidden_states, topk_ids, topk_weights = self._pad_dispatch_inputs(
            hidden_states,
            topk_output.topk_ids.to(torch.int64),
            topk_output.topk_weights,
            target_num_tokens_per_rank,
        )
        routing_map, probs = self._build_dense_routing_map(topk_ids, topk_weights)

        buffer = self._ensure_buffer(target_num_tokens_per_rank)
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            self._handle,
        ) = buffer.dispatch_with_permute(
            hidden=hidden_states,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=self.num_local_experts,
            pad_multiple=None,
        )
        assert dispatched_scaling_factor is None, (
            "HybridEP BF16 path should not produce scaling factors."
        )

        return HybridEPDispatchOutput(
            hidden_states=dispatched_hidden,
            routing_weights=dispatched_probs,
            tokens_per_expert=tokens_per_expert,
        )

    def combine(self, combine_input: HybridEPCombineInput) -> torch.Tensor:
        if self._buffer is None or self._handle is None:
            raise RuntimeError("HybridEP combine called before dispatch.")

        hidden_states, _ = self._buffer.combine_with_unpermute(
            hidden=combine_input.hidden_states,
            probs=None,
            handle=self._handle,
            pad_multiple=None,
        )
        hidden_states = hidden_states[: self._original_num_tokens]

        self._handle = None
        self._original_num_tokens = 0
        return hidden_states
