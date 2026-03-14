from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
import triton

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FlashInferFusedMoE,
    FusedMoE,
    moe_forward_piecewise_cuda_graph_impl,
)
from sglang.srt.layers.moe.rocm_moe_utils import upscale
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.token_dispatcher.moriep import MoriEPNormalCombineInput
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    NPUCompressedTensorsW4A16Int4DynamicMoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.quark.schemes import QuarkW4A4MXFp4MoE
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu, is_cuda
from sglang.srt.layers.dp_attention import get_is_extend_in_batch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferDispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
elif _is_npu:
    import torch_npu


logger = logging.getLogger(__name__)


if _is_npu:
    import torch_npu


_expert_load_logger_cached = None


def _maybe_log_expert_load(moe_layer, dispatch_output):
    """Record expert loads with zero overhead when disabled."""
    global _expert_load_logger_cached
    if _expert_load_logger_cached is None:
        from sglang.srt.layers.moe.expert_load_logger import ExpertLoadLogger

        _expert_load_logger_cached = ExpertLoadLogger.get()
    ell = _expert_load_logger_cached
    if not ell.enabled:
        return

    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        num_recv_tokens_per_expert = dispatch_output.num_recv_tokens_per_expert
    elif DispatchOutputChecker.format_is_hybridep(dispatch_output):
        tokens_per_expert = dispatch_output.tokens_per_expert
        if isinstance(tokens_per_expert, torch.Tensor):
            num_recv_tokens_per_expert = tokens_per_expert.tolist()
        else:
            num_recv_tokens_per_expert = list(tokens_per_expert)
    else:
        return

    num_local_experts = getattr(
        moe_layer, "num_local_physical_experts", moe_layer.num_local_experts
    )
    ell.init_tensors(
        num_local_experts=num_local_experts,
        ep_rank=moe_layer.moe_ep_rank,
        ep_size=moe_layer.moe_ep_size,
        global_rank=(
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        ),
    )
    ell.record(
        layer_id=moe_layer.layer_id,
        num_recv_tokens_per_expert=num_recv_tokens_per_expert,
    )


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        if _use_aiter or _is_npu:
            self.deprecate_flag = False
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False

        self.deepep_mode = get_deepep_mode()

        # UltraEP online load balancing
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        self.ultra_ep_enabled = (
            server_args is not None
            and getattr(server_args, "ultra_ep_num_redundant_per_rank", 0) > 0
            and getattr(server_args, "enable_ultra_ep", False)
        )
        self.ultra_ep_mgr = None
        self._ultra_ep_ws_event = None
        self._ultra_ep_master_registered = False

        if self.ultra_ep_enabled:
            assert not self.use_fp8_w8a8, (
                "UltraEP only supports bf16 models. FP8 quantization is not compatible."
            )
            assert not self.use_w4afp8, (
                "UltraEP only supports bf16 models. W4AFP8 quantization is not compatible."
            )

            from sglang.srt.layers.moe.ep_moe.ultra_ep_manager import (
                get_or_create_ultra_ep_manager,
            )
            from sglang.srt.distributed.parallel_state import get_tp_group

            num_redundant_per_rank = server_args.ultra_ep_num_redundant_per_rank

            self.ultra_ep_mgr = get_or_create_ultra_ep_manager(
                ep_group=get_tp_group().device_group,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=self.intermediate_size_per_partition,
                num_redundant_experts_per_rank=num_redundant_per_rank,
            )

            self.num_local_master_experts = self.num_local_experts
            self.num_local_physical_experts = (
                self.ultra_ep_mgr.num_local_physical_experts
            )

            w13_views, w2_views = self.ultra_ep_mgr.get_redundant_weight_views()
            self.eplb_w13_redundant = w13_views
            self.eplb_w2_redundant = w2_views

            # Override dispatcher expert counts so dispatch routes to physical experts.
            # moe_runner_config was already set by FusedMoE.__init__; update and
            # re-create the dispatcher with physical expert counts.
            self.moe_runner_config.num_experts = (
                self.ultra_ep_mgr.num_global_physical_experts
            )
            self.moe_runner_config.num_local_experts = (
                self.ultra_ep_mgr.num_local_physical_experts
            )
            from sglang.srt.layers.moe.fused_moe_triton.layer import (
                create_moe_dispatcher,
            )
            self.dispatcher = create_moe_dispatcher(self.moe_runner_config)

        if (
            self.deepep_mode.enable_low_latency()
            and not _is_npu
            and not (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            )
        ):
            # NPU supports low_latency deepep without deepgemm
            # FP4 quantization with flashinfer_cutedsl also supports low_latency deepep without deepgemm
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        if _use_aiter:
            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1, but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        if is_in_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for piecewise cuda graph"
            return moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def _ultra_ep_pre_dispatch(
        self, topk_output: TopKOutput
    ) -> TopKOutput:
        """Run UltraEP placement update, weight sync, and reroute before dispatch."""
        topk_ids = topk_output.topk_ids.to(torch.int64).clone()
        is_prefill = get_is_extend_in_batch()

        if is_prefill:
            # Lazily register master weight pointers after model loading
            if not self._ultra_ep_master_registered:
                self.ultra_ep_mgr.register_master_weights(
                    self.layer_id, self.w13_weight, self.w2_weight
                )
                self._ultra_ep_master_registered = True

            self.ultra_ep_mgr.update_placement_sparse(self.layer_id, topk_ids)
            self._ultra_ep_ws_event = self.ultra_ep_mgr.weight_sync(
                self.layer_id, async_finish=True
            )

        self.ultra_ep_mgr.reroute_sparse(self.layer_id, topk_ids)

        if is_prefill and self._ultra_ep_ws_event is not None:
            self._ultra_ep_ws_event.current_stream_wait()
            self._ultra_ep_ws_event = None

        return StandardTopKOutput(
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_ids,
            router_logits=getattr(topk_output, "router_logits", None),
        )

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        if self.ultra_ep_enabled:
            topk_output = self._ultra_ep_pre_dispatch(topk_output)

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        hidden_states = self.dispatcher.combine(
            combine_input=combine_input,
        )

        return hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):
        _maybe_log_expert_load(self, dispatch_output)

        if self.deprecate_flag:
            return super().run_moe_core(
                dispatch_output,
            )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_aiter(dispatch_output)
        elif _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_npu(dispatch_output)
        elif DispatchOutputChecker.format_is_flashinfer(dispatch_output):
            return self._run_moe_core_flashinfer(dispatch_output)
        elif DispatchOutputChecker.format_is_hybridep(dispatch_output):
            return self._run_moe_core_hybridep(dispatch_output)
        elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            elif not self.use_fp8_w8a8:
                output = self.forward_deepep_normal_bf16(dispatch_output)
            else:
                assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                output = self.forward_flashinfer_cutedsl(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            elif not self.use_fp8_w8a8:
                output = self.forward_deepep_ll_bf16(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )
        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def _run_moe_core_flashinfer(self, dispatch_output):
        """Handle FlashInfer A2A dispatch output with BF16 grouped GEMM.

        FlashInfer A2A preserves global expert IDs in the dispatch payload,
        whereas the grouped GEMM kernels expect local expert IDs in
        [0, num_local_experts).  We remap here: local experts are shifted to
        0-based indices, non-local expert slots are set to -1.

        Each rank computes a partial weighted sum over its local experts only.
        FlashinferDispatcher.combine routes these partial sums back to the
        original sender ranks and accumulates them into the final output.
        """
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferCombineInput,
        )

        hidden_states = dispatch_output.hidden_states
        topk_ids = dispatch_output.topk_output.topk_ids
        topk_weights = dispatch_output.topk_output.topk_weights

        assert not self.use_fp8_w8a8, (
            "FP8 models are not supported with FlashInfer A2A bf16 grouped GEMM path"
        )
        assert not self.use_w4afp8, (
            "W4AFP8 models are not supported with FlashInfer A2A bf16 grouped GEMM path"
        )

        # Remap global expert IDs → local (0-based).
        # FlashInfer A2A keeps global IDs in the payload; DeepEP converts to
        # local during dispatch. The histogram / scatter kernels in
        # deepep_grouped_gemm_preprocess index into counts[eid] of size
        # num_local_experts, so global IDs would cause OOB access.
        num_local = self._get_num_compute_experts()
        local_start = self.moe_ep_rank * num_local
        shifted = topk_ids - local_start
        topk_ids = torch.where(
            (shifted >= 0) & (shifted < num_local),
            shifted,
            -1,
        )

        output = self._run_bf16_grouped_gemm(hidden_states, topk_ids, topk_weights)
        return FlashinferCombineInput(hidden_states=output)

    def _run_moe_core_hybridep(self, dispatch_output):
        """Handle HybridEP fused-permute dispatch with BF16 grouped GEMM.

        HybridEP returns tokens already laid out as contiguous expert segments, so
        MoE compute must consume the pre-permuted layout directly. Routing
        weights are applied after the second GEMM; for BF16 inference without
        expert bias this is equivalent to weighting before the combine step.
        """
        from sglang.srt.layers.moe.token_dispatcher.hybridep import (
            HybridEPCombineInput,
        )

        assert not self.use_fp8_w8a8, (
            "FP8 models are not supported with HybridEP bf16 grouped GEMM path"
        )
        assert not self.use_w4afp8, (
            "W4AFP8 models are not supported with HybridEP bf16 grouped GEMM path"
        )

        output = self._run_bf16_grouped_gemm_prepermuted(
            dispatch_output.hidden_states,
            dispatch_output.tokens_per_expert,
            dispatch_output.routing_weights,
        )
        return HybridEPCombineInput(hidden_states=output)

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_aiter(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        hidden_states, topk_ids, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if hidden_states.shape[0] == 0:
            return hidden_states
        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_ids_copy = topk_ids.to(torch.int32)
        topk_ids_copy[topk_ids_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_ids_copy,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            quant_type=QuantType.per_128x128,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
        )

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=(hidden_states, hidden_states_scale),
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
        )
        return output

    def _get_expert_weight(self, expert_id: int):
        """Return (w13, w2) weight tensors for the given local physical expert."""
        if not self.ultra_ep_enabled or expert_id < self.num_local_master_experts:
            return self.w13_weight[expert_id], self.w2_weight[expert_id]
        r = expert_id - self.num_local_master_experts
        return self.eplb_w13_redundant[r], self.eplb_w2_redundant[r]

    def _get_num_compute_experts(self) -> int:
        if self.ultra_ep_enabled:
            return self.num_local_physical_experts
        return self.num_local_experts

    def _ensure_weight_ptrs_registered(self):
        """Lazily build device pointer tables for grouped GEMM.

        Collects data_ptr() of each expert's weight (both master and UltraEP
        redundant) into int64 device tensors.  Called once; addresses are stable
        after model loading since parameters and NVSHMEM buffers don't move.
        """
        if getattr(self, "_w13_ptrs", None) is not None:
            return

        from sglang.srt.layers.moe.ep_moe.triton_grouped_gemm import (
            build_weight_ptr_table,
        )

        self._w13_ptrs = build_weight_ptr_table(
            self.w13_weight,
            self.eplb_w13_redundant if self.ultra_ep_enabled else None,
        )
        self._w2_ptrs = build_weight_ptr_table(
            self.w2_weight,
            self.eplb_w2_redundant if self.ultra_ep_enabled else None,
        )

        self._w13_N = self.w13_weight.shape[1]  # 2 * intermediate_size
        self._w13_K = self.w13_weight.shape[2]  # hidden_size
        self._w2_N = self.w2_weight.shape[1]  # hidden_size
        self._w2_K = self.w2_weight.shape[2]  # intermediate_size

    def _build_seg_indptr_from_tokens_per_expert(
        self,
        tokens_per_expert: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        num_compute_experts = self._get_num_compute_experts()
        tokens_per_expert = tokens_per_expert.to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        if tokens_per_expert.numel() != num_compute_experts:
            raise ValueError(
                "HybridEP tokens_per_expert size does not match local physical expert count"
            )

        seg_indptr = torch.empty(
            num_compute_experts + 1, device=device, dtype=torch.int64
        )
        seg_indptr[0] = 0
        seg_indptr[1:] = torch.cumsum(tokens_per_expert, dim=0)
        return seg_indptr

    def _run_bf16_grouped_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Core BF16 grouped GEMM: permute → gate-up → SiLU·mul → down → unpermute.

        Shared by DeepEP normal and FlashInfer A2A paths. Fully on-device with
        zero host-device synchronization. The permute/unpermute kernels skip
        entries with topk_ids == -1 (invalid/padded slots), so both compact
        (DeepEP) and padded (FlashInfer) input layouts are handled correctly.
        """
        from sglang.srt.layers.moe.ep_moe.kernels import (
            deepep_grouped_gemm_preprocess,
            deepep_permute_triton_kernel,
            deepep_post_reorder_triton_kernel,
        )
        from sglang.srt.layers.moe.ep_moe.triton_grouped_gemm import (
            grouped_gemm_bf16,
        )

        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            return hidden_states

        hidden_size = hidden_states.shape[1]
        top_k = topk_ids.shape[1]
        num_compute_experts = self._get_num_compute_experts()

        self._ensure_weight_ptrs_registered()

        src2dst, seg_indptr = deepep_grouped_gemm_preprocess(
            topk_ids, num_compute_experts
        )

        # Upper-bound buffer size — avoids the D2H that seg_indptr[E].item()
        # would incur.  The grouped GEMM kernel only touches rows within each
        # expert's segment so unused rows are harmless.
        max_tokens = topk_ids.numel()

        # --- Permute input tokens by expert assignment ---
        gateup_input = torch.empty(
            (max_tokens, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)
        deepep_permute_triton_kernel[(num_tokens,)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            topk_ids,  # dummy pointer for unused a1_scales_ptr
            top_k,
            hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # --- GEMM 1: gate-up projection  [M_g, H] @ [2I, H]^T = [M_g, 2I] ---
        gate_up = torch.empty(
            (max_tokens, self._w13_N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        grouped_gemm_bf16(
            gateup_input,
            self._w13_ptrs,
            gate_up,
            seg_indptr,
            num_compute_experts,
            N=self._w13_N,
            K=self._w13_K,
        )

        # --- SiLU activation + element-wise multiply ---
        intermediate = torch.empty(
            (max_tokens, self._w2_K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        _is_cuda_device = is_cuda()
        if _is_cuda_device:
            from sgl_kernel import silu_and_mul as _silu_and_mul

            _silu_and_mul(gate_up, intermediate)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = F.silu(gate) * up

        # --- GEMM 2: down projection  [M_g, I] @ [H, I]^T = [M_g, H] ---
        # Reuse the gateup_input buffer (same shape, no longer needed)
        down_output = gateup_input
        grouped_gemm_bf16(
            intermediate,
            self._w2_ptrs,
            down_output,
            seg_indptr,
            num_compute_experts,
            N=self._w2_N,
            K=self._w2_K,
        )

        # --- Post-reorder: gather expert outputs with routing weights ---
        output = torch.zeros(
            (num_tokens, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        deepep_post_reorder_triton_kernel[(num_tokens,)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            top_k,
            hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output

    def _run_bf16_grouped_gemm_prepermuted(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        routing_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """BF16 grouped GEMM for input already permuted by expert.

        HybridEP dispatch performs both communication and expert-side permute, so
        local expert segments are already contiguous. We only need segmented GEMM
        and a final per-row routing-weight multiply before combine.
        """
        from sglang.srt.layers.moe.ep_moe.triton_grouped_gemm import (
            grouped_gemm_bf16,
        )

        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            return hidden_states

        num_compute_experts = self._get_num_compute_experts()
        self._ensure_weight_ptrs_registered()
        seg_indptr = self._build_seg_indptr_from_tokens_per_expert(
            tokens_per_expert, hidden_states.device
        )

        gate_up = torch.empty(
            (num_tokens, self._w13_N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        grouped_gemm_bf16(
            hidden_states,
            self._w13_ptrs,
            gate_up,
            seg_indptr,
            num_compute_experts,
            N=self._w13_N,
            K=self._w13_K,
        )

        intermediate = torch.empty(
            (num_tokens, self._w2_K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        _is_cuda_device = is_cuda()
        if _is_cuda_device:
            from sgl_kernel import silu_and_mul as _silu_and_mul

            _silu_and_mul(gate_up, intermediate)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = F.silu(gate) * up

        output = torch.empty(
            (num_tokens, self._w2_N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        grouped_gemm_bf16(
            intermediate,
            self._w2_ptrs,
            output,
            seg_indptr,
            num_compute_experts,
            N=self._w2_N,
            K=self._w2_K,
        )

        if routing_weights is not None:
            output.mul_(
                routing_weights.to(device=output.device, dtype=output.dtype).unsqueeze(
                    -1
                )
            )

        return output

    def forward_deepep_normal_bf16(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        (
            hidden_states,
            _hidden_states_scale,
            topk_ids,
            topk_weights,
            _num_recv_tokens_per_expert,
        ) = dispatch_output
        return self._run_bf16_grouped_gemm(hidden_states, topk_ids, topk_weights)

    def forward_deepep_ll_bf16(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        """BF16 MoE forward for DeepEP low-latency mode.

        The LL dispatch produces a [num_experts, max_tokens, hidden_size] layout
        with masked_m indicating actual token counts per expert.
        """
        (
            hidden_states,
            _hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        ) = dispatch_output

        if hidden_states.shape[0] == 0:
            return hidden_states

        num_experts = self.num_local_experts
        hidden_size = hidden_states.shape[-1]
        intermediate_size = self.w2_weight.shape[2]

        _is_cuda_device = is_cuda()
        if _is_cuda_device:
            from sgl_kernel import silu_and_mul as _silu_and_mul

        output = torch.zeros_like(hidden_states)

        for expert_id in range(num_experts):
            count = int(masked_m[expert_id].item())
            if count == 0:
                continue

            expert_input = hidden_states[expert_id, :count]
            gate_up = torch.mm(
                expert_input, self.w13_weight[expert_id].t()
            )

            if _is_cuda_device:
                intermediate = torch.empty(
                    (count, intermediate_size),
                    device=gate_up.device,
                    dtype=gate_up.dtype,
                )
                _silu_and_mul(gate_up, intermediate)
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = F.silu(gate) * up

            output[expert_id, :count] = torch.mm(
                intermediate, self.w2_weight[expert_id].t()
            )

        return output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        assert (
            envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ), "W4AFP8 does not support FP8 dispatch; please set SGLANG_DEEPEP_BF16_DISPATCH=1."
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_npu(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
            npu_fused_moe_without_routing_weights_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPNormalDispatchOutput)
            hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
                dispatch_output
            )

            group_list = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=hidden_states.device,
            )

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                input_quant = get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT")
                if not input_quant and not isinstance(
                    self.quant_method, NPUCompressedTensorsW4A16Int4DynamicMoE
                ):
                    hidden_states, hidden_states_scale = torch_npu.npu_dynamic_quant(
                        hidden_states
                    )
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLDispatchOutput)
            (
                hidden_states,
                hidden_states_scale,
                topk_ids,
                topk_weights,
                group_list,
                _,
            ) = dispatch_output

            group_list = group_list.to(torch.int64)

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        else:
            raise ValueError(f"Not Supported DeepEP format {dispatch_output.format}")

        return hidden_states


class NpuFuseEPMoE(DeepEPMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )

        self.quant_method.process_weights_after_loading = (
            self._process_weights_after_loading
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
            gmm1_permuted_weight=self.w13_weight,
            gmm1_permuted_weight_scale=self.w13_weight_scale,
            gmm2_weight=self.w2_weight,
            gmm2_weight_scale=self.w2_weight_scale,
        ).hidden_state

    def permute_w13_weight_scale(self, w: torch.Tensor, tile_n: int):
        if tile_n % 2 != 0:
            raise ValueError(f"tile_n must be even, got {tile_n}")

        *dims, n = w.shape
        if n % tile_n != 0:
            raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

        w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)

        # Permute the last two dimensions.
        perm_order = list(range(len(dims))) + [-2, -3, -1]
        w_permuted = w_reshaped.permute(perm_order)

        return w_permuted.reshape(*dims, n)

    def reshape_w13_weight(self, weight: torch.Tensor, dim: int, chunk_size: int = 64):
        # Achieving greater computing power through reshape on Ascend.
        original_shape = weight.shape
        if dim < 0:
            dim += len(original_shape)

        if original_shape[dim] % (2 * chunk_size) != 0:
            raise ValueError(
                f"Dimension {dim} size {original_shape[dim]} must be divisible by {2 * chunk_size}"
            )

        new_shape = (
            *original_shape[:dim],
            2,
            original_shape[dim] // (2 * chunk_size),
            chunk_size,
            *original_shape[dim + 1 :],
        )

        weight = weight.view(new_shape)
        weight = weight.transpose(dim, dim + 1).contiguous()

        return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])

    def _process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        cpu_w13 = layer.w13_weight.transpose(1, 2).cpu()
        w13 = self.reshape_w13_weight(cpu_w13, -1).npu()
        w13 = npu_format_cast(w13)
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)

        w2 = npu_format_cast(layer.w2_weight)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)

        w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
        w13_scale = self.permute_w13_weight_scale(w13_scale, 128)
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_scale.to(torch.float32), requires_grad=False
        )

        w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale.to(torch.float32), requires_grad=False
        )

        if hasattr(layer, "w13_weight_offset"):
            layer.w13_weight_offset = torch.nn.Parameter(
                layer.w13_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )
        if hasattr(layer, "w2_weight_offset"):
            layer.w2_weight_offset = torch.nn.Parameter(
                layer.w2_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )


class MoriEPMoE(DeepEPMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )

        assert _use_aiter, "Mori need to be used together with aiter as of now"
        self.expert_mask = torch.zeros(
            (self.num_experts),
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        expert_start_idx = self.moe_ep_rank * self.num_local_experts
        expert_end_idx = expert_start_idx + self.num_local_experts
        self.expert_mask[expert_start_idx:expert_end_idx] = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        num_token = hidden_states.shape[0]
        output_dtype = hidden_states.dtype
        scale = None
        is_fp8_quant = isinstance(self.quant_method, Fp8MoEMethod)
        is_quark_w4a4 = isinstance(self.scheme, QuarkW4A4MXFp4MoE)

        # dispatch
        dispatch_output = self.dispatcher.dispatch(
            hidden_states, topk_output
        )  # , scale=scale)

        (
            dispatch_a1,
            dispatch_scale,
            dispatch_ids,
            dispatch_weights,
            dispatch_recv_token_num,
        ) = dispatch_output

        w13_weight = self.w13_weight
        w2_weight = self.w2_weight

        w13_scale = None
        w2_scale = None

        quant_type = QuantType.No

        if not is_fp8_quant and dispatch_scale is not None:
            dispatch_a1 = upscale(
                dispatch_a1, dispatch_scale, dispatch_recv_token_num, output_dtype
            )
            dispatch_scale = None

        if is_quark_w4a4:
            if hasattr(torch, "float4_e2m1fn_x2"):
                w13_weight = self.w13_weight.view(torch.float4_e2m1fn_x2)
                w2_weight = self.w2_weight.view(torch.float4_e2m1fn_x2)

            w13_scale = self.w13_weight_scale
            w2_scale = self.w2_weight_scale
            quant_type = QuantType.per_1x32

            if hasattr(self.w13_weight, "is_shuffled"):
                w13_weight.is_shuffled = True
                w2_weight.is_shuffled = True
        elif is_fp8_quant:
            if hasattr(self, "w13_weight_scale_inv"):
                w13_scale = self.w13_weight_scale_inv
            if hasattr(self, "w2_weight_scale_inv"):
                w2_scale = self.w2_weight_scale_inv

            quant_type = QuantType.per_128x128

        # [KK TODO] should to call the apply of quant method to handle fused moe
        hidden_states = fused_moe(
            hidden_states=dispatch_a1,
            w1=w13_weight,
            w2=w2_weight,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            a1_scale=dispatch_scale,
            topk_weight=dispatch_weights,
            topk_ids=dispatch_ids,
            quant_type=quant_type,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
            num_local_tokens=dispatch_recv_token_num,
            dtype=output_dtype,
        )

        combine_input_wrapper = MoriEPNormalCombineInput
        combine_input = combine_input_wrapper(
            hidden_states=hidden_states,
            topk_ids=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
        )

        # combine
        result = self.dispatcher.combine(combine_input)

        return result[:num_token]


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    # [TODO] kk, temporary solution
    if get_moe_a2a_backend().is_mori():
        return MoriEPMoE
    if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
        return DeepEPMoE
    if get_moe_a2a_backend().is_flashinfer() or get_moe_a2a_backend().is_hybridep():
        return DeepEPMoE
    if get_moe_a2a_backend().is_ascend_fuseep():
        return NpuFuseEPMoE

    if get_moe_runner_backend().is_flashinfer_trtllm():
        # NEW: Direct FP4 detection (bypasses EP requirements)
        # Check for FP4 quantization with TRTLLM flag, regardless of EP
        # FlashInferFP4MoE must be paired with ModelOptNvFp4FusedMoEMethod.
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            from sglang.srt.layers.moe.fused_moe_triton.layer import FlashInferFP4MoE

            return FlashInferFP4MoE
        elif (
            quant_config is None
            or quant_config.get_name() == "fp8"
            or quant_config.get_name() == "modelopt_fp8"
            or quant_config.get_name() == "compressed_tensors"
        ):
            # FlashInferFusedMoE support bf16, fp8 and compressed_tensors
            return FlashInferFusedMoE

    if get_moe_runner_backend().is_flashinfer_cutlass():
        return FusedMoE
    return FusedMoE
