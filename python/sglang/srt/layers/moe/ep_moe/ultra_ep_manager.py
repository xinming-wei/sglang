"""UltraEP-based online expert load balancing manager for SGLang inference.

Wraps ultra_ep.Manager to provide per-iteration placement updates, weight sync,
and sparse topk rerouting for DeepEP normal mode.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

try:
    import ultra_ep
    from ultra_ep import EventHandle

    HAVE_ULTRA_EP = True
except ImportError:
    HAVE_ULTRA_EP = False


class UltraEPManager:
    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_redundant_experts_per_rank: int,
        num_layers: int,
    ):
        assert HAVE_ULTRA_EP, (
            "ultra_ep package is required for UltraEP integration. "
            "Install from libs/UltraEP."
        )

        self.ep_group = ep_group
        self.rank = ep_group.rank()
        self.num_ranks = ep_group.size()

        self.num_local_master_experts = num_experts // self.num_ranks
        self.num_local_redundant_experts = num_redundant_experts_per_rank
        self.num_local_physical_experts = (
            self.num_local_master_experts + self.num_local_redundant_experts
        )
        self.num_global_logical_experts = num_experts
        self.num_global_physical_experts = (
            self.num_local_physical_experts * self.num_ranks
        )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_fc1_numel = 2 * hidden_size * intermediate_size
        self.expert_fc2_numel = hidden_size * intermediate_size
        self.expert_total_numel = self.expert_fc1_numel + self.expert_fc2_numel

        self.runtime = ultra_ep.Manager(
            group=self.ep_group,
            num_layers=num_layers,
            num_local_master_experts=self.num_local_master_experts,
            num_local_redundant_experts=self.num_local_redundant_experts,
            expert_fc1_numel=self.expert_fc1_numel,
            expert_fc2_numel=self.expert_fc2_numel,
            is_train=False,
            explicitly_destroy=False,
            max_microbatches=1,
            use_gpu_solver=True,
            balance_threshold=0.0,
        )

        self.local_replica_weight_buffer: torch.Tensor = (
            self.runtime.local_replica_weight_buffer
        )

        self._master_weights_registered: Dict[int, bool] = {}

        logger.info(
            f"[UltraEPManager] Initialized: rank={self.rank}/{self.num_ranks}, "
            f"masters={self.num_local_master_experts}, "
            f"redundant={self.num_local_redundant_experts}, "
            f"physical={self.num_local_physical_experts}"
        )

    def register_master_weights(
        self,
        layer_id: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ):
        """Register master expert weight pointers for weight_sync.

        Must be called after model weights are loaded so that data_ptr() is final.
        """
        if layer_id in self._master_weights_registered:
            return

        M = self.num_local_master_experts
        fc1_weights = [w13_weight[i] for i in range(M)]
        fc2_weights = [w2_weight[i] for i in range(M)]

        self.runtime.construct_local_master_ptr_pool(
            layer_id=layer_id,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
        )
        self._master_weights_registered[layer_id] = True

    def get_redundant_weight_views(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Create weight views into the shared replica buffer for redundant experts.

        Returns (w13_views, w2_views) where each is a list of R tensors.
        w13_views[r] has shape matching w13_weight[0], backed by NVSHMEM buffer.
        w2_views[r] has shape matching w2_weight[0], backed by NVSHMEM buffer.
        """
        R = self.num_local_redundant_experts
        fc1_numel = self.expert_fc1_numel
        total_numel = self.expert_total_numel
        H = self.hidden_size
        I = self.intermediate_size

        w13_views = []
        w2_views = []
        for r in range(R):
            buf = self.local_replica_weight_buffer[r]
            w13_views.append(buf[:fc1_numel].view(2 * I, H))
            w2_views.append(buf[fc1_numel:total_numel].view(H, I))

        return w13_views, w2_views

    @torch.no_grad()
    def update_placement_sparse(self, layer_id: int, topk_ids: torch.Tensor):
        """Update expert placement from sparse topk_ids [T, K]."""
        self.runtime.update_placement_sparse(layer_id, topk_ids)

    @torch.no_grad()
    def reroute_sparse(self, layer_id: int, topk_ids: torch.Tensor):
        """In-place remap topk_ids from logical to physical expert IDs."""
        self.runtime.reroute_sparse(layer_id, topk_ids)

    def weight_sync(
        self,
        layer_id: int,
        async_finish: bool = True,
    ) -> Optional[EventHandle]:
        """Sync master weights to replicas via NVSHMEM NVLink.

        Returns an EventHandle when async_finish=True, else None.
        """
        event = self.runtime.weight_sync(
            layer_id=layer_id,
            async_finish=async_finish,
        )
        return EventHandle(event) if event else None


# Global registry keyed by EP group identity
_ultra_ep_manager_registry: Dict[int, UltraEPManager] = {}


def _get_num_hidden_layers() -> int:
    """Retrieve num_hidden_layers from the global model config."""
    try:
        from sglang.srt.model_loader.model_runner_mapping import (
            get_global_model_config,
        )
        mc = get_global_model_config()
        if mc is not None:
            return mc.hf_text_config.num_hidden_layers
    except Exception:
        pass
    return 128  # safe default


def get_or_create_ultra_ep_manager(
    ep_group: dist.ProcessGroup,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_redundant_experts_per_rank: int,
    num_layers: Optional[int] = None,
) -> UltraEPManager:
    key = id(ep_group)
    if key not in _ultra_ep_manager_registry:
        if num_layers is None:
            num_layers = _get_num_hidden_layers()
        _ultra_ep_manager_registry[key] = UltraEPManager(
            ep_group=ep_group,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_redundant_experts_per_rank=num_redundant_experts_per_rank,
            num_layers=num_layers,
        )
    return _ultra_ep_manager_registry[key]
