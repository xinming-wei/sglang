"""Low-overhead expert load logging for MoE inference benchmarking.

Records per-expert token counts during serving into pre-allocated CPU tensors.
Each rank saves its own local expert data independently; the plotting script
combines per-rank files to produce the global view (no runtime all-reduce).
By default only prefill loads are recorded; decode recording is optional.

Environment variables:
    SGLANG_ENABLE_EXPERT_LOAD_LOGGING : str
        Set to "1" to enable. Default: disabled.
    SGLANG_EXPERT_LOAD_PREFILL_ONLY : str
        Set to "1" to record prefill only, "0" to also record decode.
        Default: "1".
    SGLANG_EXPERT_LOAD_LOG_PATH : str
        Directory to save load data. Default: "./expert_loads"
    SGLANG_EXPERT_LOAD_MAX_STEPS : int
        Upper bound on total forward steps to record. Default: 100000
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from typing import List, Optional
import numpy as np
import torch
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_MAX_LAYERS = 128


class ExpertLoadLogger:
    _instance: Optional[ExpertLoadLogger] = None

    @classmethod
    def get(cls, prefill_only: Optional[bool] = None) -> ExpertLoadLogger:
        if cls._instance is not None:
            if prefill_only is not None and prefill_only != cls._instance.prefill_only:
                logger.warning(
                    "ExpertLoadLogger already initialised with prefill_only=%s; "
                    "ignoring requested prefill_only=%s",
                    cls._instance.prefill_only,
                    prefill_only,
                )
            return cls._instance

        if prefill_only is None:
            prefill_only = get_bool_env_var(
                "SGLANG_EXPERT_LOAD_PREFILL_ONLY", default="true"
            )
        cls._instance = cls(prefill_only=prefill_only)
        return cls._instance

    def __init__(self, prefill_only: bool = True):
        self.prefill_only = prefill_only
        self.record_decode = not prefill_only
        self.enabled = os.environ.get("SGLANG_ENABLE_EXPERT_LOAD_LOGGING", "0") == "1"
        if not self.enabled:
            return

        self.max_steps = int(
            os.environ.get("SGLANG_EXPERT_LOAD_MAX_STEPS", "100000")
        )
        self.save_path = os.environ.get(
            "SGLANG_EXPERT_LOAD_LOG_PATH", "./expert_loads"
        )

        self._initialized = False
        self._saving = False
        self._current_is_prefill = True
        self._record_current_forward = True
        self._prefill_idx = 0
        self._decode_idx = 0
        self._max_layer_id = 0
        self._lock = threading.Lock()

        os.makedirs(self.save_path, exist_ok=True)
        atexit.register(self._atexit_save)

        self._stop_event = threading.Event()
        self._trigger_thread = threading.Thread(
            target=self._watch_trigger, daemon=True
        )
        self._trigger_thread.start()

    # ------------------------------------------------------------------
    # Lazy initialisation (called on first record from the MoE layer)
    # ------------------------------------------------------------------

    def init_tensors(
        self,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        global_rank: int,
    ):
        if self._initialized or not self.enabled:
            return

        self._num_local_experts = num_local_experts
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._global_rank = global_rank

        self.prefill_loads = np.zeros(
            (self.max_steps, _MAX_LAYERS, num_local_experts), dtype=np.int32
        )
        self.decode_loads = None
        if self.record_decode:
            self.decode_loads = np.zeros(
                (self.max_steps, _MAX_LAYERS, num_local_experts), dtype=np.int32
            )

        self._initialized = True
        logger.info(
            "ExpertLoadLogger initialised: global_rank=%d, ep_rank=%d/%d, "
            "num_local_experts=%d, max_steps=%d, prefill_only=%s, save_path=%s",
            global_rank,
            ep_rank,
            ep_size,
            num_local_experts,
            self.max_steps,
            self.prefill_only,
            self.save_path,
        )

    # ------------------------------------------------------------------
    # Per-forward hooks (called from model_runner)
    # ------------------------------------------------------------------

    def on_forward_start(self, is_prefill: bool):
        if not self.enabled:
            return
        self._current_is_prefill = is_prefill
        self._record_current_forward = is_prefill or self.record_decode

    def should_record_current_forward(self) -> bool:
        return self.enabled and self._record_current_forward

    def on_forward_end(self):
        if (
            not self.enabled
            or not self._initialized
            or not self._record_current_forward
        ):
            return
        if self._current_is_prefill:
            self._prefill_idx += 1
        else:
            self._decode_idx += 1

    # ------------------------------------------------------------------
    # Per-layer record (called from DeepEPMoE.run_moe_core)
    # ------------------------------------------------------------------

    def record(self, layer_id: int, num_recv_tokens_per_expert: List[int]):
        if (
            not self.enabled
            or not self._initialized
            or not self._record_current_forward
        ):
            return

        if layer_id > self._max_layer_id:
            self._max_layer_id = layer_id

        n = len(num_recv_tokens_per_expert)
        if self._current_is_prefill:
            idx = self._prefill_idx
            if idx < self.max_steps:
                self.prefill_loads[idx, layer_id, :n] = num_recv_tokens_per_expert
        else:
            idx = self._decode_idx
            if idx < self.max_steps and self.decode_loads is not None:
                self.decode_loads[idx, layer_id, :n] = num_recv_tokens_per_expert

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self):
        if not self.enabled or not self._initialized:
            return
        with self._lock:
            if self._saving:
                return
            self._saving = True
        try:
            self._do_save()
        except Exception as e:
            logger.error("ExpertLoadLogger save failed: %s", e, exc_info=True)
        finally:
            with self._lock:
                self._saving = False

    def _do_save(self):
        os.makedirs(self.save_path, exist_ok=True)

        num_layers = self._max_layer_id + 1
        num_prefill = min(self._prefill_idx, self.max_steps)
        num_decode = min(self._decode_idx, self.max_steps)

        prefill_data = self.prefill_loads[:num_prefill, :num_layers, :]
        decode_data = None
        if self.decode_loads is not None:
            decode_data = self.decode_loads[:num_decode, :num_layers, :]

        metadata = {
            "ep_rank": int(self._ep_rank),
            "ep_size": int(self._ep_size),
            "global_rank": int(self._global_rank),
            "num_local_experts": int(self._num_local_experts),
            "num_layers": int(num_layers),
            "num_prefill_steps": int(num_prefill),
            "num_decode_steps": int(num_decode),
            "prefill_only": bool(self.prefill_only),
            "record_decode": bool(self.record_decode),
        }

        rank = self._global_rank
        arrays = {"prefill_loads": prefill_data}
        if decode_data is not None:
            arrays["decode_loads"] = decode_data
        np.savez_compressed(
            os.path.join(self.save_path, f"rank_{rank}_expert_loads.npz"),
            **arrays,
        )

        with open(
            os.path.join(self.save_path, f"rank_{rank}_metadata.json"), "w"
        ) as f:
            json.dump(metadata, f, indent=2)

        self._write_summary(metadata, prefill_data, decode_data)

        logger.info("ExpertLoadLogger: saved to %s (rank %d)", self.save_path, rank)

    def _write_summary(
        self,
        metadata: dict,
        prefill_data: np.ndarray,
        decode_data: Optional[np.ndarray],
    ):
        rank = self._global_rank
        path = os.path.join(self.save_path, f"rank_{rank}_summary.txt")
        with open(path, "w") as f:
            f.write(f"Expert Load Summary  —  Rank {rank}\n")
            f.write("=" * 60 + "\n")
            for k, v in metadata.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

            for name, data in [("Prefill", prefill_data), ("Decode", decode_data)]:
                if data is None:
                    f.write(f"{name}: not recorded\n\n")
                    continue
                if data.shape[0] == 0:
                    f.write(f"{name}: no data\n\n")
                    continue
                total_per_step = data.sum(axis=-1)  # [steps, layers]
                f.write(
                    f"{name} ({data.shape[0]} steps, {data.shape[1]} layers, "
                    f"{data.shape[2]} local experts):\n"
                )
                f.write(
                    f"  Total tokens/step (summed over layers & experts)  "
                    f"min={total_per_step.sum(axis=-1).min():.0f}  "
                    f"mean={total_per_step.sum(axis=-1).mean():.1f}  "
                    f"max={total_per_step.sum(axis=-1).max():.0f}\n"
                )
                per_expert = data.mean(axis=(0, 1))
                f.write(
                    f"  Per-expert avg load (over steps & layers): "
                    f"{[f'{v:.1f}' for v in per_expert]}\n\n"
                )

    # ------------------------------------------------------------------
    # Trigger / cleanup
    # ------------------------------------------------------------------

    def _atexit_save(self):
        self.save()

    def _watch_trigger(self):
        """Background thread: watch for trigger files to initiate saves.

        Loops continuously so multiple bench runs on the same server all get
        their data captured.  A save does NOT prevent future saves — each
        trigger produces a fresh snapshot of all data accumulated so far.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        trigger = os.path.join(self.save_path, ".save_triggers", f"rank_{rank}")
        while not self._stop_event.is_set():
            time.sleep(1)
            if not self.enabled:
                break
            if os.path.exists(trigger):
                logger.info("ExpertLoadLogger: trigger file detected, saving …")
                self.save()
                try:
                    os.remove(trigger)
                except OSError:
                    pass
