"""部署、目标分配与机动共享同一参数集合的混合动作 MAPPO。"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.unified_mappo.model import (
    HybridAction,
    HybridActionMask,
    HybridMAPPOConfig,
    HybridMAPPOTrainer,
    HybridRolloutBatch,
    JointTargetReturn,
)
from experiments.unified_mappo.credit_assignment import (
    normalized_counterfactual_credit,
)
from experiments.unified_mappo.causal_candidates import (
    build_factual_target_events,
)

from .red_policy import (
    GlobalStateEncoder,
    ObservationEncoder,
    PolicyTransition,
    SharedPolicy,
    UNIFIED_LOCAL_OBSERVATION_DIM,
    UNIFIED_TARGET_SLOTS,
    UNIFIED_TEAM_CONTEXT_DIM,
)


TARGET_SLOTS = UNIFIED_TARGET_SLOTS
TARGET_GEOMETRY_FEATURES = ObservationEncoder.TARGET_FEATURES
TARGET_ALLOCATION_FEATURES = 11
TARGET_SET_FEATURE_DIM = TARGET_GEOMETRY_FEATURES + 3 + TARGET_ALLOCATION_FEATURES
TARGET_HEALTH_SCALE = 16.0
TARGET_FIRE_COUNT_SCALE = 8.0
TEAM_CONTEXT_DIM = UNIFIED_TEAM_CONTEXT_DIM
RED_TYPES = {21000: "H", 21001: "M", 21002: "L"}
SEARCH_TARGET_ID = -100
FIXED_TARGET_SLOT_IDS: tuple[int | None, ...] = (
    51, 52, 53, 54, 106, 168, 169, SEARCH_TARGET_ID,
) + (None,) * (TARGET_SLOTS - 8)
TRAJECTORY_COUNTERFACTUAL_REWARD_MODE = (
    "weighted_damage_trajectory_counterfactual"
)
COUNTERFACTUAL_REWARD_MODES = {
    "weighted_damage_counterfactual",
    "weighted_damage_decision_anchored",
    TRAJECTORY_COUNTERFACTUAL_REWARD_MODE,
}
INDIVIDUAL_REWARD_MODES = {
    "weighted_damage_individual",
    "weighted_damage_guided_cow",
    *COUNTERFACTUAL_REWARD_MODES,
}




@dataclass
class _StoredSample:
    agent_id: int
    entity_id: int
    step: int
    observation: torch.Tensor
    critic_state: torch.Tensor
    target_coordinates: torch.Tensor
    target_valid_mask: torch.Tensor
    target_features: torch.Tensor
    mask_presence: bool
    mask_initial_position: bool
    mask_search_position: bool
    mask_retarget: bool
    mask_target: bool
    mask_maneuver: bool
    mask_satellite: bool
    action: HybridAction
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: float | None
    done: bool
    next_observation: torch.Tensor
    next_critic_state: torch.Tensor
    target_credits: torch.Tensor | None = None
    credit_anchor: bool = False
    search_valid_mask: torch.Tensor | None = None


class UnifiedMAPPOSharedPolicy(SharedPolicy):
    """一个模型对象统一产生生命周期、目标、机动与卫星动作。"""

    def __init__(
        self,
        *,
        seed: int,
        max_steps: int,
        observation_dim: int,
        training: bool,
        model_path: str | None = None,
    ) -> None:
        self.reward_mode = os.getenv("RED_REWARD_MODE", "")
        self.score_only = os.getenv("RED_UNIFIED_SCORE_ONLY", "0") == "1"
        self.native_snapshot_guidance = (
            os.getenv("RED_NATIVE_SNAPSHOT_GUIDANCE", "0") == "1"
        )
        self.trajectory_counterfactual = (
            self.reward_mode == TRAJECTORY_COUNTERFACTUAL_REWARD_MODE
            or self.native_snapshot_guidance
            or self.score_only
        )
        self.replay_probe = os.getenv("RED_TRAJECTORY_CF_PROBE", "0") == "1"
        if self.trajectory_counterfactual and not self.score_only:
            # Native simulator snapshots use OS copy-on-write. Keep the probe
            # single-threaded and CPU-only before policy inference starts.
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        self.local_observation_dim = int(observation_dim)
        if self.trajectory_counterfactual and self.local_observation_dim != UNIFIED_LOCAL_OBSERVATION_DIM:
            raise ValueError(
                f"真实轨迹反事实 MAPPO 的 actor 必须使用 {UNIFIED_LOCAL_OBSERVATION_DIM} 维合法局部观测"
            )
        self.global_encoder = GlobalStateEncoder(
            max_steps=max_steps,
            target_slots=TARGET_SLOTS,
        )
        actor_observation_dim = (
            self.local_observation_dim
            if self.trajectory_counterfactual
            else self.local_observation_dim + TEAM_CONTEXT_DIM
        )
        config = HybridMAPPOConfig(
            observation_dim=actor_observation_dim,
            critic_state_dim=(
                self.global_encoder.state_dim
                if self.trajectory_counterfactual
                else None
            ),
            critic_focal_observation_dim=(
                actor_observation_dim
                if self.trajectory_counterfactual
                else None
            ),
            target_slots=TARGET_SLOTS,
            target_feature_dim=TARGET_SET_FEATURE_DIM,
            hidden_dim=int(os.getenv("RED_UNIFIED_HIDDEN_DIM", "256")),
            learning_rate=float(os.getenv("RED_UNIFIED_LEARNING_RATE", "1e-4")),
            gamma=float(os.getenv("RED_UNIFIED_GAMMA", "0.99")),
            gae_lambda=float(os.getenv("RED_UNIFIED_GAE_LAMBDA", "0.95")),
            clip_ratio=float(os.getenv("RED_UNIFIED_CLIP_RATIO", "0.2")),
            value_coef=float(os.getenv("RED_UNIFIED_VALUE_COEF", "0.5")),
            entropy_coef=float(os.getenv("RED_UNIFIED_ENTROPY_COEF", "0.005")),
            deployment_policy_share=float(
                os.getenv("RED_UNIFIED_DEPLOYMENT_POLICY_SHARE", "0.5")
            ),
            max_grad_norm=float(os.getenv("RED_UNIFIED_MAX_GRAD_NORM", "0.5")),
            update_epochs=int(os.getenv("RED_UNIFIED_UPDATE_EPOCHS", "4")),
            minibatch_size=int(os.getenv("RED_UNIFIED_MINIBATCH_SIZE", "32768")),
            initial_coordinate_log_std=float(
                os.getenv("RED_UNIFIED_INITIAL_LOG_STD", "-2.0")
            ),
            search_grid_width=16,
            search_grid_height=12,
            search_target_index=FIXED_TARGET_SLOT_IDS.index(SEARCH_TARGET_ID),
            counterfactual_value_coef=float(
                os.getenv("RED_UNIFIED_COUNTERFACTUAL_VALUE_COEF", "0.1")
            ),
            seed=seed,
            # Exact stochastic prefix reconstruction and the native COW
            # snapshot require factual and replay actions to use one CPU RNG.
            device=(
                "cpu" if self.trajectory_counterfactual and not self.score_only
                else os.getenv("RED_UNIFIED_DEVICE", "auto")
            ),
        )
        self.trainer = HybridMAPPOTrainer(config)
        if self.replay_probe and self.trainer.device.type != "cpu":
            raise RuntimeError("反事实分叉 probe 必须使用 CPU 推理")
        if model_path and Path(model_path).is_file():
            loaded = HybridMAPPOTrainer.load(
                model_path,
                device=str(self.trainer.device),
                target_feature_dim=TARGET_SET_FEATURE_DIM,
            )
            if loaded.config.observation_dim != actor_observation_dim:
                raise ValueError(
                    f"统一 MAPPO checkpoint 观测维度为 {loaded.config.observation_dim}，"
                    f"当前需要 {actor_observation_dim}"
                )
            if loaded.config.critic_state_dim != config.critic_state_dim:
                raise ValueError(
                    "统一 MAPPO checkpoint 的 centralized critic 状态维度不兼容"
                )
            if loaded.config.critic_focal_observation_dim != config.critic_focal_observation_dim:
                raise ValueError(
                    "统一 MAPPO checkpoint 的 focal-agent critic 观测维度不兼容"
                )
            if loaded.config.target_feature_dim != TARGET_SET_FEATURE_DIM:
                raise ValueError("统一 MAPPO checkpoint 的目标特征维度迁移失败")
            self.trainer = loaded
            torch.manual_seed(seed)
        counterfactual_value_coef = float(
            os.getenv("RED_UNIFIED_COUNTERFACTUAL_VALUE_COEF", "0.1")
        )
        self.trainer.config = replace(
            self.trainer.config,
            counterfactual_value_coef=counterfactual_value_coef,
        )
        self.trainer.model.config = self.trainer.config
        self.training = bool(training)
        self.force_deterministic_actor = (
            os.getenv("RED_UNIFIED_FORCE_DETERMINISTIC_ACTOR", "0") == "1"
        )
        self.stochastic_actor = (
            self.training or self.replay_probe or self.score_only
        ) and not self.force_deterministic_actor
        self.dynamic_lifecycle = os.getenv(
            "RED_UNIFIED_DYNAMIC_LIFECYCLE", "0"
        ) == "1"
        self.temporal_attack_options = os.getenv(
            "RED_UNIFIED_TEMPORAL_ATTACK_OPTIONS", "0"
        ) == "1"
        self.attack_option_min_dwell_steps = max(
            1,
            int(os.getenv("RED_UNIFIED_ATTACK_OPTION_MIN_DWELL_STEPS", "60")),
        )
        self.option_control_handoff = os.getenv(
            "RED_UNIFIED_OPTION_CONTROL_HANDOFF", "0"
        ) == "1"
        self.search_option_chaining = os.getenv(
            "RED_UNIFIED_SEARCH_OPTION_CHAINING", "0"
        ) == "1"
        self.rollout_active_heads_only = os.getenv(
            "RED_UNIFIED_ROLLOUT_ACTIVE_HEADS_ONLY", "0"
        ) == "1"
        self.search_option_dwell_steps = max(
            1,
            int(os.getenv("RED_UNIFIED_SEARCH_OPTION_DWELL_STEPS", "120")),
        )
        self.search_option_reopen_distance_km = max(
            0.0,
            float(os.getenv(
                "RED_UNIFIED_SEARCH_OPTION_REOPEN_DISTANCE_KM", "15.0"
            )),
        )
        self.search_option_emergency_reopen_distance_km = max(
            0.0,
            float(os.getenv(
                "RED_UNIFIED_SEARCH_OPTION_EMERGENCY_REOPEN_DISTANCE_KM", "5.0"
            )),
        )
        self.search_reachable_mask = os.getenv(
            "RED_UNIFIED_SEARCH_REACHABLE_MASK", "0"
        ) == "1"
        self.search_leg_min_distance_km = max(
            0.0,
            float(os.getenv(
                "RED_UNIFIED_SEARCH_LEG_MIN_DISTANCE_KM", "50.0"
            )),
        )
        self.search_leg_max_distance_km = max(
            self.search_leg_min_distance_km,
            float(os.getenv(
                "RED_UNIFIED_SEARCH_LEG_MAX_DISTANCE_KM", "180.0"
            )),
        )
        self.deterministic_evasion = os.getenv(
            "RED_UNIFIED_DETERMINISTIC_EVASION", "0"
        ) == "1"
        self.deterministic_evasion_tcpa_seconds = max(
            0.0,
            float(os.getenv(
                "RED_UNIFIED_DETERMINISTIC_EVASION_TCPA_SECONDS", "120.0"
            )),
        )
        self.deterministic_evasion_dcpa_km = max(
            0.0,
            float(os.getenv(
                "RED_UNIFIED_DETERMINISTIC_EVASION_DCPA_KM", "20.0"
            )),
        )
        self.deterministic_evasion_max_track_age_steps = max(
            0,
            int(os.getenv(
                "RED_UNIFIED_DETERMINISTIC_EVASION_MAX_TRACK_AGE_STEPS", "10"
            )),
        )
        self.deterministic_evasion_pulse_steps = max(
            1,
            int(os.getenv(
                "RED_UNIFIED_DETERMINISTIC_EVASION_PULSE_STEPS", "10"
            )),
        )
        self.deterministic_evasion_cooldown_steps = max(
            0,
            int(os.getenv(
                "RED_UNIFIED_DETERMINISTIC_EVASION_COOLDOWN_STEPS", "10"
            )),
        )
        allocator_trace = os.getenv("RED_UNIFIED_TARGET_ALLOCATOR_TRACE")
        self._target_allocator_trace_path = (
            Path(allocator_trace) if allocator_trace else None
        )
        self._target_allocator_trace_stride = max(
            1,
            int(os.getenv("RED_UNIFIED_TARGET_ALLOCATOR_TRACE_STRIDE", "1")),
        )
        self._target_allocator_trace_rows: list[dict[str, Any]] = []
        if (
            self.reward_mode in {
                "weighted_damage_decision_anchored",
                TRAJECTORY_COUNTERFACTUAL_REWARD_MODE,
            }
            and not self.dynamic_lifecycle
        ):
            raise ValueError(
                "决策时刻信用分配要求 RED_UNIFIED_DYNAMIC_LIFECYCLE=1"
            )
        self.trainer.model.train(self.training)
        self.max_steps = max(1, int(max_steps))
        self._encoders: dict[int, ObservationEncoder] = {}
        self._targets: list[dict[str, Any]] = []
        self._target_slot_ids: list[int | None] = list(FIXED_TARGET_SLOT_IDS)
        self._target_slot_templates: dict[int, dict[str, Any]] = {}
        self._visible_target_ids: frozenset[int] = frozenset()
        self._public_target_ids: frozenset[int] = frozenset()
        self._actor_visible_target_ids_by_entity: dict[int, frozenset[int]] = {}
        self._full_target_runtime_states: dict[int, tuple[float, float, float]] = {}
        self._initial_target_health: dict[int, float] = {}
        self._actor_runtime_target_ids: frozenset[int] = frozenset()
        self._search_bounds = (-180.0, 180.0, -90.0, 90.0)
        self._episode_samples: list[_StoredSample] = []
        self._pending_motion: dict[int, _StoredSample] = {}
        self._prepared_maneuvers: dict[int, int] = {}
        self._deterministic_evasion_state: dict[int, dict[str, int]] = {}
        self._deterministic_evasion_trigger_count = 0
        self._deterministic_evasion_active_step_count = 0
        self._deterministic_evasion_visible_threat_count = 0
        self._prepared_lifecycle: dict[int, dict[str, Any]] = {}
        self._deployment: dict[int, dict[str, Any]] = {}
        self._episode_index = 0
        self._last_episode_score = 0.0
        self._last_ppo_update_device = str(self.device)
        self._current_team_context: np.ndarray | None = None
        self._next_team_context: np.ndarray | None = None
        self._current_global_state: np.ndarray | None = None
        self._next_global_state: np.ndarray | None = None
        self._team_type_totals: dict[int, int] = {}
        self._entity_type_by_entity: dict[int, int] = {}
        self._deployment_legal_count = 0
        self._active_deployment_count = 0
        self._maneuver_legal_count = 0
        self._maneuver_decision_count = 0
        self._alive_inference_expected_count = 0
        self._step_inference_count = 0
        self._waiting_inference_count = 0
        self._last_deployment_transition_count = 0
        self._last_maneuver_transition_count = 0
        self._last_waiting_transition_count = 0
        self._step_sample_start = 0
        self._credit_context: Mapping[str, Any] | None = None
        self._launched_agent_ids: set[int] = set()
        self._counterfactual_step_count = 0
        self._direct_fallback_step_count = 0
        self._delayed_credit_count = 0
        self._max_counterfactual_conservation_error = 0.0
        self._option_anchor_by_agent: dict[int, _StoredSample] = {}
        self._target_anchor_by_agent: dict[int, dict[int, _StoredSample]] = {}
        self._joint_target_returns: list[JointTargetReturn] = []
        self._joint_target_return_index: dict[tuple[int, tuple[int, ...]], int] = {}
        self._current_target_by_agent: dict[int, int] = {}
        self._attack_option_by_entity: dict[int, dict[str, Any]] = {}
        # Targets executed by the native/teacher controller are observation
        # context only. Keep them separate from student-owned attack options
        # so fire-allocation features cannot alter option boundaries or
        # transfer control to the student.
        self._external_target_by_entity: dict[int, int] = {}
        self._external_target_sync_count = 0
        self._external_target_feature_hit_count = 0
        self._external_target_fallback_count = 0
        self._attack_option_boundary_count = 0
        self._attack_option_keep_count = 0
        self._attack_option_retarget_count = 0
        self._attack_option_termination_count = 0
        self._attack_option_termination_reasons: dict[str, int] = {}
        self._search_option_distance_sample_count = 0
        self._search_option_min_distance_km = float("inf")
        self._search_option_last_distance_km: float | None = None
        self._search_option_emergency_boundary_count = 0
        self._search_reachable_mask_sample_count = 0
        self._search_reachable_valid_count_sum = 0
        self._search_reachable_valid_count_min = 16 * 12
        self._search_reachable_valid_count_max = 0
        self._dynamic_launched_ids: set[int] = set()
        self._target_selection_count = 0
        self._target_selection_legal_count = 0
        self._low_invalid_target_count = 0
        self._decision_anchored_assignment_count = 0
        self._decision_anchored_reward_sum = 0.0
        self._decision_anchored_delay_sum = 0
        self._decision_anchored_max_delay = 0
        self._decision_anchored_agent_ids: set[int] = set()
        self._decision_anchor_target_match_count = 0
        self._decision_anchor_indirect_count = 0
        self._decision_anchor_candidate_count = 0
        self._decision_anchor_indirect_candidate_count = 0
        self._causal_decisions: list[dict[str, Any]] = []
        self._simulator_causal_events: list[dict[str, Any]] = []
        self._factual_target_events: list[dict[str, Any]] = []
        self._target_reward_history: list[dict[str, Any]] = []
        self._applied_interventions: list[dict[str, Any]] = []
        self._applied_credit_event_ids: set[Any] = set()
        self._prepared_replay_step: int | None = None
        self._prepared_replay_semantics: dict[int, dict[str, Any]] = {}
        self._replay_branch_role = "prefix" if self.replay_probe else "none"
        self._replay_factual_pid: int | None = None
        self._last_credit_validation: dict[str, Any] = {}
        self._trajectory_credit_finalized = False
        self._guided_cow_unit_count = 0
        self._guided_cow_positive_count = 0
        self._guided_cow_reward_sum = 0.0
        replay_spec_path = os.getenv("RED_CF_SPEC")
        batch_spec_path = os.getenv("RED_CF_BATCH_SPEC")
        if batch_spec_path:
            batch_payload = json.loads(
                Path(batch_spec_path).read_text(encoding="utf-8")
            )
            self._replay_requests = tuple(
                dict(request) for request in batch_payload["requests"]
            )
        elif replay_spec_path:
            request = json.loads(
                Path(replay_spec_path).read_text(encoding="utf-8")
            )
            request.setdefault("result_path", os.environ["RED_CF_RESULT"])
            self._replay_requests = (request,)
        else:
            self._replay_requests = ()
        self._replay_spec: dict[str, Any] | None = None
        self._replay_requests_by_step: dict[int, list[dict[str, Any]]] = {}
        self._replay_children: dict[int, str] = {}
        self._replay_max_children = int(
            os.getenv("RED_UNIFIED_COW_WORKERS", "16")
        )
        self._reset_replay_schedule()

    def _reset_replay_schedule(self) -> None:
        self._replay_requests_by_step = {}
        for request in self._replay_requests:
            branch_timestep = min(
                int(row.get("timestep", row.get("tau", -1)))
                for row in request["interventions"]
            )
            self._replay_requests_by_step.setdefault(
                branch_timestep, []
            ).append(dict(request))

    def _wait_one_replay_child(self) -> None:
        child_pid, status = os.waitpid(-1, 0)
        request_id = self._replay_children.pop(int(child_pid))
        exit_code = os.waitstatus_to_exitcode(status)
        if exit_code != 0:
            raise RuntimeError(
                f"反事实子分支 {request_id} 退出码异常: {exit_code}"
            )

    def wait_for_replay_children(self) -> None:
        while self._replay_children:
            self._wait_one_replay_child()

    @property
    def replay_stop_step(self) -> int:
        if self._replay_branch_role == "counterfactual":
            return int(self._replay_spec["stop_step"])
        return max(
            (int(request["stop_step"]) for request in self._replay_requests),
            default=self.max_steps,
        )

    @property
    def replay_result_path(self) -> str:
        if self._replay_spec is None:
            raise RuntimeError("当前进程不是反事实子分支")
        return str(self._replay_spec["result_path"])

    @property
    def device(self) -> torch.device:
        return self.trainer.device

    @property
    def update_count(self) -> int:
        return self.trainer.update_count

    @property
    def transition_count(self) -> int:
        return self.trainer.transition_count

    @property
    def last_metrics(self) -> dict[str, float]:
        return self.trainer.last_metrics

    @property
    def active_entity_ids(self) -> frozenset[int]:
        return frozenset(
            entity_id
            for entity_id, decision in self._deployment.items()
            if bool(decision["presence"])
        )

    def configure_search_polygon(self, points: Sequence[Any]) -> None:
        longitudes = [float(point.lon) for point in points]
        latitudes = [float(point.lat) for point in points]
        self._search_bounds = (
            min(longitudes),
            max(longitudes),
            min(latitudes),
            max(latitudes),
        )

    def _search_xy_to_position(
        self, normalized_xy: torch.Tensor | Sequence[float]
    ) -> tuple[float, float]:
        x, y = (float(value) for value in normalized_xy)
        lon_min, lon_max, lat_min, lat_max = self._search_bounds
        return (
            lon_min + (x + 1.0) * 0.5 * (lon_max - lon_min),
            lat_min + (y + 1.0) * 0.5 * (lat_max - lat_min),
        )

    def _position_to_search_xy(
        self, lon: float, lat: float
    ) -> tuple[float, float]:
        lon_min, lon_max, lat_min, lat_max = self._search_bounds
        return (
            2.0 * (float(lon) - lon_min) / (lon_max - lon_min) - 1.0,
            2.0 * (float(lat) - lat_min) / (lat_max - lat_min) - 1.0,
        )

    def _position_to_search_index(self, lon: float, lat: float) -> int:
        x, y = self._position_to_search_xy(lon, lat)
        width = int(self.trainer.config.search_grid_width)
        height = int(self.trainer.config.search_grid_height)
        column = min(width - 1, max(0, int((x + 1.0) * 0.5 * width)))
        row = min(height - 1, max(0, int((y + 1.0) * 0.5 * height)))
        return row * width + column

    def _search_grid_reachable_mask(
        self, observation: Mapping[str, Any]
    ) -> torch.Tensor:
        """Mask SEARCH cells using only the actor's current legal position."""

        width = int(self.trainer.config.search_grid_width)
        height = int(self.trainer.config.search_grid_height)
        grid_size = width * height
        if not self.search_reachable_mask:
            return torch.ones(grid_size, dtype=torch.bool, device=self.device)
        self_info = observation.get("self") or {}
        position = self_info.get("position") or {}
        if not {"lon", "lat"} <= set(position):
            raise RuntimeError("可达搜索格点掩码缺少合法 self.position")
        current_lon = float(position["lon"])
        current_lat = float(position["lat"])
        lon_min, lon_max, lat_min, lat_max = self._search_bounds
        distances: list[float] = []
        for index in range(grid_size):
            column = index % width
            row = index // width
            lon = lon_min + (column + 0.5) * (lon_max - lon_min) / width
            lat = lat_min + (row + 0.5) * (lat_max - lat_min) / height
            mean_latitude = np.radians((current_lat + lat) / 2.0)
            east_km = (lon - current_lon) * 111.32 * np.cos(mean_latitude)
            north_km = (lat - current_lat) * 110.57
            distances.append(float(np.hypot(east_km, north_km)))
        mask = torch.tensor(
            [
                self.search_leg_min_distance_km <= distance
                <= self.search_leg_max_distance_km
                for distance in distances
            ],
            dtype=torch.bool,
            device=self.device,
        )
        if not bool(mask.any().item()):
            mask[int(np.argmin(distances))] = True
        valid_count = int(mask.sum().item())
        self._search_reachable_mask_sample_count += 1
        self._search_reachable_valid_count_sum += valid_count
        self._search_reachable_valid_count_min = min(
            self._search_reachable_valid_count_min, valid_count
        )
        self._search_reachable_valid_count_max = max(
            self._search_reachable_valid_count_max, valid_count
        )
        return mask

    def register_encoder(self, entity_id: int, encoder: ObservationEncoder) -> None:
        if encoder.observation_dim != self.local_observation_dim:
            raise ValueError(
                f"实体 {entity_id} 的局部观测维度 {encoder.observation_dim} 与所需维度 "
                f"{self.local_observation_dim} 不一致"
            )
        self._encoders[int(entity_id)] = encoder
        if self._targets:
            encoder.set_targets(self._targets_for_entity(int(entity_id)))
        self._refresh_actor_target_runtime_states()

    def register_entity_type(self, entity_id: int, entity_type: int) -> None:
        """Register immutable platform type for communication-fire features."""

        entity_type = int(entity_type)
        if entity_type not in RED_TYPES:
            raise ValueError(f"实体 {entity_id} 的红方类型 {entity_type} 非法")
        self._entity_type_by_entity[int(entity_id)] = entity_type

    @staticmethod
    def _target_dict(target: Any) -> dict[str, Any]:
        position = getattr(target, "position", None)
        if position is not None:
            return {
                "entity_id": int(target.entity_id),
                "nameChn": (
                    "区域搜索" if int(target.entity_id) == SEARCH_TARGET_ID
                    else "目标" if int(target.entity_type) == 9400
                    else "拦截阵地" if int(target.entity_type) == 9600
                    else "无人船"
                ),
                "type": int(target.entity_type),
                "health": 1.0,
                "position": {
                    "lon": float(position.lon),
                    "lat": float(position.lat),
                    "alt": float(position.alt),
                },
            }
        value = dict(target)
        value["entity_id"] = int(value.get("entity_id", value.get("id", -1)))
        return value

    def _capture_target_runtime_states(
        self,
        full_observation: Mapping[str, Any],
    ) -> None:
        """Cache simulator target state; actor injection remains catalogue-gated."""
        runtime_states: dict[int, tuple[float, float, float]] = {}
        entities = full_observation.get("entities") or {}
        for raw_target_id, entity in entities.items():
            target_id = int(raw_target_id)
            entity_type = int(entity.get("type", -1))
            if entity_type not in {9400, 9500, 9600}:
                continue
            health = max(0.0, float(entity.get("health", 0.0)))
            initial_health = self._initial_target_health.get(target_id)
            if initial_health is None:
                self._initial_target_health[target_id] = max(health, 1e-6)
            initial_health = self._initial_target_health[target_id]
            health_ratio = float(np.clip(health / initial_health, 0.0, 1.0))
            damage_ratio = float(np.clip(1.0 - health_ratio, 0.0, 1.0))
            alive = float(health > 0.0)
            runtime_states[target_id] = (
                health_ratio,
                damage_ratio,
                alive,
            )
        self._full_target_runtime_states = runtime_states
        self._refresh_actor_target_runtime_states()

    def _refresh_actor_target_runtime_states(self) -> None:
        runtime_ids: set[int] = set()
        full_states = getattr(self, "_full_target_runtime_states", {})
        for encoder in self._encoders.values():
            legal_ids = {
                int(target["entity_id"])
                for target in encoder.targets
                if int(target["entity_id"]) != SEARCH_TARGET_ID
                and bool(target.get("actor_visible", True))
                and not bool(target.get("slot_empty", False))
            }
            if getattr(self, "temporal_attack_options", False):
                # Exact health/damage belongs to the centralized critic.  The
                # actor receives only a neutral live prior; destruction is
                # represented by the legal target action mask below.
                actor_states = {
                    target_id: (1.0, 0.0, 1.0) for target_id in legal_ids
                }
            else:
                actor_states = {
                    target_id: state for target_id, state in full_states.items()
                    if target_id in legal_ids
                }
            encoder.set_target_runtime_states(actor_states)
            runtime_ids.update(actor_states)
        self._actor_runtime_target_ids = frozenset(runtime_ids)

    @staticmethod
    def _fixed_slot_record(
        slot_index: int,
        target_id: int | None,
        template: Mapping[str, Any] | None,
        *,
        actor_visible: bool,
    ) -> dict[str, Any]:
        if target_id is None:
            return {
                "entity_id": -1_000_000 - int(slot_index),
                "nameChn": "",
                "type": -1,
                "health": 0.0,
                "position": {"lon": 0.0, "lat": 0.0, "alt": 0.0},
                "actor_visible": False,
                "slot_empty": True,
                "slot_index": int(slot_index),
            }
        if actor_visible:
            value = dict(template or {})
            value.update({
                "entity_id": int(target_id),
                "actor_visible": True,
                "slot_empty": False,
                "slot_index": int(slot_index),
            })
            return value
        return {
            "entity_id": int(target_id),
            "nameChn": "",
            "type": int((template or {}).get("type", -1)),
            "health": 0.0,
            "position": {"lon": 0.0, "lat": 0.0, "alt": 0.0},
            "actor_visible": False,
            "slot_empty": False,
            "slot_index": int(slot_index),
        }

    def _targets_for_entity(
        self,
        entity_id: int,
    ) -> list[dict[str, Any]]:
        allowed_ids = self._actor_visible_target_ids_by_entity.get(
            int(entity_id),
            self._public_target_ids,
        )
        globally_known = self._visible_target_ids
        return [
            self._fixed_slot_record(
                slot_index,
                target_id,
                self._target_slot_templates.get(target_id)
                if target_id is not None else None,
                actor_visible=(
                    target_id is not None
                    and target_id in globally_known
                    and target_id in allowed_ids
                ),
            )
            for slot_index, target_id in enumerate(self._target_slot_ids)
        ]

    def set_actor_target_visibility(
        self,
        visibility: Mapping[int, Sequence[int]],
    ) -> None:
        """Apply the engine-fused communication-cluster tracks per actor."""

        globally_known = set(self._visible_target_ids)
        normalized: dict[int, frozenset[int]] = {}
        for raw_entity_id, raw_target_ids in visibility.items():
            allowed_ids = set(self._public_target_ids)
            allowed_ids.update(
                int(target_id)
                for target_id in raw_target_ids
                if int(target_id) in globally_known
            )
            normalized[int(raw_entity_id)] = frozenset(allowed_ids)
        self._actor_visible_target_ids_by_entity = normalized
        for entity_id, encoder in self._encoders.items():
            encoder.set_targets(self._targets_for_entity(entity_id))
        self._refresh_actor_target_runtime_states()

    def configure_targets(
        self,
        targets: Sequence[Any],
        *,
        reserved_targets: Sequence[Any] = (),
    ) -> None:
        visible = [self._target_dict(target) for target in targets]
        reserved = [self._target_dict(target) for target in reserved_targets]
        if not visible:
            raise ValueError("统一 MAPPO 至少需要一个已知蓝方目标")
        visible_ids = [int(target["entity_id"]) for target in visible]
        reserved_ids = [int(target["entity_id"]) for target in reserved]
        if len(visible_ids) != len(set(visible_ids)):
            raise ValueError("统一 MAPPO 可见目标目录包含重复实体 ID")
        if len(reserved_ids) != len(set(reserved_ids)):
            raise ValueError("统一 MAPPO 保留目标目录包含重复实体 ID")

        visible_by_id = {int(target["entity_id"]): target for target in visible}
        reserved_by_id = {int(target["entity_id"]): target for target in reserved}
        catalogue_ids = set(visible_by_id) | set(reserved_by_id)
        if len(catalogue_ids) > TARGET_SLOTS:
            raise ValueError(
                f"目标数量 {len(catalogue_ids)} 超过统一 MAPPO 固定容量 {TARGET_SLOTS}"
            )

        slot_ids = list(self._target_slot_ids)
        missing_ids = sorted(
            target_id
            for target_id in catalogue_ids
            if target_id not in slot_ids and target_id != SEARCH_TARGET_ID
        )
        for target_id in missing_ids:
            try:
                empty_index = slot_ids.index(None)
            except ValueError as error:
                raise ValueError(
                    f"目标 {target_id} 无可用固定槽位，容量为 {TARGET_SLOTS}"
                ) from error
            slot_ids[empty_index] = target_id

        templates = dict(getattr(self, "_target_slot_templates", {}))
        templates.update(reserved_by_id)
        templates.update(visible_by_id)
        self._target_slot_ids = slot_ids
        self._target_slot_templates = templates
        self._visible_target_ids = frozenset(visible_by_id)
        self._public_target_ids = frozenset(
            target_id
            for target_id, target in visible_by_id.items()
            if not bool(getattr(self, "dynamic_lifecycle", False))
            or int(target.get("type", -1)) != 9500
            or target_id == SEARCH_TARGET_ID
        )
        self._targets = [
            self._fixed_slot_record(
                slot_index,
                target_id,
                templates.get(target_id) if target_id is not None else None,
                actor_visible=target_id in visible_by_id,
            )
            for slot_index, target_id in enumerate(slot_ids)
        ]
        for entity_id, encoder in self._encoders.items():
            encoder.set_targets(self._targets_for_entity(entity_id))
        self._refresh_actor_target_runtime_states()

    def _target_tensors_for_entity(
        self,
        entity_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self._encoders.get(int(entity_id))
        if encoder is None:
            raise RuntimeError(f"实体 {entity_id} 尚未注册统一观测编码器")
        coordinates = np.asarray(
            encoder.last_target_coordinates, dtype=np.float32
        )
        observed_valid = torch.as_tensor(
            encoder.last_target_valid_mask,
            dtype=torch.bool,
            device=self.device,
        )
        if coordinates.shape != (TARGET_SLOTS, 2):
            raise RuntimeError(
                f"实体 {entity_id} 目标坐标维度 {coordinates.shape} 非法"
            )
        if observed_valid.shape != (TARGET_SLOTS,):
            raise RuntimeError(
                f"实体 {entity_id} 目标掩码维度 {observed_valid.shape} 非法"
            )
        valid = self._target_valid_for_entity_type(
            encoder.last_entity_type, observed_valid
        )
        return (
            torch.as_tensor(
                coordinates, dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            valid.unsqueeze(0),
        )

    @staticmethod
    def _target_bda_from_observation(
        observation: Mapping[str, Any] | None,
    ) -> dict[int, tuple[float, float, float, float]]:
        """Read BDA only from tracks present in this actor observation."""

        bda: dict[int, tuple[float, float, float, float]] = {}
        self_info = (observation or {}).get("self") or {}
        for raw_key, track in (self_info.get("detectInfo") or {}).items():
            if not bool(
                ObservationEncoder._field(track, "health_observed", False)
            ):
                continue
            target_id = int(
                ObservationEncoder._field(track, "entity_id", raw_key)
            )
            remaining = max(0.0, float(
                ObservationEncoder._field(track, "health_remaining", 0.0)
            ))
            maximum = max(remaining, float(
                ObservationEncoder._field(track, "health_max", 0.0)
            ))
            if maximum <= 0.0:
                continue
            health_ratio = float(np.clip(remaining / maximum, 0.0, 1.0))
            remaining_normalized = float(np.clip(
                remaining / TARGET_HEALTH_SCALE, 0.0, 1.0
            ))
            bda[target_id] = (
                health_ratio,
                1.0 - health_ratio,
                1.0,
                remaining_normalized,
            )
        return bda

    def _target_set_features(
        self,
        encoded: np.ndarray | torch.Tensor,
        *,
        entity_id: int,
        observation: Mapping[str, Any] | None,
    ) -> np.ndarray:
        """Build legal per-target geometry, BDA, and cluster allocation."""

        local = np.asarray(encoded, dtype=np.float32)
        if local.shape != (self.local_observation_dim,):
            raise ValueError(
                f"目标集合特征需要局部观测 {(self.local_observation_dim,)}，实际为 {local.shape}"
            )
        target_start = (
            ObservationEncoder.SELF_FEATURES
            + ObservationEncoder.IDENTITY_FEATURES
        )
        features = np.zeros(
            (TARGET_SLOTS, TARGET_SET_FEATURE_DIM), dtype=np.float32
        )
        encoder = self._encoders[int(entity_id)]
        bda = self._target_bda_from_observation(observation)
        for slot_index in range(TARGET_SLOTS):
            source_start = slot_index * encoder.target_feature_dim + target_start
            features[
                slot_index, :TARGET_GEOMETRY_FEATURES
            ] = local[
                source_start:source_start + TARGET_GEOMETRY_FEATURES
            ]
            if bool(encoder.last_target_valid_mask[slot_index]):
                features[slot_index, 12:15] = (1.0, 0.0, 0.0)
            target_id = self._target_slot_ids[slot_index]
            if target_id is not None and int(target_id) in bda:
                health_ratio, damage_ratio, alive, remaining = bda[int(target_id)]
                features[slot_index, 12:15] = (
                    health_ratio, damage_ratio, alive
                )
                features[slot_index, 20] = 1.0
                features[slot_index, 21] = remaining

        self_info = (observation or {}).get("self") or {}
        members = {
            int(member)
            for member in (self_info.get("commRangeInfo") or ())
        }
        members.add(int(entity_id))
        members.intersection_update(self._encoders)
        other_members = members - {int(entity_id)}
        entity_types = getattr(self, "_entity_type_by_entity", {})

        def member_type(member: int) -> int:
            return int(entity_types.get(
                member,
                self._encoders[member].last_entity_type,
            ))

        legacy_type_members = {
            entity_type: [
                member for member in other_members
                if int(self._encoders[member].last_entity_type) == entity_type
            ]
            for entity_type in RED_TYPES
        }
        type_members = {
            entity_type: [
                member for member in other_members
                if member_type(member) == entity_type
            ]
            for entity_type in RED_TYPES
        }
        external_targets = getattr(self, "_external_target_by_entity", {})
        self._external_target_feature_hit_count = getattr(
            self, "_external_target_feature_hit_count", 0
        ) + sum(member in external_targets for member in other_members)

        def legacy_assigned_target(member: int) -> int:
            option = self._attack_option_by_entity.get(member)
            if option is not None and not bool(option.get("completed", False)):
                return int(option.get("target_id", -1))
            return -1

        def assigned_target(member: int) -> int:
            target_id = legacy_assigned_target(member)
            if target_id >= 0:
                return target_id
            return int(external_targets.get(member, -1))

        features[:, 15] = len(other_members) / max(1, len(self._encoders))
        for slot_index, target_id in enumerate(self._target_slot_ids):
            if target_id is None or int(target_id) == SEARCH_TARGET_ID:
                continue
            legacy_assigned = [
                member
                for member in other_members
                if legacy_assigned_target(member) == int(target_id)
            ]
            features[slot_index, 16] = len(legacy_assigned) / max(
                1, len(other_members)
            )
            for offset, entity_type in enumerate(RED_TYPES, start=17):
                members_of_type = legacy_type_members[entity_type]
                assigned_type = sum(
                    int(self._encoders[member].last_entity_type) == entity_type
                    for member in legacy_assigned
                )
                features[slot_index, offset] = assigned_type / max(
                    1, len(members_of_type)
                )
            assigned = [
                member
                for member in other_members
                if assigned_target(member) == int(target_id)
            ]
            features[slot_index, 22] = min(
                1.0, len(assigned) / TARGET_FIRE_COUNT_SCALE
            )
            for offset, entity_type in enumerate(RED_TYPES, start=23):
                assigned_type = sum(
                    member_type(member) == entity_type
                    for member in assigned
                )
                features[slot_index, offset] = min(
                    1.0, assigned_type / TARGET_FIRE_COUNT_SCALE
                )
        return features

    def _attack_option_target_valid_mask(
        self,
        entity_id: int,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the environment's legal destroyed-target action mask."""

        result = target_valid.clone()
        runtime_states = self._full_target_runtime_states
        for index, target in enumerate(self._targets):
            target_id = int(target["entity_id"])
            if target_id == SEARCH_TARGET_ID:
                continue
            state = runtime_states.get(target_id)
            if state is not None and float(state[2]) <= 0.0:
                result[index] = False
        return result

    @property
    def active_student_option_entity_ids(self) -> frozenset[int]:
        """Return entities whose current attack/search option is student-owned."""

        return frozenset(
            int(entity_id)
            for entity_id, option in self._attack_option_by_entity.items()
            if option.get("controller") == "student"
        )

    def persistent_search_boundary_entity_ids(
        self,
        observations_by_entity: Mapping[int, Mapping[str, Any]],
    ) -> frozenset[int]:
        """Find SEARCH boundaries without running the Actor or encoder."""

        boundary_ids: set[int] = set()
        legal_slot_ids = {
            int(target_id)
            for target_id in self._target_slot_ids
            if target_id is not None and int(target_id) != SEARCH_TARGET_ID
        }
        for entity_id, option in self._attack_option_by_entity.items():
            entity_id = int(entity_id)
            if option.get("controller") != "student":
                continue
            observation = observations_by_entity.get(entity_id)
            if not observation:
                continue
            target_id = int(option.get("target_id", -1))
            if target_id != SEARCH_TARGET_ID:
                runtime_state = self._full_target_runtime_states.get(target_id)
                step = int(observation.get("step", 0)) + int(
                    self.trajectory_counterfactual
                )
                elapsed = step - int(option["last_decision_step"])
                if (
                    runtime_state is not None
                    and float(runtime_state[2]) <= 0.0
                ) or elapsed >= self.attack_option_min_dwell_steps:
                    boundary_ids.add(entity_id)
                continue
            self_info = observation.get("self") or {}
            for raw_target_id, track in (
                self_info.get("detectInfo") or {}
            ).items():
                target_id = int(ObservationEncoder._field(
                    track, "entity_id", raw_target_id
                ))
                if (
                    target_id not in legal_slot_ids
                    or int(ObservationEncoder._field(
                        track, "entity_type", -1
                    )) != 9500
                ):
                    continue
                if bool(ObservationEncoder._field(
                    track, "health_observed", False
                )) and float(ObservationEncoder._field(
                    track, "health_remaining", 1.0
                )) <= 0.0:
                    continue
                boundary_ids.add(entity_id)
                break
            if entity_id in boundary_ids:
                continue

            position = self_info.get("position") or {}
            if not {"lon", "lat"} <= set(position):
                continue
            lon = float(position["lon"])
            lat = float(position["lat"])
            target_lon = float(option["target_lon"])
            target_lat = float(option["target_lat"])
            mean_latitude = np.radians((lat + target_lat) / 2.0)
            east_km = (target_lon - lon) * 111.32 * np.cos(mean_latitude)
            north_km = (target_lat - lat) * 110.57
            distance_km = float(np.hypot(east_km, north_km))
            step = int(observation.get("step", 0)) + int(
                self.trajectory_counterfactual
            )
            elapsed = step - int(option["last_decision_step"])
            if distance_km <= self.search_option_emergency_reopen_distance_km:
                boundary_ids.add(entity_id)
            elif (
                elapsed >= self.search_option_dwell_steps
                and distance_km <= self.search_option_reopen_distance_km
            ):
                boundary_ids.add(entity_id)
        return frozenset(boundary_ids)

    def sync_external_target_assignment(
        self,
        entity_id: int,
        target_id: int | None,
    ) -> None:
        """Record a legally observed native target without creating an option."""

        entity_id = int(entity_id)
        if target_id is None or int(target_id) == SEARCH_TARGET_ID:
            self._external_target_by_entity.pop(entity_id, None)
            return
        target_id = int(target_id)
        if self._external_target_by_entity.get(entity_id) != target_id:
            self._external_target_sync_count += 1
        self._external_target_by_entity[entity_id] = target_id

    def legalize_external_target_command(
        self,
        entity_id: int,
        entity_type: int,
        preferred_target_id: int,
        preferred_target: Mapping[str, Any],
    ) -> tuple[int, dict[str, float]]:
        """Map a native target onto the actor's current legal local catalogue."""

        legal_targets = []
        for target in self._targets_for_entity(int(entity_id)):
            if (
                bool(target.get("slot_empty", False))
                or not bool(target.get("actor_visible", True))
            ):
                continue
            target_id = int(target["entity_id"])
            target_type = int(target.get("type", -1))
            if self.dynamic_lifecycle:
                compatible = (
                    target_type == 9500 or target_id == SEARCH_TARGET_ID
                    if int(entity_type) == 21002
                    else target_type in {9400, 9600}
                )
                if not compatible:
                    continue
            runtime_state = self._full_target_runtime_states.get(target_id)
            if (
                target_id != SEARCH_TARGET_ID
                and runtime_state is not None
                and float(runtime_state[2]) <= 0.0
            ):
                continue
            legal_targets.append(target)
        if not legal_targets:
            raise RuntimeError(
                f"实体 {int(entity_id)} 的合法局部观测中没有可执行目标"
            )

        selected = next((
            target for target in legal_targets
            if int(target["entity_id"]) == int(preferred_target_id)
        ), None)
        if selected is None:
            if int(entity_type) == 21002:
                selected = next((
                    target for target in legal_targets
                    if int(target["entity_id"]) == SEARCH_TARGET_ID
                ), None)
            selected = selected or legal_targets[0]
            self._external_target_fallback_count += 1

        selected_id = int(selected["entity_id"])
        if selected_id == int(preferred_target_id):
            command_target = {
                "x": float(preferred_target["x"]),
                "y": float(preferred_target["y"]),
                "z": float(preferred_target.get("z", 0.0)),
            }
        elif selected_id == SEARCH_TARGET_ID:
            lon, lat = self._search_xy_to_position(torch.zeros(2))
            command_target = {"x": float(lon), "y": float(lat), "z": 0.0}
        else:
            position = selected["position"]
            command_target = {
                "x": float(position["lon"]),
                "y": float(position["lat"]),
                "z": float(position.get("alt", 0.0)),
            }
        return selected_id, command_target

    def apply_prepared_external_target(
        self,
        entity_id: int,
        target_id: int,
        command_target: Mapping[str, Any],
    ) -> None:
        """Synchronize a deterministic teacher target after batched sampling."""

        entity_id = int(entity_id)
        target_id = int(target_id)
        target_index = self.target_index_for(target_id)
        sample = self._pending_motion.get(entity_id)
        if sample is None or target_index is None:
            raise RuntimeError(
                f"实体 {entity_id} 缺少已准备动作或目标槽位 {target_id}"
            )
        if not bool(sample.target_valid_mask[int(target_index)].item()):
            raise RuntimeError(
                f"实体 {entity_id} 的确定性教师目标 {target_id} 不合法"
            )
        target_lon = float(command_target["x"])
        target_lat = float(command_target["y"])
        prepared = self._prepared_lifecycle[entity_id]
        prepared.update({
            "target_index": int(target_index),
            "target_id": target_id,
            "target_lon": target_lon,
            "target_lat": target_lat,
        })
        semantics = self._prepared_replay_semantics[entity_id]
        semantics.update({
            "selected_target_id": target_id,
            "selected_index": int(target_index),
        })
        option = self._attack_option_by_entity.get(entity_id)
        if option is not None:
            option.update({
                "target_index": int(target_index),
                "target_id": target_id,
                "target_lon": target_lon,
                "target_lat": target_lat,
            })

    def _terminate_attack_option(self, entity_id: int, reason: str) -> None:
        if self._attack_option_by_entity.pop(int(entity_id), None) is None:
            return
        normalized = str(reason)
        self._attack_option_termination_count += 1
        self._attack_option_termination_reasons[normalized] = (
            self._attack_option_termination_reasons.get(normalized, 0) + 1
        )

    def mark_attack_option_complete(self, entity_id: int) -> None:
        """Expose an explicit local-policy completion hook for future internals."""

        option = self._attack_option_by_entity.get(int(entity_id))
        if option is not None:
            option["completed"] = True

    def _deterministic_evasion_maneuver(
        self,
        entity_id: int,
        observation: Mapping[str, Any],
    ) -> int:
        """Return left/straight/right using only timestamped local tracks.

        The returned value is an engine-adapter index in ``{0, 1, 2}``.
        This rule is deliberately outside the stochastic actor: its action
        mask remains false, so it contributes neither PPO log probability nor
        entropy.  A short pulse/cooldown state prevents high-frequency
        left/right chatter while still allowing a later interceptor to retrigger.
        """

        entity_id = int(entity_id)
        step = int(observation.get("step", 0)) + int(
            self.trajectory_counterfactual
        )
        state = self._deterministic_evasion_state.get(entity_id)
        if state is not None and step < int(state["pulse_until_step"]):
            self._deterministic_evasion_active_step_count += 1
            return int(state["maneuver_index"])
        if state is not None and step < int(state["cooldown_until_step"]):
            return 1

        self_info = observation.get("self") or {}
        own_position = ObservationEncoder._vector3(self_info.get("pos_ecf"))
        own_velocity = ObservationEncoder._vector3(self_info.get("vel_ecf"))
        position = self_info.get("position") or {}
        if own_position is None or own_velocity is None or not {
            "lon", "lat"
        } <= set(position):
            self._deterministic_evasion_state.pop(entity_id, None)
            return 1

        sim_time = float(observation.get("sim_time", 0.0))
        sim_step = max(1.0, float(observation.get("sim_step", 1.0)))
        time_units_per_second = 1000.0 if sim_step >= 100.0 else 1.0
        lon = float(position["lon"])
        lat = float(position["lat"])
        own_east, own_north, _ = ObservationEncoder._local_components(
            own_velocity, lon=lon, lat=lat
        )
        own_heading = (
            float(np.arctan2(own_east, own_north))
            if float(np.hypot(own_east, own_north)) > 1e-9
            else 0.0
        )

        threats: list[tuple[float, float, float, int]] = []
        for raw_track_id, track in (self_info.get("detectInfo") or {}).items():
            if int(ObservationEncoder._field(
                track, "entity_type", -1
            )) != 24000:
                continue
            age_steps = ObservationEncoder._track_age_steps(
                track,
                step=step,
                sim_time=sim_time,
                sim_step=sim_step,
            )
            if age_steps > self.deterministic_evasion_max_track_age_steps:
                continue
            track_position = ObservationEncoder._vector3(
                ObservationEncoder._field(track, "pos_ecf", None)
            )
            track_velocity = ObservationEncoder._vector3(
                ObservationEncoder._field(track, "vel_ecf", None)
            )
            if track_position is None or track_velocity is None:
                continue
            detection_time = float(ObservationEncoder._field(
                track, "time", sim_time
            ))
            age_seconds = max(
                0.0, (sim_time - detection_time) / time_units_per_second
            )
            projected_track_position = track_position + track_velocity * age_seconds
            relative_position = projected_track_position - own_position
            relative_velocity = track_velocity - own_velocity
            speed_squared = float(np.dot(relative_velocity, relative_velocity))
            closing_dot = float(np.dot(relative_position, relative_velocity))
            if speed_squared <= 1e-9 or closing_dot >= 0.0:
                continue
            tcpa = float(-closing_dot / speed_squared)
            if tcpa > self.deterministic_evasion_tcpa_seconds:
                continue
            dcpa = float(np.linalg.norm(
                relative_position + relative_velocity * tcpa
            ))
            if dcpa > self.deterministic_evasion_dcpa_km * 1000.0:
                continue
            east, north, _ = ObservationEncoder._local_components(
                relative_position, lon=lon, lat=lat
            )
            bearing = (
                float(np.arctan2(east, north))
                if float(np.hypot(east, north)) > 1e-9
                else own_heading
            )
            relative_side = float(np.sin(bearing - own_heading))
            track_id = int(ObservationEncoder._field(
                track, "entity_id", raw_track_id
            ))
            threats.append((tcpa, dcpa, relative_side, track_id))

        if not threats:
            self._deterministic_evasion_state.pop(entity_id, None)
            return 1
        self._deterministic_evasion_visible_threat_count += len(threats)
        _, _, _, _ = min(
            threats, key=lambda item: (item[0], item[1])
        )
        # The native interface exposes acceleration about the missile model's
        # Z axis, not a roll-stabilized horizontal left/right axis.  The
        # shipped reactive controller uses the positive direction; choosing
        # the sign from a horizontal bearing can command a dive into terrain.
        maneuver_index = 2
        pulse_until = step + self.deterministic_evasion_pulse_steps
        self._deterministic_evasion_state[entity_id] = {
            "maneuver_index": int(maneuver_index),
            "pulse_until_step": int(pulse_until),
            "cooldown_until_step": int(
                pulse_until + self.deterministic_evasion_cooldown_steps
            ),
        }
        self._deterministic_evasion_trigger_count += 1
        self._deterministic_evasion_active_step_count += 1
        return int(maneuver_index)

    def _attack_option_boundary(
        self,
        entity_id: int,
        target_valid: torch.Tensor,
        step: int,
        observation: Mapping[str, Any] | None = None,
    ) -> bool:
        """Implement beta using only local legality and option-local state."""

        option = self._attack_option_by_entity.get(int(entity_id))
        if option is None:
            return True
        target_index = int(option["target_index"])
        if (
            target_index < 0
            or target_index >= int(target_valid.numel())
            or not bool(target_valid[target_index].item())
        ):
            self._terminate_attack_option(entity_id, "target_illegal_or_destroyed")
            return True

        elapsed = int(step) - int(option["last_decision_step"])
        if (
            int(option.get("target_id", -1)) == SEARCH_TARGET_ID
            and bool(getattr(self, "search_option_chaining", False))
        ):
            alternatives = target_valid.clone()
            alternatives[target_index] = False
            if bool(alternatives.any().item()):
                return True
            self_info = (observation or {}).get("self") or {}
            position = self_info.get("position") or {}
            if not {"lon", "lat"} <= set(position):
                return False
            lon = float(position["lon"])
            lat = float(position["lat"])
            target_lon = float(option["target_lon"])
            target_lat = float(option["target_lat"])
            mean_latitude = np.radians((lat + target_lat) / 2.0)
            east_km = (target_lon - lon) * 111.32 * np.cos(mean_latitude)
            north_km = (target_lat - lat) * 110.57
            distance_km = float(np.hypot(east_km, north_km))
            self._search_option_distance_sample_count += 1
            self._search_option_min_distance_km = min(
                self._search_option_min_distance_km, distance_km
            )
            self._search_option_last_distance_km = distance_km
            if distance_km <= float(getattr(
                self, "search_option_emergency_reopen_distance_km", 5.0
            )):
                # Reaching an empty SEARCH waypoint makes L self-destruct in
                # the simulator.  Treat imminent completion as beta=1 even
                # before H_min; H_min still prevents ordinary mid-route turns.
                self._search_option_emergency_boundary_count += 1
                return True
            if elapsed < int(getattr(self, "search_option_dwell_steps", 120)):
                return False
            return distance_km <= float(getattr(
                self, "search_option_reopen_distance_km", 15.0
            ))
        if elapsed < self.attack_option_min_dwell_steps:
            return False

        alternatives = target_valid.clone()
        alternatives[target_index] = False
        option_completed = bool(option.get("completed", False))
        retarget_legal = bool(alternatives.any().item())
        return option_completed or retarget_legal

    def _target_valid_for_entity_type(
        self,
        entity_type: int,
        observed_valid: torch.Tensor,
    ) -> torch.Tensor:
        allowed = observed_valid.clone()
        if self.dynamic_lifecycle:
            for index, target in enumerate(self._targets):
                target_type = int(target.get("type", -1))
                target_id = int(target.get("entity_id", -1))
                compatible = (
                    target_type == 9500 or target_id == SEARCH_TARGET_ID
                    if entity_type == 21002
                    else target_type in {9400, 9600}
                )
                allowed[index] &= compatible
        if not bool(allowed.any().item()):
            raise RuntimeError(f"实体类型 {entity_type} 没有合法目标槽位")
        return allowed

    def target_index_for(self, target_id: int | None) -> int | None:
        if target_id is None:
            return None
        return next(
            (
                index
                for index, target in enumerate(self._targets)
                if int(target["entity_id"]) == int(target_id)
                and not bool(target.get("slot_empty", False))
            ),
            None,
        )

    def prepared_target_id(self, entity_id: int) -> int | None:
        """Return the legal target selected in the current prepared action."""

        sample = self._pending_motion.get(int(entity_id))
        if sample is None or not sample.mask_target:
            return None
        index = int(sample.action.target_index.item())
        if (
            index < 0
            or index >= len(self._target_slot_ids)
            or not bool(sample.target_valid_mask[index].item())
        ):
            raise RuntimeError("已准备动作包含非法目标槽位")
        target_id = self._target_slot_ids[index]
        return None if target_id is None else int(target_id)

    def prepared_legal_targets(
        self,
        entity_id: int,
        *,
        include_search: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        """Return only the locally legal target IDs and command coordinates."""

        sample = self._pending_motion.get(int(entity_id))
        if sample is None or not sample.mask_target:
            return ()
        candidates = []
        for index_tensor in sample.target_valid_mask.nonzero(as_tuple=False):
            index = int(index_tensor.item())
            target_id = self._target_slot_ids[index]
            if target_id is None:
                continue
            if int(target_id) == SEARCH_TARGET_ID:
                if not include_search:
                    continue
                lon, lat = self._search_xy_to_position(sample.action.search_xy)
            else:
                lon = float(sample.target_coordinates[index, 0].item())
                lat = float(sample.target_coordinates[index, 1].item())
            candidates.append({
                "target_id": int(target_id),
                "target": {"x": float(lon), "y": float(lat), "z": 0.0},
            })
        if not candidates and include_search:
            raise RuntimeError("已准备目标动作没有合法候选")
        return tuple(candidates)

    def sample_prepared_counterfactual_targets(
        self,
        entity_id: int,
        count: int,
        *,
        include_search: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Sample legal target references from the same frozen actor policy.

        The legacy path samples with replacement. C0a_goal reward-weighted
        continuation enables two explicit environment switches so that the
        factual action plus the references form one distinct candidate set:
        ``RED_C0A_GOAL_CANDIDATES_WITHOUT_REPLACEMENT`` and
        ``RED_C0A_GOAL_EXCLUDE_FACTUAL_TARGET``.
        """

        if count <= 0:
            raise ValueError("counterfactual target sample count must be positive")
        sample = self._pending_motion.get(int(entity_id))
        if sample is None or not sample.mask_target:
            raise RuntimeError("目标反事实抽样缺少已准备的合法目标动作")
        valid = sample.target_valid_mask.detach().clone().to(dtype=torch.bool)
        if not include_search:
            search_index = self.target_index_for(SEARCH_TARGET_ID)
            if search_index is not None:
                valid[int(search_index)] = False
        without_replacement = (
            os.getenv("RED_C0A_GOAL_CANDIDATES_WITHOUT_REPLACEMENT", "0") == "1"
        )
        exclude_factual = (
            os.getenv("RED_C0A_GOAL_EXCLUDE_FACTUAL_TARGET", "0") == "1"
        )
        if exclude_factual:
            factual_index = int(sample.action.target_index.item())
            if 0 <= factual_index < int(valid.numel()):
                valid[factual_index] = False
        if not bool(valid.any().item()):
            if without_replacement:
                return ()
            raise RuntimeError("目标反事实抽样没有合法攻击目标")

        with torch.no_grad():
            parameters = self.trainer.model.distribution_parameters(
                sample.observation.unsqueeze(0).to(self.device),
                target_features=sample.target_features.unsqueeze(0).to(self.device),
                target_valid_mask=valid.unsqueeze(0).to(self.device),
            )
            logits = parameters["target_logits"][0].masked_fill(
                ~valid.to(self.device), float("-inf")
            )
            if without_replacement:
                available = int(valid.sum().item())
                sample_count = min(int(count), available)
                if available <= int(count):
                    indices = valid.nonzero(as_tuple=False).flatten().cpu()
                else:
                    probabilities = torch.softmax(logits, dim=-1)
                    indices = torch.multinomial(
                        probabilities,
                        sample_count,
                        replacement=False,
                    ).detach().cpu()
            else:
                indices = torch.distributions.Categorical(logits=logits).sample(
                    (int(count),)
                ).detach().cpu()

        references = []
        for sample_index, index_tensor in enumerate(indices):
            index = int(index_tensor.item())
            target_id = self._target_slot_ids[index]
            if target_id is None or int(target_id) == SEARCH_TARGET_ID:
                raise RuntimeError("目标反事实抽样产生了非法攻击目标")
            references.append({
                "sample_index": int(sample_index),
                "target_index": index,
                "target_id": int(target_id),
                "target": {
                    "x": float(sample.target_coordinates[index, 0].item()),
                    "y": float(sample.target_coordinates[index, 1].item()),
                    "z": 0.0,
                },
            })
        return tuple(references)

    @staticmethod
    def _action_at_index(action: HybridAction, index: int) -> HybridAction:
        return HybridAction(
            presence=action.presence[index].detach().cpu(),
            retarget=action.retarget[index].detach().cpu(),
            initial_xy=action.initial_xy[index].detach().cpu(),
            search_xy=action.search_xy[index].detach().cpu(),
            target_xy=action.target_xy[index].detach().cpu(),
            maneuver=action.maneuver[index].detach().cpu(),
            initial_raw=action.initial_raw[index].detach().cpu(),
            search_index=action.search_index[index].detach().cpu(),
            target_index=action.target_index[index].detach().cpu(),
            maneuver_index=action.maneuver_index[index].detach().cpu(),
            satellite=action.satellite[index].detach().cpu(),
        )

    @staticmethod
    def _action_without_batch(action: HybridAction) -> HybridAction:
        return UnifiedMAPPOSharedPolicy._action_at_index(action, 0)

    def _encode_team_context(self, observation: Mapping[str, Any]) -> np.ndarray:
        entities = observation.get("entities", {})
        features: list[float] = []
        total_alive = 0
        total_entities = 0
        for entity_type in RED_TYPES:
            members = [
                entity for entity in entities.values()
                if int(entity.get("type", -1)) == entity_type
            ]
            self._team_type_totals[entity_type] = max(
                self._team_type_totals.get(entity_type, 0), len(members)
            )
            denominator = max(1, self._team_type_totals[entity_type])
            alive = [
                entity for entity in members
                if float(entity.get("health", 0.0)) > 0.0
                and bool(entity.get("isVisible", True))
            ]
            total_alive += len(alive)
            total_entities += denominator
            features.extend((
                len(alive) / denominator,
                sum(min(max(float(entity.get("health", 0.0)) / 100.0, 0.0), 1.0) for entity in alive) / denominator,
                float(np.mean([entity.get("position", {}).get("lon", 0.0) for entity in alive])) / 180.0 if alive else 0.0,
                float(np.mean([entity.get("position", {}).get("lat", 0.0) for entity in alive])) / 90.0 if alive else 0.0,
            ))
        features.extend((
            min(max(float(observation.get("step", 0)) / self.max_steps, 0.0), 1.0),
            total_alive / max(1, total_entities),
            len(self._visible_target_ids) / TARGET_SLOTS,
        ))
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (TEAM_CONTEXT_DIM,):
            raise RuntimeError(f"红方集合状态维度 {result.shape} 不等于 {(TEAM_CONTEXT_DIM,)}")
        return result

    def _augment_observation(
        self,
        encoded: np.ndarray | torch.Tensor,
        team_context: np.ndarray | None,
    ) -> np.ndarray:
        local = np.asarray(encoded, dtype=np.float32)
        if local.shape != (self.local_observation_dim,):
            raise ValueError(
                f"局部观测维度 {local.shape} 不等于 {(self.local_observation_dim,)}"
            )
        if self.trajectory_counterfactual:
            return local
        if team_context is None:
            raise RuntimeError("统一 MAPPO 动作采样前未设置红方集合状态")
        return np.concatenate((local, team_context)).astype(np.float32, copy=False)

    def _sample(
        self,
        encoded: np.ndarray,
        mask: HybridActionMask,
        *,
        agent_id: int,
        entity_id: int,
        raw_observation: Mapping[str, Any] | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = torch.as_tensor(
            self._augment_observation(encoded, self._current_team_context),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        coordinates, target_valid = self._target_tensors_for_entity(entity_id)
        target_features = torch.as_tensor(
            self._target_set_features(
                encoded, entity_id=entity_id, observation=raw_observation
            ), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            output = self.trainer.model.act(
                observation,
                coordinates,
                target_valid,
                mask,
                target_features=target_features,
                deterministic=not self.stochastic_actor,
            )
            value = output.value[0]
            if self.trajectory_counterfactual and self.training:
                if self._current_global_state is None:
                    raise RuntimeError("strict CTDE 动作采样前未设置全局 critic 状态")
                critic_state = torch.as_tensor(
                    self._current_global_state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                value = self.trainer.model.critic_value(
                    critic_state,
                    torch.tensor([agent_id], dtype=torch.long, device=self.device),
                    observation,
                )[0]
        return (
            output,
            observation[0].detach().cpu(),
            coordinates[0].detach().cpu(),
            target_valid[0].detach().cpu(),
            target_features[0].detach().cpu(),
            value.detach().cpu(),
        )

    def select_deployment(
        self,
        entity_id: int,
        red_observation: Mapping[str, Any],
        deploy_bounds: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        entity_id = int(entity_id)
        self._current_team_context = self._encode_team_context(red_observation)
        if self.trajectory_counterfactual and self._current_global_state is None:
            # Dynamic lifecycle does not train a deployment row; this fallback
            # only keeps legacy callers well-defined.
            self._current_global_state = self.global_encoder.encode(red_observation)
        encoder = self._encoders.get(entity_id)
        if encoder is None:
            raise RuntimeError(f"实体 {entity_id} 尚未注册统一观测编码器")
        entity = red_observation["entities"][entity_id]
        if self.trajectory_counterfactual and self.dynamic_lifecycle:
            # Deployment only registers legal bounds.  The actor chooses
            # x_init/y_init exactly once, conditional on LAUNCH at tau_i.
            decision = {
                "presence": 1,
                "lon": 0.0,
                "lat": 0.0,
                "defer_position": True,
                "target_index": -1,
                "target_id": -1,
                "target_lon": 0.0,
                "target_lat": 0.0,
                "deploy_bounds": tuple(float(value) for value in deploy_bounds),
            }
            self._deployment[entity_id] = decision
            self._active_deployment_count += 1
            self._deployment_legal_count += 1
            return dict(decision)
        isolated = {
            "step": 0,
            "entity_id": entity_id,
            "agent_id": encoder.agent_id,
            "self": entity,
        }
        encoded = encoder.encode(
            isolated,
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
            current_target_index=None,
            task_context=(0.0, 0.0, 0.0, 1.0, 0.0),
        )
        ones = torch.ones(1, dtype=torch.bool, device=self.device)
        zeros = torch.zeros(1, dtype=torch.bool, device=self.device)
        initial_enabled = os.getenv("RED_UNIFIED_TRAIN_INITIAL", "1") == "1"
        initial_mask = ones if initial_enabled else zeros
        deployment_presence_mask = zeros if self.dynamic_lifecycle else ones
        deployment_target_mask = zeros if self.dynamic_lifecycle else ones
        output, observation, coordinates, target_valid, target_features, value = self._sample(
            encoded,
            HybridActionMask(
                presence=deployment_presence_mask,
                initial_position=initial_mask,
                search_position=zeros,
                retarget=zeros,
                target=deployment_target_mask,
                maneuver=zeros,
                satellite=zeros,
            ),
            agent_id=int(encoder.agent_id),
            entity_id=entity_id,
            raw_observation=isolated,
        )
        presence = (
            1 if self.dynamic_lifecycle else int(output.action.presence[0].item())
        )
        normalized_xy = output.action.initial_xy[0].detach().cpu().numpy()
        lon_min, lon_max, lat_min, lat_max = deploy_bounds
        lon = lon_min + (float(normalized_xy[0]) + 1.0) * 0.5 * (lon_max - lon_min)
        lat = lat_min + (float(normalized_xy[1]) + 1.0) * 0.5 * (lat_max - lat_min)
        target_index = int(output.action.target_index[0].item())
        selected_target = (
            self._targets[target_index]
            if presence and not self.dynamic_lifecycle else None
        )
        if presence:
            self._active_deployment_count += 1
            initial_legal = (
                -1.0 <= float(normalized_xy[0]) <= 1.0
                and -1.0 <= float(normalized_xy[1]) <= 1.0
                and lon_min <= lon <= lon_max
                and lat_min <= lat <= lat_max
            )
            target_legal = (
                True if self.dynamic_lifecycle
                else (
                    0 <= target_index < len(self._targets)
                    and bool(
                        self._targets[target_index].get("actor_visible", True)
                    )
                    and not bool(
                        self._targets[target_index].get("slot_empty", False)
                    )
                )
            )
            self._deployment_legal_count += int(initial_legal and target_legal)
        decision = {
            "presence": presence,
            "lon": lon if presence else 0.0,
            "lat": lat if presence else 0.0,
            "target_index": target_index if presence else -1,
            "target_id": int(selected_target["entity_id"]) if selected_target else -1,
            "target_lon": float(selected_target["position"]["lon"]) if selected_target else 0.0,
            "target_lat": float(selected_target["position"]["lat"]) if selected_target else 0.0,
            "deploy_bounds": tuple(float(value) for value in deploy_bounds),
        }
        self._deployment[entity_id] = decision
        if self.training:
            self._episode_samples.append(
                _StoredSample(
                    agent_id=encoder.agent_id,
                    entity_id=entity_id,
                    step=-1,
                    observation=observation,
                    critic_state=torch.as_tensor(
                        self._current_global_state
                        if self.trajectory_counterfactual else observation,
                        dtype=torch.float32,
                    ),
                    target_coordinates=coordinates,
                    target_valid_mask=target_valid,
                    target_features=target_features,
                    mask_presence=bool(output.action_mask.presence[0].item()),
                    mask_initial_position=bool(output.action_mask.initial_position[0].item()),
                    mask_search_position=bool(output.action_mask.search_position[0].item()),
                    mask_retarget=bool(output.action_mask.retarget[0].item()),
                    mask_target=bool(output.action_mask.target[0].item()),
                    mask_maneuver=bool(output.action_mask.maneuver[0].item()),
                    mask_satellite=bool(output.action_mask.satellite[0].item()),
                    action=self._action_without_batch(output.action),
                    old_log_prob=output.log_prob[0].detach().cpu(),
                    old_value=value,
                    reward=None,
                    done=True,
                    next_observation=torch.zeros_like(observation),
                    next_critic_state=torch.zeros(
                        self.global_encoder.state_dim, dtype=torch.float32
                    ) if self.trajectory_counterfactual else torch.zeros_like(observation),
                    credit_anchor=not self.dynamic_lifecycle,
                )
            )
        return dict(decision)

    def deployment_target_id(self, entity_id: int) -> int | None:
        decision = self._deployment.get(int(entity_id))
        if not decision or not decision["presence"]:
            return None
        return int(decision["target_id"])

    def consume_lifecycle_decision(self, entity_id: int) -> dict[str, Any]:
        if not self.dynamic_lifecycle:
            raise RuntimeError("当前未启用逐步生命周期动作")
        entity_id = int(entity_id)
        decision = self._prepared_lifecycle.pop(entity_id, None)
        if decision is None:
            raise RuntimeError(f"实体 {entity_id} 没有待消费的生命周期动作")
        kind = str(decision["kind"])
        if kind in {"wait", "launch", "retarget"}:
            encoder = self._encoders[entity_id]
            agent_id = int(encoder.agent_id)
            target_id = int(decision["target_id"])
            if self.target_index_for(target_id) is not None:
                # WAIT may legally preselect a target before first launch.
                self._current_target_by_agent[agent_id] = target_id
            if kind == "launch":
                self._dynamic_launched_ids.add(entity_id)
            if self.training and kind in {"launch", "retarget"}:
                sample = self._pending_motion.get(entity_id)
                if sample is None:
                    raise RuntimeError(
                        f"实体 {entity_id} 的生命周期动作缺少 transition"
                    )
                sample.credit_anchor = True
                self._option_anchor_by_agent[agent_id] = sample
                self._target_anchor_by_agent.setdefault(agent_id, {})[
                    target_id
                ] = sample
        return decision

    def select_maneuver(
        self,
        entity_id: int,
        encoded: np.ndarray,
    ) -> int:
        entity_id = int(entity_id)
        if entity_id in self._prepared_maneuvers:
            return self._prepared_maneuvers.pop(entity_id)
        zeros = torch.zeros(1, dtype=torch.bool, device=self.device)
        ones = torch.ones(1, dtype=torch.bool, device=self.device)
        maneuver_mask = ones
        output, observation, coordinates, target_valid, target_features, value = self._sample(
            encoded,
            HybridActionMask(
                presence=zeros,
                initial_position=zeros,
                search_position=zeros,
                retarget=zeros,
                target=zeros,
                maneuver=maneuver_mask,
                satellite=zeros,
            ),
            agent_id=int(self._encoders[entity_id].agent_id),
            entity_id=entity_id,
        )
        if self.training:
            if entity_id in self._pending_motion:
                raise RuntimeError(f"实体 {entity_id} 存在未提交的机动 transition")
            encoder = self._encoders[entity_id]
            self._pending_motion[entity_id] = _StoredSample(
                agent_id=encoder.agent_id,
                entity_id=entity_id,
                step=int(round(float(encoded[0]) * self.max_steps)),
                observation=observation,
                critic_state=torch.as_tensor(
                    self._current_global_state
                    if self.trajectory_counterfactual else observation,
                    dtype=torch.float32,
                ),
                target_coordinates=coordinates,
                target_features=target_features,
                target_valid_mask=target_valid,
                mask_presence=False,
                mask_initial_position=False,
                mask_search_position=False,
                mask_retarget=False,
                mask_target=False,
                mask_maneuver=bool(output.action_mask.maneuver[0].item()),
                mask_satellite=False,
                action=self._action_without_batch(output.action),
                old_log_prob=output.log_prob[0].detach().cpu(),
                old_value=value,
                reward=None,
                done=False,
                next_observation=torch.zeros_like(observation),
                next_critic_state=torch.zeros(
                    self.global_encoder.state_dim, dtype=torch.float32
                ) if self.trajectory_counterfactual else torch.zeros_like(observation),
            )
        maneuver_index = int(output.action.maneuver_index[0].item())
        maneuver = int(output.action.maneuver[0].item())
        self._maneuver_decision_count += 1
        self._maneuver_legal_count += int(
            maneuver_index in (0, 1, 2) and maneuver in (-1, 0, 1)
        )
        return maneuver_index

    def observe_maneuver(self, entity_id: int, transition: PolicyTransition) -> None:
        if not self.training:
            return
        sample = self._pending_motion.pop(int(entity_id), None)
        if sample is None:
            raise RuntimeError(f"实体 {entity_id} 没有待提交的机动 transition")
        self._commit_step_sample(sample, transition)

    def has_pending_step(self, entity_id: int) -> bool:
        """当前仿真步是否已为该实体执行前向并保存样本。"""

        return int(entity_id) in self._pending_motion

    def observe_waiting(self, entity_id: int, transition: PolicyTransition) -> None:
        """提交存活但尚未发射实体的逐步共享奖励。"""

        if not self.training:
            return
        sample = self._pending_motion.pop(int(entity_id), None)
        if sample is None:
            raise RuntimeError(f"实体 {entity_id} 没有待提交的未发射 transition")
        if sample.mask_maneuver:
            raise RuntimeError(f"实体 {entity_id} 的待提交 transition 已启用机动")
        self._commit_step_sample(sample, transition)

    def _commit_step_sample(
        self,
        sample: _StoredSample,
        transition: PolicyTransition,
    ) -> None:
        sample.reward = float(transition.reward)
        sample.done = bool(transition.done)
        sample.next_observation = torch.as_tensor(
            self._augment_observation(
                transition.next_observation,
                self._next_team_context,
            ),
            dtype=torch.float32,
        )
        if self.trajectory_counterfactual:
            if self._next_global_state is None:
                raise RuntimeError("事实 transition 缺少下一 centralized state")
            sample.next_critic_state = torch.as_tensor(
                self._next_global_state,
                dtype=torch.float32,
            )
        if self.rollout_active_heads_only and not any((
            sample.mask_presence,
            sample.mask_initial_position,
            sample.mask_search_position,
            sample.mask_retarget,
            sample.mask_target,
            sample.mask_maneuver,
            sample.mask_satellite,
        )):
            return
        self._episode_samples.append(sample)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        del observation, action_mask
        raise RuntimeError("统一 MAPPO 需要携带实体标识的 select_maneuver 接口")

    def observe(self, transition: PolicyTransition) -> None:
        del transition
        raise RuntimeError("统一 MAPPO 需要携带实体标识的 observe_maneuver 接口")

    def begin_environment_step(self, full_observation: Mapping[str, Any]) -> None:
        self._cached_environment_step = False
        self._capture_target_runtime_states(full_observation)
        self._current_team_context = self._encode_team_context(full_observation)
        self._current_global_state = self.global_encoder.encode(full_observation)
        self._next_team_context = None
        self._next_global_state = None
        self._step_sample_start = len(self._episode_samples)
        self._credit_context = None

    def begin_cached_environment_step(
        self,
        full_observation: Mapping[str, Any],
    ) -> None:
        """Start a deterministic KEEP step without unused Actor/critic state."""

        self._cached_environment_step = True
        self._capture_target_runtime_states(full_observation)
        self._current_team_context = None
        self._current_global_state = None
        self._next_team_context = None
        self._next_global_state = None
        self._step_sample_start = len(self._episode_samples)
        self._credit_context = None

    def set_environment_credit_context(self, info: Mapping[str, Any]) -> None:
        self._credit_context = dict(info)

    def _teacher_command_target_index(
        self,
        command: Mapping[str, Any],
    ) -> int | None:
        target = command.get("target")
        if not target:
            return None
        target_lon = float(target["x"])
        target_lat = float(target["y"])
        candidates = []
        for index, target_id in enumerate(self._target_slot_ids):
            if target_id is None or target_id == SEARCH_TARGET_ID:
                continue
            template = self._target_slot_templates[int(target_id)]
            position = template["position"]
            distance = float(np.hypot(
                target_lon - float(position["lon"]),
                target_lat - float(position["lat"]),
            ))
            candidates.append((distance, index))
        distance, index = min(candidates)
        return int(index) if distance < 1e-4 else None

    def teacher_command_target_id(
        self,
        command: Mapping[str, Any],
        *,
        entity_type: int,
    ) -> int | None:
        """Resolve a native command to the actor-visible target catalogue."""

        index = self._teacher_command_target_index(command)
        if index is not None:
            target_id = self._target_slot_ids[index]
            return None if target_id is None else int(target_id)
        if (
            int(entity_type) == 21002
            and command.get("target")
            and self.target_index_for(SEARCH_TARGET_ID) is not None
        ):
            return SEARCH_TARGET_ID
        return None

    def build_teacher_distillation_batch(
        self,
        agent_observations: Mapping[int, Mapping[str, Any]],
        agents: Sequence[Any],
        native_actions: Sequence[Mapping[str, Any]],
        *,
        decision_step: int,
        sample_stride: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """Encode R9 actions as labels without adding them to PPO rollout."""

        commands_by_entity: dict[int, list[Mapping[str, Any]]] = {}
        for command in native_actions:
            entity_id = int(command.get("executor_id", -1))
            commands_by_entity.setdefault(entity_id, []).append(command)

        rows: dict[str, list[Any]] = {
            "observations": [],
            "target_coordinates": [],
            "target_features": [],
            "target_valid_mask": [],
            "factor_mask": [],
            "presence": [],
            "initial_xy": [],
            "search_index": [],
            "retarget": [],
            "target_index": [],
            "maneuver_index": [],
            "satellite": [],
            "agent_ids": [],
            "entity_ids": [],
            "steps": [],
        }
        skipped_target_count = 0
        for agent in agents:
            entity_id = int(getattr(agent, "entity_id", -1))
            observation = agent_observations.get(int(agent.agent_id))
            encoder = self._encoders.get(entity_id)
            if encoder is None or not observation:
                continue

            commands = commands_by_entity.get(entity_id, ())
            by_type = {
                int(command.get("commandType_id", -1)): command
                for command in commands
            }
            launch_command = by_type.get(200)
            retarget_command = by_type.get(3014)
            maneuver_command = by_type.get(3007)
            satellite_command = by_type.get(3013)
            launch_step = int(getattr(agent, "launch_step", -1))
            launched = launch_step >= 0
            current_target_index = agent._learning_target_index()
            encoded = encoder.encode(
                observation,
                launched=launched,
                launch_step=launch_step,
                satellite_used=bool(
                    observation["self"]["is_using_satellite"]
                ),
                maneuver_state=int(
                    getattr(agent, "last_learning_maneuver", 0)
                ),
                current_target_index=current_target_index,
                task_context=None,
            )
            coordinates, target_valid = self._target_tensors_for_entity(
                entity_id
            )
            target_valid = target_valid[0]
            routine = (
                (int(decision_step) + entity_id) % max(1, int(sample_stride))
                == 0
            )

            presence_mask = not launched
            initial_mask = launch_command is not None
            retarget_mask = bool(
                launched and (retarget_command is not None or routine)
            )
            target_command = launch_command or retarget_command
            teacher_target_index = (
                self._teacher_command_target_index(target_command)
                if target_command is not None
                else current_target_index
            )
            entity_type = int(observation["self"].get("type", -1))
            search_mask = bool(
                target_command is not None
                and entity_type == 21002
                and (
                    teacher_target_index is None
                    or not target_valid[int(teacher_target_index)].item()
                )
            )
            search_index = 0
            if search_mask:
                teacher_target_index = self.target_index_for(SEARCH_TARGET_ID)
                command_target = target_command["target"]
                search_index = self._position_to_search_index(
                    float(command_target["x"]),
                    float(command_target["y"]),
                )
            target_mask = bool(
                teacher_target_index is not None
                and target_valid[int(teacher_target_index)].item()
                and target_command is not None
            )
            if target_command is not None and not target_mask:
                skipped_target_count += 1

            maneuver_value = (
                float(maneuver_command.get("acc_z", 0.0))
                if maneuver_command is not None else 0.0
            )
            maneuver_index = (
                0 if maneuver_value < 0.0 else 2 if maneuver_value > 0.0 else 1
            )
            maneuver_mask = bool(
                launched
                and encoder.last_interceptor_track_count > 0
                and (maneuver_index != 1 or routine)
            )
            satellite_available = bool(
                int(observation["self"]["satellite_remaining_uses"]) > 0
                and not bool(observation["self"]["is_using_satellite"])
            )
            satellite_mask = bool(
                satellite_available
                and (satellite_command is not None or routine)
            )
            factor_mask = (
                presence_mask,
                initial_mask,
                search_mask,
                retarget_mask,
                target_mask,
                maneuver_mask,
                satellite_mask,
            )
            if not any(factor_mask):
                continue

            deploy_bounds = self._deployment[entity_id]["deploy_bounds"]
            lon_min, lon_max, lat_min, lat_max = map(float, deploy_bounds)
            position = observation["self"]["position"]
            normalized_xy = (
                2.0 * (float(position["lon"]) - lon_min) / (lon_max - lon_min) - 1.0,
                2.0 * (float(position["lat"]) - lat_min) / (lat_max - lat_min) - 1.0,
            )
            rows["observations"].append(torch.as_tensor(
                self._augment_observation(encoded, self._current_team_context),
                dtype=torch.float32,
            ))
            rows["target_coordinates"].append(coordinates[0].detach().cpu())
            rows["target_features"].append(torch.as_tensor(
                self._target_set_features(
                    encoded, entity_id=entity_id, observation=observation
                ), dtype=torch.float32
            ))
            rows["target_valid_mask"].append(target_valid.detach().cpu())
            rows["factor_mask"].append(factor_mask)
            rows["presence"].append(int(launch_command is not None))
            rows["initial_xy"].append(normalized_xy)
            rows["search_index"].append(search_index)
            rows["retarget"].append(int(retarget_command is not None))
            rows["target_index"].append(
                int(teacher_target_index) if teacher_target_index is not None else 0
            )
            rows["maneuver_index"].append(maneuver_index)
            rows["satellite"].append(int(satellite_command is not None))
            rows["agent_ids"].append(int(agent.agent_id))
            rows["entity_ids"].append(entity_id)
            rows["steps"].append(int(decision_step))

        tensor_rows = {
            "observations": torch.stack(rows["observations"]) if rows["observations"] else torch.empty((0, self.trainer.config.observation_dim), dtype=torch.float32),
            "target_coordinates": torch.stack(rows["target_coordinates"]) if rows["target_coordinates"] else torch.empty((0, TARGET_SLOTS, 2), dtype=torch.float32),
            "target_features": torch.stack(rows["target_features"]) if rows["target_features"] else torch.empty((0, TARGET_SLOTS, TARGET_SET_FEATURE_DIM), dtype=torch.float32),
            "target_valid_mask": torch.stack(rows["target_valid_mask"]) if rows["target_valid_mask"] else torch.empty((0, TARGET_SLOTS), dtype=torch.bool),
            "factor_mask": torch.as_tensor(rows["factor_mask"], dtype=torch.bool).reshape(-1, 7),
            "presence": torch.as_tensor(rows["presence"], dtype=torch.float32),
            "initial_xy": torch.as_tensor(rows["initial_xy"], dtype=torch.float32).reshape(-1, 2),
            "search_index": torch.as_tensor(rows["search_index"], dtype=torch.long),
            "retarget": torch.as_tensor(rows["retarget"], dtype=torch.float32),
            "target_index": torch.as_tensor(rows["target_index"], dtype=torch.long),
            "maneuver_index": torch.as_tensor(rows["maneuver_index"], dtype=torch.long),
            "satellite": torch.as_tensor(rows["satellite"], dtype=torch.float32),
            "agent_ids": torch.as_tensor(rows["agent_ids"], dtype=torch.long),
            "entity_ids": torch.as_tensor(rows["entity_ids"], dtype=torch.long),
            "steps": torch.as_tensor(rows["steps"], dtype=torch.long),
        }
        return tensor_rows, {"skipped_target_count": skipped_target_count}


    def _interventions_at_step(self, timestep: int) -> dict[int, dict[str, Any]]:
        if not self._replay_spec:
            return {}
        return {
            int(row["entity_id"]): dict(row)
            for row in self._replay_spec.get("interventions", ())
            if int(row.get("timestep", row.get("tau", -1))) == timestep
        }

    def _record_target_allocator_boundaries(
        self,
        candidates: Sequence[
            tuple[Any, Mapping[str, Any], np.ndarray, bool, int]
        ],
        observations: torch.Tensor,
        target_features: torch.Tensor,
        target_valid: torch.Tensor,
        output: Any,
    ) -> None:
        """Record legal local target sets without changing policy execution."""

        if self._target_allocator_trace_path is None:
            return
        selected = [
            index
            for index, (_, observation, _, _, _) in enumerate(candidates)
            if bool(output.action_mask.target[index].item())
            and int(observation.get("step", 0))
            % self._target_allocator_trace_stride == 0
        ]
        if not selected:
            return
        selected_tensor = torch.tensor(
            selected, dtype=torch.long, device=observations.device
        )
        with torch.no_grad():
            legacy_logits = self.trainer.model.distribution_parameters(
                observations.index_select(0, selected_tensor),
                target_features=target_features.index_select(0, selected_tensor),
                target_valid_mask=target_valid.index_select(0, selected_tensor),
            )["legacy_target_logits"]
        for row_index, candidate_index in enumerate(selected):
            agent, observation, _, _, _ = candidates[candidate_index]
            self._target_allocator_trace_rows.append({
                "observation": observations[candidate_index].detach().cpu(),
                "target_features": target_features[candidate_index].detach().cpu(),
                "target_valid_mask": target_valid[candidate_index].detach().cpu(),
                "legacy_target_logits": legacy_logits[row_index].detach().cpu(),
                "chosen_target_index": output.action.target_index[
                    candidate_index
                ].detach().cpu(),
                "agent_id": int(agent.agent_id),
                "entity_id": int(agent.entity_id),
                "step": int(observation.get("step", 0)),
            })

    def _flush_target_allocator_trace(self) -> None:
        if self._target_allocator_trace_path is None:
            return
        rows = self._target_allocator_trace_rows
        observation_dim = self.trainer.config.observation_dim
        target_slots = self.trainer.config.target_slots
        target_feature_dim = self.trainer.config.target_feature_dim
        payload = {
            "schema_version": 1,
            "observations": (
                torch.stack([row["observation"] for row in rows])
                if rows else torch.empty((0, observation_dim), dtype=torch.float32)
            ),
            "target_features": (
                torch.stack([row["target_features"] for row in rows])
                if rows else torch.empty(
                    (0, target_slots, target_feature_dim), dtype=torch.float32
                )
            ),
            "target_valid_mask": (
                torch.stack([row["target_valid_mask"] for row in rows])
                if rows else torch.empty((0, target_slots), dtype=torch.bool)
            ),
            "legacy_target_logits": (
                torch.stack([row["legacy_target_logits"] for row in rows])
                if rows else torch.empty((0, target_slots), dtype=torch.float32)
            ),
            "chosen_target_indices": torch.tensor(
                [int(row["chosen_target_index"]) for row in rows],
                dtype=torch.long,
            ),
            "agent_ids": torch.tensor(
                [row["agent_id"] for row in rows], dtype=torch.long
            ),
            "entity_ids": torch.tensor(
                [row["entity_id"] for row in rows], dtype=torch.long
            ),
            "steps": torch.tensor(
                [row["step"] for row in rows], dtype=torch.long
            ),
        }
        self._target_allocator_trace_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        torch.save(payload, self._target_allocator_trace_path)
        rows.clear()

    def prepare_environment_actions(
        self,
        agent_observations: Mapping[int, Mapping[str, Any]],
        agents: Sequence[Any],
        controlled_decision_types: Mapping[int, str] | None = None,
        controlled_action_components: str = "joint",
        persistent_student_entity_ids: set[int] | frozenset[int] | None = None,
    ) -> None:
        persistent_student_entity_ids = {
            int(entity_id)
            for entity_id in (persistent_student_entity_ids or ())
        }
        if self._prepared_maneuvers:
            raise RuntimeError("上一仿真步仍有未消费的批量机动动作")
        if self._prepared_lifecycle:
            raise RuntimeError("上一仿真步仍有未消费的生命周期动作")
        candidates: list[
            tuple[Any, Mapping[str, Any], np.ndarray, bool, int]
        ] = []
        expected_entity_ids = {
            int(getattr(agent, "entity_id", -1))
            for agent in agents
            if agent_observations.get(int(getattr(agent, "agent_id", -1)))
        }
        for agent in agents:
            entity_id = int(getattr(agent, "entity_id", -1))
            encoder = self._encoders.get(entity_id)
            observation = agent_observations.get(
                int(getattr(agent, "agent_id", -1))
            )
            if encoder is None or not observation:
                continue
            step = int(observation.get("step", 0))
            launch_step = int(getattr(agent, "launch_step", -1))
            maneuver_active = launch_step >= 0
            if launch_step < 0 and not self.dynamic_lifecycle:
                pending = getattr(
                    getattr(agent, "commander", None), "pending", {}
                ).get(entity_id)
                if pending is not None and int(pending[0]) <= step:
                    launch_step = step
                    maneuver_active = True
            target_index = agent._learning_target_index()
            encoded = encoder.encode(
                observation,
                launched=maneuver_active,
                launch_step=launch_step,
                satellite_used=bool(
                    observation["self"]["is_using_satellite"]
                ),
                maneuver_state=int(
                    getattr(agent, "last_learning_maneuver", 0)
                ),
                current_target_index=target_index,
                task_context=(
                    agent.commander.learning_task_context(entity_id)
                    if bool(getattr(agent, "hierarchical_learning", False))
                    else None
                ),
            )
            candidates.append((
                agent,
                observation,
                encoded,
                maneuver_active,
                int(observation["self"].get("type", -1)),
            ))
        candidate_entity_ids = {
            int(agent.entity_id) for agent, _, _, _, _ in candidates
        }
        if candidate_entity_ids != expected_entity_ids:
            missing = sorted(expected_entity_ids - candidate_entity_ids)
            raise RuntimeError(f"存活红方实体未全部进入逐步推理: {missing}")
        if not candidates:
            return

        observations = torch.as_tensor(
            np.stack([
                self._augment_observation(encoded, self._current_team_context)
                for _, _, encoded, _, _ in candidates
            ]),
            dtype=torch.float32,
            device=self.device,
        )
        target_features = torch.as_tensor(
            np.stack([
                self._target_set_features(
                    encoded, entity_id=int(agent.entity_id), observation=observation
                )
                for agent, observation, encoded, _, _ in candidates
            ]), dtype=torch.float32, device=self.device
        )
        target_views = [
            self._target_tensors_for_entity(int(agent.entity_id))
            for agent, _, _, _, _ in candidates
        ]
        coordinates = torch.cat([view[0] for view in target_views], dim=0)
        target_valid = torch.cat([view[1] for view in target_views], dim=0)
        if self.temporal_attack_options:
            target_valid = torch.stack([
                self._attack_option_target_valid_mask(
                    int(agent.entity_id), target_valid[index]
                )
                for index, (agent, _, _, _, _) in enumerate(candidates)
            ])

        zeros = torch.zeros(
            len(candidates), dtype=torch.bool, device=self.device
        )
        maneuver_active_mask = torch.tensor(
            [active for _, _, _, active, _ in candidates],
            dtype=torch.bool,
            device=self.device,
        )
        satellite_available_mask = torch.tensor(
            [
                int(observation["self"]["satellite_remaining_uses"]) > 0
                and not bool(
                    observation["self"]["is_using_satellite"]
                )
                for _, observation, _, _, _ in candidates
            ],
            dtype=torch.bool,
            device=self.device,
        )
        if self.dynamic_lifecycle:
            presence_mask = ~maneuver_active_mask
            initial_position_mask = presence_mask
            if self.temporal_attack_options:
                goal_boundaries = torch.tensor(
                    [
                        self._attack_option_boundary(
                            int(agent.entity_id),
                            target_valid[index],
                            int(observation.get("step", 0))
                            + int(self.trajectory_counterfactual),
                            observation,
                        )
                        for index, (
                            agent, observation, _, _, _
                        ) in enumerate(candidates)
                    ],
                    dtype=torch.bool,
                    device=self.device,
                )
                retarget_mask = zeros
                target_mask = goal_boundaries & target_valid.any(dim=-1)
            else:
                retarget_mask = maneuver_active_mask
                target_mask = torch.ones_like(maneuver_active_mask)
            search_position_mask = target_mask
            requested_maneuver_mask = torch.ones_like(maneuver_active_mask)
            requested_satellite_mask = satellite_available_mask
        else:
            presence_mask = zeros
            initial_position_mask = zeros
            retarget_mask = zeros
            target_mask = zeros
            search_position_mask = zeros
            requested_maneuver_mask = maneuver_active_mask
            requested_satellite_mask = (
                satellite_available_mask & maneuver_active_mask
            )
        student_goal_control_mask = torch.ones_like(target_mask)
        if controlled_decision_types is not None:
            controlled_decision_types = {
                int(entity_id): str(value).upper()
                for entity_id, value in controlled_decision_types.items()
            }
            entity_ids = [
                int(agent.entity_id) for agent, _, _, _, _ in candidates
            ]
            decision_types = [
                controlled_decision_types.get(entity_id, "")
                for entity_id in entity_ids
            ]
            selected_control_mask = torch.tensor(
                [
                    entity_id in controlled_decision_types
                    for entity_id in entity_ids
                ],
                dtype=torch.bool,
                device=self.device,
            )
            persistent_control_mask = torch.tensor(
                [
                    self.option_control_handoff
                    and entity_id in persistent_student_entity_ids
                    for entity_id in entity_ids
                ],
                dtype=torch.bool,
                device=self.device,
            )
            learn_lifecycle = controlled_action_components in {
                "lifecycle", "joint"
            }
            learn_goal = controlled_action_components in {
                "goal", "goal_position", "joint"
            }
            learn_position = controlled_action_components in {
                "position", "goal_position", "joint"
            }
            learn_search = controlled_action_components == "search"
            selected_search_control_mask = torch.tensor(
                [
                    learn_search and value in {"LAUNCH", "RETARGET"}
                    for value in decision_types
                ],
                dtype=torch.bool,
                device=self.device,
            )
            if bool(selected_search_control_mask.any().item()):
                search_target_index = self.target_index_for(SEARCH_TARGET_ID)
                if search_target_index is None:
                    raise RuntimeError("C0_search 缺少 SEARCH 固定目标槽位")
                for index in torch.nonzero(
                    selected_search_control_mask, as_tuple=False
                ).flatten().tolist():
                    if int(candidates[index][4]) != 21002:
                        raise RuntimeError("C0_search 只能控制 L 实体")
                    if not bool(
                        target_valid[index, int(search_target_index)].item()
                    ):
                        raise RuntimeError("C0_search 的 SEARCH 目标当前不合法")
                    target_valid[index] = False
                    target_valid[index, int(search_target_index)] = True
            base_target_mask = target_mask
            base_presence_mask = presence_mask
            base_initial_position_mask = initial_position_mask
            base_search_position_mask = search_position_mask
            base_maneuver_mask = requested_maneuver_mask
            base_satellite_mask = requested_satellite_mask
            selected_presence_mask = torch.tensor(
                [
                    learn_lifecycle and value == "LAUNCH"
                    for value in decision_types
                ],
                dtype=torch.bool,
                device=self.device,
            )
            presence_mask = selected_presence_mask | (
                persistent_control_mask & base_presence_mask
            )
            selected_initial_position_mask = torch.tensor(
                [
                    learn_position and value == "LAUNCH"
                    for value in decision_types
                ],
                dtype=torch.bool,
                device=self.device,
            )
            initial_position_mask = selected_initial_position_mask | (
                persistent_control_mask & base_initial_position_mask
            )
            selected_target_mask = torch.tensor(
                [
                    (learn_goal or learn_search)
                    and value in {
                        "LAUNCH", "RETARGET", "SEARCH", "REGION_SEARCH",
                        "AREA_SEARCH",
                    }
                    for value in decision_types
                ],
                dtype=torch.bool,
                device=self.device,
            ) & base_target_mask
            if self.temporal_attack_options:
                retarget_mask = zeros
            else:
                selected_retarget_mask = torch.tensor(
                    [
                        value in {
                            "RETARGET", "SEARCH", "REGION_SEARCH", "AREA_SEARCH"
                        }
                        for value in decision_types
                    ],
                    dtype=torch.bool,
                    device=self.device,
                )
                retarget_mask = (
                    selected_retarget_mask
                    | (persistent_control_mask & maneuver_active_mask)
                )
            target_mask = (
                selected_target_mask
                | (persistent_control_mask & base_target_mask)
            )
            selected_search_mask = torch.tensor(
                [
                    (
                        learn_goal
                        and value in {
                            "SEARCH", "REGION_SEARCH", "AREA_SEARCH"
                        }
                    )
                    or (
                        learn_search and value in {"LAUNCH", "RETARGET"}
                    )
                    for value in decision_types
                ],
                dtype=torch.bool,
                device=self.device,
            ) & base_target_mask
            search_position_mask = (
                selected_search_mask
                | (persistent_control_mask & base_search_position_mask)
            )
            option_control_mask = persistent_control_mask
            if self.option_control_handoff:
                option_control_mask = (
                    option_control_mask | selected_control_mask
                )
            requested_maneuver_mask = (
                base_maneuver_mask
                & option_control_mask
                & (zeros if learn_search else torch.ones_like(zeros))
            )
            requested_satellite_mask = (
                base_satellite_mask
                & option_control_mask
                & (zeros if learn_search else torch.ones_like(zeros))
            )
            student_goal_control_mask = target_mask
        search_grid_size = (
            self.trainer.config.search_grid_width
            * self.trainer.config.search_grid_height
        )
        search_valid_mask = torch.stack([
            (
                self._search_grid_reachable_mask(observation)
                if bool(search_position_mask[index].item())
                else torch.ones(
                    search_grid_size,
                    dtype=torch.bool,
                    device=self.device,
                )
            )
            for index, (_, observation, _, _, _) in enumerate(candidates)
        ])
        output = self.trainer.model.act(
            observations,
            coordinates,
            target_valid,
            HybridActionMask(
                presence=presence_mask,
                initial_position=initial_position_mask,
                search_position=search_position_mask,
                retarget=retarget_mask,
                target=target_mask,
                maneuver=requested_maneuver_mask,
                satellite=requested_satellite_mask,
            ),
            deterministic=not self.stochastic_actor,
            target_features=target_features,
            search_valid_mask=search_valid_mask,
            independent_target=self.temporal_attack_options,
        )
        self._record_target_allocator_boundaries(
            candidates,
            observations,
            target_features,
            target_valid,
            output,
        )
        if self.trajectory_counterfactual and self.training:
            if self._current_global_state is None:
                raise RuntimeError("strict CTDE 批量采样缺少全局状态")
            critic_states = torch.as_tensor(
                np.repeat(
                    self._current_global_state[None, :],
                    len(candidates),
                    axis=0,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            critic_agent_ids = torch.tensor(
                [int(row[0].agent_id) for row in candidates],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                sampled_values = self.trainer.model.critic_value(
                    critic_states, critic_agent_ids, observations
                )
        else:
            sampled_values = output.value
        current_step = int(candidates[0][1].get("step", 0)) + int(
            self.trajectory_counterfactual
        )
        self._prepared_replay_step = current_step
        self._prepared_replay_semantics.clear()

        self._alive_inference_expected_count += len(expected_entity_ids)
        self._step_inference_count += len(candidates)
        for index, (
            agent,
            observation,
            _,
            maneuver_active,
            entity_type,
        ) in enumerate(candidates):
            entity_id = int(agent.entity_id)
            action_has_maneuver = bool(
                output.action_mask.maneuver[index].item()
            )
            maneuver_index = int(
                output.action.maneuver_index[index].item()
            )
            maneuver = int(output.action.maneuver[index].item())
            if (
                self.deterministic_evasion
                and entity_type == 21002
                and entity_id in persistent_student_entity_ids
                and maneuver_active
            ):
                maneuver_index = self._deterministic_evasion_maneuver(
                    entity_id, observation
                )
                maneuver = maneuver_index - 1
            enters_field = bool(maneuver_active) or (
                self.dynamic_lifecycle
                and int(output.action.presence[index].item()) == 1
            )
            if enters_field:
                self._prepared_maneuvers[entity_id] = maneuver_index
            if action_has_maneuver:
                self._maneuver_decision_count += 1
                self._maneuver_legal_count += int(
                    maneuver_index in (0, 1, 2) and maneuver in (-1, 0, 1)
                )
            if not maneuver_active:
                self._waiting_inference_count += 1

            if self.dynamic_lifecycle:
                current_target_id = agent.commander.target_id_for(entity_id)
                active_option = self._attack_option_by_entity.get(entity_id)
                if bool(output.action_mask.target[index].item()):
                    selected_index = int(
                        output.action.target_index[index].item()
                    )
                    selected_target = self._targets[selected_index]
                    selected_target_id = int(selected_target["entity_id"])
                    self._target_selection_count += 1
                    legal_target = bool(
                        target_valid[index, selected_index].item()
                    )
                    self._target_selection_legal_count += int(legal_target)
                    if entity_type == 21002 and int(
                        selected_target.get("type", -1)
                    ) in {9400, 9600}:
                        self._low_invalid_target_count += 1
                else:
                    if self.temporal_attack_options and active_option is not None:
                        selected_index = int(active_option["target_index"])
                        selected_target_id = int(active_option["target_id"])
                    else:
                        selected_index = self.target_index_for(current_target_id)
                        selected_index = (
                            int(selected_index) if selected_index is not None else -1
                        )
                        selected_target_id = (
                            int(current_target_id)
                            if current_target_id is not None else -1
                        )
                if not maneuver_active:
                    kind = (
                        "launch"
                        if int(output.action.presence[index].item()) == 1
                        else "wait"
                    )
                elif self.temporal_attack_options:
                    search_route_boundary = bool(
                        getattr(self, "search_option_chaining", False)
                        and bool(output.action_mask.target[index].item())
                        and bool(
                            output.action_mask.search_position[index].item()
                        )
                        and selected_target_id == SEARCH_TARGET_ID
                    )
                    kind = (
                        "retarget"
                        if bool(output.action_mask.target[index].item())
                        and (
                            selected_target_id != current_target_id
                            or search_route_boundary
                        )
                        else "keep"
                    )
                else:
                    kind = (
                        "retarget"
                        if int(output.action.retarget[index].item()) == 1
                        else "keep"
                    )
                if selected_target_id == SEARCH_TARGET_ID:
                    if bool(output.action_mask.search_position[index].item()):
                        target_lon, target_lat = self._search_xy_to_position(
                            output.action.search_xy[index]
                        )
                    elif self.temporal_attack_options and active_option is not None:
                        target_lon = float(active_option["target_lon"])
                        target_lat = float(active_option["target_lat"])
                    else:
                        target_lon, target_lat = self._search_xy_to_position(
                            output.action.search_xy[index]
                        )
                elif bool(output.action_mask.target[index].item()):
                    target_lon = float(output.action.target_xy[index, 0].item())
                    target_lat = float(output.action.target_xy[index, 1].item())
                elif self.temporal_attack_options and active_option is not None:
                    target_lon = float(active_option["target_lon"])
                    target_lat = float(active_option["target_lat"])
                elif self.temporal_attack_options:
                    target_lon = 0.0
                    target_lat = 0.0
                else:
                    target_lon = float(output.action.target_xy[index, 0].item())
                    target_lat = float(output.action.target_xy[index, 1].item())
                if self.temporal_attack_options:
                    at_goal_boundary = bool(
                        output.action_mask.target[index].item()
                    )
                    if at_goal_boundary:
                        self._attack_option_boundary_count += 1
                    if kind == "keep":
                        self._attack_option_keep_count += 1
                    elif kind == "retarget":
                        self._attack_option_retarget_count += 1

                    if (
                        selected_target_id == SEARCH_TARGET_ID
                        and not bool(getattr(
                            self, "search_option_chaining", False
                        ))
                    ):
                        self._terminate_attack_option(
                            entity_id, "search_selected"
                        )
                    elif selected_index >= 0 and kind != "wait":
                        previous_controller = (
                            active_option.get("controller")
                            if active_option is not None else None
                        )
                        controller = (
                            "student"
                            if bool(student_goal_control_mask[index].item())
                            else previous_controller or "teacher"
                        )
                        previous_decision_step = (
                            int(active_option["last_decision_step"])
                            if active_option is not None
                            else current_step
                        )
                        self._attack_option_by_entity[entity_id] = {
                            "target_index": selected_index,
                            "target_id": selected_target_id,
                            "target_lon": target_lon,
                            "target_lat": target_lat,
                            "last_decision_step": (
                                current_step
                                if at_goal_boundary
                                else previous_decision_step
                            ),
                            "controller": controller,
                            "completed": False,
                            "option_kind": (
                                "search"
                                if selected_target_id == SEARCH_TARGET_ID
                                else "attack"
                            ),
                        }
                self._prepared_replay_semantics[entity_id] = {
                    "agent_id": int(agent.agent_id),
                    "kind": kind,
                    "goal_boundary": at_goal_boundary,
                    "current_target_id": current_target_id,
                    "selected_target_id": selected_target_id,
                    "selected_index": selected_index,
                    "maneuver_active": bool(maneuver_active),
                    "action_has_maneuver": bool(action_has_maneuver),
                    "raw_maneuver": int(output.action.maneuver[index].item()),
                    "satellite_request": int(
                        output.action.satellite[index].item()
                    ),
                }
                normalized_xy = output.action.initial_xy[index]
                deploy_bounds = self._deployment.get(entity_id, {}).get(
                    "deploy_bounds"
                )
                if deploy_bounds is None:
                    deploy_bounds = (-180.0, 180.0, -90.0, 90.0)
                lon_min, lon_max, lat_min, lat_max = map(float, deploy_bounds)
                initial_lon = lon_min + (
                    float(normalized_xy[0].item()) + 1.0
                ) * 0.5 * (lon_max - lon_min)
                initial_lat = lat_min + (
                    float(normalized_xy[1].item()) + 1.0
                ) * 0.5 * (lat_max - lat_min)
                self._prepared_lifecycle[entity_id] = {
                    "kind": kind,
                    "goal_boundary": at_goal_boundary,
                    "target_index": selected_index,
                    "target_id": selected_target_id,
                    "target_lon": target_lon,
                    "target_lat": target_lat,
                    "initial_lon": initial_lon if kind == "launch" else 0.0,
                    "initial_lat": initial_lat if kind == "launch" else 0.0,
                    "satellite_request": int(
                        output.action.satellite[index].item()
                    ),
                }

            if not self.training:
                continue
            if entity_id in self._pending_motion:
                raise RuntimeError(
                    f"实体 {entity_id} 存在未提交的机动 transition"
                )
            self._pending_motion[entity_id] = _StoredSample(
                agent_id=int(agent.agent_id),
                entity_id=entity_id,
                step=current_step,
                observation=observations[index].detach().cpu(),
                critic_state=torch.as_tensor(
                    self._current_global_state if self.trajectory_counterfactual
                    else observations[index].detach().cpu(),
                    dtype=torch.float32,
                ),
                target_coordinates=coordinates[index].detach().cpu(),
                target_valid_mask=target_valid[index].detach().cpu(),
                target_features=target_features[index].detach().cpu(),
                mask_presence=bool(
                    output.action_mask.presence[index].item()
                ),
                mask_initial_position=bool(
                    output.action_mask.initial_position[index].item()
                ),
                mask_search_position=bool(
                    output.action_mask.search_position[index].item()
                ),
                mask_retarget=bool(
                    output.action_mask.retarget[index].item()
                ),
                mask_target=bool(output.action_mask.target[index].item()),
                mask_maneuver=bool(
                    output.action_mask.maneuver[index].item()
                ),
                mask_satellite=bool(
                    output.action_mask.satellite[index].item()
                ),
                action=self._action_at_index(output.action, index),
                old_log_prob=output.log_prob[index].detach().cpu(),
                old_value=sampled_values[index].detach().cpu(),
                reward=None,
                done=False,
                next_observation=torch.zeros_like(
                    observations[index]
                ).cpu(),
                next_critic_state=torch.zeros(
                    self.global_encoder.state_dim, dtype=torch.float32
                ) if self.trajectory_counterfactual else torch.zeros_like(observations[index]).cpu(),
                search_valid_mask=search_valid_mask[index].detach().cpu(),
            )

    def fork_and_apply_replay_interventions(self) -> None:
        """Fork every requested NULL branch from one shared factual prefix."""

        if not self.replay_probe:
            return
        if self._prepared_replay_step is None:
            raise RuntimeError("反事实分叉前没有准备好的联合动作")
        current_step = int(self._prepared_replay_step)

        if self._replay_branch_role == "prefix":
            requests = self._replay_requests_by_step.pop(current_step, [])
            if not requests:
                return
            if self.device.type != "cpu" or torch.get_num_threads() != 1:
                raise RuntimeError("反事实 COW 分叉要求单线程 CPU 推理")
            for request in requests:
                while len(self._replay_children) >= self._replay_max_children:
                    self._wait_one_replay_child()
                python_random_state = random.getstate()
                child_pid = os.fork()
                if child_pid == 0:
                    random.setstate(python_random_state)
                    self._replay_branch_role = "counterfactual"
                    self._replay_spec = dict(request)
                    self._replay_requests = ()
                    self._replay_requests_by_step = {}
                    self._replay_children = {}
                    self._replay_factual_pid = None
                    self._applied_interventions.clear()
                    break
                self._replay_children[int(child_pid)] = str(
                    request["request_id"]
                )
            else:
                return

        if self._replay_branch_role != "counterfactual":
            raise RuntimeError(
                f"未知反事实分支角色: {self._replay_branch_role}"
            )

        interventions = self._interventions_at_step(current_step)
        for entity_id, intervention in interventions.items():
            semantic = self._prepared_replay_semantics.get(entity_id)
            if semantic is None:
                raise RuntimeError(
                    f"NULL 干预实体 {entity_id} 在 tau={current_step} 未参与推理"
                )
            requested = str(intervention.get("decision_type", "")).upper()
            is_search = requested in {"SEARCH", "REGION_SEARCH", "AREA_SEARCH"}
            is_maneuver = requested in {
                "MANEUVER", "INTERCEPT", "INTERCEPT_MANEUVER"
            }
            is_satellite = requested == "SATELLITE_REQUEST"
            semantic_match = {
                "LAUNCH": semantic["kind"] == "launch",
                "RETARGET": semantic["kind"] == "retarget",
                "TARGET_SELECT": (
                    semantic.get("goal_boundary", False)
                    and semantic["kind"] == "keep"
                ),
                "SEARCH": (
                    semantic["kind"] in {"launch", "retarget"}
                    and semantic["selected_target_id"] == SEARCH_TARGET_ID
                ),
                "REGION_SEARCH": (
                    semantic["kind"] in {"launch", "retarget"}
                    and semantic["selected_target_id"] == SEARCH_TARGET_ID
                ),
                "AREA_SEARCH": (
                    semantic["kind"] in {"launch", "retarget"}
                    and semantic["selected_target_id"] == SEARCH_TARGET_ID
                ),
                "MANEUVER": (
                    semantic["action_has_maneuver"]
                    and semantic["raw_maneuver"] != 0
                ),
                "INTERCEPT": semantic["action_has_maneuver"],
                "INTERCEPT_MANEUVER": semantic["action_has_maneuver"],
                "SATELLITE_REQUEST": semantic["satellite_request"] == 1,
            }.get(requested, False)
            if (
                str(self._replay_spec.get("kind", "single")) == "single"
                and not semantic_match
            ):
                raise RuntimeError(
                    f"NULL 干预 {requested} 与事实动作语义不一致: "
                    f"entity={entity_id}, timestep={current_step}"
                )

            lifecycle = self._prepared_lifecycle.get(entity_id)
            if requested == "LAUNCH":
                if lifecycle is None:
                    raise RuntimeError("LAUNCH NULL 缺少生命周期动作")
                lifecycle.update({
                    "kind": "wait",
                    "initial_lon": 0.0,
                    "initial_lat": 0.0,
                })
                self._prepared_maneuvers.pop(entity_id, None)
            elif requested in {"RETARGET", "TARGET_SELECT"} or is_search:
                if lifecycle is None:
                    raise RuntimeError(f"{requested} NULL 缺少生命周期动作")
                current_target_id = semantic["current_target_id"]
                current_index = self.target_index_for(current_target_id)
                lifecycle.update({
                    "kind": "keep" if semantic["maneuver_active"] else "wait",
                    "target_index": (
                        int(current_index) if current_index is not None else -1
                    ),
                    "target_id": (
                        int(current_target_id)
                        if current_target_id is not None else -1
                    ),
                    "initial_lon": 0.0,
                    "initial_lat": 0.0,
                })
                if not semantic["maneuver_active"]:
                    self._prepared_maneuvers.pop(entity_id, None)
            elif is_maneuver:
                self._prepared_maneuvers[entity_id] = 1
            elif is_satellite:
                lifecycle["satellite_request"] = 0
            else:
                raise RuntimeError(f"未知 NULL 干预类型: {requested}")

            self._applied_interventions.append({
                "entity_id": int(entity_id),
                "agent_id": int(semantic["agent_id"]),
                "timestep": current_step,
                "decision_type": str(intervention["decision_type"]),
                "null_action": str(intervention.get("null_action", "")),
                "application": "cow_tau_null_replacement",
            })

    def end_environment_step(self, full_observation: Mapping[str, Any]) -> None:
        self._capture_target_runtime_states(full_observation)
        entities = full_observation.get("entities") or {}
        for entity_id, option in tuple(self._attack_option_by_entity.items()):
            own = entities.get(entity_id, entities.get(str(entity_id)))
            if (
                own is None
                or float(own.get("health", 0.0)) <= 0.0
                or not bool(own.get("isVisible", False))
            ):
                self._terminate_attack_option(entity_id, "entity_terminated")
                continue
        if bool(getattr(self, "_cached_environment_step", False)):
            self._next_team_context = None
            self._next_global_state = None
            return
        self._next_team_context = self._encode_team_context(full_observation)
        self._next_global_state = self.global_encoder.encode(full_observation)

    def finish_environment_step(self) -> None:
        if self._prepared_lifecycle:
            raise RuntimeError(
                f"仿真步结束时仍有 {len(self._prepared_lifecycle)} 个未消费的生命周期动作"
            )
        if self._prepared_maneuvers:
            raise RuntimeError(
                f"仿真步结束时仍有 {len(self._prepared_maneuvers)} 个未消费的批量机动动作"
            )
        if self._pending_motion:
            raise RuntimeError(
                f"仿真步结束时仍有 {len(self._pending_motion)} 条未回填机动 transition"
            )
        self._prepared_replay_semantics.clear()
        self._prepared_replay_step = None
        self._cached_environment_step = False
        if self.training and self.reward_mode in COUNTERFACTUAL_REWARD_MODES:
            if self.trajectory_counterfactual:
                self._collect_trajectory_counterfactual_step()
            else:
                self._assign_counterfactual_step_credit()




    def _collect_trajectory_counterfactual_step(self) -> None:
        """Collect factual evidence; credit is computed only after CF probes."""

        if self._credit_context is None:
            raise RuntimeError("真实轨迹反事实信用缺少当前目标奖励上下文")
        for sample in self._episode_samples[self._step_sample_start:]:
            if sample.step >= 0:
                # Event-time target reward never enters the PPO rollout.
                # It is later added exactly once at the causal decision tau.
                sample.reward = 0.0
                if sample.mask_maneuver:
                    self._launched_agent_ids.add(sample.agent_id)

        timestep = int(self._credit_context.get("step", 0))
        target_rewards = {
            int(key): float(value)
            for key, value in self._credit_context.get("target_rewards", {}).items()
        }
        self._target_reward_history.append({
            "timestep": timestep,
            "target_rewards": dict(target_rewards),
        })
        self._causal_decisions.extend(
            dict(row) for row in self._credit_context.get("decision_events", ())
        )
        self._simulator_causal_events.extend(
            dict(row) for row in self._credit_context.get("causal_events", ())
        )
        if not any(value > 1e-12 for value in target_rewards.values()):
            return
        entity_to_agent = {
            int(entity_id): int(encoder.agent_id)
            for entity_id, encoder in self._encoders.items()
        }
        self._factual_target_events.extend(
            build_factual_target_events(
                episode_id=self._episode_index,
                timestep=timestep,
                target_rewards=target_rewards,
                decision_events=self._causal_decisions,
                simulator_events=self._simulator_causal_events,
                entity_to_agent=entity_to_agent,
            )
        )

    def _assign_counterfactual_step_credit(self) -> None:
        if self._credit_context is None:
            raise RuntimeError("反事实信用缺少当前仿真步的目标奖励")
        current_samples = self._episode_samples[self._step_sample_start:]
        for sample in current_samples:
            if sample.step >= 0:
                sample.reward = 0.0
                if sample.mask_maneuver:
                    self._launched_agent_ids.add(sample.agent_id)

        target_ids = [
            int(target["entity_id"])
            for target in self._targets
            if not bool(target.get("slot_empty", False))
        ]
        target_index = {
            int(target["entity_id"]): index
            for index, target in enumerate(self._targets)
            if not bool(target.get("slot_empty", False))
        }
        target_rewards_by_id = {
            int(key): float(value)
            for key, value in self._credit_context.get("target_rewards", {}).items()
        }
        if not any(value > 1e-12 for value in target_rewards_by_id.values()):
            return
        unmapped_target_ids = sorted(
            target_id
            for target_id, reward in target_rewards_by_id.items()
            if reward > 1e-12 and target_id not in target_index
        )
        if unmapped_target_ids:
            raise RuntimeError(
                f"非零奖励目标不在 centralized critic 槽位中: {unmapped_target_ids}"
            )

        deployment_samples = {
            sample.agent_id: sample
            for sample in self._episode_samples
            if sample.step < 0 and sample.mask_presence
        }
        if self.dynamic_lifecycle:
            anchors_by_agent = dict(self._option_anchor_by_agent)
            target_by_agent = dict(self._current_target_by_agent)
        else:
            anchors_by_agent = deployment_samples
            target_by_agent: dict[int, int] = {}
            for entity_id, decision in self._deployment.items():
                encoder = self._encoders.get(int(entity_id))
                if encoder is not None and bool(decision.get("presence", 0)):
                    target_by_agent[int(encoder.agent_id)] = int(
                        decision["target_id"]
                    )
        direct_by_target = {
            int(target_id): {
                int(agent_id): float(value)
                for agent_id, value in values.items()
            }
            for target_id, values in self._credit_context.get(
                "target_agent_rewards", {}
            ).items()
        }
        decision_anchored = (
            os.getenv("RED_REWARD_MODE")
            == "weighted_damage_decision_anchored"
        )
        current_step = int(self._credit_context.get("step", 0))
        if decision_anchored:
            self._assign_decision_anchored_credit(
                anchors_by_agent=anchors_by_agent,
                target_ids=target_ids,
                target_index=target_index,
                target_rewards_by_id=target_rewards_by_id,
                direct_by_target=direct_by_target,
                current_step=current_step,
            )
            return

        candidate_ids = sorted(
            agent_id
            for agent_id in anchors_by_agent
            if agent_id in self._launched_agent_ids
            and (
                target_rewards_by_id.get(target_by_agent.get(agent_id, -1), 0.0)
                > 1e-12
                or any(agent_id in values for values in direct_by_target.values())
            )
        )
        if not candidate_ids:
            raise RuntimeError("非零目标奖励没有已发射的因果候选实体")

        anchors = [anchors_by_agent[agent_id] for agent_id in candidate_ids]
        observations = torch.stack([sample.observation for sample in anchors]).to(
            self.device
        )
        actions = self._stack_actions(anchors)
        semantics = actions.semantic_tensor().to(
            self.device,
            dtype=observations.dtype,
        )
        with torch.no_grad():
            actual_q = self.trainer.model.target_action_values(
                observations,
                semantics,
            )
            null_q = self.trainer.model.target_action_values(
                observations,
                torch.zeros_like(semantics),
            )
            factual_q = actual_q.sum(dim=0, keepdim=True)
            counterfactual_q = (
                factual_q.unsqueeze(1)
                - actual_q.unsqueeze(0)
                + null_q.unsqueeze(0)
            )
            target_rewards = torch.zeros(
                1,
                TARGET_SLOTS,
                dtype=observations.dtype,
                device=self.device,
            )
            candidate_mask = torch.zeros(
                1,
                len(candidate_ids),
                TARGET_SLOTS,
                dtype=torch.bool,
                device=self.device,
            )
            direct_credit = torch.zeros_like(candidate_mask, dtype=observations.dtype)
            for raw_target_id, reward in target_rewards_by_id.items():
                index = target_index.get(raw_target_id)
                if index is None:
                    continue
                target_rewards[0, index] = reward
                direct_values = direct_by_target.get(raw_target_id, {})
                for row, agent_id in enumerate(candidate_ids):
                    candidate_mask[0, row, index] = (
                        target_by_agent.get(agent_id) == raw_target_id
                        or agent_id in direct_values
                    )
                    direct_credit[0, row, index] = direct_values.get(agent_id, 0.0)
            credit = normalized_counterfactual_credit(
                target_rewards=target_rewards,
                factual_q=factual_q,
                counterfactual_q=counterfactual_q,
                candidate_mask=candidate_mask,
                direct_damage=direct_credit,
            )

        latest_by_agent: dict[int, _StoredSample] = {}
        for sample in self._episode_samples:
            if sample.step >= 0:
                latest_by_agent[sample.agent_id] = sample
        direct_agent_ids = {
            agent_id for values in direct_by_target.values() for agent_id in values
        }
        for row, agent_id in enumerate(candidate_ids):
            reward = float(credit.agent_rewards[0, row].item())
            anchor = anchors_by_agent[agent_id]
            destination = latest_by_agent.get(agent_id, anchor)
            destination.reward = float(destination.reward or 0.0) + reward
            per_target_credit = (
                credit.coefficients[0, row] * target_rewards[0]
            ).detach().cpu().to(dtype=torch.float32)
            if anchor.target_credits is None:
                anchor.target_credits = torch.zeros(
                    TARGET_SLOTS, dtype=torch.float32
                )
            anchor.target_credits += per_target_credit
            if reward > 1e-12 and agent_id not in direct_agent_ids:
                self._delayed_credit_count += 1

        used_counterfactual = bool(
            (credit.marginal_contributions.sum(dim=1) > 1e-12).any().item()
        )
        if used_counterfactual:
            self._counterfactual_step_count += 1
        else:
            self._direct_fallback_step_count += 1
        self._max_counterfactual_conservation_error = max(
            self._max_counterfactual_conservation_error,
            float(credit.conservation_error.max().item()),
        )

    def _assign_decision_anchored_credit(
        self,
        *,
        anchors_by_agent: Mapping[int, _StoredSample],
        target_ids: list[int],
        target_index: Mapping[int, int],
        target_rewards_by_id: Mapping[int, float],
        direct_by_target: Mapping[int, Mapping[int, float]],
        current_step: int,
    ) -> None:
        candidate_ids = sorted(
            agent_id
            for agent_id in anchors_by_agent
            if agent_id in self._launched_agent_ids
        )
        if not candidate_ids:
            raise RuntimeError("非零目标奖励没有已发射的因果候选实体")

        used_counterfactual = False
        max_conservation_error = 0.0
        for target_id, reward in target_rewards_by_id.items():
            if reward <= 1e-12:
                continue
            index = target_index[target_id]
            selected_anchors: list[_StoredSample] = []
            target_match: list[bool] = []
            for agent_id in candidate_ids:
                exact_anchor = self._target_anchor_by_agent.get(
                    agent_id, {}
                ).get(target_id)
                selected_anchors.append(
                    exact_anchor if exact_anchor is not None
                    else anchors_by_agent[agent_id]
                )
                target_match.append(exact_anchor is not None)

            self._decision_anchor_candidate_count += len(candidate_ids)
            self._decision_anchor_indirect_candidate_count += sum(
                not matched for matched in target_match
            )
            stored_observations = torch.stack([
                sample.observation for sample in selected_anchors
            ])
            actions = self._stack_actions(selected_anchors)
            stored_semantics = actions.semantic_tensor().detach().cpu()
            group_key = (index, tuple(id(sample) for sample in selected_anchors))
            group_position = self._joint_target_return_index.get(group_key)
            if group_position is None:
                self._joint_target_return_index[group_key] = len(
                    self._joint_target_returns
                )
                self._joint_target_returns.append(
                    JointTargetReturn(
                        observations=stored_observations.detach().cpu(),
                        action_semantics=stored_semantics,
                        target_index=index,
                        target_return=float(reward),
                    )
                )
            else:
                existing = self._joint_target_returns[group_position]
                self._joint_target_returns[group_position] = JointTargetReturn(
                    observations=existing.observations,
                    action_semantics=existing.action_semantics,
                    target_index=existing.target_index,
                    target_return=existing.target_return + float(reward),
                )
            observations = stored_observations.to(self.device)
            semantics = stored_semantics.to(
                self.device, dtype=observations.dtype
            )
            with torch.no_grad():
                all_actual_q = self.trainer.model.target_action_values(
                    observations, semantics
                )
                all_null_q = self.trainer.model.target_action_values(
                    observations, torch.zeros_like(semantics)
                )
                actual_q = all_actual_q[:, index].reshape(1, -1, 1)
                null_q = all_null_q[:, index].reshape(1, -1, 1)
                factor_count = float(len(candidate_ids))
                factual_q = actual_q.mean(dim=1)
                counterfactual_q = (
                    factual_q.unsqueeze(1)
                    - actual_q / factor_count
                    + null_q / factor_count
                )
                target_reward = torch.tensor(
                    [[reward]], dtype=observations.dtype, device=self.device
                )
                candidate_mask = torch.ones(
                    1, len(candidate_ids), 1,
                    dtype=torch.bool, device=self.device
                )
                direct_credit = torch.zeros(
                    1, len(candidate_ids), 1,
                    dtype=observations.dtype, device=self.device
                )
                direct_values = direct_by_target.get(target_id, {})
                for row, agent_id in enumerate(candidate_ids):
                    direct_credit[0, row, 0] = direct_values.get(agent_id, 0.0)
                credit = normalized_counterfactual_credit(
                    target_rewards=target_reward,
                    factual_q=factual_q,
                    counterfactual_q=counterfactual_q,
                    candidate_mask=candidate_mask,
                    direct_damage=direct_credit,
                )

            for row, agent_id in enumerate(candidate_ids):
                destination = selected_anchors[row]
                amount = float(credit.agent_rewards[0, row].item())
                if amount <= 1e-12:
                    continue
                destination.reward = float(destination.reward or 0.0) + amount
                delay = max(0, current_step - int(destination.step))
                self._decision_anchored_assignment_count += 1
                self._decision_anchored_reward_sum += amount
                self._decision_anchored_delay_sum += delay
                self._decision_anchored_max_delay = max(
                    self._decision_anchored_max_delay, delay
                )
                self._decision_anchored_agent_ids.add(agent_id)
                if target_match[row]:
                    self._decision_anchor_target_match_count += 1
                else:
                    self._decision_anchor_indirect_count += 1
                if agent_id not in direct_values:
                    self._delayed_credit_count += 1

            used_counterfactual = used_counterfactual or bool(
                (credit.marginal_contributions.sum(dim=1) > 1e-12).any().item()
            )
            max_conservation_error = max(
                max_conservation_error,
                float(credit.conservation_error.max().item()),
            )

        if used_counterfactual:
            self._counterfactual_step_count += 1
        else:
            self._direct_fallback_step_count += 1
        self._max_counterfactual_conservation_error = max(
            self._max_counterfactual_conservation_error,
            max_conservation_error,
        )

    def counterfactual_episode_inputs(self) -> dict[str, Any]:
        """Return immutable-by-convention factual data for CF subprocesses."""

        if not self.trajectory_counterfactual:
            raise RuntimeError("当前 reward mode 不是轨迹反事实模式")
        return {
            "factual_events": tuple(dict(row) for row in self._factual_target_events),
            "target_rewards": tuple(
                {
                    "timestep": int(row["timestep"]),
                    "target_rewards": dict(row["target_rewards"]),
                }
                for row in self._target_reward_history
            ),
        }

    @property
    def applied_interventions(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._applied_interventions)

    @property
    def replay_branch_role(self) -> str:
        return self._replay_branch_role

    @property
    def replay_factual_pid(self) -> int | None:
        return self._replay_factual_pid

    def apply_trajectory_credit(self, result: Any | None) -> None:
        """Backfill measured CF credit into the sole factual PPO rollout."""

        if not self.trajectory_counterfactual or not self.training:
            raise RuntimeError("仅训练中的轨迹反事实模式可以回填信用")
        if self._trajectory_credit_finalized:
            raise RuntimeError("本回合轨迹信用已经完成，禁止重复记录")
        expected_event_ids = {
            row["event_id"] for row in self._factual_target_events
        }
        if result is None:
            if expected_event_ids:
                raise RuntimeError("存在正目标奖励时不能省略反事实信用结果")
            self._last_credit_validation = {
                "event_count": 0,
                "all_invariants_hold": True,
            }
            self._trajectory_credit_finalized = True
            return

        assignments = tuple(result.assignments)
        observed_event_ids = {
            assignment.event.event_id for assignment in assignments
        }
        if observed_event_ids != expected_event_ids:
            raise RuntimeError(
                "反事实信用事件集合与事实奖励不一致: "
                f"expected={sorted(map(str, expected_event_ids))}, "
                f"observed={sorted(map(str, observed_event_ids))}"
            )
        if self._applied_credit_event_ids & observed_event_ids:
            raise RuntimeError("检测到目标奖励事件重复回填")

        samples_by_key: dict[tuple[int, int], list[_StoredSample]] = {}
        for sample in self._episode_samples:
            samples_by_key.setdefault((sample.agent_id, sample.step), []).append(sample)
        for (raw_agent_id, raw_tau), amount in result.backfilled_rewards.items():
            key = (int(raw_agent_id), int(raw_tau))
            destinations = samples_by_key.get(key, [])
            if len(destinations) != 1:
                raise RuntimeError(
                    f"信用回填锚点 {key} 匹配 {len(destinations)} 条事实样本"
                )
            destination = destinations[0]
            destination.reward = float(destination.reward or 0.0) + float(amount)
            destination.credit_anchor = True

        for assignment in assignments:
            self._applied_credit_event_ids.add(assignment.event.event_id)
            if assignment.strategy.value == "single_counterfactual_normalization":
                self._counterfactual_step_count += 1
            else:
                self._direct_fallback_step_count += 1
            self._max_counterfactual_conservation_error = max(
                self._max_counterfactual_conservation_error,
                abs(
                    sum(item.credit for item in assignment.candidate_credits)
                    + (
                        assignment.event.reward
                        if assignment.strategy.value
                        == "no_causal_effect_zero_credit"
                        else 0.0
                    )
                    - assignment.event.reward
                ),
            )
            for item in assignment.candidate_credits:
                if item.credit <= 1e-12:
                    continue
                delay = assignment.event.timestep - item.candidate.tau
                self._decision_anchored_assignment_count += 1
                self._decision_anchored_reward_sum += item.credit
                self._decision_anchored_delay_sum += delay
                self._decision_anchored_max_delay = max(
                    self._decision_anchored_max_delay, delay
                )
                self._decision_anchored_agent_ids.add(int(item.candidate.agent_id))
                if "damage" not in item.candidate.categories:
                    self._delayed_credit_count += 1

        self._last_credit_validation = asdict(result.validation)
        if not bool(result.validation.all_invariants_hold):
            raise RuntimeError("反事实信用未通过奖励/折扣守恒校验")
        self._trajectory_credit_finalized = True

    def apply_native_handoff_credit(
        self,
        credits: Sequence[Mapping[str, Any]],
    ) -> None:
        """Write COW-gated formal target reward only to guided MAPPO rows."""

        samples = {
            (int(sample.entity_id), int(sample.step)): sample
            for sample in self._episode_samples
        }
        if self.option_control_handoff:
            # C3a records every student-controlled step of an option. Preserve
            # its real transitions so GAE can train the complete closed loop,
            # and conserve the option's COW return across those action rows.
            for sample in self._episode_samples:
                sample.reward = 0.0
                sample.credit_anchor = False
            for row in credits:
                entity_id = int(row["executor_id"])
                timestep = int(row["timestep"])
                key = (entity_id, timestep)
                root_sample = samples.get(key)
                if root_sample is None:
                    raise RuntimeError(f"C3a 信用根样本不存在: {key}")
                option_samples = sorted(
                    (
                        sample
                        for sample in self._episode_samples
                        if int(sample.entity_id) == entity_id
                        and int(sample.step) >= timestep
                    ),
                    key=lambda sample: int(sample.step),
                )
                if not option_samples:
                    raise RuntimeError(f"C3a 选项没有闭环样本: {key}")
                for previous, following in zip(
                    option_samples, option_samples[1:]
                ):
                    if int(following.step) != int(previous.step) + 1:
                        raise RuntimeError(
                            f"C3a 选项样本不连续: {previous.step}->{following.step}"
                        )
                amount = float(row["credit"])
                reward_share = amount / len(option_samples)
                for sample in option_samples:
                    sample.reward = float(sample.reward or 0.0) + reward_share
                    sample.credit_anchor = True
                target_index = self.target_index_for(int(row["target_id"]))
                if target_index is not None:
                    if root_sample.target_credits is None:
                        root_sample.target_credits = torch.zeros(
                            TARGET_SLOTS, dtype=torch.float32
                        )
                    root_sample.target_credits[target_index] += amount
        else:
            # C0/C1/C2 contain isolated controlled decision rows rather than a
            # continuous option, so each remains a one-step terminal sample.
            for sample in self._episode_samples:
                sample.done = True
                sample.next_observation = torch.zeros_like(sample.observation)
                sample.next_critic_state = torch.zeros_like(sample.critic_state)
                sample.reward = 0.0
                sample.credit_anchor = True
            for row in credits:
                key = (int(row["executor_id"]), int(row["timestep"]))
                sample = samples[key]
                amount = float(row["credit"])
                sample.reward = float(sample.reward or 0.0) + amount
                target_index = self.target_index_for(int(row["target_id"]))
                if target_index is not None:
                    if sample.target_credits is None:
                        sample.target_credits = torch.zeros(
                            TARGET_SLOTS, dtype=torch.float32
                        )
                    sample.target_credits[target_index] += amount
        self._guided_cow_unit_count = len(self._episode_samples)
        self._guided_cow_positive_count = sum(
            float(sample.reward or 0.0) > 1e-12
            for sample in self._episode_samples
        )
        self._guided_cow_reward_sum = sum(
            float(sample.reward or 0.0) for sample in self._episode_samples
        )

    @staticmethod
    def _stack_actions(samples: list[_StoredSample]) -> HybridAction:
        return HybridAction(
            presence=torch.stack([item.action.presence for item in samples]),
            retarget=torch.stack([item.action.retarget for item in samples]),
            initial_xy=torch.stack([item.action.initial_xy for item in samples]),
            search_xy=torch.stack([item.action.search_xy for item in samples]),
            target_xy=torch.stack([item.action.target_xy for item in samples]),
            maneuver=torch.stack([item.action.maneuver for item in samples]),
            initial_raw=torch.stack([item.action.initial_raw for item in samples]),
            search_index=torch.stack([item.action.search_index for item in samples]),
            target_index=torch.stack([item.action.target_index for item in samples]),
            maneuver_index=torch.stack([item.action.maneuver_index for item in samples]),
            satellite=torch.stack([item.action.satellite for item in samples]),
        )

    def finish_episode(self, official_score: float) -> dict[str, float]:
        self._last_episode_score = float(official_score)
        self._flush_target_allocator_trace()
        if not self.training:
            self._pending_motion.clear()
            self._episode_samples.clear()
            return dict(self.last_metrics)
        if (
            self.reward_mode == TRAJECTORY_COUNTERFACTUAL_REWARD_MODE
            and not self._trajectory_credit_finalized
        ):
            raise RuntimeError(
                "真实反事实轨迹尚未完成信用回填，禁止更新 PPO"
            )
        if self._prepared_maneuvers:
            raise RuntimeError(
                f"回合结束时仍有 {len(self._prepared_maneuvers)} 个未消费的批量机动动作"
            )
        if self._pending_motion:
            raise RuntimeError(
                f"回合结束时仍有 {len(self._pending_motion)} 条未提交机动 transition"
            )
        individual_reward_mode = (
            os.getenv("RED_REWARD_MODE") in INDIVIDUAL_REWARD_MODES
        )
        for sample in self._episode_samples:
            if sample.reward is None:
                if individual_reward_mode:
                    later_samples = [
                        item
                        for item in self._episode_samples
                        if item.agent_id == sample.agent_id and item.step >= 0
                    ]
                    sample.reward = 0.0
                    if later_samples:
                        first_sample = min(later_samples, key=lambda item: item.step)
                        sample.done = False
                        sample.next_observation = first_sample.observation.clone()
                        sample.next_critic_state = first_sample.critic_state.clone()
                    else:
                        sample.done = True
                else:
                    score_reward = float(official_score) / 100.0
                    sample.reward = (
                        score_reward * float(sample.action.presence.item())
                        if sample.mask_presence else score_reward
                    )
        samples = sorted(self._episode_samples, key=lambda item: (item.agent_id, item.step))
        self._last_deployment_transition_count = sum(item.mask_presence for item in samples)
        self._last_maneuver_transition_count = sum(item.mask_maneuver for item in samples)
        self._last_waiting_transition_count = sum(
            item.step >= 0
            and item.mask_presence
            and int(item.action.presence.item()) == 0
            for item in samples
        )
        if not samples:
            return dict(self.last_metrics)
        rollout = HybridRolloutBatch(
            observations=torch.stack([item.observation for item in samples]),
            target_coordinates=torch.stack([item.target_coordinates for item in samples]),
            target_valid_mask=torch.stack([item.target_valid_mask for item in samples]),
            target_features=torch.stack([item.target_features for item in samples]),
            action_mask=HybridActionMask(
                presence=torch.tensor([item.mask_presence for item in samples]),
                initial_position=torch.tensor([item.mask_initial_position for item in samples]),
                search_position=torch.tensor([item.mask_search_position for item in samples]),
                retarget=torch.tensor([item.mask_retarget for item in samples]),
                target=torch.tensor([item.mask_target for item in samples]),
                maneuver=torch.tensor([item.mask_maneuver for item in samples]),
                satellite=torch.tensor([item.mask_satellite for item in samples]),
            ),
            actions=self._stack_actions(samples),
            old_log_probs=torch.stack([item.old_log_prob for item in samples]),
            old_values=torch.stack([item.old_value for item in samples]),
            rewards=torch.tensor([float(item.reward) for item in samples]),
            dones=torch.tensor([float(item.done) for item in samples]),
            next_observations=torch.stack([item.next_observation for item in samples]),
            critic_states=torch.stack([item.critic_state for item in samples]),
            next_critic_states=torch.stack([item.next_critic_state for item in samples]),
            agent_ids=torch.tensor([item.agent_id for item in samples]),
            steps=torch.tensor([item.step for item in samples]),
            search_valid_mask=torch.stack([
                (
                    item.search_valid_mask
                    if item.search_valid_mask is not None
                    else torch.ones(
                        self.trainer.config.search_grid_width
                        * self.trainer.config.search_grid_height,
                        dtype=torch.bool,
                    )
                )
                for item in samples
            ]),
            target_credits=torch.stack([
                item.target_credits
                if item.target_credits is not None
                else torch.zeros(TARGET_SLOTS, dtype=torch.float32)
                for item in samples
            ]),
            credit_anchor_mask=torch.tensor([
                item.credit_anchor for item in samples
            ]),
            joint_target_returns=tuple(self._joint_target_returns),
            joint_target_mode=(
                os.getenv("RED_REWARD_MODE")
                == "weighted_damage_decision_anchored"
            ),
            per_agent_advantage_normalization=self.trajectory_counterfactual,
        )
        rollout_output = os.getenv("RED_UNIFIED_ROLLOUT_OUTPUT")
        if rollout_output:
            output_path = Path(rollout_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": 2, "rollout": rollout}, output_path)
            metrics = {
                "rollout_sample_count": float(rollout.batch_size),
                "rollout_reward_sum": float(rollout.rewards.sum().item()),
                "update_deferred": 1.0,
            }
            self.trainer.last_metrics = metrics
            self._episode_samples.clear()
            self._joint_target_returns.clear()
            self._joint_target_return_index.clear()
            self._episode_index += 1
            return metrics
        rollout_device = self.device
        requested_update_device = os.getenv(
            "RED_UNIFIED_UPDATE_DEVICE", "cpu"
        )
        update_device = self.trainer.resolve_runtime_device(
            requested_update_device
        )
        try:
            self.trainer.move_runtime_device(update_device)
            metrics = self.trainer.update(rollout)
        finally:
            # Factual inference and the next episode's frozen replay prefix must
            # use the checkpoint's CPU RNG stream. Never persist a CUDA runtime
            # device into the trajectory-counterfactual checkpoint contract.
            self.trainer.move_runtime_device(rollout_device)
        self._last_ppo_update_device = str(update_device)
        metrics = dict(metrics)
        metrics["ppo_update_used_cuda"] = float(update_device.type == "cuda")
        self.trainer.last_metrics = metrics
        self._episode_samples.clear()
        self._joint_target_returns.clear()
        self._joint_target_return_index.clear()
        self._episode_index += 1
        return metrics

    def diagnostics(self) -> dict[str, Any]:
        visible_targets = [
            target
            for target in self._targets
            if bool(target.get("actor_visible", True))
            and not bool(target.get("slot_empty", False))
        ]
        target_histogram: dict[str, int] = {}
        if self.dynamic_lifecycle:
            target_ids = self._current_target_by_agent.values()
        else:
            target_ids = (
                int(decision["target_id"])
                for decision in self._deployment.values()
                if decision["presence"]
            )
        for raw_target_id in target_ids:
            target_id = str(int(raw_target_id))
            target_histogram[target_id] = target_histogram.get(target_id, 0) + 1
        deployment_count = len(self._deployment)
        active_count = len(self.active_entity_ids)
        participating_count = (
            len(self._dynamic_launched_ids)
            if self.dynamic_lifecycle else active_count
        )
        return {
            "algorithm": (
                "target_conditioned_ctde_mappo_v8"
                if self.trajectory_counterfactual
                else "unified_hybrid_mappo"
            ),
            "agent_identity_map": {
                str(entity_id): int(encoder.agent_id)
                for entity_id, encoder in sorted(self._encoders.items())
            },
            "training": self.training,
            "episode_index": self._episode_index,
            "deployment_count": deployment_count,
            "active_count": active_count,
            "presence_rate": (
                participating_count / deployment_count
                if deployment_count else 0.0
            ),
            "target_count": len(visible_targets),
            "target_ids": [
                int(target["entity_id"]) for target in visible_targets
            ],
            "fixed_target_slots": True,
            "target_slot_ids": list(self._target_slot_ids),
            "target_slot_visibility": [
                bool(target.get("actor_visible", True))
                and not bool(target.get("slot_empty", False))
                for target in self._targets
            ],
            "actor_visible_target_ids_by_entity": {
                str(entity_id): sorted(int(target_id) for target_id in target_ids)
                for entity_id, target_ids in sorted(
                    self._actor_visible_target_ids_by_entity.items()
                )
            },
            "actor_runtime_target_ids": sorted(
                int(target_id) for target_id in self._actor_runtime_target_ids
            ),
            "actor_target_state_features": [
                "health_ratio", "damage_ratio", "alive",
            ],
            "actor_target_state_discovery_gated": True,
            "target_histogram": target_histogram,
            "update_count": self.update_count,
            "transition_count": self.transition_count,
            "last_episode_score": self._last_episode_score,
            "deployment_legal_rate": (
                self._deployment_legal_count / self._active_deployment_count
                if self._active_deployment_count else 1.0
            ),
            "maneuver_legal_rate": (
                self._maneuver_legal_count / self._maneuver_decision_count
                if self._maneuver_decision_count else 1.0
            ),
            "maneuver_decision_count": self._maneuver_decision_count,
            "deterministic_evasion": self.deterministic_evasion,
            "deterministic_evasion_trigger_count": (
                self._deterministic_evasion_trigger_count
            ),
            "deterministic_evasion_active_step_count": (
                self._deterministic_evasion_active_step_count
            ),
            "deterministic_evasion_visible_threat_count": (
                self._deterministic_evasion_visible_threat_count
            ),
            "alive_inference_expected_count": self._alive_inference_expected_count,
            "step_inference_count": self._step_inference_count,
            "waiting_inference_count": self._waiting_inference_count,
            "last_deployment_transition_count": self._last_deployment_transition_count,
            "last_maneuver_transition_count": self._last_maneuver_transition_count,
            "last_waiting_transition_count": self._last_waiting_transition_count,
            "counterfactual_step_count": self._counterfactual_step_count,
            "direct_fallback_step_count": self._direct_fallback_step_count,
            "delayed_credit_count": self._delayed_credit_count,
            "max_counterfactual_conservation_error": (
                self._max_counterfactual_conservation_error
            ),
            "dynamic_lifecycle": self.dynamic_lifecycle,
            "temporal_attack_options": self.temporal_attack_options,
            "attack_option_boundary_count": self._attack_option_boundary_count,
            "attack_option_keep_count": self._attack_option_keep_count,
            "attack_option_retarget_count": self._attack_option_retarget_count,
            "attack_option_termination_count": (
                self._attack_option_termination_count
            ),
            "attack_option_termination_reasons": dict(
                self._attack_option_termination_reasons
            ),
            "search_option_distance_sample_count": (
                self._search_option_distance_sample_count
            ),
            "search_option_min_distance_km": (
                self._search_option_min_distance_km
                if np.isfinite(self._search_option_min_distance_km) else None
            ),
            "search_option_last_distance_km": (
                self._search_option_last_distance_km
            ),
            "search_option_emergency_boundary_count": (
                self._search_option_emergency_boundary_count
            ),
            "search_reachable_mask": self.search_reachable_mask,
            "rollout_active_heads_only": self.rollout_active_heads_only,
            "search_leg_min_distance_km": self.search_leg_min_distance_km,
            "search_leg_max_distance_km": self.search_leg_max_distance_km,
            "search_reachable_mask_sample_count": (
                self._search_reachable_mask_sample_count
            ),
            "search_reachable_valid_count_mean": (
                self._search_reachable_valid_count_sum
                / self._search_reachable_mask_sample_count
                if self._search_reachable_mask_sample_count else None
            ),
            "search_reachable_valid_count_min": (
                self._search_reachable_valid_count_min
                if self._search_reachable_mask_sample_count else None
            ),
            "search_reachable_valid_count_max": (
                self._search_reachable_valid_count_max
                if self._search_reachable_mask_sample_count else None
            ),
            "attack_option_active_count": len(self._attack_option_by_entity),
            "student_attack_option_active_count": len(
                self.active_student_option_entity_ids
            ),
            "external_target_assignment_count": len(
                self._external_target_by_entity
            ),
            "external_target_sync_count": self._external_target_sync_count,
            "external_target_feature_hit_count": (
                self._external_target_feature_hit_count
            ),
            "external_target_fallback_count": (
                self._external_target_fallback_count
            ),
            "attack_option_min_dwell_steps": (
                self.attack_option_min_dwell_steps
            ),
            "option_control_handoff": self.option_control_handoff,
            "retarget_interval": 0 if self.temporal_attack_options else 1,
            "target_selection_legal_rate": (
                self._target_selection_legal_count
                / self._target_selection_count
                if self._target_selection_count else 1.0
            ),
            "low_invalid_target_count": self._low_invalid_target_count,
            "decision_anchored_assignment_count": (
                self._decision_anchored_assignment_count
            ),
            "decision_anchored_reward_sum": self._decision_anchored_reward_sum,
            "decision_anchored_agent_count": len(
                self._decision_anchored_agent_ids
            ),
            "decision_anchored_mean_delay": (
                self._decision_anchored_delay_sum
                / self._decision_anchored_assignment_count
                if self._decision_anchored_assignment_count else 0.0
            ),
            "decision_anchored_max_delay": self._decision_anchored_max_delay,
            "decision_anchor_target_match_count": (
                self._decision_anchor_target_match_count
            ),
            "decision_anchor_indirect_count": self._decision_anchor_indirect_count,
            "decision_anchor_candidate_count": self._decision_anchor_candidate_count,
            "decision_anchor_indirect_candidate_count": (
                self._decision_anchor_indirect_candidate_count
            ),
            "team_context_dim": TEAM_CONTEXT_DIM,
            "model_observation_dim": self.trainer.config.observation_dim,
            "critic_state_dim": self.trainer.config.critic_state_dim,
            "strict_ctde": self.trainer.model.strict_ctde,
            "actor_uses_team_context": not self.trajectory_counterfactual,
            "critic_focal_observation_dim": (
                self.trainer.config.critic_focal_observation_dim
            ),
            "rollout_device": str(self.device),
            "ppo_update_device": self._last_ppo_update_device,
            "factual_target_event_count": len(self._factual_target_events),
            "causal_decision_event_count": len(self._causal_decisions),
            "simulator_causal_event_count": len(self._simulator_causal_events),
            "applied_intervention_count": len(self._applied_interventions),
            "trajectory_credit_finalized": self._trajectory_credit_finalized,
            "trajectory_credit_validation": dict(self._last_credit_validation),
            "guided_cow_unit_count": self._guided_cow_unit_count,
            "guided_cow_positive_count": self._guided_cow_positive_count,
            "guided_cow_reward_sum": self._guided_cow_reward_sum,
            "last_metrics": dict(self.last_metrics),
        }

    def reset_episode(self) -> None:
        if self._pending_motion:
            raise RuntimeError("重置时仍有未提交的统一 MAPPO 机动 transition")
        self._episode_samples.clear()
        self._prepared_maneuvers.clear()
        self._deterministic_evasion_state.clear()
        self._deterministic_evasion_trigger_count = 0
        self._deterministic_evasion_active_step_count = 0
        self._deterministic_evasion_visible_threat_count = 0
        self._prepared_lifecycle.clear()
        self._actor_visible_target_ids_by_entity.clear()
        for entity_id, encoder in self._encoders.items():
            encoder.set_targets(self._targets_for_entity(entity_id))
            encoder.set_target_runtime_states({})
        self._deployment.clear()
        self._current_team_context = None
        self._full_target_runtime_states.clear()
        self._initial_target_health.clear()
        self._actor_runtime_target_ids = frozenset()
        self._next_team_context = None
        self._current_global_state = None
        self._next_global_state = None
        self._team_type_totals.clear()
        self._deployment_legal_count = 0
        self._active_deployment_count = 0
        self._maneuver_legal_count = 0
        self._maneuver_decision_count = 0
        self._alive_inference_expected_count = 0
        self._step_inference_count = 0
        self._waiting_inference_count = 0
        self._last_deployment_transition_count = 0
        self._last_maneuver_transition_count = 0
        self._last_waiting_transition_count = 0
        self._step_sample_start = 0
        self._credit_context = None
        self._launched_agent_ids.clear()
        self._counterfactual_step_count = 0
        self._direct_fallback_step_count = 0
        self._delayed_credit_count = 0
        self._max_counterfactual_conservation_error = 0.0
        self._option_anchor_by_agent.clear()
        self._target_anchor_by_agent.clear()
        self._joint_target_returns.clear()
        self._joint_target_return_index.clear()
        self._current_target_by_agent.clear()
        self._attack_option_by_entity.clear()
        self._external_target_by_entity.clear()
        self._external_target_sync_count = 0
        self._external_target_feature_hit_count = 0
        self._external_target_fallback_count = 0
        self._attack_option_boundary_count = 0
        self._attack_option_keep_count = 0
        self._attack_option_retarget_count = 0
        self._attack_option_termination_count = 0
        self._attack_option_termination_reasons.clear()
        self._search_option_distance_sample_count = 0
        self._search_option_min_distance_km = float("inf")
        self._search_option_last_distance_km = None
        self._search_option_emergency_boundary_count = 0
        self._search_reachable_mask_sample_count = 0
        self._search_reachable_valid_count_sum = 0
        self._search_reachable_valid_count_min = 16 * 12
        self._search_reachable_valid_count_max = 0
        self._dynamic_launched_ids.clear()
        self._target_selection_count = 0
        self._target_selection_legal_count = 0
        self._low_invalid_target_count = 0
        self._decision_anchored_assignment_count = 0
        self._decision_anchored_reward_sum = 0.0
        self._decision_anchored_delay_sum = 0
        self._decision_anchored_max_delay = 0
        self._decision_anchored_agent_ids.clear()
        self._decision_anchor_target_match_count = 0
        self._decision_anchor_indirect_count = 0
        self._decision_anchor_candidate_count = 0
        self._decision_anchor_indirect_candidate_count = 0
        self._causal_decisions.clear()
        self._simulator_causal_events.clear()
        self._factual_target_events.clear()
        self._target_reward_history.clear()
        self._applied_interventions.clear()
        self._prepared_replay_step = None
        self._prepared_replay_semantics.clear()
        self._replay_branch_role = "prefix" if self.replay_probe else "none"
        self._replay_factual_pid = None
        self._replay_spec = None
        self._replay_children.clear()
        self._reset_replay_schedule()
        self._applied_credit_event_ids.clear()
        self._last_credit_validation = {}
        self._trajectory_credit_finalized = False
        self._guided_cow_unit_count = 0
        self._guided_cow_positive_count = 0
        self._guided_cow_reward_sum = 0.0

    def set_training(self, training: bool) -> None:
        self.training = bool(training)
        self.stochastic_actor = (
            self.training or self.replay_probe or self.score_only
        ) and not getattr(self, "force_deterministic_actor", False)
        self.trainer.model.train(self.training)

    def save(self, path: str) -> None:
        self.trainer.save(path)

    def load(self, path: str) -> None:
        loaded = HybridMAPPOTrainer.load(
            path,
            device=str(self.device),
            target_feature_dim=TARGET_SET_FEATURE_DIM,
        )
        if loaded.config.observation_dim != self.trainer.config.observation_dim:
            raise ValueError(
                f"统一 MAPPO checkpoint 观测维度为 {loaded.config.observation_dim}，"
                f"当前需要 {self.trainer.config.observation_dim}"
            )
        if loaded.config.critic_state_dim != self.trainer.config.critic_state_dim:
            raise ValueError(
                "统一 MAPPO checkpoint 的 centralized critic 状态维度不兼容"
            )
        if (
            loaded.config.critic_focal_observation_dim
            != self.trainer.config.critic_focal_observation_dim
        ):
            raise ValueError(
                "统一 MAPPO checkpoint 的 focal-agent critic 观测维度不兼容"
            )
        if loaded.config.target_feature_dim != TARGET_SET_FEATURE_DIM:
            raise ValueError("统一 MAPPO checkpoint 的目标特征维度迁移失败")
        self.trainer = loaded
