"""BC-ERCA feasibility top layer with a frozen low-level motion policy.

The top layer only consumes the commander's legal red-side state.  Environment
ground truth is used solely as a scalar training reward by the caller; it is
never encoded into :class:`BCERCAHighLevelPolicy`'s deployment observation.
"""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baselines import (
    Assignment,
    BaselineObservation,
    BaselineRules,
    PlatformState,
    TargetPrior,
    distance_km,
    engagement_effectiveness,
)


_TARGET_HEALTH = {9400: 32.0, 9500: 1.6, 9600: 20.0}
_TARGET_REFERENCE_DAMAGE = {9400: 16.0, 9500: 0.8, 9600: 12.0}
_EXPECTED_DAMAGE = {
    "H": {9400: 16.0, 9600: 12.0},
    "M": {9400: 4.0, 9600: 3.0},
    "L": {9500: 0.8},
}


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass
class BCERCAConfig:
    """All train/deploy knobs for the feasibility top layer."""

    global_dim: int = 10
    platform_dim: int = 5
    target_dim: int = 9
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 32
    replay_size: int = 4096
    warmup_transitions: int = 32
    train_interval: int = 20
    target_tau: float = 0.005
    regularization: float = 1e-4
    rho: float = 1.0
    eta: float = 1.0
    kappa: float = 0.02
    gumbel_temperature: float = 0.50
    gumbel_min_temperature: float = 0.10
    exploration_decay_decisions: int = 1000
    terminal_prior_hm: float = 0.10
    terminal_prior_l: float = 0.05
    distance_weight: float = 0.02
    release_fraction: float = 0.30
    seed: int = 20260904
    max_steps: int = 1200
    device: str = "auto"

    @classmethod
    def from_env(cls, *, max_steps: int, seed: int) -> "BCERCAConfig":
        return cls(
            hidden_dim=_env_int("RED_TOP_HIDDEN_DIM", 64),
            learning_rate=_env_float("RED_TOP_LEARNING_RATE", 3e-4),
            gamma=_env_float("RED_TOP_GAMMA", 0.99),
            batch_size=_env_int("RED_TOP_BATCH_SIZE", 32),
            replay_size=_env_int("RED_TOP_REPLAY_SIZE", 4096),
            warmup_transitions=_env_int("RED_TOP_WARMUP_TRANSITIONS", 32),
            train_interval=_env_int("RED_TOP_TRAIN_INTERVAL", 20),
            target_tau=_env_float("RED_TOP_TARGET_TAU", 0.005),
            regularization=_env_float("RED_TOP_REGULARIZATION", 1e-4),
            rho=_env_float("RED_TOP_RHO", 1.0),
            eta=_env_float("RED_TOP_ETA", 1.0),
            kappa=_env_float("RED_TOP_KAPPA", 0.02),
            gumbel_temperature=_env_float("RED_TOP_GUMBEL_TEMPERATURE", 0.50),
            gumbel_min_temperature=_env_float(
                "RED_TOP_GUMBEL_MIN_TEMPERATURE", 0.10
            ),
            exploration_decay_decisions=_env_int(
                "RED_TOP_EXPLORATION_DECAY_DECISIONS", 1000
            ),
            terminal_prior_hm=_env_float("RED_TOP_TERMINAL_PRIOR_HM", 0.10),
            terminal_prior_l=_env_float("RED_TOP_TERMINAL_PRIOR_L", 0.05),
            distance_weight=_env_float("RED_TOP_DISTANCE_WEIGHT", 0.02),
            release_fraction=_env_float("RED_TOP_RELEASE_FRACTION", 0.30),
            seed=int(seed),
            max_steps=max(1, int(max_steps)),
            device=os.getenv("RED_TOP_DEVICE", "auto"),
        )

    def validate(self) -> None:
        if self.batch_size <= 0 or self.replay_size < self.batch_size:
            raise ValueError("BC-ERCA requires replay_size >= batch_size > 0")
        if self.train_interval <= 0 or self.exploration_decay_decisions <= 0:
            raise ValueError("BC-ERCA interval/decay settings must be positive")
        if not (0.0 < self.target_tau <= 1.0):
            raise ValueError("BC-ERCA target_tau must lie in (0, 1]")
        if self.rho <= 0.0 or self.eta <= 0.0 or self.kappa < 0.0:
            raise ValueError("BC-ERCA requires rho/eta > 0 and kappa >= 0")
        if not (0.0 < self.terminal_prior_hm <= 1.0):
            raise ValueError("terminal_prior_hm must lie in (0, 1]")
        if not (0.0 < self.terminal_prior_l <= 1.0):
            raise ValueError("terminal_prior_l must lie in (0, 1]")
        if not (0.0 < self.release_fraction <= 1.0):
            raise ValueError("release_fraction must lie in (0, 1]")


class BCERCANetwork(nn.Module):
    """Small set encoder with a scalar V head and bounded edge residual head."""

    def __init__(self, config: BCERCAConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.platform_encoder = nn.Sequential(
            nn.Linear(config.platform_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(config.target_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        context_input = config.global_dim + hidden * 4
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.edge_head = nn.Sequential(
            nn.Linear(hidden + config.platform_dim + config.target_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Exact zero-residual initialization recovers the deterministic anchor.
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)

    @staticmethod
    def _pool(encoded: torch.Tensor, hidden: int, device: torch.device) -> torch.Tensor:
        if encoded.numel() == 0:
            return torch.zeros(hidden * 2, dtype=torch.float32, device=device)
        return torch.cat((encoded.mean(dim=0), encoded.max(dim=0).values), dim=0)

    def context(self, snapshot: Mapping[str, Any], device: torch.device) -> torch.Tensor:
        global_features = torch.as_tensor(
            snapshot["global_features"], dtype=torch.float32, device=device
        )
        platforms = torch.as_tensor(
            [item["features"] for item in snapshot["platforms"]],
            dtype=torch.float32,
            device=device,
        ).reshape(-1, self.config.platform_dim)
        targets = torch.as_tensor(
            [item["features"] for item in snapshot["targets"]],
            dtype=torch.float32,
            device=device,
        ).reshape(-1, self.config.target_dim)
        hidden = self.config.hidden_dim
        platform_pool = self._pool(self.platform_encoder(platforms), hidden, device)
        target_pool = self._pool(self.target_encoder(targets), hidden, device)
        return self.context_encoder(torch.cat((global_features, platform_pool, target_pool)))

    def value(self, snapshot: Mapping[str, Any], device: torch.device) -> torch.Tensor:
        return self.value_head(self.context(snapshot, device)).squeeze(-1)

    def residual_matrix(
        self, snapshot: Mapping[str, Any], device: torch.device
    ) -> torch.Tensor:
        platforms = snapshot["platforms"]
        targets = snapshot["targets"]
        if not platforms:
            return torch.empty((0, len(targets) + 1), dtype=torch.float32, device=device)
        context = self.context(snapshot, device)
        rows: list[torch.Tensor] = []
        zero_target = [0.0] * self.config.target_dim
        for platform in platforms:
            platform_features = platform["features"]
            edge_rows = []
            for target in targets:
                edge_rows.append(platform_features + target["features"] + [0.0])
            edge_rows.append(platform_features + zero_target + [1.0])
            raw = torch.as_tensor(edge_rows, dtype=torch.float32, device=device)
            repeated_context = context.expand(raw.shape[0], -1)
            rows.append(
                self.config.rho
                * torch.tanh(self.edge_head(torch.cat((repeated_context, raw), dim=-1)).squeeze(-1))
            )
        return torch.stack(rows)


class BCERCAHighLevelPolicy:
    """Event-level structured Double-Q policy for target/HOLD assignment."""

    def __init__(
        self,
        initial_targets: tuple[TargetPrior, ...],
        rules: BaselineRules,
        config: BCERCAConfig | None = None,
        *,
        training: bool = False,
        stage: str = "double_q",
        model_path: str | None = None,
    ):
        self.config = config or BCERCAConfig()
        self.config.validate()
        if stage not in {"v_warmup", "double_q"}:
            raise ValueError(f"Unsupported BC-ERCA stage: {stage}")
        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.rules = rules
        self.initial_target_ids = frozenset(item.entity_id for item in initial_targets)
        self.training = bool(training)
        self.stage = stage
        self.online = BCERCANetwork(self.config).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay: list[dict[str, Any]] = []
        self.update_count = 0
        self.value_update_count = 0
        self.transition_count = 0
        self.decision_count = 0
        self.double_q_decision_count = 0
        self.total_commit_count = 0
        self.total_edit_count = 0
        self.last_metrics: dict[str, float] = {}
        self.satellite_platform_ids: frozenset[int] = frozenset()
        self._episode_transitions: list[dict[str, Any]] = []
        self._current_snapshot: dict[str, Any] | None = None
        self._current_action: dict[str, Any] | None = None
        self._reward_since_decision = 0.0
        self._pending: dict[int, tuple[int, float]] = {}
        self._inflight: dict[int, tuple[int, float]] = {}
        self._processed: set[int] = set()
        self._committed_counts: dict[int, int] = {}
        self._last_episode_metrics: dict[str, float] = {}
        if model_path and Path(model_path).is_file():
            self.load(model_path)
        self.set_training(training)

    def set_training(self, training: bool) -> None:
        self.training = bool(training)
        self.online.train(self.training)
        self.target.eval()

    def reset_episode(self) -> None:
        self._episode_transitions.clear()
        self._current_snapshot = None
        self._current_action = None
        self._reward_since_decision = 0.0
        self._pending.clear()
        self._inflight.clear()
        self._processed.clear()
        self._committed_counts.clear()
        self.satellite_platform_ids = frozenset()

    def observe_reward(self, reward: float) -> None:
        if self.training:
            self._reward_since_decision += float(reward)

    def on_assignment_accepted(self, assignment: Assignment, platform: PlatformState) -> None:
        target_id = int(assignment.target_id)
        quality = self._quality(platform.kind, target_id=None, target_type=None)
        # The exact quality depends on target type and is refreshed at launch.
        self._pending[int(assignment.platform_id)] = (target_id, quality)
        self._committed_counts[target_id] = self._committed_counts.get(target_id, 0) + 1

    def mark_launched(self, platform_id: int, target: TargetPrior, kind: str) -> None:
        self._pending.pop(int(platform_id), None)
        quality = self._quality(kind, target.entity_id, target.entity_type)
        self._inflight[int(platform_id)] = (int(target.entity_id), quality)

    def _sync_lifecycle(self, observation: BaselineObservation) -> None:
        by_id = {item.entity_id: item for item in observation.platforms}
        for platform_id, payload in list(self._pending.items()):
            platform = by_id.get(platform_id)
            if platform is None or not platform.alive:
                self._pending.pop(platform_id, None)
        for platform_id, payload in list(self._inflight.items()):
            platform = by_id.get(platform_id)
            if platform is not None and platform.alive:
                continue
            if platform_id not in self._processed:
                self._processed.add(platform_id)
            self._inflight.pop(platform_id, None)

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        self._sync_lifecycle(observation)
        snapshot = self._make_snapshot(observation)

        baseline = self._solve(snapshot, network=None, explore=False, baseline=None)
        snapshot["baseline_action"] = baseline
        if self.training and self._current_snapshot is not None and self._current_action is not None:
            self._record_transition(snapshot, terminal=False)
        if self.training and self.stage == "double_q":
            action = self._solve(
                snapshot,
                network=self.online,
                explore=True,
                baseline=baseline,
            )
        else:
            action = baseline if self.stage == "v_warmup" else self._solve(
                snapshot,
                network=self.online,
                explore=False,
                baseline=baseline,
            )
        self.decision_count += 1
        if self.training and self.stage == "double_q":
            self.double_q_decision_count += 1
        edits = self._edit_count(action, baseline)
        commits = sum(target_id is not None for target_id in action["targets"].values())
        self.total_edit_count += edits
        self.total_commit_count += commits
        self._current_snapshot = snapshot
        self._current_action = action
        self._reward_since_decision = 0.0
        return self._postprocess(observation, action)

    def finish_episode(self, *, terminal: bool = True) -> None:
        if not self.training or self._current_snapshot is None or self._current_action is None:
            return
        self._record_transition(None, terminal=terminal)
        if self.stage == "v_warmup":
            self._update_value_from_episode()
        elif len(self.replay) >= max(self.config.batch_size, self.config.warmup_transitions):
            self._update_double_q()
        self._last_episode_metrics = {
            "episode_event_transitions": float(len(self._episode_transitions)),
            "episode_return": float(sum(item["reward"] for item in self._episode_transitions)),
        }
        self._episode_transitions.clear()
        self._current_snapshot = None
        self._current_action = None
        self._reward_since_decision = 0.0

    def _record_transition(
        self, next_snapshot: dict[str, Any] | None, *, terminal: bool
    ) -> None:
        assert self._current_snapshot is not None
        assert self._current_action is not None
        transition = {
            "state": self._current_snapshot,
            "action": self._current_action,
            "reward": float(self._reward_since_decision),
            "next_state": next_snapshot,
            "terminal": bool(terminal),
        }
        self.transition_count += 1
        self._episode_transitions.append(transition)
        if self.stage == "double_q":
            self.replay.append(transition)
            if len(self.replay) > self.config.replay_size:
                del self.replay[: len(self.replay) - self.config.replay_size]
            if (
                len(self.replay) >= max(self.config.batch_size, self.config.warmup_transitions)
                and self.transition_count % self.config.train_interval == 0
            ):
                self._update_double_q()

    def _update_value_from_episode(self) -> None:
        if not self._episode_transitions:
            return
        returns: list[float] = []
        running = 0.0
        for transition in reversed(self._episode_transitions):
            running = float(transition["reward"]) + running
            returns.append(running)
        returns.reverse()
        losses = []
        for _ in range(4):
            predicted = torch.stack(
                [self.online.value(item["state"], self.device) for item in self._episode_transitions]
            )
            targets = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            loss = F.smooth_l1_loss(predicted, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.target.load_state_dict(self.online.state_dict())
        self.value_update_count += 1
        self.last_metrics.update({
            "value_loss": float(np.mean(losses)),
            "value_grad_norm": float(grad_norm.detach().cpu()),
        })

    def _update_double_q(self) -> None:
        if len(self.replay) < self.config.batch_size:
            return
        indices = self.rng.choice(len(self.replay), size=self.config.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        predictions = torch.stack(
            [self._q_value(item["state"], item["action"], self.online) for item in batch]
        )
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for item in batch:
                if item["terminal"] or item["next_state"] is None:
                    targets.append(torch.tensor(item["reward"], dtype=torch.float32, device=self.device))
                    continue
                next_state = item["next_state"]
                baseline = next_state["baseline_action"]
                next_action = self._solve(
                    next_state,
                    network=self.online,
                    explore=False,
                    baseline=baseline,
                )
                next_q = self._q_value(next_state, next_action, self.target)
                targets.append(float(item["reward"]) + self.config.gamma * next_q)
            target_tensor = torch.stack(targets)
        td_loss = F.smooth_l1_loss(predictions, target_tensor)
        penalties = []
        for item in batch:
            residuals = self.online.residual_matrix(item["state"], self.device)
            if residuals.numel():
                penalties.append(residuals.square().mean())
        regularization = (
            torch.stack(penalties).mean()
            if penalties
            else torch.zeros((), dtype=torch.float32, device=self.device)
        )
        loss = td_loss + self.config.regularization * regularization
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target.parameters(), self.online.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.target_tau)
                target_parameter.add_(online_parameter, alpha=self.config.target_tau)
        self.update_count += 1
        self.last_metrics.update({
            "td_loss": float(td_loss.detach().cpu()),
            "regularization": float(regularization.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "q_mean": float(predictions.detach().mean().cpu()),
            "target_q_mean": float(target_tensor.detach().mean().cpu()),
        })

    def _q_value(
        self,
        snapshot: Mapping[str, Any],
        action: Mapping[str, Any],
        network: BCERCANetwork,
    ) -> torch.Tensor:
        value = network.value(snapshot, self.device)
        platforms = snapshot["platforms"]
        if not platforms:
            return value
        residuals = network.residual_matrix(snapshot, self.device)
        target_index = {
            int(item["entity_id"]): index for index, item in enumerate(snapshot["targets"])
        }
        hold_index = len(snapshot["targets"])
        baseline = snapshot["baseline_action"]
        selected_terms = []
        baseline_terms = []
        for row, platform in enumerate(platforms):
            platform_id = str(platform["entity_id"])
            selected_target = action["targets"].get(platform_id)
            baseline_target = baseline["targets"].get(platform_id)
            selected_terms.append(
                residuals[row, target_index.get(int(selected_target), hold_index)]
                if selected_target is not None
                else residuals[row, hold_index]
            )
            baseline_terms.append(
                residuals[row, target_index.get(int(baseline_target), hold_index)]
                if baseline_target is not None
                else residuals[row, hold_index]
            )
        residual_difference = torch.stack(selected_terms).sum() - torch.stack(baseline_terms).sum()
        base_difference = float(action["f0"] - baseline["f0"])
        edits = float(self._edit_count(action, baseline))
        advantage = (
            self.config.eta * base_difference
            + residual_difference
            - self.config.kappa * edits
        ) / max(1, len(platforms))
        return value + advantage

    def _make_snapshot(self, observation: BaselineObservation) -> dict[str, Any]:
        ready = tuple(
            sorted(
                (
                    item
                    for item in observation.platforms
                    if item.alive
                    and not item.launched
                    and item.entity_id not in self._pending
                ),
                key=lambda item: item.entity_id,
            )
        )
        total = max(1, observation.metrics.total_count)
        pending_count = len(self._pending)
        inflight_count = len(self._inflight)
        kinds = {kind: sum(item.kind == kind for item in ready) for kind in ("H", "M", "L")}
        global_features = [
            np.clip(observation.step / self.config.max_steps, 0.0, 1.0),
            observation.metrics.pressure,
            observation.metrics.launched_count / total,
            len(ready) / total,
            pending_count / total,
            inflight_count / total,
            kinds["H"] / total,
            kinds["M"] / total,
            kinds["L"] / total,
            min(1.0, len(observation.targets) / 10.0),
        ]
        platforms = [
            {
                "entity_id": item.entity_id,
                "kind": item.kind,
                "position": [item.position.lon, item.position.lat],
                "features": [
                    float(item.kind == "H"),
                    float(item.kind == "M"),
                    float(item.kind == "L"),
                    item.position.lon / 180.0,
                    item.position.lat / 90.0,
                ],
            }
            for item in ready
        ]
        targets = []
        for item in sorted(observation.targets, key=lambda target: target.entity_id):
            capacity = self._target_capacity(item.entity_type)
            committed = self._committed_counts.get(item.entity_id, 0)
            targets.append({
                "entity_id": item.entity_id,
                "entity_type": item.entity_type,
                "position": [item.position.lon, item.position.lat],
                "value": item.value,
                "capacity": capacity,
                "committed": committed,
                "features": [
                    float(item.entity_type == 9400),
                    float(item.entity_type == 9500),
                    float(item.entity_type == 9600),
                    item.position.lon / 180.0,
                    item.position.lat / 90.0,
                    item.value / 10.0,
                    min(1.0, capacity / 100.0),
                    min(2.0, committed / max(1, capacity)),
                    float(item.entity_id not in self.initial_target_ids),
                ],
            })
        release_budget = min(
            len(platforms),
            max(1, math.ceil(len(platforms) * self.config.release_fraction))
            if platforms
            else 0,
        )
        return {
            "step": observation.step,
            "global_features": [float(value) for value in global_features],
            "platforms": platforms,
            "targets": targets,
            "release_budget": release_budget,
        }

    def _solve(
        self,
        snapshot: Mapping[str, Any],
        *,
        network: BCERCANetwork | None,
        explore: bool,
        baseline: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        platforms = snapshot["platforms"]
        targets = snapshot["targets"]
        if not platforms:
            return {"targets": {}, "f0": 0.0}
        residuals = None
        if network is not None:
            with torch.no_grad():
                residuals = network.residual_matrix(snapshot, self.device).detach().cpu().numpy()
        target_index = {int(item["entity_id"]): index for index, item in enumerate(targets)}
        hold_index = len(targets)
        baseline_targets = baseline["targets"] if baseline is not None else {}
        temperature = self._exploration_temperature() if explore else 0.0
        noise = (
            self.rng.gumbel(0.0, temperature, size=(len(platforms), len(targets) + 1))
            if temperature > 0.0
            else np.zeros((len(platforms), len(targets) + 1), dtype=np.float64)
        )

        source = "source"
        gate = "commit_gate"
        sink = "sink"
        graph = nx.DiGraph()
        graph.add_node(source, demand=-len(platforms))
        graph.add_node(sink, demand=len(platforms))
        graph.add_node(gate, demand=0)
        graph.add_edge(gate, sink, capacity=int(snapshot["release_budget"]), weight=0)
        base_by_edge: dict[tuple[str, str], float] = {}
        for row, platform in enumerate(platforms):
            platform_node = f"p:{platform['entity_id']}"
            graph.add_node(platform_node, demand=0)
            graph.add_edge(source, platform_node, capacity=1, weight=0)
            hold_score = 0.0
            if residuals is not None:
                hold_score += float(residuals[row, hold_index])
            hold_score += float(noise[row, hold_index])
            if baseline is not None and baseline_targets.get(str(platform["entity_id"])) is not None:
                hold_score -= self.config.kappa
            graph.add_edge(
                platform_node,
                sink,
                capacity=1,
                weight=self._flow_cost(hold_score, tie=0),
            )
            for target in targets:
                if not self._compatible(platform["kind"], int(target["entity_type"])):
                    continue
                target_id = int(target["entity_id"])
                for offset in range(1, int(snapshot["release_budget"]) + 1):
                    rank = int(target["committed"]) + offset
                    slot_node = f"t:{target_id}:{rank}"
                    if slot_node not in graph:
                        graph.add_node(slot_node, demand=0)
                        graph.add_edge(slot_node, gate, capacity=1, weight=0)
                    base_score = self._base_edge_score(platform, target, rank)
                    score = self.config.eta * base_score if network is not None else base_score
                    if residuals is not None:
                        score += float(residuals[row, target_index[target_id]])
                    score += float(noise[row, target_index[target_id]])
                    if (
                        baseline is not None
                        and baseline_targets.get(str(platform["entity_id"])) != target_id
                    ):
                        score -= self.config.kappa
                    tie = 1 + target_index[target_id] * 1000 + rank
                    graph.add_edge(
                        platform_node,
                        slot_node,
                        capacity=1,
                        weight=self._flow_cost(score, tie=tie),
                    )
                    base_by_edge[(platform_node, slot_node)] = base_score
        flow = nx.min_cost_flow(graph)
        chosen: dict[str, int | None] = {}
        f0 = 0.0
        for platform in platforms:
            platform_node = f"p:{platform['entity_id']}"
            selected_target = None
            for node, amount in flow[platform_node].items():
                if amount <= 0 or not node.startswith("t:"):
                    continue
                selected_target = int(node.split(":", 2)[1])
                f0 += base_by_edge[(platform_node, node)]
                break
            chosen[str(platform["entity_id"])] = selected_target
        return {"targets": chosen, "f0": float(f0)}

    @staticmethod
    def _flow_cost(score: float, *, tie: int) -> int:
        # Integer costs make NetworkX deterministic. HOLD has tie=0 and wins
        # exact score ties, matching the feasibility plan's conservative rule.
        return -int(round(float(score) * 1_000_000.0)) * 10_000 + int(tie)

    def _base_edge_score(
        self, platform: Mapping[str, Any], target: Mapping[str, Any], rank: int
    ) -> float:
        capacity = max(1, int(target["capacity"]))
        target_type = int(target["entity_type"])
        marginal = float(target["value"]) / capacity if rank <= capacity else 0.0
        quality = self._quality(str(platform["kind"]), int(target["entity_id"]), target_type)
        correction = (float(target["value"]) / capacity) * (min(quality, 1.0) - 1.0)
        platform_position = platform["position"]
        target_position = target["position"]
        left = type("Point", (), {"lon": platform_position[0], "lat": platform_position[1]})()
        right = type("Point", (), {"lon": target_position[0], "lat": target_position[1]})()
        distance_penalty = self.config.distance_weight * distance_km(left, right) / 100.0
        return float(marginal + correction - distance_penalty)

    def _postprocess(
        self, observation: BaselineObservation, action: Mapping[str, Any]
    ) -> tuple[Assignment, ...]:
        platform_by_id = {item.entity_id: item for item in observation.platforms}
        target_by_id = {item.entity_id: item for item in observation.targets}
        grouped: dict[int, list[PlatformState]] = {}
        for raw_platform_id, target_id in action["targets"].items():
            if target_id is None:
                continue
            platform = platform_by_id.get(int(raw_platform_id))
            if platform is None or target_id not in target_by_id:
                continue
            grouped.setdefault(int(target_id), []).append(platform)
        satellite: set[int] = set()
        assignments: list[Assignment] = []
        for target_id in sorted(grouped):
            group = sorted(grouped[target_id], key=lambda item: (item.kind, item.entity_id))
            high = [item for item in group if item.kind == "H"]
            medium = [item for item in group if item.kind == "M"]
            paired = min(len(high), len(medium))
            paired_high = {item.entity_id for item in high[:paired]}
            target = target_by_id[target_id]
            if high and target.entity_type in {9400, 9600}:
                satellite.add(high[0].entity_id)
            for platform in group:
                if platform.kind == "M":
                    delay = 0
                elif platform.entity_id in paired_high:
                    delay = self.rules.package_high_delay
                else:
                    delay = self.rules.wave_delay(platform.kind)
                assignments.append(
                    Assignment(
                        platform_id=platform.entity_id,
                        target_id=target_id,
                        launch_step=observation.step + delay,
                        score=0.0,
                    )
                )
        self.satellite_platform_ids = frozenset(satellite)
        return tuple(sorted(assignments, key=lambda item: (item.launch_step, item.platform_id)))

    def _target_capacity(self, target_type: int) -> int:
        health = _TARGET_HEALTH.get(int(target_type), 1.0)
        reference = _TARGET_REFERENCE_DAMAGE.get(int(target_type), 1.0)
        prior = self.config.terminal_prior_l if int(target_type) == 9500 else self.config.terminal_prior_hm
        return max(1, int(math.ceil(health / max(reference * prior, 1e-8))))

    def _quality(
        self, kind: str, target_id: int | None, target_type: int | None
    ) -> float:
        del target_id
        if target_type is None:
            return 0.0
        damage = _EXPECTED_DAMAGE.get(kind, {}).get(int(target_type), 0.0)
        reference = _TARGET_REFERENCE_DAMAGE.get(int(target_type), 1.0)
        return float(damage / max(reference, 1e-8))

    @staticmethod
    def _compatible(kind: str, target_type: int) -> bool:
        return engagement_effectiveness(kind, int(target_type)) > 0.0 and not (
            kind == "L" and int(target_type) in {9400, 9600}
        )

    @staticmethod
    def _edit_count(action: Mapping[str, Any], baseline: Mapping[str, Any]) -> int:
        return sum(
            target_id != baseline["targets"].get(platform_id)
            for platform_id, target_id in action["targets"].items()
        )

    def _exploration_temperature(self) -> float:
        fraction = min(1.0, self.double_q_decision_count / self.config.exploration_decay_decisions)
        return float(
            self.config.gumbel_temperature
            + fraction
            * (self.config.gumbel_min_temperature - self.config.gumbel_temperature)
        )

    def diagnostics(self) -> dict[str, Any]:
        with torch.no_grad():
            final_edge_layer = self.online.edge_head[-1]
            residual_parameter_norm = float(
                (final_edge_layer.weight.square().sum() + final_edge_layer.bias.square().sum())
                .sqrt()
                .cpu()
            )
        decisions = max(1, self.decision_count)
        result: dict[str, Any] = {
            "algorithm": "bc_erca_top",
            "training": self.training,
            "stage": self.stage,
            "device": str(self.device),
            "decision_count": self.decision_count,
            "double_q_decision_count": self.double_q_decision_count,
            "transition_count": self.transition_count,
            "update_count": self.update_count,
            "value_update_count": self.value_update_count,
            "replay_size": len(self.replay),
            "commit_count": self.total_commit_count,
            "edit_count": self.total_edit_count,
            "edit_rate": self.total_edit_count / decisions,
            "residual_parameter_norm": residual_parameter_norm,
            "pending_count": len(self._pending),
            "inflight_count": len(self._inflight),
            "processed_count": len(self._processed),
            "last_metrics": {**self.last_metrics, **self._last_episode_metrics},
        }
        return result

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "bc_erca_top",
                "config": asdict(self.config),
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "replay": self.replay,
                "update_count": self.update_count,
                "value_update_count": self.value_update_count,
                "transition_count": self.transition_count,
                "decision_count": self.decision_count,
                "double_q_decision_count": self.double_q_decision_count,
                "total_commit_count": self.total_commit_count,
                "total_edit_count": self.total_edit_count,
                "last_metrics": self.last_metrics,
                "rng_state": self.rng.bit_generator.state,
            },
            destination,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != "bc_erca_top":
            raise ValueError(f"Checkpoint {path} is not a BC-ERCA top model")
        saved_config = checkpoint.get("config", {})
        for field in ("global_dim", "platform_dim", "target_dim", "hidden_dim"):
            if int(saved_config.get(field, getattr(self.config, field))) != int(
                getattr(self.config, field)
            ):
                raise ValueError(f"BC-ERCA checkpoint/config mismatch for {field}")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.learning_rate
        self.replay = list(checkpoint.get("replay", []))[-self.config.replay_size :]
        self.update_count = int(checkpoint.get("update_count", 0))
        self.value_update_count = int(checkpoint.get("value_update_count", 0))
        self.transition_count = int(checkpoint.get("transition_count", 0))
        self.decision_count = int(checkpoint.get("decision_count", 0))
        self.double_q_decision_count = int(
            checkpoint.get("double_q_decision_count", self.update_count * self.config.train_interval)
        )
        self.total_commit_count = int(checkpoint.get("total_commit_count", 0))
        self.total_edit_count = int(checkpoint.get("total_edit_count", 0))
        self.last_metrics = dict(checkpoint.get("last_metrics", {}))
        if "rng_state" in checkpoint:
            self.rng.bit_generator.state = checkpoint["rng_state"]

