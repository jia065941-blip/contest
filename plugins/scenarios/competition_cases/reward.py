"""Unified destruction/time scoring for the nine competition scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


DESTROYED_TARGET_WEIGHT = 0.8
TIME_EFFICIENCY_WEIGHT = 0.2

# 不同蓝方根节点的任务价值。普通目标建筑是主要打击对象，
# 雷达/拦截阵地和无人船统一按基础价值计分。
OBJECTIVE_KIND_WEIGHTS = {
    "target": 3.0,
    "site": 1.0,
    "ship": 1.0,
}
DEFAULT_OBJECTIVE_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class RewardPolicy:
    scenario_id: str
    objective_ids: tuple[int, ...]
    max_steps: int
    scenario_path: Path
    objective_weights: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    score: float
    raw_score: float
    K: float
    T: float
    completed: bool
    destroyed_ids: tuple[int, ...]
    destroyed_weight: float
    total_objective_weight: float
    objective_weights: tuple[tuple[int, float], ...]
    destruction_steps: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["destroyed_ids"] = list(self.destroyed_ids)
        value["objective_weights"] = {
            str(entity_id): weight
            for entity_id, weight in self.objective_weights
        }
        value["destruction_steps"] = {
            str(entity_id): step
            for entity_id, step in self.destruction_steps
        }
        return value


def _entity_health(
    entities: Mapping[int | str, Mapping[str, Any]],
    entity_id: int,
) -> float:
    entity = entities.get(entity_id, entities.get(str(entity_id)))
    if entity is None:
        return float("inf")
    return float(entity.get("health", float("inf")))


def load_reward_policy(scenario: str | Path) -> RewardPolicy | None:
    """Load judge inputs from the ``case_info.json`` beside a scenario.

    Non-competition scenarios do not have ``case_info.json`` and therefore
    return ``None`` instead of changing their original execution behaviour.
    """
    raw_path = str(scenario)
    if raw_path.startswith(("http://", "https://")):
        return None

    scenario_path = Path(raw_path).resolve()
    info_path = scenario_path.with_name("case_info.json")
    if not info_path.is_file():
        return None

    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected_hash = str(info.get("scenario_sha256", ""))
    scenario_bytes = scenario_path.read_bytes()
    # Git may check JSON out with CRLF on Windows. Treat newline conversion as
    # transport-level normalization while still rejecting content changes.
    lf_bytes = scenario_bytes.replace(b"\r\n", b"\n")
    hash_candidates = {
        hashlib.sha256(scenario_bytes).hexdigest(),
        hashlib.sha256(lf_bytes).hexdigest(),
        hashlib.sha256(lf_bytes.replace(b"\n", b"\r\n")).hexdigest(),
    }
    if expected_hash and expected_hash not in hash_candidates:
        raise ValueError(
            f"场景校验失败：{scenario_path} 与 case_info.json 的 SHA-256 不一致"
        )

    objective_ids = tuple(int(value) for value in info["objective_ids"])
    roots_by_id = {
        int(root["id"]): root
        for root in info.get("roots", [])
        if "id" in root
    }
    objective_weights = tuple(
        (
            entity_id,
            float(
                OBJECTIVE_KIND_WEIGHTS.get(
                    str(roots_by_id.get(entity_id, {}).get("kind", "")),
                    DEFAULT_OBJECTIVE_WEIGHT,
                )
            ),
        )
        for entity_id in objective_ids
    )
    max_steps = int(info["max_steps"])
    if not objective_ids:
        raise ValueError(f"{info_path} 没有计分目标")
    if max_steps <= 0:
        raise ValueError(f"{info_path} 的 max_steps 必须大于 0")
    if any(weight <= 0 for _, weight in objective_weights):
        raise ValueError(f"{info_path} 包含非正数的目标权重")

    return RewardPolicy(
        scenario_id=str(info["scenario_id"]),
        objective_ids=objective_ids,
        max_steps=max_steps,
        scenario_path=scenario_path,
        objective_weights=objective_weights,
    )


def calculate_reward(
    *,
    policy: RewardPolicy,
    target_health: Mapping[int | str, float],
    destruction_steps: Mapping[int | str, int] | None = None,
) -> RewardBreakdown:
    """Calculate ``100 * clip(0.8K + 0.2T, 0, 1)``.

    ``K`` is the destroyed objective *value* ratio instead of a plain count.
    ``T`` is weighted time efficiency.  A destroyed objective earns more time
    credit when it is destroyed earlier; an undestroyed objective earns none.

    ``destruction_steps`` supplies each objective's first destroyed step.
    Missing timing information earns no time credit.
    """
    destroyed_ids = tuple(
        entity_id
        for entity_id in policy.objective_ids
        if float(
            target_health.get(
                entity_id,
                target_health.get(str(entity_id), float("inf")),
            )
        )
        <= 0.0
    )
    weight_pairs = policy.objective_weights or tuple(
        (entity_id, DEFAULT_OBJECTIVE_WEIGHT)
        for entity_id in policy.objective_ids
    )
    weight_by_id = dict(weight_pairs)
    total_objective_weight = sum(weight_by_id.values())
    if total_objective_weight <= 0:
        raise ValueError("计分目标总权重必须大于 0")

    destroyed_weight = sum(weight_by_id[entity_id] for entity_id in destroyed_ids)
    k_value = destroyed_weight / total_objective_weight
    completed = len(destroyed_ids) == len(policy.objective_ids)

    supplied_steps = destruction_steps or {}
    normalized_steps: dict[int, int] = {}
    earned_time_credit = 0.0
    for entity_id in destroyed_ids:
        raw_step = supplied_steps.get(
            entity_id,
            supplied_steps.get(str(entity_id)),
        )
        if raw_step is None:
            raw_step = policy.max_steps
        step = max(0, min(policy.max_steps, int(raw_step)))
        normalized_steps[entity_id] = step
        remaining_fraction = 1.0 - step / policy.max_steps
        earned_time_credit += weight_by_id[entity_id] * remaining_fraction

    t_value = earned_time_credit / total_objective_weight
    t_value = max(0.0, min(1.0, t_value))
    raw_score = (
        DESTROYED_TARGET_WEIGHT * k_value
        + TIME_EFFICIENCY_WEIGHT * t_value
    )
    return RewardBreakdown(
        score=100.0 * max(0.0, min(1.0, raw_score)),
        raw_score=raw_score,
        K=k_value,
        T=t_value,
        completed=completed,
        destroyed_ids=destroyed_ids,
        destroyed_weight=destroyed_weight,
        total_objective_weight=total_objective_weight,
        objective_weights=tuple(weight_pairs),
        destruction_steps=tuple(sorted(normalized_steps.items())),
    )


class RewardTracker:
    """Accumulate one round of state without changing the simulator."""

    def __init__(self, policy: RewardPolicy) -> None:
        self.policy = policy
        self.destruction_steps: dict[int, int] = {}

    def check_completion(
        self,
        step: int,
        observation: Mapping[str, Any],
    ) -> None:
        """Record each objective's first destroyed step.

        Repeated observations do not overwrite the first destroyed step.
        """
        entities = observation.get("entities", {})
        for entity_id in self.policy.objective_ids:
            if (
                entity_id not in self.destruction_steps
                and _entity_health(entities, entity_id) <= 0
            ):
                self.destruction_steps[entity_id] = int(step)

    def finish(self, observation: Mapping[str, Any]) -> RewardBreakdown:
        entities = observation.get("entities", {})
        target_health = {
            entity_id: _entity_health(entities, entity_id)
            for entity_id in self.policy.objective_ids
        }
        return calculate_reward(
            policy=self.policy,
            target_health=target_health,
            destruction_steps=self.destruction_steps,
        )
