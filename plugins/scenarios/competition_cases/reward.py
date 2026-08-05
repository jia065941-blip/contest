"""Unified K/D/T scoring for the nine competition scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


DESTROYED_TARGET_WEIGHT = 0.7
MISSILE_SURVIVAL_WEIGHT = 0.3
TIME_PENALTY_WEIGHT = 0.2

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
    D: float
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
    entity = entities.get(entity_id, entities.get(str(entity_id), {}))
    return float(entity.get("health", 0.0))


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
    actual_hash = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    if expected_hash and expected_hash != actual_hash:
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
    surviving_red_missile_count: int,
    initial_red_missile_count: int,
    completion_step: int | None = None,
    destruction_steps: Mapping[int | str, int] | None = None,
) -> RewardBreakdown:
    """Calculate ``100 * clip(0.7K + 0.3D - 0.2T, 0, 1)``.

    ``K`` is the destroyed objective *value* ratio instead of a plain count.
    ``T`` is the remaining weighted time debt.  Every destroyed objective
    reduces that debt according to its value and destruction time, so time
    credit no longer depends on destroying every enemy root node.

    ``completion_step`` is retained for callers using the former API.  The
    tracker supplies per-objective ``destruction_steps`` for the new formula.
    """
    if initial_red_missile_count <= 0:
        raise ValueError("initial_red_missile_count 必须大于 0")
    if not 0 <= surviving_red_missile_count <= initial_red_missile_count:
        raise ValueError("surviving_red_missile_count 必须在 0 和初始数量之间")

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
    d_value = surviving_red_missile_count / initial_red_missile_count
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
            # 兼容旧调用：只有“全部完成时刻”时，把它作为已摧毁目标的
            # 保守近似；没有任何时间信息则不授予提前完成奖励。
            raw_step = completion_step if completion_step is not None else policy.max_steps
        step = max(0, min(policy.max_steps, int(raw_step)))
        normalized_steps[entity_id] = step
        remaining_fraction = 1.0 - step / policy.max_steps
        earned_time_credit += weight_by_id[entity_id] * remaining_fraction

    t_value = 1.0 - earned_time_credit / total_objective_weight
    t_value = max(0.0, min(1.0, t_value))
    raw_score = (
        DESTROYED_TARGET_WEIGHT * k_value
        + MISSILE_SURVIVAL_WEIGHT * d_value
        - TIME_PENALTY_WEIGHT * t_value
    )
    return RewardBreakdown(
        score=100.0 * max(0.0, min(1.0, raw_score)),
        raw_score=raw_score,
        K=k_value,
        D=d_value,
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

    def __init__(
        self,
        policy: RewardPolicy,
        red_missile_ids: tuple[int, ...],
    ) -> None:
        if not red_missile_ids:
            raise ValueError("场景中没有可评分的红方导弹")
        self.policy = policy
        self.red_missile_ids = red_missile_ids
        self.completion_step: int | None = None
        self.destruction_steps: dict[int, int] = {}
        self.final_step = 0

    def check_completion(
        self,
        step: int,
        observation: Mapping[str, Any],
    ) -> None:
        """Record each objective's first destroyed step.

        ``completion_step`` remains an informational field for result files;
        it no longer gates the time component of the score.
        """
        entities = observation.get("entities", {})
        self.final_step = int(step)
        for entity_id in self.policy.objective_ids:
            if (
                entity_id not in self.destruction_steps
                and _entity_health(entities, entity_id) <= 0
            ):
                self.destruction_steps[entity_id] = int(step)
        if (
            self.completion_step is None
            and len(self.destruction_steps) == len(self.policy.objective_ids)
        ):
            self.completion_step = int(step)

    def finish(self, observation: Mapping[str, Any]) -> RewardBreakdown:
        entities = observation.get("entities", {})
        surviving_red_missile_count = sum(
            _entity_health(entities, entity_id) > 0
            for entity_id in self.red_missile_ids
        )
        target_health = {
            entity_id: _entity_health(entities, entity_id)
            for entity_id in self.policy.objective_ids
        }
        return calculate_reward(
            policy=self.policy,
            target_health=target_health,
            surviving_red_missile_count=surviving_red_missile_count,
            initial_red_missile_count=len(self.red_missile_ids),
            completion_step=self.completion_step,
            destruction_steps=self.destruction_steps,
        )
