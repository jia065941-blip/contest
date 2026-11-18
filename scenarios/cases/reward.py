# 比赛得分按完整摧毁的计分目标权重计算。

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


# 不同蓝方根节点的任务价值：暗巢:哨卫巢:计分母巢 = 1:2:5。
OBJECTIVE_KIND_WEIGHTS = {
    "target": 5.0,
    "site": 2.0,
    "ship": 1.0,
}
OBJECTIVE_ENTITY_KINDS = {9400: "target", 9600: "site", 9500: "ship"}
DEFAULT_OBJECTIVE_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class RewardPolicy:
    scenario_id: str
    objective_ids: tuple[int, ...]
    max_steps: int
    scenario_path: Path
    objective_weights: tuple[tuple[int, float], ...] = ()
    objective_initial_health: tuple[tuple[int, float], ...] = ()


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
    """读取并检查本地场景，从实体初态组装计分规则。"""
    raw_path = str(scenario)
    if raw_path.startswith(("http://", "https://")):
        return None

    scenario_path = Path(raw_path).resolve()
    from ..scenario_checker import check_scenario

    data = check_scenario(scenario_path)
    imagine = data["imagineProfile"]
    max_steps = int(imagine["simEndLogicTime"] // imagine["simStep"])
    return _reward_policy_from_scene(scenario_path, data, max_steps)


def _reward_policy_from_scene(
    scenario_path: Path, data: dict[str, Any], max_steps: int,
) -> RewardPolicy | None:
    """按实体类型确定计分目标，直接使用场景中的初始血量。"""
    objectives = [
        item["entity"]
        for item in data["imagineProfile"]["entityList"]
        if item["entity"]["entityType"] in OBJECTIVE_ENTITY_KINDS
    ]
    if not objectives:
        return None
    objective_ids = tuple(entity["id"] for entity in objectives)
    objective_weights = tuple(
        (
            entity["id"],
            OBJECTIVE_KIND_WEIGHTS[OBJECTIVE_ENTITY_KINDS[entity["entityType"]]],
        )
        for entity in objectives
    )
    objective_initial_health = tuple(
        (entity["id"], float(entity["survivePoints"])) for entity in objectives
    )
    if max_steps <= 0:
        raise ValueError(f"{scenario_path} 的总步数必须大于 0")
    if any(health <= 0 for _, health in objective_initial_health):
        raise ValueError(f"{scenario_path} 包含非正数的计分目标初始血量")

    return RewardPolicy(
        scenario_id=data.get("metadata", {}).get("id", scenario_path.stem),
        objective_ids=objective_ids,
        max_steps=max_steps,
        scenario_path=scenario_path,
        objective_weights=objective_weights,
        objective_initial_health=objective_initial_health,
    )


def calculate_reward(
    *,
    policy: RewardPolicy,
    target_health: Mapping[int | str, float],
    destruction_steps: Mapping[int | str, int] | None = None,
) -> RewardBreakdown:
    """按血量归零的目标权重计算0至100分，存活目标得零分。

    K与raw_score记录已摧毁目标的权重占比；T恒为零。
    摧毁时刻仅用于结果记录，不影响得分。
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
    for entity_id in destroyed_ids:
        raw_step = supplied_steps.get(
            entity_id,
            supplied_steps.get(str(entity_id)),
        )
        if raw_step is None:
            raw_step = policy.max_steps
        step = max(0, min(policy.max_steps, int(raw_step)))
        normalized_steps[entity_id] = step

    t_value = 0.0
    raw_score = max(0.0, min(1.0, k_value))
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
