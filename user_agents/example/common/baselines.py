"""一个通信群内的 R0—R9 任务分配规则与输入输出值对象。

策略读取已整理的平台、目标和指标，返回 Assignment；规划器负责把结果转成引擎动作。
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from math import hypot

from .contracts import Position


# 文件入口和训练配置共用的任务层名称。R9 的学习差异位于执行层，任务分配沿用 R8。
RED_POLICY_CHOICES = (
    "r0_random",
    "r1_priority",
    "r2_static_assignment",
    "r3_static_priority",
    "r4_rolling_rules",
    "r5_event_rolling",
    "r6_pressure_assignment",
    "r7_strike_packages",
    "r7_static_search",
    "r8_satellite_packages",
    "r9_hierarchical_learning",
)

# 压力系数与规则中的 cautious_pressure 配合使用，指标范围为 0 到该权重。
INTERCEPTOR_PRESSURE_WEIGHT = 0.65


@dataclass(frozen=True)
class PlatformState:
    """规划时使用的本机摘要：编号、H/M/L 类型、三维位置及当前可用标记。"""
    entity_id: int
    kind: str
    position: Position
    alive: bool = True


@dataclass(frozen=True)
class TargetPrior:
    """公开目录或已收到轨迹形成的目标摘要，value 是任务分配使用的价值。"""

    entity_id: int
    entity_type: int
    position: Position
    value: float = 1.0
    alive: bool = True


@dataclass(frozen=True)
class Assignment:
    """任务层输出：选择目标或显式搜索目的地，并附带评分和搜索编组编号。"""
    platform_id: int
    # 攻击任务填写 target_id；搜索任务由 destination 给出位置。
    target_id: int | None
    score: float
    destination: Position | None = None
    search_group_id: int | None = None


@dataclass(frozen=True)
class TacticalMetrics:
    """从本群成员报告整理的威胁、存活、事件和已有任务占用指标。"""

    observed_interceptor_count: int = 0
    alive_count: int = 0
    event_revision: int = 0
    # (目标编号, 已分配平台数)，供滚动任务分配考虑当前覆盖程度。
    assigned_target_counts: tuple[tuple[int, int], ...] = ()

    @property
    def pressure(self) -> float:
        """以已知自爆虫数相对存活成员数形成压力，截断到 1 后乘压力权重。"""

        interceptor_pressure = min(1.0, self.observed_interceptor_count / max(1, self.alive_count))
        return INTERCEPTOR_PRESSURE_WEIGHT * interceptor_pressure


@dataclass(frozen=True)
class BaselineObservation:
    """一帧任务算法输入，包含本群平台、已知目标和协同统计指标快照。"""
    step: int
    platforms: tuple[PlatformState, ...]
    targets: tuple[TargetPrior, ...]
    metrics: TacticalMetrics = field(default_factory=TacticalMetrics)

    def active_platforms(self) -> tuple[PlatformState, ...]:
        """提取本帧可参与分配的平台。"""
        return tuple(item for item in self.platforms if item.alive)

    def active_targets(self) -> tuple[TargetPrior, ...]:
        """提取当前目录中仍标记可用的目标。"""
        return tuple(item for item in self.targets if item.alive)


@dataclass(frozen=True)
class BaselineRules:
    """示例任务算法的公共参数，时间间隔以仿真步数计，距离以千米计。"""
    # 容量与评分：capacity 限制单目标占用，覆盖惩罚鼓励分散分配。
    target_capacity: int = 10_000
    value_weight: float = 1.0
    distance_weight: float = 0.25
    coverage_weight: float = 0.75
    # 规划时序：普通周期、事件冷却和自适应周期均为仿真步数。
    replan_interval: int = 20
    event_replan_cooldown: int = 5
    adaptive_replan_interval: int = 5
    # 阶段与预算：先按窗口探测，再依据压力阈值选择每轮可改派的比例。
    probe_window: int = 10
    cautious_pressure: float = 0.20
    probe_reassignment_fraction: float = 0.15
    commit_reassignment_fraction: float = 0.30
    cautious_reassignment_fraction: float = 0.12
    # 静态搜索：三机编组、组间距离与低空成员投入比例；实际规模还受可用数和预算限制。
    static_search_group_size: int = 3
    static_search_group_spacing_km: float = 5.0
    static_search_low_fraction: float = 0.06
    static_search_min_platforms: int = 9
    static_search_max_range_km: float = 48.0
    static_search_danger_radius_km: float = 20.0


def target_is_compatible(platform_kind: str, target_type: int) -> bool:
    """限定 H/M 分配到母巢或哨卫，L 分配到已知暗巢。"""
    return target_type in ({9400, 9600} if platform_kind in {"H", "M"} else {9500} if platform_kind == "L" else set())

def engagement_effectiveness(platform_kind: str, target_type: int) -> float:
    """返回优化分配使用的示例攻击效果系数，参与任务评分。"""

    return {
        "H": {9400: 0.8, 9600: 0.6, 9500: 0.0},
        "M": {9400: 0.8, 9600: 0.6, 9500: 0.0},
        "L": {9400: 0.05, 9600: 0.05, 9500: 0.8},
    }.get(platform_kind, {}).get(target_type, 0.0)


def distance_km(left: Position, right: Position) -> float:
    """计算两点的平面水平距离，米制位置换算为千米，不计高度差。"""
    return hypot(left.x - right.x, left.y - right.y) / 1000.0


class RedBaseline:
    """示例任务层约定：批量分配任务，并可按需支持飞行中的即时目标认领。"""
    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """根据当前群快照生成本轮需处理的任务。"""
        raise NotImplementedError

    def claim_retarget(
        self,
        platform: PlatformState,
        observed_targets: tuple[TargetPrior, ...],
    ) -> TargetPrior | None:
        """按需认领本机已知目标；默认保持当前任务，搜索策略可覆写此方法。"""

        return None


class R0RandomPolicy(RedBaseline):
    """R0：按步数和本机编号独立播种，在适配目标中可复现随机选取。"""
    def __init__(self, rules: BaselineRules, seed: int) -> None:
        """保存分配容量和随机种子，随机源在每个平台选择时局部创建。"""
        self.rules = rules
        self.seed = seed

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """按编号遍历可用平台，为每架平台随机选择尚有容量的适配目标。"""
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts: dict[int, int] = {}
        assignments: list[Assignment] = []
        for platform in sorted(observation.active_platforms(), key=lambda item: item.entity_id):
            candidates = [target for target in targets if target_is_compatible(platform.kind, target.entity_type) and counts.get(target.entity_id, 0) < self.rules.target_capacity]
            if not candidates:
                continue
            # 每个平台的随机选择由本步和自身编号确定，不依赖全局随机调用顺序。
            rng = random.Random(self.seed + observation.step * 1009 + platform.entity_id)
            target = candidates[rng.randrange(len(candidates))]
            assignments.append(Assignment(platform.entity_id, target.entity_id, 0.0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
        return tuple(assignments)


class R1PriorityPolicy(RedBaseline):
    """R1：按任务价值与攻击效果加分，按距离和目标已有分配数扣分。"""
    def __init__(self, rules: BaselineRules) -> None:
        """保存各项评分权重和目标容量。"""
        self.rules = rules

    def score(self, platform: PlatformState, target: TargetPrior, assigned_count: int = 0) -> float:
        """计算价值×攻击效果、水平距离惩罚与分配拥挤惩罚的加权和。"""
        return self.rules.value_weight * target.value * engagement_effectiveness(platform.kind, target.entity_type) - self.rules.distance_weight * distance_km(platform.position, target.position) / 100.0 - self.rules.coverage_weight * assigned_count

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """按类型、编号顺序贪心选目标，并在本轮逐架增加目标占用数。"""
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts: dict[int, int] = {}
        assignments: list[Assignment] = []
        # 固定处理次序使同一快照产生相同分配；后续平台会看到前面已占用的目标数。
        platforms = sorted(observation.active_platforms(), key=lambda item: (item.kind, item.entity_id))
        for platform in platforms:
            candidates = [target for target in targets if target_is_compatible(platform.kind, target.entity_type) and counts.get(target.entity_id, 0) < self.rules.target_capacity]
            if not candidates:
                continue
            target = max(candidates, key=lambda item: (self.score(platform, item, counts.get(item.entity_id, 0)), -item.entity_id))
            score = self.score(platform, target, counts.get(target.entity_id, 0))
            assignments.append(Assignment(platform.entity_id, target.entity_id, score))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
        return tuple(assignments)


class R2StaticAssignmentPolicy(R1PriorityPolicy):
    """R2：首次生成容量约束分配并缓存，后续仅过滤已不可用的平台或目标。"""

    def __init__(self, rules: BaselineRules) -> None:
        """初始化空计划，首次 decide 时计算。"""
        super().__init__(rules)
        self._plan: tuple[Assignment, ...] | None = None

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """为每个平台建立按收益排序的候选集，求解一次静态分配。"""
        # 缓存建立后保持原任务；当前快照只用于剔除已不可用的平台和目标。
        if self._plan is not None:
            active_platform_ids = {item.entity_id for item in observation.active_platforms()}
            active_target_ids = {item.entity_id for item in observation.active_targets()}
            return tuple(item for item in self._plan if item.platform_id in active_platform_ids and item.target_id in active_target_ids)

        platforms = tuple(sorted(observation.active_platforms(), key=lambda item: item.entity_id))
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        # 预先计算各平台的适配目标与收益排序，供精确或贪心分支复用。
        options = tuple(
            tuple(
                sorted(
                    ((self.score(platform, target), target) for target in targets if target_is_compatible(platform.kind, target.entity_type)),
                    key=lambda item: (-item[0], item[1].entity_id),
                )
            )
            for platform in platforms
        )
        selected = self._solve_static_pairs(platforms, options)
        selected.sort(key=lambda item: (item[1].kind, item[1].entity_id))
        self._plan = tuple(
            Assignment(platform.entity_id, target.entity_id, score)
            for score, platform, target in selected
        )
        return self._plan

    def _solve_static_pairs(
        self,
        platforms: tuple[PlatformState, ...],
        options: tuple[tuple[tuple[float, TargetPrior], ...], ...],
    ) -> list[tuple[float, PlatformState, TargetPrior]]:
        """平台数不超过 12 时枚举最优正收益组合，较大规模采用确定性贪心。"""

        # 大规模避免组合枚举：按平台顺序取首个正收益且未满容量的候选。
        if len(platforms) > 12:
            counts: dict[int, int] = {}
            selected: list[tuple[float, PlatformState, TargetPrior]] = []
            for platform, platform_options in zip(platforms, options):
                candidate = next(
                    (
                        (score, target)
                        for score, target in platform_options
                        if score > 0 and counts.get(target.entity_id, 0) < self.rules.target_capacity
                    ),
                    None,
                )
                if candidate is None:
                    continue
                score, target = candidate
                counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
                selected.append((score, platform, target))
            return selected

        # 小规模穷举允许平台保持未分配，只有正收益边才进入候选组合。
        best_score = 0.0
        best: tuple[tuple[float, PlatformState, TargetPrior], ...] = ()
        counts: dict[int, int] = {}
        chosen: list[tuple[float, PlatformState, TargetPrior]] = []

        def visit(index: int, score_total: float) -> None:
            """递归选择跳过当前平台或占用一个目标，回溯时撤回临时占用。"""
            nonlocal best_score, best
            if index == len(platforms):
                candidate = tuple(chosen)
                if score_total > best_score + 1e-12:
                    best_score, best = score_total, candidate
                return
            # 先访问“不分配”分支；相同收益保留先遇到的组合，排序保证结果可复现。
            visit(index + 1, score_total)
            for score, target in options[index]:
                if score <= 0 or counts.get(target.entity_id, 0) >= self.rules.target_capacity:
                    continue
                counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
                chosen.append((score, platforms[index], target))
                visit(index + 1, score_total + score)
                chosen.pop()
                counts[target.entity_id] -= 1

        visit(0, 0.0)
        return list(best)


class R4RollingRulePolicy(R1PriorityPolicy):
    """R4：按固定间隔刷新优先级任务，在间隔内沿用缓存结果。"""
    def __init__(self, rules: BaselineRules) -> None:
        """初始化最近规划帧和缓存任务。"""
        super().__init__(rules)
        self._last_replan_step: int | None = None
        self._decision: tuple[Assignment, ...] = ()

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """到达规划周期时重新运行 R1，其余调用仅过滤失效任务。"""
        if self._last_replan_step is None or observation.step - self._last_replan_step >= self.rules.replan_interval:
            self._decision = super().decide(observation)
            self._last_replan_step = observation.step
        active_platform_ids = {item.entity_id for item in observation.active_platforms()}
        active_target_ids = {item.entity_id for item in observation.active_targets()}
        return tuple(item for item in self._decision if item.platform_id in active_platform_ids and item.target_id in active_target_ids)


class R5EventRollingPolicy(R1PriorityPolicy):
    """R5：复用 R1 分配，事件版本变化与冷却判断由 Commander 控制。"""


class R3StaticPriorityPolicy(R1PriorityPolicy):
    """R3：首次运行 R1 贪心分配并固定计划，后续只过滤失效项。"""

    def __init__(self, rules: BaselineRules) -> None:
        """初始化尚未生成的静态优先级计划。"""
        super().__init__(rules)
        self._plan: tuple[Assignment, ...] | None = None

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """首次计算后保持原分配，不为失效任务补选新目标。"""
        if self._plan is None:
            self._plan = super().decide(observation)

        active_platform_ids = {item.entity_id for item in observation.active_platforms()}
        active_target_ids = {item.entity_id for item in observation.active_targets()}
        return tuple(
            item
            for item in self._plan
            if item.platform_id in active_platform_ids and item.target_id in active_target_ids
        )


class AdaptiveRollingPolicy(R1PriorityPolicy):
    """R6—R9 共用的阶段判断、重分配预算和优先级分配工具。"""

    def _stage(self, observation: BaselineObservation) -> str:
        """存在 L 平台时先探测，再按压力进入吸收阶段；其余进入集中攻击阶段。"""
        # probe 窗口优先于压力判断；窗口过后仍有 L 平台且压力较大时进入 absorb。
        low_remaining = any(item.kind == "L" for item in observation.active_platforms())
        metrics = observation.metrics
        if low_remaining and observation.step < self.rules.probe_window:
            return "probe"
        if low_remaining and metrics.pressure >= self.rules.cautious_pressure:
            return "absorb"
        return "commit"

    def _reassignment_budget(self, observation: BaselineObservation, stage: str) -> int:
        """按阶段选取存活平台的一定比例，向上取整作为本轮任务数量上限。"""
        remaining = len(observation.active_platforms())
        if remaining == 0:
            return 0
        if stage == "probe":
            fraction = self.rules.probe_reassignment_fraction
        elif observation.metrics.pressure >= self.rules.cautious_pressure:
            fraction = self.rules.cautious_reassignment_fraction
        else:
            fraction = self.rules.commit_reassignment_fraction
        # 尚有可用平台时至少允许调整一架，避免小群的比例预算被舍入为零。
        return max(1, math.ceil(remaining * fraction))

    def _best_target(
        self,
        platform: PlatformState,
        targets: tuple[TargetPrior, ...],
        counts: dict[int, int],
    ) -> TargetPrior | None:
        """按当前占用数筛选适配目标，再按 R1 评分及编号稳定选取。"""
        candidates = [item for item in targets if target_is_compatible(platform.kind, item.entity_type) and counts.get(item.entity_id, 0) < self.rules.target_capacity]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (self.score(platform, item, counts.get(item.entity_id, 0)), -item.entity_id),
        )

    def _priority_assignments(
        self,
        observation: BaselineObservation,
        platforms: list[PlatformState],
        limit: int,
        counts: dict[int, int] | None = None,
    ) -> tuple[Assignment, ...]:
        """按调用方给出的平台顺序，逐个选择目标直至用完预算。"""
        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        # 无显式计数时复制当前任务占用；编组算法传入的计数则继续原地累计。
        counts = dict(observation.metrics.assigned_target_counts) if counts is None else counts
        assignments: list[Assignment] = []
        for platform in platforms:
            if len(assignments) >= limit:
                break
            target = self._best_target(platform, targets, counts)
            if target is None:
                continue
            score = self.score(platform, target, counts.get(target.entity_id, 0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            assignments.append(Assignment(platform.entity_id, target.entity_id, score))
        return tuple(assignments)


class R6PressureAssignmentPolicy(AdaptiveRollingPolicy):
    """R6：探测和高压阶段优先 L 平台，集中阶段按 H/M/L 顺序调整任务。"""

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """先确定阶段与候选顺序，再在阶段预算内执行优先级分配。"""
        stage = self._stage(observation)
        active = tuple(sorted(observation.active_platforms(), key=lambda item: item.entity_id))
        if stage in {"probe", "absorb"}:
            candidates = [item for item in active if item.kind == "L"]
        else:
            candidates = [item for kind in ("H", "M", "L") for item in active if item.kind == kind]
        return self._priority_assignments(observation, candidates, self._reassignment_budget(observation, stage))


class R7StrikePackagePolicy(AdaptiveRollingPolicy):
    """R7：集中阶段优先将 H/M 配成同目标攻击组，其余阶段先处理 L 任务。"""

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """在预算内先分配整对 H/M，再用剩余预算处理尚未入组的平台。"""
        stage = self._stage(observation)
        active = tuple(sorted(observation.active_platforms(), key=lambda item: item.entity_id))
        budget = self._reassignment_budget(observation, stage)
        if stage in {"probe", "absorb"}:
            low = [item for item in active if item.kind == "L"]
            return self._priority_assignments(observation, low, budget)

        targets = tuple(sorted(observation.active_targets(), key=lambda item: item.entity_id))
        counts = dict(observation.metrics.assigned_target_counts)
        assignments: list[Assignment] = []
        high = [item for item in active if item.kind == "H"]
        medium = [item for item in active if item.kind == "M"]
        low = [item for item in active if item.kind == "L"]
        # H/M 按编号一一配对，每对共同占用一个目标并消耗两个名额。
        paired = min(len(high), len(medium))
        processed_pairs = 0
        for index in range(paired):
            if len(assignments) + 2 > budget:
                break
            h_platform, m_platform = high[index], medium[index]
            target = self._best_target(h_platform, targets, counts)
            if target is None:
                break
            h_score = self.score(h_platform, target, counts.get(target.entity_id, 0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            m_score = self.score(m_platform, target, counts.get(target.entity_id, 0))
            counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
            assignments.extend((
                Assignment(h_platform.entity_id, target.entity_id, h_score),
                Assignment(m_platform.entity_id, target.entity_id, m_score),
            ))
            processed_pairs += 1

        # 仅排除实际完成分配的配对成员，其余平台继续参与剩余预算的普通分配。
        paired_ids = {item.entity_id for item in high[:processed_pairs] + medium[:processed_pairs]}
        remaining = [item for item in high + medium + low if item.entity_id not in paired_ids]
        if len(assignments) < budget:
            assignments.extend(self._priority_assignments(observation, remaining, budget - len(assignments), counts))
        return tuple(sorted(assignments, key=lambda item: item.platform_id))


class R8SatellitePackagePolicy(R7StrikePackagePolicy):
    """R8：沿用 R7 编组，并为每个本轮攻击目标选择一架 H 平台申请天眼。"""

    def __init__(self, rules: BaselineRules) -> None:
        """初始化本轮需要使用天眼的平台集合。"""
        super().__init__(rules)
        self.satellite_platform_ids: frozenset[int] = frozenset()

    def decide(self, observation: BaselineObservation) -> tuple[Assignment, ...]:
        """完成 R7 分配后，按目标去重挑选执行天眼的 H 平台。"""
        assignments = super().decide(observation)
        platform_by_id = {item.entity_id: item for item in observation.platforms}
        target_by_id = {item.entity_id: item for item in observation.targets}
        # 每个本轮目标最多挑选一架 H 平台；执行组件另维护本机是否已使用过天眼。
        selected: set[int] = set()
        covered_targets: set[int] = set()
        for assignment in assignments:
            platform = platform_by_id[assignment.platform_id]
            target = target_by_id[assignment.target_id]
            if (
                platform.kind == "H"
                and target.entity_type in {9400, 9600}
                and assignment.target_id not in covered_targets
            ):
                selected.add(assignment.platform_id)
                covered_targets.add(assignment.target_id)
        self.satellite_platform_ids = frozenset(selected)
        return assignments


class R9HierarchicalLearningPolicy(R8SatellitePackagePolicy):
    """R9 的任务层沿用 R8；学习 Actor 在执行层依据本机观测和任务上下文选择机动。"""


def build_red_baseline(
    name: str,
    rules: BaselineRules,
    seed: int,
    search_polygon: tuple[Position, ...] | None = None,
) -> RedBaseline:
    """按示例策略名称创建任务算法，静态搜索另需公开搜索边界。"""
    if name == "r0_random":
        return R0RandomPolicy(rules, seed)
    if name == "r1_priority":
        return R1PriorityPolicy(rules)
    if name == "r2_static_assignment":
        return R2StaticAssignmentPolicy(rules)
    if name == "r3_static_priority":
        return R3StaticPriorityPolicy(rules)
    if name == "r4_rolling_rules":
        return R4RollingRulePolicy(rules)
    if name == "r5_event_rolling":
        return R5EventRollingPolicy(rules)
    if name == "r6_pressure_assignment":
        return R6PressureAssignmentPolicy(rules)
    if name == "r7_strike_packages":
        return R7StrikePackagePolicy(rules)
    # 搜索几何实现按需导入，普通目标分配无需创建搜索路线工具。
    if name == "r7_static_search":
        if search_polygon is None:
            raise ValueError("r7_static_search requires a public search polygon")
        from ..coordination.search import R7StaticSearchPolicy

        return R7StaticSearchPolicy(rules, search_polygon)
    if name == "r8_satellite_packages":
        return R8SatellitePackagePolicy(rules)
    if name == "r9_hierarchical_learning":
        return R9HierarchicalLearningPolicy(rules)
    raise ValueError(f"Unknown red baseline '{name}'")
