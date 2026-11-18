"""一个当前通信群内的任务分配与动作生成。

输入为公开目标初值及群成员提交的单机报告，输出为目标引导坐标。
"""

from __future__ import annotations

import math

from .baselines import (
    BaselineObservation,
    BaselineRules,
    PlatformState,
    RED_POLICY_CHOICES,
    TacticalMetrics,
    TargetPrior,
    build_red_baseline,
    distance_km,
    engagement_effectiveness,
    target_is_compatible,
)
from .contracts import Position
from .targets import ReceivedTrackFusion
from .waypoints import bounded_waypoint


# 引擎实体类型映射到示例任务层使用的 H/M/L 分组；动作码沿用引擎接口。
_KIND_BY_TYPE = {21000: "H", 21001: "M", 21002: "L"}
_ACTION_SET_TARGET = 1.0


class RedBaselineCommander:
    """把群内报告转换为基线输入，维护分配计数并为各成员缓存待发送指令。

    每帧先接收报告与融合轨迹，再按策略节奏规划；各成员请求动作时消费自己的指令。
    """

    def __init__(
        self,
        targets: tuple[TargetPrior, ...],
        policy_name: str,
        seed: int = 1,
        search_polygon: tuple[Position, ...] | None = None,
    ) -> None:
        """保存策略与公开初值，创建规则参数，并初始化本局群内状态。"""
        if policy_name not in RED_POLICY_CHOICES:
            raise ValueError(f"Unsupported red baseline '{policy_name}'")
        # 每个新通信群以同一份公开目录起步，后续轨迹由该群收到的报告补充。
        self.initial_targets = tuple(item for item in targets if item.entity_type in (9400, 9600))
        self.policy_name = policy_name
        self.seed = seed
        self.search_polygon = search_polygon
        self.rules = BaselineRules()
        self.expected_platform_ids: set[int] = set()
        self.reset()

    def register_platform(self, entity_id: int) -> None:
        """登记群内预期实体，用于识别本帧实际报告与缺报成员。"""
        self.expected_platform_ids.add(int(entity_id))

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        """汇总本帧成员报告、更新轨迹及缺报状态，并生成自适应策略指标。"""

        observed_ids = {int(item["entity_id"]) for item in observations}
        if self.expected_platform_ids and not observed_ids <= self.expected_platform_ids:
            raise ValueError("Commander received a platform outside its communication domain")
        self.current_snapshot_ids = frozenset(observed_ids)
        self.absent_platform_ids = frozenset(self.expected_platform_ids - observed_ids)
        # 先写入本机状态，再融合其目标轨迹；位置或缺报变化会增加事件版本。
        changed = False
        for observation in observations:
            self.report(observation)
            changed = self.track_fusion.ingest(observation) or changed
        for entity_id in self.absent_platform_ids:
            # 缺报成员退出本次规划快照，并撤销旧分配；缺报本身只表示本帧没有报告。
            if self.reports.pop(entity_id, None) is not None:
                changed = True
            self.observed_target_ids.pop(entity_id, None)
            self.guidance.pop(entity_id, None)
            self.last_guidance.pop(entity_id, None)
            previous_target = self.target_by_platform.pop(entity_id, None)
            if previous_target is not None:
                self.assigned_by_target[previous_target] = max(
                    0, self.assigned_by_target.get(previous_target, 1) - 1
                )
        self.targets = self.track_fusion.targets
        if changed:
            self.event_revision += 1
        self.tactical_metrics = self._build_tactical_metrics(observations)

    @staticmethod
    def _track_field(track, name: str, default):
        """从字典或 SDK 记录读取同名轨迹字段。"""
        return track.get(name, default) if isinstance(track, dict) else getattr(track, name, default)

    def report(self, observation: dict) -> None:
        """读取报告中的本机类型、位置、存活状态及已知目标编号。"""
        own = observation.get("self", {})
        entity_id = int(observation.get("entity_id", -1))
        kind = _KIND_BY_TYPE.get(int(own.get("type", 0)))
        position = own.get("pos", {})
        if entity_id < 0 or kind is None or not isinstance(position, dict):
            return
        if self.expected_platform_ids and entity_id not in self.expected_platform_ids:
            raise ValueError("Platform is not registered in this communication domain")
        # 保存该成员实际收到的目标编号，供暗巢攻击动作的可用性检查使用。
        self.observed_target_ids[entity_id] = {
            int(self._track_field(track, "entity_id", raw_id))
            for raw_id, track in (own.get("detectInfo") or {}).items()
            if int(self._track_field(track, "entity_type", -1)) in (9400, 9500, 9600)
        }
        self.reports[entity_id] = PlatformState(
            entity_id=entity_id,
            kind=kind,
            position=Position(
                x=float(position.get("x", 0.0)),
                y=float(position.get("y", 0.0)),
                z=float(position.get("z", 0.0)),
            ),
            alive=float(own.get("health", 0)) > 0 and bool(own.get("isVisible", True)),
        )

    def _build_tactical_metrics(self, observations: tuple[dict, ...]) -> TacticalMetrics:
        """对本群收到的自爆虫轨迹去重，形成压力、成员数与任务计数。"""
        # 多个成员可能探测到同一自爆虫，压力指标按实体编号去重。
        interceptor_ids: set[int] = set()
        for observation in observations:
            for raw_id, track in (observation.get("self", {}).get("detectInfo") or {}).items():
                if int(self._track_field(track, "entity_type", -1)) == 24000:
                    interceptor_ids.add(int(self._track_field(track, "entity_id", raw_id)))
        alive_count = sum(item.alive for item in self.reports.values())
        return TacticalMetrics(
            observed_interceptor_count=len(interceptor_ids),
            alive_count=alive_count,
            event_revision=self.event_revision,
            assigned_target_counts=tuple(sorted(self.assigned_by_target.items())),
        )

    def action_for(self, entity_id: int, step: int) -> list[float] | None:
        """按需触发本帧规划，返回并消费本机待发指令，或尝试搜索策略的即时重定向。"""

        entity_id = int(entity_id)
        if self.current_snapshot_ids is None:
            raise RuntimeError("begin_step must provide the domain snapshot before planning")
        if entity_id not in self.current_snapshot_ids:
            return None
        if self._should_plan(step):
            self._plan(step)
        # 每条计划动作只发送一次；后续帧保持引擎已执行的任务，直到计划发生变化。
        action = self.guidance.pop(entity_id, None)
        return action if action is not None else self._retarget_action_for(entity_id)

    def waypoint_action_for(self, entity_id: int, step: int, map_area: dict) -> list[float] | None:
        action = self.action_for(entity_id, step)
        if action is None:
            return None
        x, y = bounded_waypoint(action[2], action[3], map_area)
        # 任务延续和去重使用实际提交位置，融合轨迹仍保留原始观测。
        target_id = self.last_guidance[entity_id][0]
        self.last_guidance[entity_id] = (target_id, x, y)
        return [action[0], action[1], x, y]

    def _should_plan(self, step: int) -> bool:
        """依据静态、周期、事件和自适应策略的不同节奏判断是否重新分配。"""
        if not self.current_snapshot_ids:
            return False
        if self.last_plan_step < 0:
            return True
        # 静态策略只做初次规划；R6 以上使用较短周期，R5 还要求事件版本前进。
        if self.policy_name in {"r2_static_assignment", "r3_static_priority"}:
            return False
        if self.policy_name in {
            "r6_pressure_assignment", "r7_strike_packages", "r7_static_search",
            "r8_satellite_packages", "r9_hierarchical_learning",
        }:
            return step - self.last_plan_step >= self.rules.adaptive_replan_interval
        if self.policy_name == "r5_event_rolling":
            return (
                self.event_revision > self.planned_revision
                and step - self.last_plan_step >= self.rules.event_replan_cooldown
            )
        return step - self.last_plan_step >= self.rules.replan_interval

    def _plan(self, step: int) -> None:
        """调用任务算法，更新选中成员的目标、引导航点与待发动作。"""
        platforms = tuple(sorted(self.reports.values(), key=lambda item: item.entity_id))
        # 算法只接收本群整理后的不可变值对象，返回分配结果而不直接操纵引擎。
        decision = self.policy.decide(BaselineObservation(step, platforms, self.targets, self.tactical_metrics))
        initial_plan = self.last_plan_step < 0
        # R4 后续每轮仅更新四分之一存活成员，减少整群同时换目标。
        if not initial_plan and self.policy_name == "r4_rolling_rules":
            decision = decision[:math.ceil(sum(item.alive for item in platforms) * 0.25)]
        assignments = {item.platform_id: item for item in decision}
        targets = {item.entity_id: item for item in self.targets}
        # 初次规划覆盖全部存活成员；后续仅处理算法选中的成员，其余沿用原任务。
        if initial_plan:
            selected = tuple(item for item in platforms if item.alive)
            counts: dict[int, int] = {}
        else:
            selected = tuple(item for item in platforms if item.alive and item.entity_id in assignments)
            counts = dict(self.assigned_by_target)
            # 先扣除本次待替换成员的旧任务占用，再按新目标补回计数。
            for platform in selected:
                previous = self.target_by_platform.get(platform.entity_id)
                if previous is not None:
                    counts[previous] = max(0, counts.get(previous, 1) - 1)

        # 显式搜索结果优先；攻击结果无法用于本机时，退回可用目标或公开区域搜索。
        for platform in selected:
            assignment = assignments.get(platform.entity_id)
            target = targets.get(assignment.target_id) if assignment is not None else None
            is_search = assignment is not None and assignment.destination is not None
            if is_search:
                target = None
                destination = assignment.destination
            else:
                if target is None or not self._target_is_available(platform, target):
                    target = self._best_target(platform, counts)
                destination = target.position if target is not None else self._search_waypoint(platform)
            if target is not None:
                self.target_by_platform[platform.entity_id] = target.entity_id
                counts[target.entity_id] = counts.get(target.entity_id, 0) + 1
                # 签名同时包含内部目标编号和位置；目标移动后需重发坐标动作。
                signature = (target.entity_id, destination.x, destination.y)
            else:
                self.target_by_platform.pop(platform.entity_id, None)
                signature = (None, destination.x, destination.y)
            action = [_ACTION_SET_TARGET, float(platform.entity_id), destination.x, destination.y]
            # 仅把变化后的任务放入待发队列，避免每次规划重复发送同一动作。
            if initial_plan or self.last_guidance.get(platform.entity_id) != signature:
                self.guidance[platform.entity_id] = action
            self.last_guidance[platform.entity_id] = signature
        self.assigned_by_target = {target_id: count for target_id, count in counts.items() if count > 0}
        self.last_plan_step = step
        self.planned_revision = self.event_revision

    def _target_is_available(self, platform: PlatformState, target: TargetPrior) -> bool:
        """检查任务目标的存活、类型适配及本机暗巢情报条件。"""
        return (
            target.alive
            and target_is_compatible(platform.kind, target.entity_type)
            and (target.entity_type in (9400, 9600)
                 or target.entity_id in self.observed_target_ids.get(platform.entity_id, set()))
        )

    def _best_target(self, platform: PlatformState, counts: dict[int, int]) -> TargetPrior | None:
        """在本机可用目标中，按价值、预期攻击效果、距离与已分配数选择目标。"""
        candidates = [
            target for target in self.targets
            if self._target_is_available(platform, target)
            and counts.get(target.entity_id, 0) < self.rules.target_capacity
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda target: (
            self.rules.value_weight * target.value * engagement_effectiveness(platform.kind, target.entity_type)
            - self.rules.distance_weight * distance_km(platform.position, target.position) / 100.0
            - self.rules.coverage_weight * counts.get(target.entity_id, 0),
            -target.entity_id,
        ))

    def _search_waypoint(self, platform: PlatformState) -> Position:
        """以公开区域中心和编号对应顶点的中点生成稳定搜索航点。"""
        if not self.search_polygon:
            return platform.position
        center = Position(
            sum(point.x for point in self.search_polygon) / len(self.search_polygon),
            sum(point.y for point in self.search_polygon) / len(self.search_polygon),
            platform.position.z,
        )
        vertex = self.search_polygon[platform.entity_id % len(self.search_polygon)]
        return Position((center.x + vertex.x) / 2.0, (center.y + vertex.y) / 2.0, platform.position.z)

    def _retarget_action_for(self, entity_id: int) -> list[float] | None:
        """让支持即时重定向的策略认领本机已知目标，并同步分配计数。"""
        platform = self.reports.get(entity_id)
        if platform is None:
            return None
        # 即时重定向按距离、编号排序，由具体搜索策略决定是否认领其中一个目标。
        candidates = tuple(sorted(
            (target for target in self.targets if self._target_is_available(platform, target)),
            key=lambda target: (distance_km(platform.position, target.position), target.entity_id),
        ))
        target = self.policy.claim_retarget(platform, candidates)
        if target is None:
            return None
        previous = self.target_by_platform.get(entity_id)
        if previous is not None:
            self.assigned_by_target[previous] = max(0, self.assigned_by_target.get(previous, 1) - 1)
        self.target_by_platform[entity_id] = target.entity_id
        self.assigned_by_target[target.entity_id] = self.assigned_by_target.get(target.entity_id, 0) + 1
        self.last_guidance[entity_id] = (target.entity_id, target.position.x, target.position.y)
        return [_ACTION_SET_TARGET, float(entity_id), target.position.x, target.position.y]

    def target_id_for(self, platform_id: int) -> int | None:
        """查询本群规划中分配给指定成员的目标编号。"""
        return self.target_by_platform.get(int(platform_id))

    def should_use_satellite(self, platform_id: int) -> bool:
        """查询任务策略为该成员安排的天眼使用标记。"""
        return int(platform_id) in getattr(self.policy, "satellite_platform_ids", frozenset())

    def learning_task_context(self, platform_id: int) -> tuple[float, float, float, float, float]:
        """为 R9 返回五维群内任务特征。

        依次为三类目标的 one-hot、群内压力，以及本机是否有有效报告。
        """

        target_id = self.target_id_for(platform_id)
        target = next((item for item in self.targets if item.entity_id == target_id), None)
        entity_type = target.entity_type if target is not None else -1
        return (
            float(entity_type == 9400), float(entity_type == 9500), float(entity_type == 9600),
            self.tactical_metrics.pressure,
            1.0 if self.reports.get(int(platform_id)) is not None else 0.0,
        )

    def reset(self) -> None:
        """清空群内报告与动作缓存，并从公开初值重建任务策略和轨迹融合器。"""
        # 群内动态快照与分配账本：分别保存成员状态、目标占用及成员当前任务。
        self.reports: dict[int, PlatformState] = {}
        self.current_snapshot_ids: frozenset[int] | None = None
        self.absent_platform_ids: frozenset[int] = frozenset()
        self.assigned_by_target: dict[int, int] = {}
        self.target_by_platform: dict[int, int] = {}
        # guidance 是一次性待发队列，last_guidance 保存已计算任务的签名以去重。
        self.guidance: dict[int, list[float]] = {}
        self.last_guidance: dict[int, tuple[int | None, float, float]] = {}
        self.observed_target_ids: dict[int, set[int]] = {}
        # 帧号控制周期重规划；事件版本控制 R5 的变化检测与冷却。
        self.last_plan_step = -1
        self.event_revision = 0
        self.planned_revision = -1
        self.tactical_metrics = TacticalMetrics()
        self.policy = build_red_baseline(self.policy_name, self.rules, self.seed, self.search_polygon)
        self.targets = self.initial_targets
        self.track_fusion = ReceivedTrackFusion(self.initial_targets)
