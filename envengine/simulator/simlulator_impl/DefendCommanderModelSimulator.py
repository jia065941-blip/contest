"""Simulation boundary for an injected built-in blue defense controller."""

import logging
import math
from numbers import Real
from types import MappingProxyType
from typing import Callable

from envengine.sdk.Util import UtilsPy
from envengine.common.BaseStruct import Vector3d
from envengine.common import SimmerCommandType, InterceptorLaunchCmd, PosVelAcc
from envengine.sdk.base_struct.Basic import DetectInfo
from envengine.sdk.base_struct.Entity import EntityExt
from envengine.sdk.base_struct.Message import Command
from envengine.simulator.decorator import Simulator
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory
from blue_defense.base import (
    BaseBlueDefense,
    BlueClock,
    BlueDecision,
    BlueObservation,
    InterceptorAssignment,
    InterceptorState,
    OwnEntityState,
    ScheduledInterceptorAssignment,
    ThreatTrack
)


@Simulator.register("DefendCommanderS")
class DefendCommanderModelSimulator(ISimulator):
    """Build observations, validate launches and execute controller decisions."""

    def __init__(
        self,
        entity_ext: EntityExt,
        send_commands: Callable[[list[Command]], None],
        send_events: Callable[[dict], None],
        simulator_factory: SimulatorFactory = None,
    ):
        super().__init__(entity_ext, send_commands, send_events, simulator_factory)
        self.launched_list: dict[int, list[int]] = {}
        self.pending_launches: dict[int, ScheduledInterceptorAssignment] = {}
        self._executed_coordination_members: dict[str, set[int]] = {}
        self.coordination_metrics = self._new_coordination_metrics()
        # 场景初始化仅创建实体；防御策略由运行阶段显式注入。
        self.defense_controller: BaseBlueDefense | None = None
        self._missing_controller_warned = False
        self._blue_clock = self._new_blue_clock()
        self._new_tracks: list[ThreatTrack] = []
        self._observed_target_ids: set[int] = set()

    def set_defense_controller(
        self,
        controller: BaseBlueDefense | None,
    ) -> None:
        """为当前控制节点注入一个配置完成的虫族控制器。"""

        if controller is not None and not isinstance(controller, BaseBlueDefense):
            raise TypeError("controller must inherit BaseBlueDefense")
        self.defense_controller = controller
        self._missing_controller_warned = False
        self.pending_launches = {}
        self._new_tracks = []
        self._observed_target_ids = set()
        if controller is None:
            logging.warning("Blue defense controller disabled")
        else:
            controller.bind_clock(self._blue_clock)
            logging.info("Blue defense controller activated: %s", controller.name)

    @property
    def simulator_sim_step(self) -> float:
        return 1000

    def update(self) -> None:
        if (
            self.defense_controller is not None
            and self.defense_controller.plans_launch_time
        ):
            self._execute_due_launches()
        self._decide_from_information()

    def set_position(self, pos: Vector3d) -> None:
        pass

    def set_speed(self, speed: float) -> None:
        pass

    def handel_detect_info(self, new_detect_info: dict[int, DetectInfo]) -> None:
        """原始探测即时交给蓝方，另收集本次决策期间收到的记录。"""
        tracks = [self._detect_info_to_threat_track(item) for item in new_detect_info.values()]
        if self.defense_controller is None:
            return
        self.defense_controller.receive_tracks(tracks, self.current_step)
        self._observed_target_ids.update(track.target_id for track in tracks)
        self._new_tracks.extend(tracks)

    def _decide_from_information(self) -> None:
        controller = self.defense_controller
        if controller is None:
            self._warn_missing_controller()
            return
        if controller.plans_launch_time:
            observation = self._build_blue_observation()
            decision = controller.decide(observation)
            self._validate_decision(decision)
            self._new_tracks = []
            if decision is None:
                return
            decisions = decision.assignments
            if decision.solve_stats:
                self._record_solve_stats(decision.solve_stats)
            immediate_decisions = [
                item
                for item in decisions
                if isinstance(item, InterceptorAssignment)
            ]
            if immediate_decisions:
                self.pending_launches = {}
                self._execute_immediate_assignments(immediate_decisions)
                return
            planned_launches = {
                item.interceptor_id: item
                for item in decisions
            }
            if controller.preserves_pending_launches:
                self.pending_launches.update(planned_launches)
            else:
                self.pending_launches = planned_launches
            self._execute_due_launches()
            return

        observation = self._build_blue_observation()
        decision = controller.decide(observation)
        self._validate_decision(decision)
        self._new_tracks = []
        if decision is None:
            return
        self._execute_immediate_assignments(list(decision.assignments))

    def _validate_decision(self, decision: BlueDecision | None) -> None:
        """在修改待释放队列或释放前，检查策略输出的公共结构。"""
        if decision is None:
            return
        if not isinstance(decision, BlueDecision):
            raise TypeError("blue defense decide must return BlueDecision or None")
        if not isinstance(decision.assignments, tuple):
            raise TypeError("BlueDecision.assignments must be a tuple")
        if not isinstance(decision.solve_stats, dict):
            raise TypeError("BlueDecision.solve_stats must be a dict")
        for name, value in decision.solve_stats.items():
            if not isinstance(name, str) or not isinstance(value, (int, float, str, bool)):
                raise TypeError("BlueDecision.solve_stats must contain string keys and numeric, string or bool values")

        assignment_kind = None
        interceptor_ids = set()
        for index, assignment in enumerate(decision.assignments):
            prefix = f"BlueDecision.assignments[{index}]"
            if isinstance(assignment, ScheduledInterceptorAssignment):
                kind = ScheduledInterceptorAssignment
                if not self.defense_controller.plans_launch_time:
                    raise ValueError("blue policy without plans_launch_time cannot return scheduled assignments")
                for name in ("decision_step", "planned_launch_step", "estimated_intercept_step"):
                    self._validate_finite_number(getattr(assignment, name), f"{prefix}.{name}")
                for name in ("coordination_id", "coordination_mode", "spatial_mode"):
                    value = getattr(assignment, name)
                    if value is not None and not isinstance(value, str):
                        raise TypeError(f"{prefix}.{name} must be a string or None")
                for name in ("launch_pos", "estimated_intercept_pos"):
                    value = getattr(assignment, name)
                    if value is not None:
                        self._validate_assignment_vector(value, f"{prefix}.{name}")
            elif isinstance(assignment, InterceptorAssignment):
                kind = InterceptorAssignment
            else:
                raise TypeError(f"{prefix} must be an immediate or scheduled interceptor assignment")
            if assignment_kind is not None and assignment_kind is not kind:
                raise ValueError("BlueDecision cannot mix immediate and scheduled assignments")
            assignment_kind = kind
            for name in ("interceptor_id", "target_id"):
                if type(getattr(assignment, name)) is not int:
                    raise TypeError(f"{prefix}.{name} must be an int, excluding bool")
            if assignment.interceptor_id in interceptor_ids:
                raise ValueError(f"BlueDecision contains duplicate interceptor {assignment.interceptor_id}")
            interceptor_ids.add(assignment.interceptor_id)

    @staticmethod
    def _validate_finite_number(value, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite number, excluding bool")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must be finite")

    @classmethod
    def _validate_assignment_vector(cls, value, name: str) -> None:
        if not isinstance(value, Vector3d):
            raise TypeError(f"{name} must be a blue defense Vector3")
        for axis in ("x", "y", "z"):
            cls._validate_finite_number(getattr(value, axis), f"{name}.{axis}")

    def _execute_immediate_assignments(
        self,
        decisions: list[InterceptorAssignment],
    ) -> None:
        resolved = []
        for item in decisions:
            target = self._simulator_factory.get_blue_track(item.target_id, self.current_step)
            if target is not None and self._assignment_is_launch_eligible(item, target):
                resolved.append((item, target))
        assignments = [item for item, _ in resolved]
        command_list = [self._assignment_to_command(item, target) for item, target in resolved]

        for item in assignments:
            launched = self.launched_list.setdefault(item.target_id, [])
            if item.interceptor_id not in launched:
                launched.append(item.interceptor_id)
        self.coordination_metrics["actual_launches"] += len(assignments)
        self.coordination_metrics["immediate_launches"] += len(assignments)

        if command_list:
            self._send_commands(command_list)

    def _build_blue_observation(self) -> BlueObservation:
        controller = self.defense_controller
        if controller is None:
            raise RuntimeError("blue defense observation requested without a controller")
        launched_interceptor_ids = {item for items in self.launched_list.values() for item in items}
        reserved_interceptor_ids = (
            set(self.pending_launches)
            if controller.preserves_pending_launches
            else set()
        )
        interceptors = []
        for simulator in self._simulator_factory.get_simulators_by_type(24000):
            entity = simulator.entity_ext.entity
            max_launch_angle = self._parent_max_launch_angle(entity.parentId)
            interceptors.append(
                InterceptorState(
                    interceptor_id=entity.id,
                    pos=self._copy_vector(entity.posEcf),
                    vel=self._copy_vector(entity.velEcf),
                    available=(
                        entity.survivePoints > 0
                        and entity.id not in launched_interceptor_ids
                        and entity.id not in reserved_interceptor_ids
                        and max_launch_angle > 0
                    ),
                    max_launch_angle_deg=max_launch_angle,
                )
            )

        own_entities = []
        for simulator in self._simulator_factory.get_simulators_by_side(1):
            entity = simulator.entity_ext.entity
            own_entities.append(OwnEntityState(
                entity_id=entity.id, name=entity.nameChn, type=entity.entityType,
                pos=self._copy_vector(entity.posEcf),
                vel=self._copy_vector(entity.velEcf),
                health=entity.survivePoints, is_visible=entity.isVisible,
            ))

        launched_map = {
            key: value.copy()
            for key, value in self.launched_list.items()
        }
        if controller.preserves_pending_launches:
            for assignment in self.pending_launches.values():
                launched_map.setdefault(assignment.target_id, []).append(
                    assignment.interceptor_id
                )

        if getattr(self, "_blue_clock", None) is None:
            self._blue_clock = self._new_blue_clock()
        return BlueObservation(
            step=self.current_step, clock=self._blue_clock,
            targets=tuple(self._new_tracks), interceptors=tuple(interceptors),
            own_entities=tuple(own_entities),
            launched_map=MappingProxyType({key: tuple(value) for key, value in launched_map.items()}),
        )

    def _new_blue_clock(self) -> BlueClock:
        if self._simulator_factory is None:
            return BlueClock()
        profile = self._simulator_factory.profile.imagineProfile
        return BlueClock(
            start_time_ms=profile.simTime,
            step_duration_ms=profile.simStep,
        )

    def _detect_info_to_threat_track(self, detect_info: DetectInfo) -> ThreatTrack:
        radar_ids = {radar.entity_ext.entity.id
                     for radar in self._simulator_factory.get_simulators_by_type(44000)
                     if radar.entity_ext.entity.sideId == 1}
        if detect_info.detect_from not in radar_ids:
            raise ValueError("虫族信息库只接受己方视野塔的感知记录")

        track = ThreatTrack(
            target_id=detect_info.entity_id,
            name=detect_info.nameChn,
            type=detect_info.entity_type,
            pos=self._copy_vector(detect_info.pos),
            vel=self._copy_vector(detect_info.vel),
            detect_step=detect_info.step,
            source_id=detect_info.detect_from,
        )
        if (track.target_id <= 0 or track.source_id <= 0
                or track.type not in (21000, 21001, 21002)
                or not math.isfinite(track.detect_step)
                or track.detect_step < 0 or track.detect_step > self.current_step):
            raise ValueError("蓝方探测记录的身份、来源或时间无效")
        for vector in (track.pos, track.vel):
            if not all(math.isfinite(value) for value in (vector.x, vector.y, vector.z)):
                raise ValueError("蓝方探测记录包含无效坐标或速度")
        return track

    def _assignment_to_command(self, assignment: InterceptorAssignment, target=None) -> Command:
        if target is None:
            target:ThreatTrack = self._simulator_factory.get_blue_track(assignment.target_id, self.current_step)
        if target is None:
            raise ValueError("blue assignment requires a known target state")
        target_pos = target.pos
        target_vel = target.vel
        return Command(
            executorId=assignment.interceptor_id,
            commandTypeId=SimmerCommandType.INTERCEPTOR_LAUNCH,
            commandAttributes=InterceptorLaunchCmd(
                weaponId=assignment.interceptor_id,
                targetId=assignment.target_id,
                targetPosInfo=PosVelAcc(
                    cPos=Vector3d(target_pos.x, target_pos.y, target_pos.z),
                    cVel=Vector3d(target_vel.x, target_vel.y, target_vel.z),
                ),
            ),
        )

    def _execute_due_launches(self) -> None:
        current_step = self._blue_clock.continuous_step_at(self.sim_time)
        self.pending_launches = {
            interceptor_id: assignment
            for interceptor_id, assignment in self.pending_launches.items()
            if self._current_assignment(assignment) is not None
        }
        due = sorted(
            (
                assignment
                for assignment in self.pending_launches.values()
                if assignment.planned_launch_step <= current_step
            ),
            key=lambda item: item.planned_launch_step,
        )
        command_list = []
        for scheduled in due:
            self.pending_launches.pop(scheduled.interceptor_id, None)
            assignment = self._current_assignment(scheduled)
            if assignment is None:
                continue
            target = self._simulator_factory.get_blue_track(assignment.target_id, self.current_step)
            if target is None or not self._assignment_is_launch_eligible(assignment, target):
                continue
            command_list.append(self._assignment_to_command(assignment, target))
            launched = self.launched_list.setdefault(assignment.target_id, [])
            launched.append(assignment.interceptor_id)
            self.coordination_metrics["actual_launches"] += 1
            coordination_id = scheduled.coordination_id
            if coordination_id is not None:
                members = self._executed_coordination_members.setdefault(
                    coordination_id,
                    set(),
                )
                members.add(scheduled.interceptor_id)
                if len(members) == 2:
                    self.coordination_metrics["executed_coordination_pairs"] += 1
                    temporal_mode = scheduled.coordination_mode
                    spatial_mode = scheduled.spatial_mode
                    if temporal_mode in ("staggered", "synchronized"):
                        self.coordination_metrics["executed_temporal_pairs"] += 1
                        self.coordination_metrics[
                            f"executed_{temporal_mode}_pairs"
                        ] += 1
                    if spatial_mode in ("concentrated", "layered", "crossfire"):
                        self.coordination_metrics["executed_spatial_pairs"] += 1
                        self.coordination_metrics[
                            f"executed_{spatial_mode}_pairs"
                        ] += 1
                    if temporal_mode is not None and spatial_mode is not None:
                        self.coordination_metrics[
                            "executed_spatiotemporal_pairs"
                        ] += 1
            planned_delay = scheduled.planned_launch_step - scheduled.decision_step
            self.coordination_metrics["planned_delay_step_total"] += planned_delay
            if planned_delay > 0:
                self.coordination_metrics["delayed_launches"] += 1
            else:
                self.coordination_metrics["immediate_launches"] += 1
        if command_list:
            self._send_commands(command_list)

    def _current_assignment(
        self,
        scheduled: ScheduledInterceptorAssignment,
    ) -> InterceptorAssignment | None:
        interceptor = self._simulator_factory.get_simulator_by_id(scheduled.interceptor_id)
        target = self._simulator_factory.get_blue_track(scheduled.target_id, self.current_step)
        if interceptor is None or target is None:
            return None
        launched_interceptor_ids = {
            item
            for items in self.launched_list.values()
            for item in items
        }
        if (
            scheduled.interceptor_id in launched_interceptor_ids
            or interceptor.entity_ext.entity.survivePoints <= 0
        ):
            return None
        return InterceptorAssignment(
            interceptor_id=scheduled.interceptor_id,
            target_id=scheduled.target_id,
        )

    def _assignment_is_launch_eligible(self, assignment: InterceptorAssignment, target=None) -> bool:
        """Preserve the original engine's platform angle/range launch gate."""
        interceptor = self._simulator_factory.get_simulator_by_id(assignment.interceptor_id)
        if interceptor is None:
            return False
        if target is None:
            target:ThreatTrack = self._simulator_factory.get_blue_track(assignment.target_id, self.current_step)
        if (interceptor.entity_ext.entity.survivePoints <= 0
                or any(assignment.interceptor_id in ids for ids in self.launched_list.values())
                or target is None):
            return False
        return self._interceptor_is_launch_eligible(
            interceptor.entity_ext.entity,
            target.pos,
            target.vel,
        )

    def _interceptor_is_launch_eligible(self, interceptor_entity, target_pos, target_vel) -> bool:
        parent_id = interceptor_entity.parentId
        for entity_type, max_angle in ((9500, 30.0), (9600, 20.0)):
            for platform in self._simulator_factory.get_simulators_by_type(entity_type):
                entity = platform.entity_ext.entity
                if entity.id != parent_id:
                    continue
                relative = self._subtract(entity.posEcf, target_pos)
                if self._vector_length(relative) >= 600_000:
                    return False
                return self._vector_angle(relative, target_vel) < max_angle
        return False

    def reset(self) -> None:
        super().reset()
        self._new_tracks = []
        self._observed_target_ids = set()
        self._blue_clock = self._new_blue_clock()
        if self.defense_controller is not None:
            self.defense_controller.reset()
            self.defense_controller.bind_clock(self._blue_clock)
        self.launched_list = {}
        self.pending_launches = {}
        self._executed_coordination_members = {}
        self.coordination_metrics = self._new_coordination_metrics()
        self._missing_controller_warned = False

    def init_model(self):
        pass

    def command_received(self, command: Command) -> None:
        if command.commandTypeId == SimmerCommandType.RADAR_DETECT_STATUS_UPDATE:
            self.handel_detect_info(command.commandAttributes)
        else:
            super().command_received(command)

    def _parent_max_launch_angle(self, parent_id: int) -> float:
        for entity_type, max_angle in ((9500, 30.0), (9600, 20.0)):
            for platform in self._simulator_factory.get_simulators_by_type(entity_type):
                if platform.entity_ext.entity.id == parent_id:
                    return max_angle
        return 0.0

    @staticmethod
    def _new_coordination_metrics() -> dict[str, float | int]:
        return {
            "optimization_runs": 0,
            "optimal_runs": 0,
            "feasible_runs": 0,
            "fallback_runs": 0,
            "candidate_count_max": 0,
            "observed_target_count_max": 0,
            "candidate_target_count_max": 0,
            "selected_target_count_max": 0,
            "selected_target_count_total": 0,
            "selected_assignment_count_total": 0,
            "solve_time_s_total": 0.0,
            "solve_time_s_max": 0.0,
            "mip_gap_max": 0.0,
            "actual_launches": 0,
            "immediate_launches": 0,
            "delayed_launches": 0,
            "planned_delay_step_total": 0.0,
            "direct_fire_assignments": 0,
            "successive_first_plans": 0,
            "coordination_pair_plans": 0,
            "temporal_pair_plans": 0,
            "staggered_pair_plans": 0,
            "synchronized_pair_plans": 0,
            "executed_temporal_pairs": 0,
            "executed_staggered_pairs": 0,
            "executed_synchronized_pairs": 0,
            "intercept_gap_step_total": 0.0,
            "spatial_pair_plans": 0,
            "concentrated_pair_plans": 0,
            "layered_pair_plans": 0,
            "crossfire_pair_plans": 0,
            "spatiotemporal_pair_plans": 0,
            "executed_coordination_pairs": 0,
            "executed_spatial_pairs": 0,
            "executed_concentrated_pairs": 0,
            "executed_layered_pairs": 0,
            "executed_crossfire_pairs": 0,
            "executed_spatiotemporal_pairs": 0,
            "intercept_separation_m_total": 0.0,
            "approach_angle_deg_total": 0.0,
            "asset_clearance_m_total": 0.0,
        }

    def _record_solve_stats(
        self,
        stats: dict[str, float | int | str | bool],
    ) -> None:
        # 策略可提供稀疏统计或文字诊断；只有数值项参与累计。
        stats = {name: value for name, value in stats.items() if isinstance(value, (int, float))}
        metrics = self.coordination_metrics
        metrics["optimization_runs"] += 1
        if stats.get("optimal", False):
            metrics["optimal_runs"] += 1
        else:
            metrics["feasible_runs"] += 1
        if stats.get("fallback", False):
            metrics["fallback_runs"] += 1
        metrics["candidate_count_max"] = max(
            metrics["candidate_count_max"],
            stats.get("candidate_count", 0),
        )
        metrics["observed_target_count_max"] = max(
            metrics["observed_target_count_max"],
            stats.get("observed_target_count", 0),
        )
        metrics["candidate_target_count_max"] = max(
            metrics["candidate_target_count_max"],
            stats.get("candidate_target_count", 0),
        )
        metrics["selected_target_count_max"] = max(
            metrics["selected_target_count_max"],
            stats.get("selected_target_count", 0),
        )
        metrics["selected_target_count_total"] += stats.get(
            "selected_target_count",
            0,
        )
        metrics["selected_assignment_count_total"] += stats.get(
            "selected_assignment_count",
            0,
        )
        metrics["direct_fire_assignments"] += stats.get(
            "selected_direct_fire_count",
            0,
        )
        metrics["successive_first_plans"] += stats.get(
            "selected_successive_first_count",
            0,
        )
        metrics["solve_time_s_total"] += stats.get("solve_time_s", 0.0)
        metrics["solve_time_s_max"] = max(
            metrics["solve_time_s_max"],
            stats.get("solve_time_s", 0.0),
        )
        metrics["mip_gap_max"] = max(metrics["mip_gap_max"], stats.get("mip_gap", 0.0))
        metrics["coordination_pair_plans"] += stats.get(
            "selected_pair_count",
            0,
        )
        metrics["temporal_pair_plans"] += stats.get(
            "selected_temporal_pair_count",
            0,
        )
        metrics["staggered_pair_plans"] += stats.get(
            "selected_staggered_pair_count",
            0,
        )
        metrics["synchronized_pair_plans"] += stats.get(
            "selected_synchronized_pair_count",
            0,
        )
        metrics["intercept_gap_step_total"] += stats.get(
            "intercept_gap_step_total",
            0.0,
        )
        metrics["spatial_pair_plans"] += stats.get(
            "selected_spatial_pair_count",
            0,
        )
        for mode in ("concentrated", "layered", "crossfire"):
            metrics[f"{mode}_pair_plans"] += stats.get(
                f"selected_{mode}_pair_count",
                0,
            )
        metrics["spatiotemporal_pair_plans"] += stats.get(
            "selected_spatiotemporal_pair_count",
            0,
        )
        for metric in (
            "intercept_separation_m_total",
            "approach_angle_deg_total",
            "asset_clearance_m_total",
        ):
            metrics[metric] += stats.get(metric, 0.0)

    def get_coordination_metrics(self) -> dict[str, float | int | str]:
        metrics = dict(self.coordination_metrics)
        optimization_runs = metrics["optimization_runs"]
        actual_launches = metrics["actual_launches"]
        metrics["solve_time_s_mean"] = (
            metrics["solve_time_s_total"] / optimization_runs
            if optimization_runs
            else 0.0
        )
        metrics["selected_target_count_mean"] = (
            metrics["selected_target_count_total"] / optimization_runs
            if optimization_runs
            else 0.0
        )
        metrics["planned_delay_step_mean"] = (
            metrics["planned_delay_step_total"] / actual_launches
            if actual_launches
            else 0.0
        )
        temporal_pair_plans = metrics["temporal_pair_plans"]
        metrics["intercept_gap_step_mean"] = (
            metrics["intercept_gap_step_total"] / temporal_pair_plans
            if temporal_pair_plans
            else 0.0
        )
        spatial_pair_plans = metrics["spatial_pair_plans"]
        for total_name, mean_name in (
            ("intercept_separation_m_total", "intercept_separation_m_mean"),
            ("approach_angle_deg_total", "approach_angle_deg_mean"),
            ("asset_clearance_m_total", "asset_clearance_m_mean"),
        ):
            metrics[mean_name] = (
                metrics[total_name] / spatial_pair_plans
                if spatial_pair_plans
                else 0.0
            )
        controller = self.defense_controller
        if controller is not None and hasattr(controller, "temporal_mode"):
            metrics["temporal_mode"] = controller.temporal_mode
        if controller is not None and hasattr(controller, "spatial_mode"):
            metrics["spatial_mode"] = controller.spatial_mode
        if controller is not None and hasattr(controller, "spatiotemporal_mode"):
            metrics["spatiotemporal_mode"] = (
                controller.spatiotemporal_mode
            )
        return metrics

    @staticmethod
    def _copy_vector(value) -> Vector3d:
        return Vector3d(
            float(DefendCommanderModelSimulator._get_coord(value, "x")),
            float(DefendCommanderModelSimulator._get_coord(value, "y")),
            float(DefendCommanderModelSimulator._get_coord(value, "z")),
        )

    @staticmethod
    def _get_coord(value, name: str) -> float:
        coord = getattr(value, name, 0.0)
        return coord() if callable(coord) else coord

    @staticmethod
    def _subtract(left, right) -> Vector3d:
        return Vector3d(
            DefendCommanderModelSimulator._get_coord(left, "x") - right.x,
            DefendCommanderModelSimulator._get_coord(left, "y") - right.y,
            DefendCommanderModelSimulator._get_coord(left, "z") - right.z,
        )

    @staticmethod
    def _vector_length(value) -> float:
        return math.sqrt(value.x ** 2 + value.y ** 2 + value.z ** 2)

    @staticmethod
    def _vector_angle(left, right) -> float:
        left_length = DefendCommanderModelSimulator._vector_length(left)
        right_length = DefendCommanderModelSimulator._vector_length(right)
        if left_length == 0 or right_length == 0:
            return 0.0
        cosine = (left.x * right.x + left.y * right.y + left.z * right.z) / (left_length * right_length)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _warn_missing_controller(self) -> None:
        if self._missing_controller_warned:
            return
        logging.warning(
            "DefendCommander %s has no blue defense controller; decision disabled",
            self.entity_ext.entity.id,
        )
        self._missing_controller_warned = True
