from math import ceil

from ..common.baselines import distance_km
from ..common.commander import RedBaselineCommander
from ..common.contracts import Position
from .strike_coordination import StrikeCoordination
from .task_memory import PlatformTask


class SpatialCommander(RedBaselineCommander):
    def __init__(self, targets, seed, search_polygon, deployment):
        super().__init__(targets, "r7_strike_packages", seed, search_polygon)
        self.deployment = deployment
        self.coordination = StrikeCoordination(
            dict(deployment.report, assignments=deployment.assignments), targets,
        )
        self._losses = {}
        self._redirects = {}
        self._claims = {}
        self._strike_report_snapshot = ()
        self._satellite_claimed = False
        # 各通信组使用同一公开部署清单，错开请求步数，避免同一步重复扣除全局额度。
        first_request = max(1, ceil(self.coordination.satellite_start))
        self._satellite_request_steps = {
            key: first_request + offset for offset, key in enumerate(sorted(deployment.assignments))
        }
        self.current_step = 0

    def restore_tasks(self, tasks: tuple[PlatformTask, ...]) -> None:
        for task in tasks:
            if task.entity_id not in self.expected_platform_ids:
                raise ValueError('任务不能进入其他通信群')
            if task.target_id is not None:
                self.target_by_platform[task.entity_id] = task.target_id
            if task.guidance is not None:
                self.last_guidance[task.entity_id] = task.guidance
        for task in tasks:
            self._losses.update(task.confirmed_losses)
            self.coordination.restore_reports(task.strike_reports)
            self._claims.update(task.search_claims)
            for key, old, new in task.redirects:
                self._redirects[key] = (old, new)
        for key, target_id in self._losses.items():
            self.coordination.remaining[target_id] -= self.coordination.damage(
                self.deployment.assignments[key]['kind'], target_id,
            )
        for key, (old, new) in self._redirects.items():
            kind = self.deployment.assignments[key]['kind']
            self.coordination.remaining[old] -= self.coordination.damage(kind, old)
            self.coordination.remaining[new] += self.coordination.damage(kind, new)
        self.coordination.accounted_losses.update(self._losses)
        self.coordination.redirected.update(self._redirects)

    def task_for(self, entity_id: int) -> PlatformTask | None:
        if entity_id not in self.reports:
            return None
        entry = self.deployment.assignments[entity_id]
        return PlatformTask(
            entity_id, self.target_by_platform.get(entity_id, entry['target_id']),
            self.last_guidance.get(entity_id), self.current_step,
            Position(**entry['destination']) if entry['role'] != 'strike' else None,
            entity_id in self._claims,
            tuple(sorted(self._losses.items())),
            tuple((key, old, new) for key, (old, new) in sorted(self._redirects.items())),
            tuple(sorted(self._claims.items())),
            self._strike_report_snapshot,
        )

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        super().begin_step(observations)
        if not observations:
            return
        self.current_step = int(observations[0]['step'])
        self.last_plan_step = self.current_step
        if any(item.get('is_using_satellite', False) for item in observations):
            self._satellite_claimed = True
        for key, report in self.reports.items():
            entry = self.deployment.assignments[key]
            self.target_by_platform.setdefault(key, entry['target_id'])
            if (not report.alive and report.kind in ('H', 'M')
                    and key not in self.coordination.accounted_losses):
                target_id = self.target_by_platform[key]
                aim = self.last_guidance.get(key)
                position = Position(aim[1], aim[2]) if aim else Position(**entry['destination'])
                if distance_km(report.position, position) > 2:
                    self._losses[key] = target_id
        aims = {key: Position(value[1], value[2]) for key, value in self.last_guidance.items()}
        local_tracks = {
            int(item['entity_id']): {
                int(track['entity_id']): (int(track['detected_step']), Position(
                    track['pos'].x, track['pos'].y, track['pos'].z))
                for track in item['self']['detectInfo'].values()
                if int(track['entity_type']) in (9400, 9600)
            } for item in observations
        }
        old_targets = dict(self.target_by_platform)
        velocities = {int(item['entity_id']): (item['self']['vel']['x'], item['self']['vel']['y'])
                      for item in observations}
        stages = {int(item['entity_id']): item['self']['stage'] for item in observations}
        redirects = self.coordination.repair_losses(
            self.reports, self.target_by_platform, aims, local_tracks, self.current_step, velocities, stages,
        )
        for key, (target_id, _) in redirects.items():
            self._redirects[key] = (old_targets[key], target_id)
            self.target_by_platform[key] = target_id
        # 同一通信群的不可变报告只生成一次，供各成员保存自己的任务记忆。
        self._strike_report_snapshot = tuple((key, target, seen, distance)
            for key, (target, seen, distance) in sorted(self.coordination.strike_reports.items()))

    def action_for(self, entity_id: int, step: int) -> list[float] | None:
        platform = self.reports.get(entity_id)
        if platform is None or not platform.alive:
            return None
        entry = self.deployment.assignments[entity_id]
        target_id = self.target_by_platform.get(entity_id)
        if entry['role'] == 'search' and target_id is None:
            candidates = sorted(
                (target for target in self.targets if target.entity_type == 9500
                 and target.entity_id in self.observed_target_ids.get(entity_id, ())),
                key=lambda target: (distance_km(platform.position, target.position), target.entity_id),
            )
            for capacity in (2, 3):
                selected = next((target for target in candidates
                                 if sum(value == target.entity_id for value in self._claims.values()) < capacity), None)
                if selected is not None:
                    target_id = selected.entity_id
                    self._claims[entity_id] = target_id
                    self.target_by_platform[entity_id] = target_id
                    break
        if target_id is None:
            destination = Position(**entry['destination'])
            signature = (None, destination.x, destination.y)
            action = [1.0, float(entity_id), destination.x, destination.y]
        else:
            if (target_id not in {item.entity_id for item in self.initial_targets}
                    and target_id not in self.observed_target_ids.get(entity_id, ())):
                return None
            target = next(item for item in self.targets if item.entity_id == target_id)
            if target.entity_type == 9500 and target_id not in self.observed_target_ids.get(entity_id, ()):
                return None
            signature = (target_id, target.position.x, target.position.y)
            previous = self.last_guidance.get(entity_id)
            if previous is not None and previous[0] == target_id:
                if target_id not in self.observed_target_ids.get(entity_id, ()):
                    return None
                if distance_km(Position(previous[1], previous[2]), target.position) < 0.010:
                    return None
            action = [1.0, float(entity_id), target.position.x, target.position.y]
        if self.last_guidance.get(entity_id) == signature:
            return None
        self.last_guidance[entity_id] = signature
        return action

    def should_use_satellite(self, entity_id: int) -> bool:
        if self._satellite_claimed or self.current_step != self._satellite_request_steps[entity_id]:
            return False
        self._satellite_claimed = True
        return True
