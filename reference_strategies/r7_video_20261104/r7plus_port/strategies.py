from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .legacy.baselines import distance_km
from .legacy.spatial_commander import SpatialCommander
from .legacy.spatial_waves import SpatialWaveDeployment
from .legacy.task_memory import PlatformTask


def coverage_deployment(original, observation, variant):
    if variant not in ('coverage', 'recovery'):
        raise ValueError(f'未知迭代方案：{variant}')
    assignments = deepcopy(original.assignments)
    positions = {int(row[1]): list(row) for row in original.actions}
    low_ids = sorted(key for key, item in observation['entities'].items() if item['type'] == 21002)
    old_cover = sorted(key for key in low_ids if assignments[key]['role'] == 'cover')
    search_count = math.ceil(len(low_ids) / 2)
    selected = np.linspace(0, len(old_cover) - 1, len(low_ids) - search_count, dtype=int)
    cover_ids = {old_cover[index] for index in selected}
    search_ids = sorted(set(low_ids) - cover_ids)
    polygon = observation['coordinates'][0]
    south, north = min(point[1] for point in polygon), max(point[1] for point in polygon)
    front, east = max(point[0] for point in polygon), observation['map_area']['lonMax']
    pairs = search_count // 2
    for index, key in enumerate(search_ids):
        lane = index // 2 if index < pairs * 2 else pairs // 2
        latitude = south + (lane + .5) / pairs * (north - south)
        longitude = front - .001 - (index % 2) * .003 if index < pairs * 2 else front - .007
        positions[key] = [0, key, longitude, latitude]
        assignments[key].update(role='search', target_id=None,
            destination={'lon': east - .001, 'lat': latitude}, package_id=None)
    report = deepcopy(original.report)
    report.update(variant=variant, role_counts=dict(Counter(item['role'] for item in assignments.values())),
                  search_stripe_spacing_km=(north - south) * 111 / pairs)
    report['escort_alignment'] = [row for row in report['escort_alignment'] if row['cover_id'] in cover_ids]
    return SpatialWaveDeployment([positions[key] for key in sorted(positions)], assignments, report)


class CoverageCommander(SpatialCommander):
    def _allocate(self, capacity):
        free = sorted(key for key, unit in self.reports.items()
                      if unit.alive and self.deployment.assignments[key]['role'] == 'search'
                      and self.target_by_platform.get(key) is None)
        claims = Counter(self._claims.values())
        targets = {target.entity_id: target for target in self.targets if target.entity_type == 9500}
        slots = [key for key in sorted(targets) for _ in range(max(0, capacity - claims[key]))]
        if not free or not slots:
            return
        costs = np.full((len(free), len(slots) + len(free)), 1_000_000.0)
        for row, key in enumerate(free):
            unit = self.reports[key]
            known = self.observed_target_ids.get(key, ())
            for column, target_id in enumerate(slots):
                target = targets[target_id]
                distance = distance_km(unit.position, target.position)
                forward = (target.position.lon - unit.position.lon) * 111 * math.cos(math.radians(unit.position.lat))
                if target_id not in known or forward < -.1 or self.current_step + distance / .1 > 1800:
                    costs[row, column] = 2_000_000
                else:
                    costs[row, column] = distance * 1000
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if column < len(slots) and costs[row, column] < 1_000_000:
                key, target_id = free[row], slots[column]
                self._claims[key] = target_id
                self.target_by_platform[key] = target_id

    def begin_step(self, observations):
        super().begin_step(observations)
        self._allocate(2)

    def action_for(self, entity_id, step):
        entry = self.deployment.assignments[entity_id]
        if entry['role'] == 'search' and self.target_by_platform.get(entity_id) is None:
            unit = self.reports.get(entity_id)
            if unit is None or not unit.alive:
                return None
            destination = entry['destination']
            signature = (None, destination['lon'], destination['lat'])
            if self.last_guidance.get(entity_id) == signature:
                return None
            self.last_guidance[entity_id] = signature
            return [2.0, float(entity_id), destination['lon'], destination['lat']]
        return super().action_for(entity_id, step)


@dataclass(frozen=True)
class RecoveryTask(PlatformTask):
    claim_leases: tuple[tuple[int, int, float], ...] = ()


class RecoveryCommander(CoverageCommander):
    def __init__(self, *args):
        super().__init__(*args)
        self.claim_leases = {}

    def restore_tasks(self, tasks):
        super().restore_tasks(tasks)
        for task in tasks:
            for key, target, expires in task.claim_leases:
                previous = self.claim_leases.get(key)
                if previous is None or expires > previous[1]:
                    self.claim_leases[key] = (target, expires)

    def task_for(self, entity_id):
        task = super().task_for(entity_id)
        if task is None:
            return None
        values = {field.name: getattr(task, field.name) for field in fields(PlatformTask)}
        return RecoveryTask(**values, claim_leases=tuple(
            (key, target, expires) for key, (target, expires) in sorted(self.claim_leases.items())))

    def begin_step(self, observations):
        SpatialCommander.begin_step(self, observations)
        targets = {target.entity_id: target for target in self.targets}
        for key, (target_id, expires) in tuple(self.claim_leases.items()):
            if self.current_step > expires:
                self.claim_leases.pop(key)
                self._claims.pop(key, None)
        # 只给本群仍有当前报告的搜索单位续期；失联不被认定为死亡或命中。
        for key, unit in self.reports.items():
            target_id = self.target_by_platform.get(key)
            if unit.alive and self.deployment.assignments[key]['role'] == 'search' and target_id in targets:
                arrival = self.current_step + distance_km(unit.position, targets[target_id].position) / .1
                self.claim_leases[key] = (target_id, arrival + 45)
                self._claims[key] = target_id
        self._claims = {key: target for key, target in self._claims.items() if key in self.claim_leases}
        self._allocate(2)
        # 全部可达暗巢已有两名执行者时，再安排一名冗余，避免一直空闲。
        self._allocate(3)
        for key, target_id in self._claims.items():
            if key not in self.claim_leases and key in self.reports:
                arrival = self.current_step + distance_km(self.reports[key].position, targets[target_id].position) / .1
                self.claim_leases[key] = (target_id, arrival + 45)
