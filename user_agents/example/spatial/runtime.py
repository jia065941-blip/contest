from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from weakref import WeakValueDictionary

import numpy as np

from user_agents.base import BaseAgent, BaseDeployment
from ..common.config import RedAgentConfig
from ..common.contracts import Position
from ..common.targets import initial_targets_from_observation
from .coordination import TaskDirectory
from .geometry import public_geometry
from .short_evasion import ShortThreatEvasion
from .spatial_waves import SpatialWaveDeployment, build_spatial_wave_deployment
from .strategies import coverage_deployment


@dataclass(frozen=True)
class _Plan:
    seed: int
    types: dict[int, int]
    public_targets: dict
    map_area: dict
    deployment_bounds: tuple[float, float, float, float]
    deployment: SpatialWaveDeployment


# 部署观测没有运行编号。每个入口保留最近一次公开计划，作战实例再按运行编号隔离状态。
_PLANS: dict[str, _Plan] = {}
_SESSIONS = WeakValueDictionary()


class _Session:
    def __init__(self, run_id, variant, plan):
        self.run_id, self.variant, self.plan = run_id, variant, plan
        self.members = set()
        area = plan.map_area
        polygon = tuple(Position(x, y) for x, y in (
            (area['xMin'], area['yMin']), (area['xMax'], area['yMin']),
            (area['xMax'], area['yMax']), (area['xMin'], area['yMax'])))
        self.directory = TaskDirectory(
            frozenset(plan.types), initial_targets_from_observation({'entities': plan.public_targets}),
            RedAgentConfig(policy_name='r7_strike_packages', motion_policy='straight', seed=plan.seed),
            polygon, plan.deployment)

    def register(self, entity_id, observation):
        if entity_id not in self.plan.types or entity_id in self.members:
            raise ValueError('受控单位不在部署清单中或重复注册')
        if (observation['seed'] != self.plan.seed or observation['map_area'] != self.plan.map_area
                or observation['entities'] != self.plan.public_targets
                or observation['self']['type'] != self.plan.types[entity_id]):
            raise ValueError('策略初态与本次公开部署计划不一致')
        position = observation['self']['pos']
        if any(type(position[axis]) not in (int, float) or not isfinite(position[axis])
               for axis in ('x', 'y', 'z')):
            raise ValueError('本机初始位置必须是有限数值')
        west, east, south, north = self.plan.deployment_bounds
        if not west <= position['x'] <= east or not south <= position['y'] <= north:
            raise ValueError('引擎回传的本机初始位置超出合法部署区域')
        # 提交位置与引擎初始化结果可能不同；作战使用本机合法观测的实际位置。
        self.members.add(entity_id)


class SpatialDeployment(BaseDeployment):
    variant = 'baseline'

    def deploy(self, observation):
        seed = observation['seed']
        if type(seed) is not int:
            raise ValueError('空间波次策略需要整数种子')
        map_area, step_duration_ms = public_geometry(observation['coordinates'])
        deployment = build_spatial_wave_deployment(
            observation['entities'], observation['public_targets'], observation['coordinates'],
            map_area, step_duration_ms)
        if self.variant != 'baseline':
            deployment = coverage_deployment(deployment, dict(observation, map_area=map_area), self.variant)
        self.deployment = deployment
        coordinates = observation['coordinates']
        deployment_bounds = (min(point[0] for point in coordinates), max(point[0] for point in coordinates),
                             min(point[1] for point in coordinates), max(point[1] for point in coordinates))
        _PLANS[self.variant] = _Plan(seed, {key: item['type'] for key, item in observation['entities'].items()},
                                    deepcopy(observation['public_targets']), map_area, deployment_bounds, deployment)
        return {int(row[1]): (float(row[2]), float(row[3])) for row in deployment.actions}


class SpatialAttack(BaseAgent):
    variant = 'baseline'

    def __init__(self, entity_id, init_observation):
        super().__init__(entity_id, init_observation, history_length=0)
        run_id = init_observation['run_id']
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('作战策略需要非空运行编号')
        key = (run_id, self.variant)
        session = _SESSIONS.get(key)
        if session is None:
            if self.variant not in _PLANS:
                raise RuntimeError('请先使用同一文件的 Deployment 完成空间波次部署')
            session = _Session(run_id, self.variant, _PLANS[self.variant])
        session.register(entity_id, init_observation)
        _SESSIONS[key] = session
        self._resource = session
        self._communications = session.directory.endpoint(entity_id)
        self._closed = False
        self._reset_state()

    @property
    def deployment(self):
        return self._resource.plan.deployment

    def _reset_state(self):
        self._commander = self._mailbox = None
        self.launch_step = -1
        self.sat_used = False
        self._satellite_inactive_seen = False
        self.short_evasion = ShortThreatEvasion()

    def get_obs(self, observation):
        # 引擎启动时会连续返回开启误报；先看到关闭状态，再接受后续的开启状态。
        active = observation.get('is_using_satellite', False)
        if not active:
            self._satellite_inactive_seen = True
        elif not self._satellite_inactive_seen:
            observation = dict(observation, is_using_satellite=False)
        super().get_obs(observation)
        self._mailbox = self._communications.receive(observation)

    def get_action(self):
        observation = self.latest_observation
        if observation is None or self._mailbox is None:
            raise RuntimeError('策略必须先收到本机观测')
        step = observation['step']
        self._commander = self._mailbox.prepare(step)
        if self.launch_step < 0:
            self.launch_step = step
        self._commander.report(observation)
        command = self._commander.waypoint_action_for(self.entity_id, step, self.init_observation['map_area'])
        actions = [] if command is None else [command]
        if observation.get('is_using_satellite', False):
            self.sat_used = True
        if not self.sat_used and self._commander.should_use_satellite(self.entity_id):
            actions.append([2, self.entity_id, 0, 0])
            self.sat_used = True
        acceleration = self.short_evasion.command(observation)
        if acceleration is not None:
            actions.append([3, self.entity_id, acceleration, 0])
        return np.asarray(actions, dtype=np.float64).reshape(-1, 4)

    def on_reset(self):
        self._communications.reset()
        self._reset_state()

    def close(self):
        if self._closed:
            return
        self._closed = True
        session = self._resource
        session.members.remove(self.entity_id)
        self._communications.reset()
        self.latest_observation = None
        self._mailbox = self._commander = None
        if not session.members:
            session.directory.reset()
            _SESSIONS.pop((session.run_id, session.variant), None)
            if _PLANS.get(session.variant) is session.plan:
                _PLANS.pop(session.variant)
