from copy import deepcopy

import numpy as np

from user_agents.base import BaseAgent, BaseDeployment

from .coordination import TaskDirectory
from .legacy.config import RedAgentConfig
from .legacy.contracts import Position
from .legacy.priors import initial_targets_from_observation
from .legacy.short_evasion import ShortThreatEvasion
from .legacy.spatial_waves import build_spatial_wave_deployment
from .strategies import coverage_deployment


_SESSIONS = {}


def _run_id(observation):
    value = observation['run_id']
    if not isinstance(value, str) or not value:
        raise ValueError('原R7+部署与作战需要本局非空run_id')
    return value


class _Session:
    def __init__(self, observation, deployment, history_length):
        self.run_id = _run_id(observation)
        self.expected_ids = frozenset(observation['entities'])
        self.public_targets = deepcopy(observation['public_targets'])
        self.map_area = dict(observation['map_area'])
        self.seed = observation['seed']
        if type(self.seed) is not int:
            raise ValueError('R7+对照必须显式提供整数种子')
        self.deployment = deployment
        self.history_length = history_length
        self.members = set()
        config = RedAgentConfig(
            policy_name='r7_spatial_packages', motion_policy='short_evasion',
            seed=self.seed, max_steps=3000,
        )
        area = self.map_area
        polygon = tuple(Position(lon, lat) for lon, lat in (
            (area['lonMin'], area['latMin']), (area['lonMax'], area['latMin']),
            (area['lonMax'], area['latMax']), (area['lonMin'], area['latMax']),
        ))
        self.directory = TaskDirectory(
            self.expected_ids,
            initial_targets_from_observation({'entities': self.public_targets}),
            config, polygon, deployment,
        )

    def register(self, entity_id, observation):
        if entity_id not in self.expected_ids:
            raise ValueError(f'实体{entity_id}不属于本局部署清单')
        if entity_id in self.members:
            raise ValueError(f'实体{entity_id}重复注册')
        if (observation['seed'] != self.seed or observation['map_area'] != self.map_area
                or observation['entities'] != self.public_targets):
            raise ValueError('部署与作战的公开初态或种子不一致')
        if observation['max_steps'] != 3000:
            raise ValueError('本次原R7+对照使用固定3000步配置')
        self.members.add(entity_id)


class Deployment(BaseDeployment):
    def __init__(self, map_area, variant, history_length=0):
        self.deployment = None
        self.run_id = None
        self.map_area = dict(map_area)
        self.variant = variant
        self.history_length = history_length

    def deploy(self, observation):
        # 地图边界属于赛题公开静态资料，当前部署接口没有重复提供这一项。
        observation = dict(observation, map_area=self.map_area)
        run_id = _run_id(observation)
        if run_id in _SESSIONS:
            raise ValueError(f'本局原R7+已经部署：{run_id}')
        public_targets = observation['public_targets']
        if any(entity['type'] not in (9400, 9600) for entity in public_targets.values()):
            raise ValueError('部署公开目标目录不能包含暗巢或其他目标')
        deployment = build_spatial_wave_deployment(
            observation['entities'], public_targets,
            observation['coordinates'], observation['map_area'],
        )
        if self.variant != 'baseline':
            deployment = coverage_deployment(deployment, observation, self.variant)
        session = _Session(observation, deployment, self.history_length)
        self.deployment = deployment
        self.run_id = run_id
        _SESSIONS[run_id] = session
        return {int(row[1]): (float(row[2]), float(row[3]))
                for row in deployment.actions if int(row[0]) == 0}


class RedAttack(BaseAgent):
    policy_name = 'r7_spatial_packages'
    motion_policy = 'short_evasion'

    def __init__(self, agent_id, entity_id, init_observation):
        run_id = _run_id(init_observation)
        if run_id not in _SESSIONS:
            raise RuntimeError('本局必须先通过原R7+部署入口生成整队计划')
        session = _SESSIONS[run_id]
        super().__init__(agent_id, entity_id, init_observation, history_length=session.history_length)
        session.register(entity_id, init_observation)
        self._resource = session
        self._communications = session.directory.endpoint(entity_id)
        self._closed = False
        self._reset_state()

    @property
    def deployment(self):
        return self._resource.deployment

    def _reset_state(self):
        self._commander = None
        self._mailbox = None
        self.launch_step = -1
        self.sat_used = False
        self.short_evasion = ShortThreatEvasion()

    def get_obs(self, observation):
        super().get_obs(observation)
        self._mailbox = self._communications.receive(observation)

    def get_action(self):
        observation = self.latest_observation
        if observation is None or self._mailbox is None:
            raise RuntimeError('作战策略必须先收到真实隔离观测')
        self._commander = self._mailbox.prepare(int(observation['step']))
        actions = []
        step = observation['step']
        if self.launch_step < 0:
            self.launch_step = step
        self._commander.report(observation)
        command = self._commander.action_for(self.entity_id, step)
        if command is not None:
            actions.append(command)
        if observation.get('is_using_satellite', False):
            self.sat_used = True
        if (self.launch_step >= 0 and not self.sat_used
                and self._commander.should_use_satellite(self.entity_id)):
            actions.append([3, self.entity_id, 0, 0])
            self.sat_used = True
        acceleration = self.short_evasion.command(observation)
        if acceleration is not None:
            actions.append([5, self.entity_id, acceleration, 0])
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
            close_run(session.run_id)


def close_run(run_id):
    session = _SESSIONS.pop(run_id, None)
    if session is not None:
        session.directory.reset()
