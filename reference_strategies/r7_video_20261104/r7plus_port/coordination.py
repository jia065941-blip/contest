from dataclasses import replace

from user_agents.example.common.coordination import (
    _CommunicationDirectory as RemoteDirectory,
    _GroupMailbox as RemoteGroupMailbox,
    _LocalMailbox as RemoteLocalMailbox,
)

from .legacy.spatial_commander import SpatialCommander
from .strategies import CoverageCommander, RecoveryCommander


class TaskGroupMailbox(RemoteGroupMailbox):
    def __init__(self, members, registered_entities, targets, config, search_polygon, deployment):
        super().__init__(
            members, registered_entities, targets,
            replace(config, policy_name='r7_strike_packages'), search_polygon,
        )
        variant = deployment.report['variant']
        commander_type = {'r7plus_coordinated_v2': SpatialCommander,
                          'coverage': CoverageCommander, 'recovery': RecoveryCommander}[variant]
        self.commander = commander_type(targets, config.seed, search_polygon, deployment)
        for entity_id in sorted(self.members & registered_entities):
            self.commander.register_platform(entity_id)
        self._inherited = {}
        self._restored = False

    def receive(self, observation, task=None):
        super().receive(observation)
        entity_id = int(observation['entity_id'])
        if not self._restored and task is not None:
            if task.entity_id != entity_id:
                raise ValueError('只能传递本机已经持有的任务')
            self._inherited[entity_id] = task

    def prepare(self, step):
        if not self._restored:
            if self._inherited:
                self.commander.restore_tasks(tuple(self._inherited.values()))
            self._inherited.clear()
            self._restored = True
        # 保留远端prepare的所有权与释放行为，修复前后使用同一继承入口。
        return super().prepare(step)


class TaskLocalMailbox(RemoteLocalMailbox):
    def __init__(self, entity_id, receive, reset):
        super().__init__(entity_id, receive, reset)
        self._group = None

    def receive(self, observation):
        task = self._group.commander.task_for(self._entity_id) if self._group is not None else None
        self._group = self._receive(self._entity_id, observation, task)
        return self._group

    def reset(self):
        super().reset()
        self._group = None


class TaskDirectory(RemoteDirectory):
    def __init__(self, registered_entities, targets, config, search_polygon, deployment):
        super().__init__(registered_entities, targets, config, search_polygon)
        self.deployment = deployment

    def endpoint(self, entity_id):
        if entity_id not in self._registered_entities:
            raise ValueError('邮箱端点必须对应本局受控单位')
        return TaskLocalMailbox(entity_id, self.receive, self.reset_member)

    def receive(self, entity_id, observation, task=None):
        if int(observation['entity_id']) != entity_id:
            raise ValueError('只能上报本机观测')
        step = int(observation['step'])
        if step != self._step:
            self._previous = self._current
            self._current = {}
            self._step = step
        peers = observation.get('self', {}).get('commRangeInfo') or ()
        key = tuple(sorted({entity_id, *(int(peer) for peer in peers)}))
        mailbox = self._current.get(key)
        if mailbox is None:
            mailbox = self._previous.get(key)
            if mailbox is None:
                mailbox = TaskGroupMailbox(
                    key, self._registered_entities, self._targets,
                    self._config, self._search_polygon, self.deployment,
                )
            self._current[key] = mailbox
        mailbox.receive(observation, task)
        return mailbox
