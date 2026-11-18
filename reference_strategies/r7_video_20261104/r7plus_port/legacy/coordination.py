"""Example-owned mailboxes for engine-authorized communication groups.

AgentManager already delivers every local observation before requesting any
action. Agents use that barrier to exchange reports inside their current group;
the environment never prepares a planner or computes an example action.
"""

from copy import deepcopy

from .commander import RedBaselineCommander


class _LocalMailbox:
    """A platform can submit only its own report and use its current group."""

    def __init__(self, entity_id, receive, reset):
        self._entity_id = entity_id
        self._receive = receive
        self._reset = reset
        self._group = None

    def receive(self, observation: dict):
        task = self._group.commander.task_for(self._entity_id) if self._group is not None else None
        self._group = self._receive(self._entity_id, observation, task)
        return self._group

    def reset(self) -> None:
        self._reset(self._entity_id)
        self._group = None


class _GroupMailbox:
    def __init__(self, members, registered_entities, targets, config, search_polygon, deployment):
        self.members = frozenset(members)
        if config.policy_name == "r7_spatial_packages":
            from .spatial_commander import SpatialCommander

            if deployment is None:
                raise RuntimeError("R7+ requires its wave deployment before combat")
            self.commander = SpatialCommander(targets, config.seed, search_polygon, deployment)
        else:
            self.commander = RedBaselineCommander(
                targets, policy_name=config.policy_name, seed=config.seed,
                search_polygon=search_polygon,
            )
        for entity_id in sorted(self.members & registered_entities):
            self.commander.register_platform(entity_id)
        self._reports: dict[int, dict] = {}
        self._prepared_step: int | None = None
        self._inherited = {}
        self._restored = False

    def receive(self, observation: dict, task) -> None:
        entity_id = int(observation["entity_id"])
        if entity_id not in self.members:
            raise ValueError("A report cannot enter another communication group's mailbox")
        self._reports[entity_id] = deepcopy(observation)
        if not self._restored and task is not None:
            if task.entity_id != entity_id:
                raise ValueError("A platform can only carry its own task")
            self._inherited[entity_id] = task

    def prepare(self, step: int) -> RedBaselineCommander:
        if self._prepared_step != step:
            if not self._restored:
                if self._inherited:
                    self.commander.restore_tasks(tuple(self._inherited.values()))
                self._inherited.clear()
                self._restored = True
            reports = tuple(
                self._reports[entity_id]
                for entity_id in sorted(self._reports)
                if int(self._reports[entity_id]["step"]) == step
            )
            self.commander.begin_step(deepcopy(reports))
            self._prepared_step = step
        return self.commander


class _CommunicationDirectory:
    """Route local reports to separate mailboxes within one example instance.

    This directory stores no fleet-wide observation snapshot. A mailbox owns
    only the reports admitted by the engine's communication membership. Group
    changes rebuild a planner from each member's own task and legal reports.
    """

    def __init__(self, registered_entities, targets, config, search_polygon=None):
        self._registered_entities = frozenset(registered_entities)
        self._targets = targets
        self._config = config
        self._search_polygon = search_polygon
        self._step: int | None = None
        self._current: dict[tuple[int, ...], _GroupMailbox] = {}
        self._previous: dict[tuple[int, ...], _GroupMailbox] = {}
        self.deployment = None

    def endpoint(self, entity_id: int) -> _LocalMailbox:
        if entity_id not in self._registered_entities:
            raise ValueError("A mailbox endpoint must belong to a registered platform")
        return _LocalMailbox(entity_id, self.receive, self.reset_member)

    def receive(self, entity_id: int, observation: dict, task=None) -> _GroupMailbox:
        if int(observation["entity_id"]) != entity_id:
            raise ValueError("An agent can only report its own observation")
        step = int(observation["step"])
        if step != self._step:
            self._previous = self._current
            self._current = {}
            self._step = step
        peers = observation.get("self", {}).get("commRangeInfo") or ()
        key = tuple(sorted({entity_id, *(int(peer) for peer in peers)}))
        mailbox = self._current.get(key)
        if mailbox is None:
            mailbox = self._previous.get(key)
            if mailbox is None:
                mailbox = _GroupMailbox(
                    key, self._registered_entities, self._targets,
                    self._config, self._search_polygon,
                    self.deployment,
                )
            self._current[key] = mailbox
        mailbox.receive(observation, task)
        return mailbox

    def reset_member(self, entity_id: int) -> None:
        discarded = set()
        for groups in (self._current, self._previous):
            for key in tuple(groups):
                if entity_id in key:
                    mailbox = groups.pop(key)
                    if id(mailbox) not in discarded:
                        mailbox.commander.reset()
                        discarded.add(id(mailbox))
        if not self._current and not self._previous:
            self._step = None
