"""示例内部按当前通信成员建立邮箱，并在观测接收完成后进行群内规划。

AgentManager 先向所有实例调用 get_obs，再收集 get_action。示例利用这个顺序，
在 get_obs 阶段提交本机报告，在首个 get_action 阶段整理同帧群内报告。
"""

from copy import deepcopy

from .commander import RedBaselineCommander


def _planning_report(observation: dict) -> dict:
    """保存规划需要的公开值，隔离调用方修改，省去整群通信表的重复复制。"""
    own = observation["self"]
    tracks = {}
    for identifier, track in (own.get("detectInfo") or {}).items():
        # saved = {name: track[name] for name in ("entity_id", "entity_type", "detected_step")
        #          if name in track}
        # if track.get("pos") is not None:
        #     saved["pos"] = dict(track["pos"])

        tracks[identifier] = {
            "entity_id": track.entity_id,
            "entity_type": track.entity_type,
            "detected_step": track.step,
            "pos": track.pos,
        }
    state = {name: own[name] for name in ("type", "health", "isVisible") if name in own}
    state.update(pos=dict(own.get("pos", {})), detectInfo=tracks)
    return {"entity_id": observation["entity_id"], "step": observation["step"],
            "is_using_satellite": observation.get("is_using_satellite", False), "self": state}


class _LocalMailbox:
    """绑定单个实体编号的邮箱端点，将本机报告路由到当前通信群。"""

    def __init__(self, entity_id, receive, reset):
        """保存实体身份以及目录提供的接收、重置回调。"""
        self._entity_id = entity_id
        self._receive = receive
        self._reset = reset

    def receive(self, observation: dict):
        """向目录提交观测，返回该观测对应的群邮箱。"""
        return self._receive(self._entity_id, observation)

    def reset(self) -> None:
        """清理含本机成员的群邮箱状态。"""
        self._reset(self._entity_id)


class _GroupMailbox:
    """保存一个固定成员集合的报告、规划器和本帧准备状态。"""
    def __init__(self, members, registered_entities, targets, config, search_polygon):
        """按成员集合创建规划器，并登记其中已构造的红方实例。"""
        self.members = frozenset(members)
        # 两个空间 R7 入口也使用 r7_strike_packages 配置；其他策略保持原复制逻辑。
        self._compact_reports = config.policy_name in (
            "r0_random", "r7_strike_packages", "r7_static_search",
        )
        self.commander = RedBaselineCommander(
            targets, policy_name=config.policy_name, seed=config.seed,
            search_polygon=search_polygon,
        )
        for entity_id in sorted(self.members & registered_entities):
            self.commander.register_platform(entity_id)
        self._reports: dict[int, dict] = {}
        self._prepared_step: int | None = None

    def receive(self, observation: dict) -> None:
        """复制成员最新报告，避免后续对调用方观测的改动影响已收信息。"""
        entity_id = int(observation["entity_id"])
        if entity_id not in self.members:
            raise ValueError("A report cannot enter another communication group's mailbox")
        self._reports[entity_id] = (
            _planning_report(observation) if self._compact_reports else deepcopy(observation)
        )

    def prepare(self, step: int) -> RedBaselineCommander:
        """每帧只汇总一次同帧报告，后续成员复用该帧规划上下文。"""
        # 首个成员请求动作时，各成员的 get_obs 已完成；只采纳 step 一致的最新报告。
        if self._prepared_step != step:
            reports = tuple(
                self._reports[entity_id]
                for entity_id in sorted(self._reports)
                if int(self._reports[entity_id]["step"]) == step
            )
            self.commander.begin_step(reports if self._compact_reports else deepcopy(reports))
            # 规划器已提取本帧任务所需状态，释放包含整群通信快照的原始报告。
            self._reports.clear()
            self._prepared_step = step
        return self.commander


class _CommunicationDirectory:
    """按观测中的 commRangeInfo 将本机报告分到各通信群。

    成员集合完全相同的群复用上一帧邮箱；集合发生变化时从公开初值创建新规划器。
    """

    def __init__(self, registered_entities, targets, config, search_polygon=None):
        """保存公开初值、配置和成员身份；分别维护当前帧与上一帧的群索引。"""
        self._registered_entities = frozenset(registered_entities)
        self._targets = targets
        self._config = config
        self._search_polygon = search_polygon
        self._step: int | None = None
        self._current: dict[tuple[int, ...], _GroupMailbox] = {}
        self._previous: dict[tuple[int, ...], _GroupMailbox] = {}

    def endpoint(self, entity_id: int) -> _LocalMailbox:
        """为已登记的实体创建绑定身份的本机端点。"""
        if entity_id not in self._registered_entities:
            raise ValueError("A mailbox endpoint must belong to a registered platform")
        return _LocalMailbox(entity_id, self.receive, self.reset_member)

    def register_member(self, entity_id: int) -> None:
        """示例实例构造时登记本机身份。"""
        self._registered_entities = self._registered_entities | {entity_id}

    def unregister_member(self, entity_id: int) -> None:
        """实例退出时撤销登记，并清理含本机的邮箱。"""
        self.reset_member(entity_id)
        self._registered_entities = self._registered_entities - {entity_id}

    def receive(self, entity_id: int, observation: dict) -> _GroupMailbox:
        """从本机通信成员构造稳定键，取得本帧群邮箱并提交报告。"""
        if int(observation["entity_id"]) != entity_id:
            raise ValueError("An agent can only report its own observation")
        step = int(observation["step"])
        # 进入新帧时保留上一帧索引用于复用，当前帧只登记本帧实际出现的群。
        if step != self._step:
            self._previous = self._current
            self._current = {}
            self._step = step
        # 本机编号始终属于自己的群；排序后的成员元组同时作为群身份与复用键。
        peers = observation.get("self", {}).get("commRangeInfo") or ()
        key = tuple(sorted({entity_id, *(int(peer) for peer in peers)}))
        mailbox = self._current.get(key)
        if mailbox is None:
            # 完全相同的成员集合保留分配历史；合并或分裂后创建独立的新规划状态。
            mailbox = self._previous.get(key)
            if mailbox is None:
                mailbox = _GroupMailbox(
                    key, self._registered_entities, self._targets,
                    self._config, self._search_polygon,
                )
            self._current[key] = mailbox
        mailbox.receive(observation)
        return mailbox

    def reset_member(self, entity_id: int) -> None:
        """移除包含指定实体的当前、上一帧邮箱，重置其规划状态。"""
        # 同一邮箱可能同时出现在两帧索引中，用对象身份去重，避免重复重置。
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

    def reset(self) -> None:
        """在回合边界清空邮箱索引，并对共享引用的规划器去重重置。"""
        discarded = set()
        for groups in (self._current, self._previous):
            for mailbox in groups.values():
                if id(mailbox) not in discarded:
                    mailbox.commander.reset()
                    discarded.add(id(mailbox))
            groups.clear()
        self._step = None
