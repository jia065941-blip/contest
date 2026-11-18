"""汇总蓝方探测记录，生成带时效和位置预测的观测快照。"""

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from ..base import BlueClock, BlueObservation, DefendedAsset, InterceptorState, OwnEntityState, ThreatTrack, distance_m
from envengine.common.BaseStruct import Vector3d


@dataclass(frozen=True)
class PolicyObservation(BlueObservation):
    """内置 B 算法的整理结果，保护对象与记忆规则保留在实现内部。"""

    assets: Sequence[DefendedAsset] = ()


class BlueInformationStore:
    """内置蓝方自己持有的目标记忆与算法快照。"""

    # 信息时效按最后探测时刻计算，单位为时间步
    TRACK_LIFETIME_STEPS = 600

    def __init__(self, clock: BlueClock | None = None):
        # 目标按编号合并；己方资源保存为本帧状态，释放清单按目标分组。
        self._tracks: dict[int, ThreatTrack] = {}
        self._interceptors: tuple[InterceptorState, ...] = ()
        self._assets: tuple[DefendedAsset, ...] = ()
        self._own_entities: tuple[OwnEntityState, ...] = ()
        self._launched: dict[int, tuple[int, ...]] = {}
        self._clock = clock

    def bind_clock(self, clock: BlueClock) -> None:
        """在首次物理预测前注入场景时钟，不使用猜测的步长。"""
        if not isinstance(clock, BlueClock):
            raise TypeError("blue clock must be a BlueClock")
        self._clock = clock

    def reset(self) -> None:
        """清空上一局的目标信息、己方资源与释放记录。"""
        self._tracks.clear()
        self._interceptors = ()
        self._assets = ()
        self._own_entities = ()
        self._launched.clear()

    def receive_tracks(self, tracks: Iterable[ThreatTrack], sim_step: int) -> None:
        """校验一批探测记录，并按时间与来源编号合并到信息库。"""
        # 先检查整批身份、来源、探测时间以及位置速度，再更新库内状态。
        received = tuple(tracks)
        for track in received:
            if (track.target_id <= 0 or track.source_id <= 0
                    or track.type not in (21000, 21001, 21002)
                    or not math.isfinite(track.detect_step)
                    or track.detect_step < 0 or track.detect_step > sim_step):
                raise ValueError("蓝方探测记录的身份、来源或时间无效")
            if not all(math.isfinite(value) for vector in (track.pos, track.vel)
                       for value in (vector.x, vector.y, vector.z)):
                raise ValueError("蓝方探测记录包含无效坐标或速度")
        # 仅保留时效内记录；同目标优先采用新探测，相同时刻采用较小来源编号。
        self._expire(sim_step)
        for track in received:
            if sim_step - track.detect_step > self.TRACK_LIFETIME_STEPS:
                continue
            previous = self._tracks.get(track.target_id)
            if previous is not None and (track.detect_step, -track.source_id) <= (
                    previous.detect_step, -previous.source_id):
                continue
            self._tracks[track.target_id] = track

    def update_own_state(self, interceptors: Iterable[InterceptorState],
                         assets: Iterable[DefendedAsset],
                         launched_map: Mapping[int, Sequence[int]], *,
                         own_entities: Iterable[OwnEntityState] = ()) -> None:
        """更新己方自爆虫、保护对象与已释放清单。"""
        # 拷贝容器，形成信息库本次快照使用的稳定集合。
        self._interceptors = tuple(interceptors)
        self._assets = tuple(assets)
        self._own_entities = tuple(own_entities)
        self._launched = {target_id: tuple(ids) for target_id, ids in launched_map.items()}

    def get_track(self, target_id: int, sim_step: int) -> ThreatTrack | None:
        """读取有效目标记录，并预测到当前仿真时刻。"""
        track = self._tracks.get(target_id)
        if track is None:
            return None
        if sim_step - track.detect_step > self.TRACK_LIFETIME_STEPS:
            del self._tracks[target_id]
            return None
        return self._predict(track, sim_step)

    def get_tracks(self, sim_step: int) -> tuple[ThreatTrack, ...]:
        """查询全部有效目标，并预测到指定步号。"""
        self._expire(sim_step)
        return tuple(self._predict(self._tracks[target_id], sim_step)
                     for target_id in sorted(self._tracks))

    def snapshot(self, sim_step: int, *, clock: BlueClock | None = None) -> PolicyObservation:
        """按目标编号生成观测，并以只读映射提供已释放清单。"""
        if clock is not None:
            self.bind_clock(clock)
        if self._clock is None:
            raise RuntimeError("bind the scenario clock before querying blue tracks")
        return PolicyObservation(
            step=sim_step,
            targets=self.get_tracks(sim_step),
            interceptors=self._interceptors,
            assets=self._assets,
            own_entities=self._own_entities,
            launched_map=MappingProxyType(dict(self._launched)),
            clock=self._clock,
        )

    def nearest_track(self, position: Vector3d, max_range_m: float, sim_step: int) -> ThreatTrack | None:
        """选择指定距离内最近的有效目标，相同距离按编号排序。"""
        # 查询参数始终为步号；物理距离单位仍为米。
        targets = self.get_tracks(sim_step)
        candidates = [(distance_m(position, track.pos), track.target_id, track)
                      for track in targets]
        candidates = [item for item in candidates if item[0] <= max_range_m]
        return min(candidates, key=lambda item: item[:2])[2] if candidates else None

    def _expire(self, sim_step: int) -> None:
        """统一清理超过600步有效期的记录。"""
        self._tracks = {target_id: track for target_id, track in self._tracks.items()
                        if sim_step - track.detect_step <= self.TRACK_LIFETIME_STEPS}

    def _predict(self, track: ThreatTrack, sim_step: int) -> ThreatTrack:
        """按最后探测速度匀速外推 局部坐标中的位置。"""
        if self._clock is None:
            raise RuntimeError("bind the scenario clock before predicting blue tracks")
        elapsed_seconds = self._clock.steps_to_seconds(max(0.0, sim_step - track.detect_step))
        return replace(track, pos=Vector3d(
            track.pos.x + track.vel.x * elapsed_seconds,
            track.pos.y + track.vel.y * elapsed_seconds,
            track.pos.z + track.vel.z * elapsed_seconds,
        ))
