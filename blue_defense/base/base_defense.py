"""蓝方策略的公共基类与运行生命周期。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from .types import BlueClock, BlueDecision, BlueObservation, ThreatTrack


class BaseBlueDefense(ABC):
    """接收蓝方观测，输出自爆虫调度决策，并管理本局策略状态。"""

    # 环境通过这两个标志选择计划释放流程，以及合并或替换待释放计划。
    name = "base"
    plans_launch_time = False
    preserves_pending_launches = False

    def __init__(
        self,
        *,
        seed: int | None = None,
        asset_values: Mapping[int, float] | None = None,
    ):
        # 每个运行实例保存自己的种子与价值映射副本。
        self.seed = seed
        self.asset_values = dict(asset_values) if asset_values is not None else {}
        # 默认辅助实现只保存最新原始探测；策略可覆盖接收和查询方法。
        self._received_tracks: dict[int, ThreatTrack] = {}
        self._clock: BlueClock | None = None

    def bind_clock(self, clock: BlueClock) -> None:
        """由环境注册及重置时提供本局步长，既有策略无需覆写。"""
        if not isinstance(clock, BlueClock):
            raise TypeError("blue clock must be a BlueClock")
        self._clock = clock

    def receive_tracks(self, tracks: Iterable[ThreatTrack], sim_step: int) -> None:
        """接收即时探测更新，不受 decide 的调用间隔限制。"""
        for track in tracks:
            previous = self._received_tracks.get(track.target_id)
            if previous is None or (track.detect_step, -track.source_id) > (
                previous.detect_step, -previous.source_id
            ):
                self._received_tracks[track.target_id] = track

    def get_tracks(self, sim_step: int) -> tuple[ThreatTrack, ...]:
        """供执行端查询蓝方目标状态；默认不设有效期、不作位置外推。"""
        return tuple(self._received_tracks[key] for key in sorted(self._received_tracks))

    def get_track(self, target_id: int, sim_step: int) -> ThreatTrack | None:
        """按编号查询；自定义记忆和预测可统一覆盖 get_tracks。"""
        return next((track for track in self.get_tracks(sim_step)
                     if track.target_id == target_id), None)

    @abstractmethod
    def decide(self, observation: BlueObservation) -> BlueDecision | None:
        """评估当前观测；等待下一决策时刻时返回 None。"""

    def reset(self) -> None:
        """在环境重置时清理本局策略状态。"""
        self._received_tracks.clear()

    def close(self) -> None:
        """由运行编排层调用，释放策略自身持有的资源。"""

    def asset_value(self, entity_id: int, entity_type: int) -> float:
        """优先读取实体的配置价值，其余使用对应类型的默认价值。"""
        # 场景可以为具体实体指定价值；默认值按母巢、哨卫巢、暗巢区分。
        if entity_id in self.asset_values:
            return float(self.asset_values[entity_id])
        return {
            9400: 5.0,
            9600: 2.0,
            9500: 1.0,
        }.get(entity_type, 1.0)
