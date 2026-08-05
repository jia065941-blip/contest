# -*-coding:utf-8 -*-
import json
import logging
import random
from pathlib import Path

import numpy as np

from envengine import TrainingEnv
from envengine.sdk.log import LogManager
from envengine.simulator.simlulator_impl.InterceptorSimulator import InterceptorSimulator
from envengine.simulator.simlulator_impl.CompCruiseMissileHSimulator import (
    CompCruiseMissileHSimulator,
)
from envengine.simulator.simlulator_impl.CompCruiseMissileLSimulator import (
    CompCruiseMissileLSimulator,
)
from envengine.simulator.simlulator_impl.CompCruiseMissileMSimulator import (
    CompCruiseMissileMSimulator,
)
from main import read_profile
from user_agents import AttackMissileAgent, DeployAgent

RED_SIMULATORS = (
    CompCruiseMissileHSimulator,
    CompCruiseMissileMSimulator,
    CompCruiseMissileLSimulator,
)
RED_ENTITY_TYPES = (21000, 21001, 21002)


def collect_trajectories(
    output_dir: Path,
    scenario: str,
    rounds: int,
    max_steps: int,
    seed: int,
    side: str = "blue",
) -> Path:
    if side not in ("blue", "red"):
        raise ValueError("side必须为blue或red")
    random.seed(seed)
    np.random.seed(seed)
    LogManager(color_enabled=True)

    profile = read_profile(scenario)
    env = TrainingEnv(profile, render_mode="none")
    _register_agents(env, profile)
    env.reset()

    collector = TrajectoryCollector(
        output_dir,
        {
            "scenario": scenario,
            "seed": seed,
            "rounds": rounds,
            "max_steps": max_steps,
            "engine_step_ms": profile.imagineProfile.simStep,
            "side": side,
            "motion": (
                "pac2_fixed_target_pursuit"
                if side == "blue"
                else "red_cruise_missile_motion"
            ),
            "sample": "one_sample_per_TrainingEnv.step",
        },
        side,
    )
    try:
        for round_index in range(rounds):
            env.red_model_deploy()
            collector.start_round(env.current_round)
            count = 0
            for _ in range(max_steps):
                _, _, done, _ = env.step()
                count += collector.collect(env)
                if done:
                    break
            collector.end_round()
            logging.info(
                f"[轨迹采集] 第 {round_index + 1}/{rounds} 轮完成，共 {count} 条记录"
            )
            if round_index + 1 < rounds:
                env.reset()
    finally:
        collector.close()
        env.close()
    return output_dir


class TrajectoryCollector:
    def __init__(self, output_dir: Path, metadata: dict, side: str):
        self.output_dir = output_dir
        self.side = side
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.round_file = None
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    def start_round(self, round_id: int) -> None:
        if self.round_file is not None:
            raise RuntimeError("上一轮轨迹文件尚未关闭")
        self.round_file = (self.output_dir / f"round_{round_id}.jsonl").open(
            "w", encoding="utf-8"
        )

    def collect(self, env) -> int:
        if self.round_file is None:
            raise RuntimeError("采集前必须调用 start_round")

        count = 0
        factory = env.engine.simulator_factory
        entity_types = (24000,) if self.side == "blue" else RED_ENTITY_TYPES
        for entity_type in entity_types:
            for simulator in factory.get_simulators_by_type(entity_type):
                count += self._collect_simulator(env, simulator)
        return count

    def _collect_simulator(self, env, simulator) -> int:
        if self.side == "red":
            if not isinstance(simulator, RED_SIMULATORS):
                raise TypeError(f"实体 {simulator.entity_ext.entity.id} 不是红方进攻弹仿真器")
            if not _active_red_missile(simulator):
                return 0
            self.round_file.write(
                json.dumps(_red_record(env, simulator), ensure_ascii=False) + "\n"
            )
            return 1

        factory = env.engine.simulator_factory
        if self.side == "blue":
            if not isinstance(simulator, InterceptorSimulator):
                raise TypeError(f"实体 {simulator.entity_ext.entity.id} 不是拦截弹仿真器")
            if not _active_interceptor(simulator):
                return 0
            if simulator.target_id is None:
                raise RuntimeError(f"已发射拦截弹 {simulator.entity_ext.entity.id} 没有目标")

            target = factory.get_simulator_by_id(simulator.target_id)
            if target is None:
                raise RuntimeError(f"找不到目标实体 {simulator.target_id}")
            target_entity = target.entity_ext.entity
            if not target_entity.isVisible or target_entity.survivePoints <= 0:
                return 0

            self.round_file.write(
                json.dumps(_record(env, simulator, target), ensure_ascii=False) + "\n"
            )
            return 1
        raise ValueError("未知预测对象")

    def end_round(self) -> None:
        if self.round_file is None:
            raise RuntimeError("当前没有已打开的轨迹文件")
        self.round_file.close()
        self.round_file = None

    def close(self) -> None:
        if self.round_file is not None:
            self.round_file.close()
            self.round_file = None


def _register_agents(env: TrainingEnv, profile) -> None:
    initial_targets = env._get_init_ship_observation()
    for index, simulator in enumerate(env.engine.simulator_factory.get_all_simulators()):
        entity = simulator.entity_ext.entity
        if entity.entityType in (21000, 21001, 21002):
            env.agent_manager.register_agent(
                AttackMissileAgent(index + 1, entity.id, initial_targets)
            )
    env.agent_manager.register_agent(
        DeployAgent(
            -1,
            -1,
            {},
            profile.imagineProfile.redArea.coordinates,
            profile.imagineProfile.redArea.coordinatesHM,
        )
    )


def _active_interceptor(simulator: InterceptorSimulator) -> bool:
    entity = simulator.entity_ext.entity
    return (
        simulator.launched == 1
        and entity.isVisible
        and entity.survivePoints > 0
        and simulator.ret < 0
        and not (entity.posEcf.x == 0 and entity.posEcf.y == 0 and entity.posEcf.z == 0)
    )


def _active_red_missile(simulator) -> bool:
    entity = simulator.entity_ext.entity
    return (
        simulator.launch == 1
        and entity.isVisible
        and entity.survivePoints > 0
        and simulator.ret < 0
        and not (entity.posEcf.x == 0 and entity.posEcf.y == 0 and entity.posEcf.z == 0)
    )


def _record(env, interceptor: InterceptorSimulator, target) -> dict:
    interceptor_entity = interceptor.entity_ext.entity
    target_entity = target.entity_ext.entity
    return {
        "round": env.current_round,
        "step": env.current_step,
        "sim_time_ms": env.engine.sim_time,
        "interceptor_id": interceptor_entity.id,
        "target_id": target_entity.id,
        "interceptor": _entity_state(interceptor_entity),
        "target": _entity_state(target_entity),
    }


def _red_record(env, simulator) -> dict:
    entity = simulator.entity_ext.entity
    return {
        "round": env.current_round,
        "step": env.current_step,
        "sim_time_ms": env.engine.sim_time,
        "missile_id": entity.id,
        "entity_type": entity.entityType,
        "missile": _entity_state(entity),
    }


def _entity_state(entity) -> dict:
    return {
        "position_ecf": _vector(entity.posEcf),
        "velocity_ecf": _vector(entity.velEcf),
        "survive_points": entity.survivePoints,
        "is_visible": entity.isVisible,
    }


def _vector(vector) -> dict:
    return {"x": float(vector.x), "y": float(vector.y), "z": float(vector.z)}
