# -*-coding:utf-8 -*-

import copy
import json
import logging
import os
from typing import Optional

import numpy as np

from envengine.agent_manager import AgentManager
from envengine.agent_manager.actions.deploy_action import SetLLA, DeployCompleted
from envengine.engine import Engine
from envengine.environment.command_adapter import CommandAdapter
from envengine.environment.command_converter import CommandConverter
from envengine.render import Renderer
from envengine.sdk.base_struct.profile.profile import Profile
from envengine.sdk.writer import write_state, write_ai_action, write_config, write_immediately
from envengine.simulator.util import SpeedDistributor
from user_agents.base_agent import AgentType
from envengine.common import SimmerCommandType, Vector3d
from envengine.environment.individual_reward import (
    allocate_direct_credit,
    target_reward_vector,
    target_reward_vector_fixed_21,
)

logger = logging.getLogger(__name__)

_OBJECTIVE_TYPE_WEIGHTS = {9400: 5.0, 9500: 1.0, 9600: 2.0}
_DAMAGE_SHAPING_WEIGHT = 0.01
_APPROACH_SHAPING_WEIGHT = 0.10
_DESTROYED_OBJECTIVE_WEIGHT = 0.8
_TIME_EFFICIENCY_WEIGHT = 0.2
_REWARD_MODES = frozenset({
    "baseline_original",
    "custom",
    "weighted_damage",
    "weighted_damage_individual",
    "weighted_damage_counterfactual",
    "weighted_damage_decision_anchored",
    "weighted_damage_trajectory_counterfactual",
    "weighted_damage_guided_cow",
})


def _vector3_dict(value) -> dict[str, float]:
    return {
        "x": float(value.x),
        "y": float(value.y),
        "z": float(value.z),
    }


def _learning_runtime_fields(simulator, simulator_factory) -> dict:
    entity_id = int(simulator.entity_ext.entity.id)
    used_count = int(simulator_factory.satellite_use_count(entity_id))
    max_use_count = int(simulator_factory.red_sat_max_use_count)
    return {
        "vel_ecf": _vector3_dict(simulator.entity_ext.entity.velEcf),
        "is_using_satellite": bool(
            simulator_factory.is_using_satellite(entity_id)
        ),
        "satellite_use_count": used_count,
        "satellite_max_use_count": max_use_count,
        "satellite_remaining_uses": max_use_count - used_count,
    }


class TrainingEnv:
    """
    训练环境 - 封装Engine，提供标准RL接口
    支持多智能体隔离观测
    """

    def __init__(self, profile: Profile, render_mode: Optional[str] = None):
        self.engine = Engine(profile, 0)
        self.current_round = 0
        self.current_step = 0
        self.max_steps = self.engine.total_steps
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            map_area = profile.imagineProfile.mapArea
            self.renderer = Renderer(map_area.lonMin, map_area.lonMax, map_area.latMin, map_area.latMax)
            self.renderer.start()

        # 初始化智能体管理器
        self.agent_manager = AgentManager()
        # 初始化线性分配器
        polygon = profile.imagineProfile.redArea.coordinates[0]
        lons = [point[0] for point in polygon]
        lats = [point[1] for point in polygon]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        self.speed_distributor = SpeedDistributor(min_lon=min_lon, max_lon=max_lon,
                                                  min_lat=min_lat, max_lat=max_lat)

        # 初始化上一帧状态缓存（用于奖励计算）
        self._last_observation = None
        self._last_actions = None
        self._shared_learning_policy = None
        default_reward_mode = (
            "baseline_original"
            if os.getenv("RED_MOTION_POLICY") == "ppo_baseline"
            else "custom"
        )
        self.reward_mode = os.getenv("RED_REWARD_MODE", default_reward_mode)
        if self.reward_mode not in _REWARD_MODES:
            raise ValueError(
                f"Unsupported RED_REWARD_MODE {self.reward_mode!r}; "
                f"expected one of {sorted(_REWARD_MODES)}"
            )
        self._objective_weights = {
            int(entity_id): float(weight)
            for entity_id, weight in json.loads(
                os.getenv("BLUE_ASSET_VALUES", "{}")
            ).items()
        }
        self._initial_objective_health: dict[int, float] = {}
        self.last_team_reward = 0.0
        self.last_official_team_reward = 0.0
        self.episode_official_return = 0.0
        self.last_target_rewards: dict[int, float] = {}
        self.last_target_agent_rewards: dict[int, dict[int, float]] = {}
        self.episode_target_agent_rewards: dict[int, dict[int, float]] = {}
        self.last_agent_reward_sum = 0.0
        self.last_credit_conservation_error = 0.0
        self.max_credit_conservation_error = 0.0
        self.last_credit_nonzero_count = 0
        self.episode_agent_reward_sum = 0.0
        self.episode_rewarded_agent_ids: set[int] = set()
        self._target_hit_offsets: dict[int, int] = {}
        self.last_causal_events: tuple[dict, ...] = ()
        self.last_decision_events: tuple[dict, ...] = ()
        self.target_reward_history: list[dict] = []
        self._trajectory_credit_return = 0.0
        self.record_native_trajectory = (
            os.getenv("RED_RECORD_NATIVE_TRAJECTORY", "0") == "1"
        )
        self.native_deployment_actions: list[dict] = []
        self.native_trajectory_steps: list[dict] = []
        self._teacher_override_entities: set[int] = set()
        self._sticky_option_unit_by_entity: dict[int, str] = {}

    def set_shared_learning_policy(self, policy) -> None:
        """Register a training-only centralized critic lifecycle hook.

        Platform agents still receive only isolated observations.  The shared
        policy is supplied the full state solely when explicitly training.
        """

        self._shared_learning_policy = policy

    def reset(self) -> dict:
        """
        重置环境
        :return: 初始观测
        """
        if self._shared_learning_policy is not None:
            self._shared_learning_policy.reset_episode()

        # 重置引擎
        self.engine.reset()
        self.current_step = 0
        self.last_team_reward = 0.0
        self.last_official_team_reward = 0.0
        self.episode_official_return = 0.0
        self.last_target_rewards = {}
        self.last_target_agent_rewards = {}
        self.episode_target_agent_rewards = {}
        self.last_agent_reward_sum = 0.0
        self.last_credit_conservation_error = 0.0
        self.max_credit_conservation_error = 0.0
        self.last_credit_nonzero_count = 0
        self.episode_agent_reward_sum = 0.0
        self.episode_rewarded_agent_ids.clear()
        self._target_hit_offsets.clear()
        self.last_causal_events = ()
        self.last_decision_events = ()
        self.target_reward_history.clear()
        self._trajectory_credit_return = 0.0
        self.native_deployment_actions.clear()
        self.native_trajectory_steps.clear()
        self._teacher_override_entities.clear()
        self._sticky_option_unit_by_entity.clear()
        self.current_round += 1

        # 重置所有智能体
        self.agent_manager.reset_all()

        # 获取初始态势作为观测
        observation = self._get_observation()
        if self.reward_mode in {
            "custom",
            "weighted_damage",
            "weighted_damage_individual",
            "weighted_damage_counterfactual",
            "weighted_damage_decision_anchored",
            "weighted_damage_trajectory_counterfactual",
            "weighted_damage_guided_cow",
        }:
            self._reset_reward_tracking(observation)

        # 将想定配置数据写入文件
        imagine_profile = self.engine.profile.imagineProfile
        red_area = imagine_profile.redArea
        map_area = imagine_profile.mapArea
        write_config({"mapArea": map_area.__dict__, "redArea": red_area.__dict__}, str(self.current_round) + ".json")

        # 将态势数据写入文件
        write_state(observation, str(self.current_round) + ".json")

        # 重置上一帧状态缓存
        self._last_observation = observation.copy()
        self._last_actions = None

        logger.info(f"[训练环境] 环境已重置")
        if self.render_mode == 'human' and self.renderer:
            self.renderer.update_data(observation["entities"])
            # 更新红蓝方区域
            self.renderer.update_area(self.engine.profile.imagineProfile.redArea.coordinates, self.engine.profile.imagineProfile.redArea.coordinatesHM)
        return observation

    def red_model_deploy(self):
        """
        红方模型部署
        :return:
        """

        while True:
            np_action: dict = self._generate_actions_from_deploy_agent()
            # 转换np.array动作到结构体动作
            action = []
            for row in np_action:
                action_type = int(row[0])

                if action_type == 0:  # 移动动作
                    action.append(
                        SetLLA(
                            executor_id=int(row[1]),
                            lla=Vector3d(
                                x=float(row[2]),
                                y=float(row[3]),
                                z=float(row[4])
                            )
                        ).to_dict()
                    )
                elif action_type == 1:  # 完成动作
                    action.append(DeployCompleted().to_dict())
                elif action_type == 2:  # 单网络策略选择不参与本回合
                    simulator = self.engine.get_simulator_by_id(int(row[1]))
                    if simulator is None:
                        raise ValueError(f"Unknown red entity {int(row[1])} in deployment action")
                    simulator.entity_ext.entity.isVisible = False
                    simulator.entity_ext.entity.survivePoints = 0
            # 判断是否部署结束
            deploy_completed_flag = False
            for action_item in action:
                if action_item["commandType_id"] == SimmerCommandType.DEPLOY_COMPLETED:
                    deploy_completed_flag = True
                if not deploy_completed_flag:
                    if self.record_native_trajectory:
                        self.native_deployment_actions.append(
                            copy.deepcopy(action_item)
                        )
                    # 将AI动作数据写入文件
                    write_ai_action(action_item, str(self.current_round) + ".json")

                    # 判断部署位置是否在区域内
                    # in_region = self.speed_distributor.is_in_region(action_item["lla"]["x"], action_item["lla"]["y"])
                    # if not in_region:
                    #     logger.error(f"[训练环境] 部署位置不在区域内")
                    #     continue

                    # 更改模型位置
                    self.engine.simulator_factory.modify_simulator_position(action_item["executor_id"],
                                                                            action_item["lla"])
                    # 更改模型初始速度
                    # speed = self.speed_distributor.get_speed(action_item["lla"]["x"])
                    # self.engine.simulator_factory.modify_simulator_speed(action_item["executor_id"], speed)
                    # 渲染
                    if self.render_mode == 'human' and self.renderer:
                        # 获取观测
                        observation = self._get_observation()
                        self.renderer.update_data(observation["entities"])
            if deploy_completed_flag:
                break

    def red_model_deploy_from_native(self, actions: list[dict]) -> None:
        """Replay deployment commands produced by this simulator."""

        if self.record_native_trajectory:
            self.native_deployment_actions.extend(copy.deepcopy(actions))
        for action in actions:
            self.engine.simulator_factory.modify_simulator_position(
                int(action["executor_id"]),
                action["lla"],
            )

    def step(
        self,
        native_actions: list[dict] | None = None,
        guided_student_entity_ids: set[int] | None = None,
    ) -> tuple[dict, dict, bool, dict]:
        """
        执行一步
        """

        guided_student_entity_ids = {
            int(entity_id) for entity_id in (guided_student_entity_ids or ())
        }
        teacher_prefix = native_actions is not None and not guided_student_entity_ids
        if teacher_prefix:
            action = copy.deepcopy(native_actions)
            self.last_decision_events = ()
        elif native_actions is not None:
            action = copy.deepcopy(native_actions)
        else:
            action = self._generate_actions_from_agents()
            self._apply_teacher_target_override(action)
            self.last_decision_events = self._consume_commander_decision_events()

        # 记录数据：将AI动作数据写入文件
        for i in action:
            write_ai_action(i, str(self.current_round) + ".json")

        # Apply the learned initial position immediately before its same-frame
        # LAUNCH. DEPLOY itself is not a physics command.
        physics_actions: list[dict] = []
        for command in action:
            if int(command.get("commandType_id", -1)) == int(SimmerCommandType.DEPLOY):
                self.engine.simulator_factory.modify_simulator_position(
                    int(command["executor_id"]),
                    command["lla"],
                )
            else:
                physics_actions.append(command)


        # 对AI指令到引擎可接收指令进行适配
        action_adapter = CommandAdapter.common_adapter(physics_actions)
        # 执行仿真步进
        self.engine.step(action_adapter)
        self.current_step += 1
        self.last_causal_events = tuple(
            dict(row)
            for row in self.engine.simulator_factory.consume_causal_events()
        )
        if self.record_native_trajectory:
            self.native_trajectory_steps.append({
                "step": int(self.current_step),
                "actions": copy.deepcopy(action),
                "causal_events": copy.deepcopy(list(self.last_causal_events)),
            })

        # 获取观测
        observation = self._get_observation()

        # 记录数据：将态势数据写入文件
        write_state(observation, str(self.current_round) + ".json")

        # 计算奖励
        rewards = self._compute_reward(observation)
        self.episode_official_return += float(self.last_official_team_reward)
        self.last_agent_reward_sum = float(sum(rewards.values()))
        self.episode_agent_reward_sum += self.last_agent_reward_sum
        for target_id, allocations in self.last_target_agent_rewards.items():
            cumulative = self.episode_target_agent_rewards.setdefault(
                int(target_id), {}
            )
            for agent_id, value in allocations.items():
                cumulative[int(agent_id)] = (
                    cumulative.get(int(agent_id), 0.0) + float(value)
                )
        self.episode_rewarded_agent_ids.update(
            int(agent_id)
            for agent_id, reward in rewards.items()
            if float(reward) > 1e-12
        )
        if self.reward_mode in {
            "weighted_damage_individual",
            "weighted_damage_counterfactual",
            "weighted_damage_decision_anchored",
        }:
            self.last_credit_conservation_error = abs(
                self.last_agent_reward_sum - self.last_official_team_reward
            )
            self.max_credit_conservation_error = max(
                self.max_credit_conservation_error,
                self.last_credit_conservation_error,
            )
        # 判断是否结束
        done = self.get_is_done()
        if done:
            # No next begin_step follows a terminal physics frame.
            commanders = {
                agent.commander
                for agent in self.agent_manager.get_all_agents()
                if getattr(agent, "commander", None) is not None
            }
            for commander in commanders:
                reconcile = getattr(commander, "reconcile_terminal", None)
                if not callable(reconcile):
                    continue
                isolated = []
                for agent in self.agent_manager.get_all_agents():
                    if getattr(agent, "commander", None) is commander:
                        value = self.agent_manager.extract_observation_for_agent(
                            observation, agent.agent_id
                        )
                        if value:
                            isolated.append(value)
                reconcile(tuple(isolated))
                assert_contract_ok = getattr(commander, "assert_contract_ok", None)
                if callable(assert_contract_ok):
                    assert_contract_ok()


        # 额外信息
        info = {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "done": done,
            "target_rewards": dict(self.last_target_rewards),
            "target_agent_rewards": {
                int(target_id): dict(values)
                for target_id, values in self.last_target_agent_rewards.items()
            },
            "agent_reward_sum": self.last_agent_reward_sum,
            "credit_conservation_error": self.last_credit_conservation_error,
            "credit_nonzero_count": self.last_credit_nonzero_count,
            "causal_events": tuple(dict(row) for row in self.last_causal_events),
            "decision_events": tuple(dict(row) for row in self.last_decision_events),
        }

        if self._shared_learning_policy is not None and not teacher_prefix:
            self._shared_learning_policy.end_environment_step(observation)
            set_credit_context = getattr(
                self._shared_learning_policy,
                "set_environment_credit_context",
                None,
            )
            if callable(set_credit_context):
                set_credit_context(info)

        # 通知每个智能体它们的奖励，并记录历史
        recorded_agents = (
            ()
            if teacher_prefix
            else tuple(
                agent
                for agent in self.agent_manager.get_all_agents()
                if not guided_student_entity_ids
                or int(agent.entity_id) in guided_student_entity_ids
            )
        )
        for agent in recorded_agents:
            agent_id = agent.agent_id
            reward = rewards.get(agent_id, 0.0)

            # 为该智能体提取隔离观测
            isolated_obs = self.agent_manager.extract_observation_for_agent(
                observation, agent_id
            )

            # 获取该智能体刚才执行的动作
            entity_id = agent.entity_id
            agent_action = None
            for cmd in action:
                if isinstance(cmd, dict) and "executor_id" in cmd and cmd["executor_id"] == entity_id:
                    agent_action = cmd
                    break
            # 记录这一步的经验
            agent.record_step(
                observation=isolated_obs,
                action=agent_action,
                reward=reward,
                info=info
            )

        if self._shared_learning_policy is not None and not teacher_prefix:
            self._shared_learning_policy.finish_environment_step()

        if self.render_mode == 'human' and self.renderer:
            self.renderer.update_data(observation["entities"])

        # 更新上一帧状态
        self._last_observation = observation.copy()
        self._last_actions = action.copy() if action else None
        return observation, rewards, done, info

    def prepare_native_distillation_batch(
        self,
        native_actions: list[dict],
        *,
        decision_step: int,
        sample_stride: int,
    ):
        """Capture student-visible state and native teacher action labels."""

        full_observation = self._get_observation()
        self._shared_learning_policy.begin_environment_step(full_observation)
        agent_observations = {}
        agents = tuple(self.agent_manager.get_all_agents())
        for agent in agents:
            isolated = self.agent_manager.extract_observation_for_agent(
                full_observation,
                agent.agent_id,
            )
            if isolated:
                isolated["is_using_satellite"] = bool(
                    isolated["self"].get("is_using_satellite", False)
                )
                agent_observations[agent.agent_id] = isolated

        commanders = {
            agent.commander
            for agent in agents
            if getattr(agent, "commander", None) is not None
        }
        for commander in commanders:
            commander.begin_step(tuple(agent_observations.values()))
        return self._shared_learning_policy.build_teacher_distillation_batch(
            agent_observations,
            agents,
            native_actions,
            decision_step=decision_step,
            sample_stride=sample_stride,
        )

    def prepare_native_handoff_actions(
        self,
        native_actions: list[dict],
        controlled_units: list[dict],
        controlled_action_components: str = "joint",
        *,
        persistent_option_control: bool = False,
        target_head_counterfactual: bool = False,
        dense_target_counterfactual: bool = False,
        target_counterfactual_samples: int = 0,
        persistent_inference_entity_ids: set[int] | None = None,
    ) -> tuple[list[dict], set[int], list[dict]]:
        """Replace native actions with boundary or student-owned option actions."""

        full_observation = self._get_observation()
        cached_keep_step = bool(
            persistent_option_control
            and persistent_inference_entity_ids is not None
            and not controlled_units
            and not persistent_inference_entity_ids
        )
        if cached_keep_step:
            self._shared_learning_policy.begin_cached_environment_step(
                full_observation
            )
        else:
            self._shared_learning_policy.begin_environment_step(
                full_observation
            )
        agent_observations = {}
        agents = tuple(self.agent_manager.get_all_agents())
        for agent in agents:
            isolated = self.agent_manager.extract_observation_for_agent(
                full_observation,
                agent.agent_id,
            )
            if isolated:
                agent_observations[agent.agent_id] = isolated

        commanders = {
            agent.commander
            for agent in agents
            if getattr(agent, "commander", None) is not None
        }
        for commander in commanders:
            commander.begin_step(tuple(agent_observations.values()))

        decision_types = {
            int(unit["executor_id"]): (
                "LAUNCH" if int(unit["command_type"]) == 200 else "RETARGET"
            )
            for unit in controlled_units
        }
        teacher_commands = {
            (
                int(command.get("executor_id", -1)),
                int(command.get("commandType_id", -1)),
            ): command
            for command in native_actions
        }
        if persistent_option_control:
            for unit in controlled_units:
                self._sticky_option_unit_by_entity[
                    int(unit["executor_id"])
                ] = str(unit["unit_id"])
        persistent_entity_ids = (
            set(self._shared_learning_policy.active_student_option_entity_ids)
            if persistent_option_control else set()
        )
        inference_entity_ids = persistent_entity_ids
        if persistent_inference_entity_ids is not None:
            inference_entity_ids = persistent_entity_ids & {
                int(entity_id)
                for entity_id in persistent_inference_entity_ids
            }
        student_entity_ids = set(decision_types) | inference_entity_ids
        selected_agents = tuple(
            agent for agent in agents
            if (
                int(agent.entity_id) in student_entity_ids
                and int(agent.agent_id) in agent_observations
            )
        )
        live_student_entity_ids = {
            int(agent.entity_id) for agent in selected_agents
        }
        partial_target_commands = {}
        if controlled_action_components in {"lifecycle", "position"}:
            units_by_entity = {
                int(unit["executor_id"]): unit for unit in controlled_units
            }
            for agent in selected_agents:
                entity_id = int(agent.entity_id)
                unit = units_by_entity.get(entity_id)
                if unit is None:
                    continue
                teacher_command = teacher_commands.get((
                    entity_id, int(unit["command_type"])
                ))
                if teacher_command is None:
                    raise RuntimeError(
                        f"部分动作控制缺少实体 {entity_id} 的教师命令"
                    )
                observation = agent_observations[int(agent.agent_id)]
                preferred_target_id = (
                    self._shared_learning_policy.teacher_command_target_id(
                        teacher_command,
                        entity_type=int(observation["self"].get("type", -1)),
                    )
                )
                if preferred_target_id is None:
                    raise RuntimeError(
                        f"实体 {entity_id} 的教师目标不在合法局部目录"
                    )
                teacher_target_id, command_target = (
                    self._shared_learning_policy.legalize_external_target_command(
                        entity_id,
                        int(observation["self"].get("type", -1)),
                        preferred_target_id,
                        teacher_command["target"],
                    )
                )
                agent.commander.target_by_platform[entity_id] = int(
                    teacher_target_id
                )
                self._shared_learning_policy.sync_external_target_assignment(
                    entity_id, teacher_target_id
                )
                partial_target_commands[(
                    entity_id, int(unit["command_type"])
                )] = command_target
        self._shared_learning_policy.prepare_environment_actions(
            agent_observations,
            selected_agents,
            controlled_decision_types=decision_types,
            controlled_action_components=controlled_action_components,
            persistent_student_entity_ids=inference_entity_ids,
        )
        for (entity_id, _), command_target in partial_target_commands.items():
            target_id = int(
                next(
                    agent.commander.target_by_platform[entity_id]
                    for agent in selected_agents
                    if int(agent.entity_id) == entity_id
                )
            )
            self._shared_learning_policy.apply_prepared_external_target(
                entity_id, target_id, command_target
            )
        student_entity_ids = live_student_entity_ids

        raw_student_actions = []
        for agent in selected_agents:
            agent.set_observation(agent_observations[agent.agent_id])
            raw_student_actions.append(agent.get_action())
        converted = CommandConverter.common_converter(raw_student_actions)
        student_command_types = {200, 3014}
        persistent_search_route = (
            persistent_option_control
            and controlled_action_components == "search"
        )
        deterministic_evasion = bool(
            persistent_search_route
            and getattr(
                self._shared_learning_policy,
                "deterministic_evasion",
                False,
            )
        )
        if persistent_option_control and (
            not persistent_search_route or deterministic_evasion
        ):
            student_command_types.add(
                int(SimmerCommandType.SET_DESIRED_ACC_Z)
            )
        if persistent_option_control and not persistent_search_route:
            student_command_types.update({
                int(SimmerCommandType.EXECUTE_SATELLITE_DETECTION),
            })
        if controlled_action_components in {
            "position", "goal_position", "joint"
        }:
            student_command_types.add(int(SimmerCommandType.DEPLOY))
        student_lifecycle = [
            command for command in converted
            if int(command.get("commandType_id", -1)) in student_command_types
            and int(command.get("executor_id", -1)) in student_entity_ids
        ]
        if controlled_action_components in {"lifecycle", "position"}:
            for command in student_lifecycle:
                command_type = int(command.get("commandType_id", -1))
                if command_type not in {200, 3014}:
                    continue
                command["target"] = copy.deepcopy(partial_target_commands[
                    (int(command["executor_id"]), command_type)
                ])

        unit_keys = {
            (int(unit["executor_id"]), int(unit["command_type"]))
            for unit in controlled_units
        }
        persistent_command_types = {
            int(SimmerCommandType.MISSILE_LAUNCH),
            int(SimmerCommandType.CHANGE_MISSILE_TARGET),
        }
        if deterministic_evasion:
            persistent_command_types.add(
                int(SimmerCommandType.SET_DESIRED_ACC_Z)
            )
        if not persistent_search_route:
            persistent_command_types.update({
                int(SimmerCommandType.SET_DESIRED_ACC_Z),
                int(SimmerCommandType.EXECUTE_SATELLITE_DETECTION),
                int(SimmerCommandType.DEPLOY),
            })
        merged = [
            copy.deepcopy(command)
            for command in native_actions
            if (
                (
                    int(command.get("executor_id", -1)),
                    int(command.get("commandType_id", -1)),
                ) not in unit_keys
                and not (
                    persistent_option_control
                    and int(command.get("executor_id", -1))
                    in persistent_entity_ids
                    and int(command.get("commandType_id", -1))
                    in persistent_command_types
                )
            )
        ]
        merged.extend(copy.deepcopy(student_lifecycle))
        self.last_decision_events = self._consume_commander_decision_events()

        overrides = []
        controlled_entity_ids = {
            int(unit["executor_id"]) for unit in controlled_units
        }
        prepared_replay_step = getattr(
            self._shared_learning_policy,
            "_prepared_replay_step",
            None,
        )
        current_timestep = int(
            self.current_step + 1
            if prepared_replay_step is None
            else prepared_replay_step
        )
        for unit in controlled_units:
            entity_id = int(unit["executor_id"])
            student_actions = [
                copy.deepcopy(command)
                for command in student_lifecycle
                if int(command.get("executor_id", -1)) == entity_id
            ]
            override = {
                "unit_id": str(unit["unit_id"]),
                "timestep": int(unit["timestep"]),
                "executor_id": entity_id,
                "teacher_command_type": int(unit["command_type"]),
                "actions": copy.deepcopy(student_actions),
            }
            if target_head_counterfactual:
                student_target_id = (
                    self._shared_learning_policy.prepared_target_id(entity_id)
                )
                if student_target_id is None:
                    raise RuntimeError("目标头反事实缺少学生合法目标")
                override["student_target_id"] = int(student_target_id)
                if target_counterfactual_samples > 0:
                    sampled_rows = []
                    for candidate in (
                        self._shared_learning_policy
                        .sample_prepared_counterfactual_targets(
                            entity_id,
                            int(target_counterfactual_samples),
                            include_search=False,
                        )
                    ):
                        actions = copy.deepcopy(student_actions)
                        for command in actions:
                            if int(command.get("commandType_id", -1)) not in {
                                200, 3014
                            }:
                                continue
                            command["target"] = copy.deepcopy(candidate["target"])
                        sampled_rows.append({
                            "sample_index": int(candidate["sample_index"]),
                            "target_index": int(candidate["target_index"]),
                            "target_id": int(candidate["target_id"]),
                            "actions": actions,
                        })
                    override["counterfactual_samples"] = sampled_rows
                else:
                    teacher_command = teacher_commands.get(
                        (
                            entity_id,
                            int(unit["command_type"]),
                        )
                    )
                    if teacher_command is None or "target" not in teacher_command:
                        raise RuntimeError("目标头反事实缺少教师目标命令")
                    counterfactual_actions = copy.deepcopy(student_actions)
                    for command in counterfactual_actions:
                        if int(command.get("commandType_id", -1)) not in {
                            200, 3014
                        }:
                            continue
                        command["target"] = copy.deepcopy(
                            teacher_command["target"]
                        )
                    override["counterfactual_actions"] = counterfactual_actions
                if dense_target_counterfactual:
                    candidate_actions = {}
                    for candidate in (
                        self._shared_learning_policy.prepared_legal_targets(
                            entity_id,
                            include_search=False,
                        )
                    ):
                        actions = copy.deepcopy(student_actions)
                        for command in actions:
                            if int(command.get("commandType_id", -1)) not in {
                                200, 3014
                            }:
                                continue
                            command["target"] = copy.deepcopy(candidate["target"])
                        candidate_actions[str(candidate["target_id"])] = actions
                    override["counterfactual_actions_by_target"] = candidate_actions
            if persistent_option_control:
                override["teacher_command_types"] = sorted(
                    persistent_command_types
                )
            overrides.append(override)
        if persistent_option_control:
            for entity_id in sorted(
                live_student_entity_ids - controlled_entity_ids
            ):
                root_unit_id = self._sticky_option_unit_by_entity.get(
                    entity_id
                )
                if root_unit_id is None:
                    continue
                overrides.append({
                    "unit_id": root_unit_id,
                    "timestep": current_timestep,
                    "executor_id": entity_id,
                    "teacher_command_type": -1,
                    "teacher_command_types": sorted(
                        persistent_command_types
                    ),
                    "actions": [
                        copy.deepcopy(command)
                        for command in student_lifecycle
                        if int(command.get("executor_id", -1)) == entity_id
                    ],
                })
        live_persistent_entity_ids = {
            int(agent.entity_id)
            for agent in agents
            if (
                int(agent.entity_id) in persistent_entity_ids
                and int(agent.agent_id) in agent_observations
            )
        }
        return (
            merged,
            live_persistent_entity_ids | live_student_entity_ids,
            overrides,
        )

    def persistent_search_boundary_entity_ids(
        self,
        entity_ids: set[int],
    ) -> set[int]:
        """Return locally legal SEARCH boundaries without neural inference."""

        full_observation = self._get_observation()
        observations_by_entity = {}
        for agent in self.agent_manager.get_all_agents():
            entity_id = int(agent.entity_id)
            if entity_id not in entity_ids:
                continue
            isolated = self.agent_manager.extract_observation_for_agent(
                full_observation, int(agent.agent_id)
            )
            if isolated:
                observations_by_entity[entity_id] = isolated
        return set(
            self._shared_learning_policy
            .persistent_search_boundary_entity_ids(observations_by_entity)
        )

    def _apply_teacher_target_override(self, actions: list[dict]) -> None:
        raw_target_id = os.getenv("RED_TEACHER_OVERRIDE_TARGET_ID")
        if raw_target_id is None:
            return
        target_id = int(raw_target_id)
        target_simulator = self.engine.get_simulator_by_id(target_id)
        if target_simulator is None:
            raise ValueError(f"Unknown teacher override target {target_id}")
        target = target_simulator.entity_ext.entity.lla
        limit = int(os.getenv("RED_TEACHER_OVERRIDE_COUNT", "12"))
        allowed_types = {
            int(value) for value in os.getenv(
                "RED_TEACHER_OVERRIDE_ENTITY_TYPES", "21002"
            ).split(",")
        }
        for command in actions:
            if len(self._teacher_override_entities) >= limit:
                break
            if int(command.get("commandType_id", -1)) != int(
                SimmerCommandType.MISSILE_LAUNCH
            ):
                continue
            entity_id = int(command["executor_id"])
            if entity_id in self._teacher_override_entities:
                continue
            simulator = self.engine.get_simulator_by_id(entity_id)
            if simulator is None or int(
                simulator.entity_ext.entity.entityType
            ) not in allowed_types:
                continue
            command["target"] = {
                "x": float(target.x),
                "y": float(target.y),
                "z": float(target.z),
            }
            self._teacher_override_entities.add(entity_id)

    def _consume_commander_decision_events(self) -> tuple[dict, ...]:
        """Consume each shared commander's newly emitted causal decisions."""

        commanders = {
            agent.commander
            for agent in self.agent_manager.get_all_agents()
            if getattr(agent, "commander", None) is not None
        }
        events: list[dict] = []
        for commander in commanders:
            consume = getattr(commander, "consume_decision_events", None)
            if callable(consume):
                events.extend(dict(row) for row in consume())
        return tuple(events)

    def get_is_done(self) -> bool:
        if self.current_step >= self.max_steps:
            return True

        # 所有 红方弹、拦截弹 结束之后，仿真结束
        for h in self.engine.simulator_factory.get_simulators_by_type(21000):
            if h.entity_ext.entity.isVisible:
                return False

        for l in self.engine.simulator_factory.get_simulators_by_type(21001):
            if l.entity_ext.entity.isVisible:
                return False

        for m in self.engine.simulator_factory.get_simulators_by_type(21002):
            if m.entity_ext.entity.isVisible:
                return False

        for la in self.engine.simulator_factory.get_simulators_by_type(24000):
            if la.entity_ext.entity.isVisible:
                return False

        return True

    def _generate_actions_from_agents(self) -> list[np.array]:
        """
        使用智能体工厂生成所有智能体的动作
        :return: actions
        """
        # 获取当前完整观测
        full_observation = self._get_observation()
        if self._shared_learning_policy is not None:
            self._shared_learning_policy.begin_environment_step(full_observation)

        # 为每个智能体(前提是模型存活)提取隔离观测并收集动作
        agent_observations = {}
        for agent in self.agent_manager.get_all_agents():
            isolated_obs = self.agent_manager.extract_observation_for_agent(
                full_observation,
                agent.agent_id
            )

            if isolated_obs:
                isolated_obs["is_using_satellite"] = bool(
                    isolated_obs["self"].get("is_using_satellite", False)
                )
                agent_observations[agent.agent_id] = isolated_obs

        commanders = {
            agent.commander
            for agent in self.agent_manager.get_all_agents()
            if getattr(agent, "commander", None) is not None
        }
        for commander in commanders:
            begin_step = getattr(commander, "begin_step", None)
            if begin_step is not None:
                begin_step(tuple(agent_observations.values()))

        if self._shared_learning_policy is not None:
            prepare_actions = getattr(
                self._shared_learning_policy,
                "prepare_environment_actions",
                None,
            )
            if callable(prepare_actions):
                prepare_actions(
                    agent_observations,
                    tuple(self.agent_manager.get_all_agents()),
                )
            fork_and_apply = getattr(
                self._shared_learning_policy,
                "fork_and_apply_replay_interventions",
                None,
            )
            if callable(fork_and_apply):
                fork_and_apply()

        # PAOS treats every bottom-policy/agent failure as a fatal experiment
        # contract violation.  Other commanders retain the historical
        # best-effort AgentManager behavior.
        fail_fast = any(
            bool(getattr(commander, "fail_fast_agent_errors", False))
            for commander in commanders
        )
        # 收集所有智能体的动作
        actions = self.agent_manager.collect_actions_from_agents(
            agent_observations,
            fail_fast=fail_fast,
        )
        # Surface an atomic PAOS batch failure only after every agent action
        # has been collected, and before command conversion or engine step.
        for commander in commanders:
            assert_contract_ok = getattr(commander, "assert_contract_ok", None)
            if callable(assert_contract_ok):
                assert_contract_ok()
        # 转换所有智能体动作到结构体类型
        return CommandConverter.common_converter(actions)


    def prepare_commander_no_step_probe(self, commander) -> dict:
        """Build legal PAOS input without agent actions or an engine step."""

        full_observation = self._get_observation()
        observations = []
        for agent in self.agent_manager.get_all_agents():
            if getattr(agent, "commander", None) is not commander:
                continue
            isolated = self.agent_manager.extract_observation_for_agent(
                full_observation, agent.agent_id
            )
            if isolated:
                observations.append(isolated)
        result = commander.no_step_probe(tuple(observations))
        assert_contract_ok = getattr(commander, "assert_contract_ok", None)
        if callable(assert_contract_ok):
            assert_contract_ok()
        return result
    def _generate_actions_from_deploy_agent(self) -> np.array:
        """
        使用智能体工厂生成部署智能体的动作
        :return: actions
        """
        # 为部署智能体提取隔离观测并收集动作
        agent_observations = {}

        for agent in self.agent_manager.get_agent_by_type(AgentType.DEPLOY):
            agent_observations[agent.agent_id] = self._get_red_observation()

        # 收集部署智能体的动作
        action = self.agent_manager.collect_actions_from_deploy_agent(agent_observations)

        return action

    def _get_init_ship_observation(self) -> dict:
        """获取初始化时，给红方 AI 的蓝方信息
        仅传递，蓝方高中价值舰船（即：目标、拦截阵地）

        9400：目标
        9500：无人船
        9600：拦截阵地
        """

        simulators = self.engine.simulator_factory.get_all_simulators()

        observation = {
            "step": self.current_step,
            "sim_time": self.engine.sim_time,
            "sim_step": self.engine.sim_step,
            "entities": {}
        }

        for simulator in simulators:
            entity_data = simulator.entity_ext.entity
            if entity_data.entityType not in (9400, 9500, 9600):
                continue

            observation["entities"][entity_data.id] = {
                "nameChn": entity_data.nameChn,
                "position": {
                    "lon": entity_data.lla.x,
                    "lat": entity_data.lla.y,
                    "alt": entity_data.lla.z
                },
                "health": entity_data.survivePoints,
                "isVisible": entity_data.isVisible,
                "type": entity_data.entityType,
                "side": entity_data.sideId,
                "detectInfo": entity_data.detectInfo,
                "commRangeInfo": entity_data.commRangeInfo
            }

        return observation

    def _get_observation(self) -> dict:
        """获取当前观测（态势）"""
        simulators = self.engine.simulator_factory.get_all_simulators()

        observation = {
            "step": self.current_step,
            "sim_time": self.engine.sim_time,
            "sim_step": self.engine.sim_step,
            "entities": {}
        }

        for simulator in simulators:
            entity_data = simulator.entity_ext.entity
            observation["entities"][entity_data.id] = {
                "nameChn": entity_data.nameChn,
                "position": {
                    "lon": entity_data.lla.x,
                    "lat": entity_data.lla.y,
                    "alt": entity_data.lla.z
                },
                "pos_ecf":{
                    "x":entity_data.posEcf.x,
                    "y":entity_data.posEcf.y,
                    "z":entity_data.posEcf.z,
                },
                "stage":entity_data.stage,
                "health": entity_data.survivePoints,
                "isVisible": entity_data.isVisible,
                "type": entity_data.entityType,
                "side": entity_data.sideId,
                "detectInfo": entity_data.detectInfo,
                "commRangeInfo": entity_data.commRangeInfo,
                **_learning_runtime_fields(simulator, self.engine.simulator_factory),
            }

        return observation

    def _get_red_observation(self) -> dict:
        """获取红方观测（态势）"""
        simulators = self.engine.simulator_factory.get_simulators_by_side(0)

        observation = {
            "step": self.current_step,
            "sim_time": self.engine.sim_time,
            "sim_step": self.engine.sim_step,
            "entities": {}
        }

        for simulator in simulators:
            entity_data = simulator.entity_ext.entity
            observation["entities"][entity_data.id] = {
                "nameChn": entity_data.nameChn,
                "position": {
                    "lon": entity_data.lla.x,
                    "lat": entity_data.lla.y,
                    "alt": entity_data.lla.z
                },
                "health": entity_data.survivePoints,
                "type": entity_data.entityType,
                "side": entity_data.sideId,
                "detectInfo": entity_data.detectInfo,
                **_learning_runtime_fields(simulator, self.engine.simulator_factory),
            }

        return observation

    @staticmethod
    def _lookup_entity(entities: dict, entity_id: int) -> dict | None:
        return entities.get(entity_id, entities.get(str(entity_id)))

    @staticmethod
    def _distance_km(first: dict, second: dict) -> float:
        mean_latitude = np.radians(
            (float(first.get("lat", 0.0)) + float(second.get("lat", 0.0))) / 2.0
        )
        east_km = (
            float(second.get("lon", 0.0)) - float(first.get("lon", 0.0))
        ) * 111.32 * np.cos(mean_latitude)
        north_km = (
            float(second.get("lat", 0.0)) - float(first.get("lat", 0.0))
        ) * 110.57
        return float(np.hypot(east_km, north_km))

    def _reset_reward_tracking(self, observation: dict) -> None:
        entities = observation.get("entities", {})
        if not self._objective_weights:
            self._objective_weights = {
                int(entity_id): _OBJECTIVE_TYPE_WEIGHTS[int(entity["type"])]
                for entity_id, entity in entities.items()
                if int(entity.get("side", -1)) == 1
                and int(entity.get("type", -1)) in _OBJECTIVE_TYPE_WEIGHTS
            }
        self._initial_objective_health = {
            entity_id: max(
                0.0,
                float((self._lookup_entity(entities, entity_id) or {}).get("health", 0.0)),
            )
            for entity_id in self._objective_weights
        }
        relation = self.engine.simulator_factory.target_hit_relation
        self._target_hit_offsets = {
            int(target_id): len(relation.get(int(target_id), ()))
            for target_id in self._objective_weights
        }

    def _target_rewards(self, observation: dict) -> dict[int, float]:
        rewards = target_reward_vector(
            current_entities=observation.get("entities", {}),
            previous_entities=(self._last_observation or {}).get("entities", {}),
            initial_health=self._initial_objective_health,
            objective_weights=self._objective_weights,
        )
        self.last_target_rewards = dict(rewards)
        self.last_target_agent_rewards = {}
        return rewards

    def _target_rewards_fixed_21(self, observation: dict) -> dict[int, float]:
        """Exact ``w_j / 21`` per-target score increment."""

        rewards = target_reward_vector_fixed_21(
            current_entities=observation.get("entities", {}),
            previous_entities=(self._last_observation or {}).get("entities", {}),
            initial_health=self._initial_objective_health,
            objective_weights=self._objective_weights,
        )
        self.last_target_rewards = dict(rewards)
        self.last_target_agent_rewards = {}
        return rewards


    def _consume_target_hit_events(self) -> dict[int, tuple[dict, ...]]:
        relation = self.engine.simulator_factory.target_hit_relation
        result: dict[int, tuple[dict, ...]] = {}
        for target_id in self._objective_weights:
            events = relation.get(int(target_id), ())
            offset = self._target_hit_offsets.get(int(target_id), 0)
            result[int(target_id)] = tuple(events[offset:])
            self._target_hit_offsets[int(target_id)] = len(events)
        return result

    def _compute_weighted_damage_reward(self, observation: dict) -> dict[int, float]:
        """返回正式加权毁伤评分势函数的逐步增量。"""

        reward = float(sum(self._target_rewards(observation).values()))
        self.last_team_reward = float(reward)
        self.last_official_team_reward = float(reward)
        self.last_credit_nonzero_count = (
            len(self.agent_manager.get_all_agents()) if reward > 1e-12 else 0
        )
        return {
            agent.agent_id: float(reward)
            for agent in self.agent_manager.get_all_agents()
        }

    def _compute_weighted_damage_individual_reward(
        self,
        observation: dict,
    ) -> dict[int, float]:
        """将每个目标奖励仅分配给本步实际毁伤来源。"""

        target_rewards = self._target_rewards(observation)
        agents = self.agent_manager.get_all_agents()
        entity_to_agent = {
            int(agent.entity_id): int(agent.agent_id)
            for agent in agents
            if int(agent.entity_id) >= 0
        }
        allocation = allocate_direct_credit(
            target_rewards=target_rewards,
            hit_events=self._consume_target_hit_events(),
            entity_to_agent=entity_to_agent,
            agent_ids=tuple(int(agent.agent_id) for agent in agents),
        )
        reward = float(sum(target_rewards.values()))
        self.last_team_reward = reward
        self.last_official_team_reward = reward
        self.last_credit_conservation_error = allocation.conservation_error
        self.last_credit_nonzero_count = len(allocation.nonzero_agent_ids)
        self.last_target_agent_rewards = {
            int(target_id): dict(values)
            for target_id, values in allocation.target_agent_rewards.items()
        }
        return allocation.agent_rewards

    def _compute_weighted_damage_trajectory_counterfactual_reward(
        self,
        observation: dict,
    ) -> dict[int, float]:
        """Record target rewards while factual PPO rows remain reward-free.

        Completed-episode counterfactual credit is later written exactly once
        at each causal decision timestep, never at this damage timestep.
        """

        target_rewards = self._target_rewards_fixed_21(observation)
        reward = float(sum(target_rewards.values()))
        self.last_team_reward = reward
        self.last_official_team_reward = reward
        self.last_credit_nonzero_count = 0
        self.target_reward_history.append({
            "timestep": int(self.current_step),
            "target_rewards": dict(target_rewards),
        })
        return {
            int(agent.agent_id): 0.0
            for agent in self.agent_manager.get_all_agents()
        }

    def _compute_weighted_damage_guided_cow_reward(
        self,
        observation: dict,
    ) -> dict[int, float]:
        """Record only formal target-damage reward; credit is backfilled by COW."""

        target_rewards = self._target_rewards_fixed_21(observation)
        reward = float(sum(target_rewards.values()))
        self.last_team_reward = reward
        self.last_official_team_reward = reward
        self.last_credit_nonzero_count = 0
        self.target_reward_history.append({
            "timestep": int(self.current_step),
            "target_rewards": dict(target_rewards),
        })
        return {
            int(agent.agent_id): 0.0
            for agent in self.agent_manager.get_all_agents()
        }


    def _compute_original_baseline_reward(self) -> dict[int, float]:
        """Preserve the Git HEAD reward: +1 for every surviving agent entity."""
        rewards = {}
        for agent in self.agent_manager.get_all_agents():
            simulator = self.engine.get_simulator_by_id(agent.entity_id)
            rewards[agent.agent_id] = (
                1.0
                if simulator is not None
                and simulator.entity_ext.entity.survivePoints > 0
                else 0.0
            )
        return rewards

    def _compute_reward(self, observation: dict) -> dict[int, float]:
        """Return a shared reward aligned with the official 0.8K + 0.2T score."""
        if self.reward_mode == "baseline_original":
            rewards = self._compute_original_baseline_reward()
            self.last_team_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            self.last_official_team_reward = 0.0
            return rewards
        if self.reward_mode == "weighted_damage":
            return self._compute_weighted_damage_reward(observation)
        if self.reward_mode == "weighted_damage_individual":
            return self._compute_weighted_damage_individual_reward(observation)
        if self.reward_mode in {
            "weighted_damage_counterfactual",
            "weighted_damage_decision_anchored",
        }:
            return self._compute_weighted_damage_individual_reward(observation)
        if self.reward_mode == "weighted_damage_trajectory_counterfactual":
            return self._compute_weighted_damage_trajectory_counterfactual_reward(
                observation
            )
        if self.reward_mode == "weighted_damage_guided_cow":
            return self._compute_weighted_damage_guided_cow_reward(observation)

        current_entities = observation.get("entities", {})
        previous_entities = (self._last_observation or {}).get("entities", {})
        total_weight = sum(self._objective_weights.values())
        team_reward = 0.0
        official_team_reward = 0.0

        if total_weight > 0.0:
            for entity_id, weight in self._objective_weights.items():
                current = self._lookup_entity(current_entities, entity_id)
                previous = self._lookup_entity(previous_entities, entity_id)
                if current is None or previous is None:
                    continue

                initial_health = max(
                    self._initial_objective_health.get(entity_id, 0.0), 1e-6
                )
                previous_health = max(0.0, float(previous.get("health", 0.0)))
                current_health = max(0.0, float(current.get("health", 0.0)))
                normalized_weight = weight / total_weight

                # Small dense shaping helps PPO learn before the terminal kill
                # without overpowering the official destruction/time score.
                damage_delta = max(0.0, previous_health - current_health) / initial_health
                team_reward += (
                    _DAMAGE_SHAPING_WEIGHT * normalized_weight * damage_delta
                )

                if previous_health > 0.0 and current_health <= 0.0:
                    remaining_fraction = max(
                        0.0,
                        1.0 - self.current_step / max(1, self.max_steps),
                    )
                    official_team_reward += normalized_weight * (
                        _DESTROYED_OBJECTIVE_WEIGHT
                        + _TIME_EFFICIENCY_WEIGHT * remaining_fraction
                    )

        team_reward += official_team_reward
        self.last_team_reward = float(team_reward)
        self.last_official_team_reward = float(official_team_reward)
        rewards = {
            agent.agent_id: team_reward
            for agent in self.agent_manager.get_all_agents()
        }
        for agent in self.agent_manager.get_all_agents():
            commander = getattr(agent, "commander", None)
            target_lookup = getattr(commander, "target_id_for", None)
            if not callable(target_lookup):
                continue
            target_id = target_lookup(agent.entity_id)
            previous_platform = self._lookup_entity(previous_entities, agent.entity_id)
            current_platform = self._lookup_entity(current_entities, agent.entity_id)
            previous_target = self._lookup_entity(previous_entities, target_id)
            current_target = self._lookup_entity(current_entities, target_id)
            if any(
                item is None
                for item in (
                    previous_platform,
                    current_platform,
                    previous_target,
                    current_target,
                )
            ):
                continue
            if (
                float(current_platform.get("health", 0.0)) <= 0.0
                or float(current_target.get("health", 0.0)) <= 0.0
            ):
                continue
            previous_distance = self._distance_km(
                previous_platform.get("position", {}),
                previous_target.get("position", {}),
            )
            current_distance = self._distance_km(
                current_platform.get("position", {}),
                current_target.get("position", {}),
            )
            normalized_progress = np.clip(
                (previous_distance - current_distance) / 1000.0,
                -0.01,
                0.01,
            )
            rewards[agent.agent_id] += float(
                _APPROACH_SHAPING_WEIGHT * normalized_progress
            )
        return rewards

    def close(self):
        """关闭环境"""
        if self.renderer:
            self.renderer.stop()
        print("[TrainingEnv] 环境已关闭")

    # 委托方法
    def get_simulator(self, entity_id: int):
        return self.engine.get_simulator(entity_id)

    def get_simulators_by_type(self, entity_type: int):
        return self.engine.get_simulators_by_type(entity_type)

    def get_simulators_by_side(self, side_id: int):
        return self.engine.get_simulators_by_side(side_id)
