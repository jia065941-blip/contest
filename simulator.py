"""仿真编排：公开配置、调用循环和资源生命周期。"""

from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
import logging
import random

import yaml

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
# 运行入口保留开始和结束提示，其他模块的常规 INFO 由 LogManager 的级别控制。
logger.setLevel(logging.INFO)


class Simulator:
    """维护环境、虫族策略、用户策略和日志，组织仿真运行。"""

    def __init__(self, config:dict):
        # 公开运行配置：保存运行选项，不保存场景内部资料。
        self.config = config

        # 环境实例：维护系统状态，提供观测和步进。
        self._env = None

        # 虫族（蓝方）策略：仅在 run 时实例化。
        self._blue_policy = None

        # 用户策略：仅在 run 时实例化，内部管理部署与各实体决策。
        self._user_policy = None

        # 日志组件：复用 SDK LogManager 管理控制台提示及日志缓存。
        # SDK writer 在 run 中局部持有，直接接收环境已有的过程记录。
        self._logger = None

        # 运行标志：同一实例只执行一次，避免重复初始化。
        self._started = False

        # 赛后归因器只读取可信引擎和策略已经产生的任务，不向任何策略回灌真值。
        self._attribution = None

    def run(self) -> None:
        """组织初始化、推进到环境结束、保存结果和资源释放。"""
        # 检查运行状态，防止同一实例重复启动。
        if self._started:
            raise RuntimeError("A Simulator instance can only run once")
        self._started = True
        writer = None
        try:
            # 先配置输出通道，确保环境初始化和策略执行都使用本次日志选项。
            self._logger = self._create_logger()
            writer = self._create_writer()
            scenario = self._initialize()
            logger.info(
                "仿真开始：场景=%s，红方=%s，蓝方=%s，seed=%s",
                self.config["scenario"].stem, self.config["agent"].stem,
                self.config["blue_mode"].stem, self.config["seed"],
            )

            # 示例输出通道：按需汇总结果；仅在本次 run 中使用，默认关闭。
            example_evaluation = self._create_example_evaluation(scenario)
            self._attribution = self._create_attribution(scenario)
            if self._attribution is not None:
                self._attribution.start()

            # 主循环只遵循环境返回的 done，不解释时限或其他终止规则。
            done = False
            observation = {}
            info = {}
            while not done:
                observation, _, done, info = self._step()
                if self._attribution is not None:
                    self._attribution.update(info["step"])
                if info["step"] % 200 == 0:
                    print(f"current step: {info['step']}")
                if example_evaluation is not None:
                    example_evaluation.update(info["step"], observation)

            # 示例输出通道：正常结束时写一次结果，统计内容可由使用者替换。
            if example_evaluation is not None:
                result = example_evaluation.build(
                    observation,
                    termination_reason=info.get("termination_reason", "unknown"),
                    red_launched=None,
                )
                example_evaluation.write(self.config["output_dir"], result, filename="example_evaluation.json")
            if self._attribution is not None:
                self._attribution.finish(info.get("termination_reason", "unknown"))

            # 过程文件由环境与模型写入 SDK；结束时等待缓冲记录落盘。
            writer.flush()
            logger.info("仿真结束：共推进 %s 步", info["step"])
        except Exception as error:
            if self._attribution is not None:
                self._attribution.abort(error)
            logger.error("仿真运行失败：%s", error)
            raise
        finally:
            # 无论初始化、推进或保存是否失败，都释放已取得的资源。
            self._close(writer)

    def _resolve_seed(self) -> None:
        """仅生成一次共享种子，各组件如何应用种子由各自实现决定。"""
        if self.config["seed"] is None:
            self.config["seed"] = random.SystemRandom().randrange(2**32)

    def _initialize(self):
        """加载场景、创建环境与双方策略，并完成本局部署。"""
        from scenarios import load_scenario

        # 场景侧完成读取、检查和配置组装，上层只在初始化期间传递配置。
        self._resolve_seed()
        scenario = load_scenario(self.config["scenario"])
        self._env = self._create_environment(scenario)
        self._blue_policy = self._create_blue_policy(scenario)

        # 蓝方保持控制流程，环境重置后先完成独立部署。
        self._env.register_blue_policy(self._blue_policy)
        self._env.reset()
        self._deploy()

        # 策略实例从部署后的本机初态创建；初始布局记录由管理器写入 SDK。
        self._user_policy = self._create_user_policy()
        self._user_policy.reset()
        return scenario

    def _deploy(self) -> dict:
        """调用一次部署入口并应用方案，局部对象在此阶段结束后关闭。"""
        from user_agents.red_checker import check_deployment

        deployment_class = check_deployment(self.config["agent"])
        deployment = deployment_class()
        try:
            manager = self._env.agent_manager
            observation = manager.get_deployment_observation(seed=self.config["seed"])
            plan = deployment.deploy(observation)
            manager.apply_deployment(plan)
            return self._env.get_observation()
        finally:
            deployment.close()

    def _create_environment(self, scenario):
        """将场景配置和运行选项交给环境，创建本局实例。"""
        from envengine.environment.training_env import TrainingEnv

        return TrainingEnv(
            scenario,
            render_mode=self.config["render_mode"],
            record_logs=self.config["logging"]["enabled"],
        )

    def _create_blue_policy(self, scenario):
        """根据蓝方 Python 文件路径创建虫族策略实例。"""
        from blue_defense.blue_checker import check_blue_policy

        policy_class = check_blue_policy(self.config["blue_mode"])
        return policy_class(seed=self.config["seed"], asset_values=scenario.asset_values)

    def _create_user_policy(self):
        """检查红方文件入口，交给管理器创建并注册本局单机策略。"""
        from user_agents.red_checker import check_red_agent

        agent_class = check_red_agent(self.config["agent"])
        return self._env.agent_manager.register_user_agent(
            agent_class, seed=self.config["seed"], max_steps=self._env.max_steps,
        )

    def _create_logger(self):
        """复用日志管理器，默认控制台保留运行提示和错误。"""
        from envengine.sdk.log import LogManager

        return LogManager(color_enabled=False, level=logging.ERROR)

    def _create_writer(self):
        """将公开记录选项交给现成 SDK writer，逐步数据由环境直接写入。"""
        from envengine.sdk.writer import WriteConfig, init_writer

        log = self.config["logging"]
        enabled = log["enabled"]
        # 即使详细记录关闭也显式配置禁用通道，避免便利函数使用默认的全开配置。
        return init_writer(WriteConfig(
            enable_config=enabled,
            enable_state=enabled,
            enable_event=enabled and log["events"],
            enable_ai_action=enabled and log["actions"],
            output_dir=str(self.config["output_dir"]),
            log_frequency=log["state_interval_steps"],
            verbose=False,
        ))

    def _create_example_evaluation(self, scenario):
        """创建可选的示例结果输出通道，评分规则由场景提供。"""
        if not self.config["logging"]["example_evaluation"]:
            return None

        from example_evaluation import RunSummary
        from scenarios.cases import RewardTracker

        rule = scenario.reward_policy
        channel = RunSummary(
            scenario=rule.scenario_id if rule is not None else str(scenario.path),
            policies={"red": self.config["agent"].stem, "blue": self.config["blue_mode"].stem},
            reward_tracker=RewardTracker(rule) if rule is not None else None,
        )
        channel.start(self._env.get_observation())
        return channel

    def _create_attribution(self, scenario):
        """创建只读赛后归因通道；该通道不是红方观测接口的一部分。"""
        if not self.config["logging"].get("attribution_evaluation", False):
            return None

        from evaluation.attribution import AttributionTracker

        return AttributionTracker(
            env=self._env,
            scenario=scenario,
            output_dir=self.config["output_dir"],
            seed=self.config["seed"],
        )

    def _step(self) -> tuple:
        """环境组织策略调用、推进及过程记录，返回本步状态与结束信息。"""
        return self._env.step()

    def _close(self, writer=None) -> None:
        """关闭运行组件与本次 writer，任一关闭失败仍继续清理其余资源。"""
        # 环境和策略先退出，writer 收尾落盘，最后恢复调用方的控制台日志配置。
        resources = (self._env, self._user_policy, self._blue_policy, writer, self._logger)
        self._env = self._user_policy = self._blue_policy = self._logger = self._attribution = None
        with ExitStack() as stack:
            for resource in reversed(resources):
                if resource is not None:
                    stack.callback(resource.close)
