"""红方示例的程序化训练工具，管理回合、Critic全态、更新与模型保存。"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime
import logging
from pathlib import Path
import random
import time

from scenarios import ScenarioConfig, load_scenario

from ..common.config import RedAgentConfig


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BLUE_MODE = ROOT / "blue_defense/immediate/b0_fixed_ratio_random.py"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_training(
    scenario: ScenarioConfig | str | Path,
    config: RedAgentConfig,
    *,
    rounds: int = 1,
    output_dir: str | Path = "results/training",
    max_steps: int | None = None,
    blue_mode: str | Path = DEFAULT_BLUE_MODE,
    plan_path: str | Path | None = None,
) -> Path:
    """执行多局训练，保存模型并返回本次结果目录。

    config指定任务层、PPO/MAPPO、种子和模型路径，并开启training。
    max_steps只截断本次实验，默认沿用场景时限；模型的时间特征使用同一尺度。
    plan_path是示例自身的可选网格方案，部署观测使用管理器的五字段接口。
    """
    if not config.training:
        raise ValueError("Training config must enable training")
    if not config.model_path:
        raise ValueError("model_path is required to save the trained model")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None")

    import numpy as np

    from blue_defense.blue_checker import check_blue_policy
    from envengine import TrainingEnv
    from envengine.sdk.log import LogManager
    from envengine.sdk.writer import WriteConfig, init_writer
    from example_evaluation import RunSummary
    from scenarios.cases import RewardTracker

    from ..assignment.r0_random import RedAttack
    from ..deployment import Deployment

    # 场景和策略文件各自负责装配；一次训练复用同一个随机流和模型资源。
    scene = scenario if isinstance(scenario, ScenarioConfig) else load_scenario(scenario)
    blue_class = check_blue_policy(Path(blue_mode).expanduser().resolve())
    seed = random.SystemRandom().randrange(2**32) if config.seed is None else config.seed
    random.seed(seed)
    np.random.seed(seed)
    model_path = Path(config.model_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    log_manager = writer = env = group = blue_policy = deployer = None
    try:
        # 训练默认只打印进度并保存示例结果，关闭详细过程文件，避免大量输出。
        log_manager = LogManager(color_enabled=False, level=logging.ERROR)
        writer = init_writer(WriteConfig(
            output_dir=str(destination), verbose=False,
            enable_config=False, enable_state=False, enable_event=False, enable_ai_action=False,
        ))
        env = TrainingEnv(scene, render_mode="none", record_logs=False)
        step_limit = env.max_steps if max_steps is None else min(max_steps, env.max_steps)
        env.max_steps = step_limit
        training_config = replace(config, seed=seed, max_steps=step_limit, model_path=str(model_path))
        blue_policy = blue_class(seed=seed, asset_values=scene.asset_values)
        env.register_blue_policy(blue_policy)
        deployer = Deployment()
        deployer.plan_path = plan_path

        logger.info("Training scenario: %s; rounds: %s; steps: %s; seed: %s",
                    scene.path, rounds, step_limit, seed)
        for round_index in range(rounds):
            # 回合重置后先布局；部署接收步数、公开清单、目标、区域和随机种子。
            env.reset()
            observation = env.agent_manager.get_deployment_observation(seed=seed)
            env.agent_manager.apply_deployment(deployer.deploy(observation))
            initial = env.get_observation()
            if group is None:
                # 学习配置仅交给本示例的私有资源层，普通Actor仍逐机接收观测。
                group = env.agent_manager.register_user_agent(
                    RedAttack, seed=seed, options={"config": asdict(training_config)},
                    max_steps=step_limit,
                )
                group.reset()
            runtime = group.agents[0]._resource
            summary_builder = RunSummary(
                scenario=scene.reward_policy.scenario_id if scene.reward_policy else str(scene.path),
                policies={"red": config.policy_name, "red_motion": config.motion_policy, "blue": blue_policy.name},
                reward_tracker=RewardTracker(scene.reward_policy) if scene.reward_policy else None,
            )
            summary_builder.start(initial)
            final_observation = initial
            termination_reason = "time_limit"
            started = time.perf_counter()
            for step in range(1, step_limit + 1):
                # Critic前态→环境分发观测并推进→下一全态和经验→模型更新。
                runtime.begin_training_step(env.get_observation())
                final_observation, _, done, _ = env.step()
                runtime.finish_training_step(final_observation)
                summary_builder.update(step, final_observation)
                if step % 200 == 0:
                    logger.info("Round %s, step %s", round_index + 1, step)
                if done:
                    termination_reason = "time_limit" if step >= step_limit else "environment_done"
                    break
            writer.flush()
            red_launched = sum(agent._execution.launch_step >= 0 for agent in group.agents)
            summary = summary_builder.build(
                final_observation, termination_reason=termination_reason, red_launched=red_launched,
            )
            filename = "summary.json" if rounds == 1 else f"summary_round_{round_index + 1}.json"
            path = summary_builder.write(destination, summary, filename)
            logger.info("Round %s finished in %.3f seconds; summary: %s",
                        round_index + 1, time.perf_counter() - started, path)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        runtime.save_model(str(model_path))
        logger.info("Learning checkpoint saved: %s", model_path)
        return destination
    finally:
        # 任一创建、训练或落盘失败时仍尝试关闭全部已取得的资源。
        with ExitStack() as cleanup:
            for resource in (log_manager, writer, deployer, blue_policy, group, env):
                if resource is not None:
                    cleanup.callback(resource.close)
