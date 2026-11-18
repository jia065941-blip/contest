"""Explicit centralized-training driver for the red example agents.

The competition environment knows only observations, actions, and simulation
steps. This trusted caller owns the learning lifecycle and may read full state
for the MAPPO Critic; agents continue to receive isolated observations only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import random
import time

from blue_defense import build_blue_config
from scenarios import ScenarioBundle, read_profile, resolve_deployment_constraints

from .config import RedAgentConfig, validate_red_arguments


RED_PLATFORM_TYPES = (21000, 21001, 21002)


def run_training(args: argparse.Namespace, bundle: ScenarioBundle) -> None:
    """Run the existing PPO/MAPPO training loop outside main and the environment.

    ``main.py --red-learning-train`` forwards its validated execution arguments
    here. Programmatic callers must likewise resolve paths and round limits.
    """

    validate_red_arguments(args)
    if not args.red_learning_train:
        raise ValueError("The red training entry point requires --red-learning-train")

    import numpy as np

    from blue_defense import BlueDefenseController
    from envengine import TrainingEnv
    from envengine.sdk.log import LogManager
    from envengine.sdk.writer import WriteConfig, init_writer, write_immediately
    from evaluation import RunSummary
    from scenarios.cases import RewardTracker

    from .factory import RedAgentFactory

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2**32)
    random.seed(args.seed)
    np.random.seed(args.seed)
    profile = read_profile(str(bundle.path))
    coordinates, coordinates_hm, launch_altitude = resolve_deployment_constraints(
        profile, bundle.deployment_area, args.deployment_plan,
    )
    blue_config = build_blue_config(args, bundle)
    agent_factory = RedAgentFactory(RedAgentConfig.from_arguments(args))

    LogManager(color_enabled=args.enable_log_color)
    write_config = WriteConfig(
        output_dir=str(args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
        log_frequency=args.log_frequency,
        verbose=args.verbose,
        batch_size=args.batch_size,
        enable_config=args.enable_config,
        enable_state=args.enable_state,
        enable_event=args.enable_event,
        enable_ai_action=args.enable_ai_action,
    )
    writer = init_writer(write_config)
    env = None
    try:
        logging.info("Scenario: %s; rounds: %s; steps: %s; seed: %s",
                     bundle.path, args.total_rounds, args.max_steps, args.seed)
        logging.info("Training policies: red=%s, motion=%s, blue=%s",
                     args.red_policy, args.red_motion_policy, blue_config.policy_name)
        env = TrainingEnv(profile, render_mode=args.render_mode)
        env.max_steps = args.max_steps
        env.set_deployment_constraints(coordinates, coordinates_hm, launch_altitude)
        simulators = env.engine.simulator_factory.get_all_simulators()
        for simulator in env.engine.simulator_factory.get_simulators_by_type(35000):
            simulator.set_defense_controller(BlueDefenseController(blue_config))

        public_observation = env.get_public_targets_observation()
        area = profile.imagineProfile.mapArea
        agents = agent_factory.create_agents(
            public_observation,
            (
                (index + 1, simulator.entity_ext.entity.id)
                for index, simulator in enumerate(simulators)
                if simulator.entity_ext.entity.entityType in RED_PLATFORM_TYPES
            ),
            search_polygon=(
                (area.lonMin, area.latMin), (area.lonMax, area.latMin),
                (area.lonMax, area.latMax), (area.lonMin, area.latMax),
            ),
        )
        for agent in agents:
            env.agent_manager.register_agent(agent)

        entities = {}
        if args.deployment_plan is not None:
            entities = {
                item.entity.id: {
                    "type": item.entity.entityType,
                    "position": {"alt": launch_altitude if launch_altitude is not None else item.entity.lla.z},
                }
                for item in profile.imagineProfile.entityList
                if item.entity.entityType in RED_PLATFORM_TYPES
            }
        deploy_agent = agent_factory.create_deploy_agent(
            entities, coordinates, coordinates_hm,
            plan_path=args.deployment_plan, launch_altitude=launch_altitude,
        )
        env.agent_manager.register_agent(deploy_agent)

        for round_index in range(args.total_rounds):
            agent_factory.reset_training_episode()
            initial_observation = env.reset()
            summary_builder = RunSummary(
                scenario=bundle.reward_policy.scenario_id if bundle.reward_policy else str(args.scenario),
                policies={"red": args.red_policy, "red_motion": args.red_motion_policy, "blue": blue_config.policy_name},
                reward_tracker=RewardTracker(bundle.reward_policy) if bundle.reward_policy else None,
            )
            summary_builder.start(initial_observation)
            final_observation = initial_observation
            termination_reason = "time_limit"
            env.red_model_deploy()
            start_time = time.perf_counter()
            for step in range(1, args.max_steps + 1):
                agent_factory.begin_training_step(env.get_observation())
                final_observation, _, done, _ = env.step()
                agent_factory.finish_training_step(final_observation)
                summary_builder.update(step, final_observation)
                if step % 200 == 0:
                    logging.info("Round %s, step %s", round_index + 1, step)
                if done:
                    termination_reason = "time_limit" if step >= args.max_steps else "environment_done"
                    break
            write_immediately()
            red_launched = sum(
                getattr(agent, "launch_step", -1) >= 0
                for agent in env.agent_manager.get_all_agents()
            )
            summary = summary_builder.build(
                final_observation, termination_reason=termination_reason, red_launched=red_launched,
            )
            filename = "summary.json" if args.total_rounds == 1 else f"summary_round_{round_index + 1}.json"
            path = summary_builder.write(write_config.output_dir, summary, filename)
            print("FINAL_SUMMARY " + json.dumps(summary, ensure_ascii=False))
            logging.info("Round %s finished in %.3f seconds; summary: %s",
                         round_index + 1, time.perf_counter() - start_time, path)

        args.red_learning_model.parent.mkdir(parents=True, exist_ok=True)
        agent_factory.save_model(str(args.red_learning_model))
        logging.info("Learning checkpoint saved: %s", args.red_learning_model)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            writer.close()
