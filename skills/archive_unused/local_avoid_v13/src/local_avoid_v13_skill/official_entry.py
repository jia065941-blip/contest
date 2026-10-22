"""Explicit, process-local adapter for the unmodified official environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .model import default_bundle_path
from .original import build_original_policy
from .policy import LocalAvoidControlPolicy


def _scenario_path(root: Path, value: str) -> Path:
    bundled = root / "scenarios" / "cases" / value / "scenario.json"
    if bundled.is_file():
        return bundled.resolve()
    supplied = Path(value).expanduser()
    if supplied.is_file() and supplied.suffix.lower() == ".json":
        return supplied.resolve()
    raise ValueError(f"Unknown official scenario: {value}")


def _case_max_steps(scenario: Path) -> int:
    metadata = scenario.with_name("case_info.json")
    if not metadata.is_file():
        return 1000
    return int(json.loads(metadata.read_text(encoding="utf-8"))["max_steps"])


def _deployment_plan(scenario: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    if value == "preset":
        plan = scenario.with_name("deployment_plan.example.json")
    else:
        plan = Path(value).expanduser()
    if not plan.is_file():
        raise ValueError(f"Deployment plan is unavailable: {plan}")
    return plan.resolve()


def run_official(args: argparse.Namespace) -> None:
    root = args.competition_root.resolve(strict=True)
    core = root / "core"
    scenario = _scenario_path(root, args.scenario)
    max_steps = args.max_steps or _case_max_steps(scenario)
    deployment_plan = _deployment_plan(scenario, args.deployment_plan)
    if args.red_policy != "r9_hierarchical_learning":
        raise ValueError("local_avoid_v13 official runs require r9_hierarchical_learning")
    output = (args.output_root / args.run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    for location in (root, core):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    old_cwd = Path.cwd()
    os.chdir(core)
    policy = None
    try:
        import main as official_main
        from user_agents.attack_missile_agent import AttackMissileAgent as OfficialAgent

        original_summary = official_main.RunSummary

        def build_policy(name: str, *, seed=None, model_path=None, max_steps=1000,
                         training=False, observation_dim=85):
            del seed, model_path, observation_dim
            nonlocal policy
            if name != "ppo" or training:
                raise ValueError("V13 adapter accepts inference-only internal policy 'ppo'")
            if args.implementation == "skill":
                policy = LocalAvoidControlPolicy(args.bundle, max_steps=max_steps)
            else:
                if args.jmy_contest_root is None or args.checkpoint is None:
                    raise ValueError("original mode requires --jmy-contest-root and --checkpoint")
                policy = build_original_policy(
                    args.jmy_contest_root, args.checkpoint, max_steps=max_steps)
            return policy

        class V13Agent(OfficialAgent):
            def __init__(self, *agent_args, **agent_kwargs):
                super().__init__(*agent_args, **agent_kwargs)
                if isinstance(self.learning_policy, LocalAvoidControlPolicy):
                    self.learning_policy.register_agent(self.agent_id, self.entity_id)
                else:
                    self.learning_policy.register_agent(
                        self.agent_id, self.entity_id, self.learning_policy.max_steps)

            def _set_acc_z_learning(self, actions, observation):
                if self.launch_step < 0:
                    return
                target_id = self.commander.target_id_for(self.entity_id)
                target = next((item for item in self.commander.targets
                               if item.entity_id == target_id), None)
                goal = target.position if target is not None else None
                if isinstance(self.learning_policy, LocalAvoidControlPolicy):
                    action = self.learning_policy.action_for(
                        agent_id=self.agent_id, entity_id=self.entity_id,
                        observation=observation, launch_step=self.launch_step,
                        maneuver_state=self.set_acc_z_z, goal_position=goal)
                else:
                    encoded = self.learning_policy.encode_motion_observation(
                        observation, agent_id=self.agent_id, entity_id=self.entity_id,
                        launch_step=self.launch_step, maneuver_state=self.set_acc_z_z,
                        goal_position=goal)
                    mask = self.learning_policy.prepare_action_mask(
                        encoded, np.ones(3, dtype=np.bool_))
                    action = self.learning_policy.select_action(encoded, mask)
                maneuver = {0: -1.0, 1: 0.0, 2: 1.0}[action]
                actions.append([0, self.entity_id, maneuver, 0.0])
                self.set_acc_z_z = int(maneuver)

            def reset(self):
                super().reset()
                if isinstance(self.learning_policy, LocalAvoidControlPolicy):
                    self.learning_policy.reset_agent(self.agent_id)
                else:
                    encoder = self.learning_policy._encoders.get(self.agent_id)
                    if encoder is not None:
                        encoder.reset()
                    self.learning_policy._recovery_pending.pop(self.agent_id, None)

        def labeled_summary(*summary_args, **summary_kwargs):
            policies = dict(summary_kwargs.get("policies") or {})
            if policies.get("red_motion") == "ppo":
                policies["red_motion"] = (
                    "local_avoid_v13" if args.implementation == "skill"
                    else "local_avoid_v13_original")
            summary_kwargs["policies"] = policies
            return original_summary(*summary_args, **summary_kwargs)

        official_main.build_learning_motion_policy = build_policy
        official_main.AttackMissileAgent = V13Agent
        official_main.RunSummary = labeled_summary
        os.environ.update({
            "BLUE_POLICY": args.blue_policy,
            "RED_POLICY": args.red_policy,
            "RED_MOTION_POLICY": "ppo",
            "RED_LEARNING_MODEL": str(Path(args.bundle).resolve()),
            "RED_LEARNING_TRAIN": "0",
            "BLUE_POLICY_SEED": str(args.seed),
            "RED_POLICY_SEED": str(args.seed),
            "SIMULATION_SEED": str(args.seed),
            "LOCAL_AVOID_GATE_MODE": "mission_v12",
        })
        reward_policy = official_main.load_reward_policy(str(scenario))
        if reward_policy is not None:
            os.environ["BLUE_ASSET_VALUES"] = json.dumps(
                dict(reward_policy.objective_weights))
        core_args = list(args.core_args)
        if core_args[:1] == ["--"]:
            core_args = core_args[1:]
        deployment_args = (["--deployment-plan", str(deployment_plan)]
                           if deployment_plan is not None else [])
        sys.argv = ["main.py", "--scenario", str(scenario), "--output-dir", str(output),
                    "--max-steps", str(max_steps), "--total-rounds", str(args.total_rounds),
                    "--render-mode", args.render_mode, *deployment_args, *core_args]
        official_main.main()
    finally:
        if isinstance(policy, LocalAvoidControlPolicy):
            policy.close()
        os.chdir(old_cwd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--competition-root", type=Path, required=True)
    run.add_argument("--bundle", type=Path, default=default_bundle_path())
    run.add_argument("--implementation", choices=("skill", "original"), default="skill")
    run.add_argument("--jmy-contest-root", type=Path)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--scenario", required=True)
    run.add_argument("--deployment-plan", metavar="preset|PATH",
                     help="Use 'preset' for the selected Final20 case plan, or an explicit JSON path")
    run.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    run.add_argument("--red-policy", default="r9_hierarchical_learning")
    run.add_argument("--seed", type=int, default=701)
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-root", type=Path, required=True,
                     help="Explicit result root; keep it outside the official checkout")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--total-rounds", type=int, default=1)
    run.add_argument("--render-mode", choices=("none", "human"), default="none")
    run.add_argument("core_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_official(args)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"LOCAL_AVOID_V13_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
