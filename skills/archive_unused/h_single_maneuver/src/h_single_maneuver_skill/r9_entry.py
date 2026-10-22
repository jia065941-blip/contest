"""Run R9 with either H-only rules or the original V13 for every missile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .runtime_adapter import HSingleManeuverRuntimePolicy, default_profile_path


def _scenario_path(root: Path, value: str) -> Path:
    bundled = root / "scenarios/cases" / value / "scenario.json"
    supplied = Path(value).expanduser()
    path = bundled if bundled.is_file() else supplied
    if not path.is_file():
        raise ValueError(f"Unknown scenario: {value}")
    return path.resolve()


def _max_steps(scenario: Path) -> int:
    metadata = scenario.with_name("case_info.json")
    return int(json.loads(metadata.read_text(encoding="utf-8"))["max_steps"])


def run(args: argparse.Namespace) -> None:
    from local_avoid_v13_skill.original import build_original_policy

    root = args.environment_root.resolve(strict=True)
    core = (
        root / "core"
        if (root / "core/envengine").is_dir()
        else root
    )
    scenario = _scenario_path(root, args.scenario)
    profile = (args.profile or default_profile_path()).resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    contest_root = args.jmy_contest_root.resolve(strict=True)
    max_steps = args.max_steps or _max_steps(scenario)
    output = (args.output_root / args.run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)

    sys.dont_write_bytecode = True
    inserted = [str(root), str(core)]
    sys.path[:0] = inserted
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    old_environment = {
        name: os.environ.get(name)
        for name in (
            "BLUE_POLICY",
            "BLUE_INTERCEPTOR_RATIO",
            "RED_POLICY",
            "RED_MOTION_POLICY",
            "RED_LEARNING_MODEL",
            "RED_LEARNING_TRAIN",
            "BLUE_POLICY_SEED",
            "RED_POLICY_SEED",
            "SIMULATION_SEED",
            "LOCAL_AVOID_GATE_MODE",
            "H_SINGLE_MANEUVER_TRACE",
        )
    }
    h_policy = HSingleManeuverRuntimePolicy(max_steps=max_steps)
    h_policy.load(profile)
    local_policy = build_original_policy(
        contest_root, checkpoint, max_steps=max_steps
    )
    os.chdir(core)
    try:
        import main as official_main
        from user_agents.attack_missile_agent import AttackMissileAgent as OfficialAgent

        original_summary = official_main.RunSummary

        def build_policy(name: str, **_kwargs):
            if name != "ppo":
                raise ValueError("R9 comparison uses the internal ppo slot")
            return local_policy

        class HybridAgent(OfficialAgent):
            def __init__(self, *agent_args, **agent_kwargs):
                super().__init__(*agent_args, **agent_kwargs)
                local_policy.register_agent(
                    self.agent_id, self.entity_id, local_policy.max_steps
                )

            def _set_acc_z_learning(self, actions, observation):
                if self.launch_step < 0:
                    return
                target_id = self.commander.target_id_for(self.entity_id)
                target = next(
                    (
                        item
                        for item in self.commander.targets
                        if item.entity_id == target_id
                    ),
                    None,
                )
                goal = target.position if target is not None else None
                entity_type = int((observation.get("self") or {}).get("type", -1))
                if args.lower_policy == "hybrid" and entity_type == 21000:
                    encoded = h_policy.encode_motion_observation(
                        observation,
                        agent_id=self.agent_id,
                        entity_id=self.entity_id,
                        launch_step=self.launch_step,
                        maneuver_state=self.set_acc_z_z,
                        goal_position=goal,
                    )
                    mask = h_policy.prepare_action_mask(
                        encoded, np.ones(3, dtype=np.bool_)
                    )
                    action = h_policy.select_action(encoded, mask)
                else:
                    encoded = local_policy.encode_motion_observation(
                        observation,
                        agent_id=self.agent_id,
                        entity_id=self.entity_id,
                        launch_step=self.launch_step,
                        maneuver_state=self.set_acc_z_z,
                        goal_position=goal,
                    )
                    mask = local_policy.prepare_action_mask(
                        encoded, np.ones(3, dtype=np.bool_)
                    )
                    action = local_policy.select_action(encoded, mask)
                maneuver = {0: -1.0, 1: 0.0, 2: 1.0}[int(action)]
                actions.append([0, self.entity_id, maneuver, 0.0])
                self.set_acc_z_z = int(maneuver)

            def reset(self):
                super().reset()
                encoder = local_policy._encoders.get(self.agent_id)
                if encoder is not None:
                    encoder.reset()
                local_policy._recovery_pending.pop(self.agent_id, None)

        def labeled_summary(*summary_args, **summary_kwargs):
            policies = dict(summary_kwargs.get("policies") or {})
            policies["red_motion"] = (
                "hybrid_h_rule_ml_local_v13"
                if args.lower_policy == "hybrid"
                else "local_v13_all"
            )
            summary_kwargs["policies"] = policies
            return original_summary(*summary_args, **summary_kwargs)

        official_main.build_learning_motion_policy = build_policy
        official_main.AttackMissileAgent = HybridAgent
        official_main.RunSummary = labeled_summary
        os.environ.update(
            {
                "BLUE_POLICY": args.blue_policy,
                "BLUE_INTERCEPTOR_RATIO": str(args.blue_interceptor_ratio),
                "RED_POLICY": "r9_hierarchical_learning",
                "RED_MOTION_POLICY": "ppo",
                "RED_LEARNING_MODEL": str(checkpoint),
                "RED_LEARNING_TRAIN": "0",
                "BLUE_POLICY_SEED": str(args.seed),
                "RED_POLICY_SEED": str(args.seed),
                "SIMULATION_SEED": str(args.seed),
                "LOCAL_AVOID_GATE_MODE": "mission_v12",
                "H_SINGLE_MANEUVER_TRACE": str(output / "maneuver_decisions.jsonl"),
            }
        )
        core_args = list(args.core_args)
        if core_args[:1] == ["--"]:
            core_args = core_args[1:]
        sys.argv = [
            "main.py",
            "--scenario",
            str(scenario),
            "--output-dir",
            str(output),
            "--max-steps",
            str(max_steps),
            "--total-rounds",
            "1",
            "--render-mode",
            args.render_mode,
            *core_args,
        ]
        official_main.main()
    finally:
        h_policy.close()
        os.chdir(old_cwd)
        sys.argv = old_argv
        del sys.path[: len(inserted)]
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--jmy-contest-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--lower-policy", choices=("hybrid", "local-all"), required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--blue-interceptor-ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument("core_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"H_SINGLE_R9_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
