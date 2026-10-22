"""Run the release environment's native R9 above an external local policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .original import build_external_policy


def _scenario_path(root: Path, value: str) -> Path:
    bundled = root / "scenarios/cases" / value / "scenario.json"
    supplied = Path(value).expanduser()
    path = bundled if bundled.is_file() else supplied
    if not path.is_file():
        raise ValueError(f"Unknown release scenario: {value}")
    return path.resolve()


def _deployment_plan(scenario: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = scenario.with_name("deployment_plan.example.json") if value == "preset" else Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"Deployment plan is unavailable: {path}")
    return path.resolve()


def _read_release_profile(path: str):
    """Load packaged legacy scenes into the release's dict-based comm schema."""
    from envengine import Profile

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in data.get("imagineProfile", {}).get("entityList", ()):
        entity = item.get("entity", {}) if isinstance(item, dict) else {}
        if isinstance(entity.get("commRangeInfo"), list):
            entity["commRangeInfo"] = {}
    return Profile.from_dict(data)


def run(args: argparse.Namespace) -> None:
    root = args.competition_root.resolve(strict=True)
    scenario = _scenario_path(root, args.scenario)
    deployment_plan = _deployment_plan(scenario, args.deployment_plan)
    checkpoint = args.checkpoint.resolve(strict=True)
    contest_root = args.jmy_contest_root.resolve(strict=True)
    output = (args.output_root / args.run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    old_cwd = Path.cwd()
    policy = None
    try:
        import main as release_main
        from user_agents.red import attack_missile_agent, factory

        release_main.read_profile = _read_release_profile

        max_steps = args.max_steps
        if max_steps is None:
            metadata = scenario.with_name("case_info.json")
            max_steps = int(json.loads(metadata.read_text(encoding="utf-8"))["max_steps"])
        policy = build_external_policy(contest_root, checkpoint, max_steps=max_steps)

        registered: set[int] = set()
        original_init = attack_missile_agent.AttackMissileAgent.__init__

        def patched_init(self, *agent_args, **agent_kwargs):
            original_init(self, *agent_args, **agent_kwargs)
            policy.register_agent(self.agent_id, self.entity_id, policy.max_steps)
            registered.add(self.agent_id)

        def patched_maneuver(self, actions, observation):
            if self.launch_step < 0:
                return
            target_id = self._commander.target_id_for(self.entity_id)
            target = next(
                (item for item in self._commander.targets if item.entity_id == target_id),
                None,
            )
            goal = target.position if target is not None else None
            encoded = policy.encode_motion_observation(
                observation,
                agent_id=self.agent_id,
                entity_id=self.entity_id,
                launch_step=self.launch_step,
                maneuver_state=self.set_acc_z_z,
                goal_position=goal,
            )
            mask = policy.prepare_action_mask(encoded, np.ones(3, dtype=np.bool_))
            action = int(policy.select_action(encoded, mask))
            maneuver = (-1.0, 0.0, 1.0)[action]
            actions.append([0, self.entity_id, maneuver, 0.0])
            self.set_acc_z_z = int(maneuver)

        def build_policy(name: str, **kwargs):
            if name != "ppo" or kwargs.get("training", False):
                raise ValueError("External R9 adapter accepts inference-only motion policy 'ppo'")
            return policy

        attack_missile_agent.AttackMissileAgent.__init__ = patched_init
        attack_missile_agent.AttackMissileAgent._set_acc_z_learning = patched_maneuver
        factory.build_learning_motion_policy = build_policy

        argv = [
            "--scenario", str(scenario),
            "--red-policy", "r9_hierarchical_learning",
            "--red-motion-policy", "ppo",
            "--red-learning-model", str(checkpoint),
            "--blue-policy", args.blue_policy,
            "--blue-interceptor-ratio", str(args.blue_interceptor_ratio),
            "--seed", str(args.seed),
            "--total-rounds", "1",
            "--max-steps", str(max_steps),
            "--render-mode", args.render_mode,
            "--output-dir", str(output),
        ]
        if deployment_plan is not None:
            argv += ["--deployment-plan", str(deployment_plan)]
        release_args = list(args.release_args)
        if release_args[:1] == ["--"]:
            release_args = release_args[1:]
        argv += release_args
        old_cwd = Path.cwd()
        __import__("os").chdir(root)
        release_main.main(argv)
    finally:
        if policy is not None:
            close = getattr(policy, "close", None)
            if callable(close):
                close()
        __import__("os").chdir(old_cwd)
        if sys.path and sys.path[0] == str(root):
            del sys.path[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--jmy-contest-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--deployment-plan", metavar="preset|PATH", default="preset")
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--blue-interceptor-ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument("release_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"LOCAL_AVOID_EXTERNAL_R9_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
