"""Run PlanSpec-v2 with either original or InferenceSkill V13 numerics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from .checkpoint import EXPECTED_CHECKPOINT_SHA256, file_sha256
from .original import build_original_policy
from .policy import LocalAvoidControlPolicy


class _SkillEvasionAdapter(LocalAvoidControlPolicy):
    instances: list["_SkillEvasionAdapter"] = []

    def __init__(self, max_steps: int = 3000) -> None:
        super().__init__(max_steps=max_steps)
        self.instances.append(self)

    def load(self, checkpoint: str | Path) -> None:
        actual = file_sha256(Path(checkpoint))
        if actual != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(f"Unexpected V13 checkpoint SHA-256: {actual}")


class _OriginalEvasionAdapter:
    def __init__(self, *, snapshot: Path, max_steps: int = 3000) -> None:
        self.snapshot = snapshot
        self.max_steps = int(max_steps)
        self.delegate = None

    def load(self, checkpoint: str | Path) -> None:
        self.delegate = build_original_policy(
            self.snapshot, checkpoint, max_steps=self.max_steps)

    def set_training(self, training: bool) -> None:
        if training:
            raise ValueError("V13 plan replay is inference-only")

    def __getattr__(self, name):
        if self.delegate is None:
            raise AttributeError(name)
        return getattr(self.delegate, name)


def _resolve_scenario(snapshot: Path, value: str) -> Path:
    bundled = snapshot / "scenarios/cases" / value / "scenario.json"
    supplied = Path(value).expanduser()
    path = bundled if bundled.is_file() else supplied
    if not path.is_file():
        raise ValueError(f"Unknown Plan Anchor scenario: {value}")
    return path.resolve()


def _resolve_plan(release: Path, scenario: Path, value: str) -> Path:
    supplied = Path(value).expanduser()
    path = supplied if supplied.is_file() else release / "plans" / scenario.parent.name / f"{value}.json"
    if not path.is_file():
        raise ValueError(f"Unknown PlanSpec: {value}")
    return path.resolve()


def _max_steps(scenario: Path) -> int:
    metadata = scenario.with_name("case_info.json")
    return int(json.loads(metadata.read_text(encoding="utf-8"))["max_steps"])


def _install_lifecycle_bridge():
    """Bridge the published phased policy onto the bundled legacy TrainingEnv."""
    from envengine.agent_manager.agent_manager import AgentManager
    from envengine.environment.training_env import TrainingEnv
    from policies.red.learning.plan_skill import DeterministicPlanPolicy

    original_collect = AgentManager.collect_actions_from_agents
    original_reconcile = DeterministicPlanPolicy.reconcile_episode_telemetry
    added_methods = []
    added_training_methods = []

    def begin_environment_step(policy, full_observation):
        policy.begin_observation_phase(full_observation)

    def end_environment_step(_policy, _observation):
        return None

    def finish_environment_step(policy):
        policy.finish_commit_phase()

    for name, method in (("begin_environment_step", begin_environment_step),
                         ("end_environment_step", end_environment_step),
                         ("finish_environment_step", finish_environment_step)):
        if not hasattr(DeterministicPlanPolicy, name):
            setattr(DeterministicPlanPolicy, name, method)
            added_methods.append(name)

    if not hasattr(TrainingEnv, "build_episode_telemetry"):
        def build_episode_telemetry(_environment, score):
            return {
                "engine_lifecycle_available": False,
                "engine_lifecycle_events": [],
                "score": dict(score),
            }
        TrainingEnv.build_episode_telemetry = build_episode_telemetry
        added_training_methods.append("build_episode_telemetry")

    def reconcile_if_available(policy, telemetry):
        if not telemetry.get("engine_lifecycle_available", True):
            policy._engine_reconciled = False
            policy._engine_launch_ids = set()
            return
        original_reconcile(policy, telemetry)

    DeterministicPlanPolicy.reconcile_episode_telemetry = reconcile_if_available

    def collect_phased(self, agent_observations):
        plan_agents = [agent for agent in self._agents.values()
                       if isinstance(getattr(agent, "policy", None), DeterministicPlanPolicy)]
        if not plan_agents:
            return original_collect(self, agent_observations)
        policy = plan_agents[0].policy
        for agent_id, observation in agent_observations.items():
            agent = self._agents.get(agent_id)
            if agent is not None:
                agent.set_observation(observation)
        for agent in plan_agents:
            if agent.agent_id in agent_observations:
                agent.report()
        step = int(next(iter(agent_observations.values()), {}).get("step", 0))
        order = policy.begin_decision_phase(step, agent_observations.keys())
        actions = []
        records = []
        for agent_id in order:
            agent = self._agents.get(agent_id)
            if agent is None or agent_id not in agent_observations:
                continue
            commands = agent.get_action()
            actions.append(commands)
            records.append(SimpleNamespace(
                agent_id=agent_id, step=step, commands=commands))
        policy.end_decision_phase(step)
        satellite_commands, _receipt = policy.settle_satellite_requests()
        if satellite_commands:
            actions.append(np.asarray(satellite_commands, dtype=np.float64))
        policy.observe_decision_batch(SimpleNamespace(records=tuple(records)))
        return actions

    AgentManager.collect_actions_from_agents = collect_phased

    def restore() -> None:
        AgentManager.collect_actions_from_agents = original_collect
        DeterministicPlanPolicy.reconcile_episode_telemetry = original_reconcile
        for name in added_methods:
            delattr(DeterministicPlanPolicy, name)
        for name in added_training_methods:
            delattr(TrainingEnv, name)

    return restore


def run(args: argparse.Namespace) -> None:
    plan_root = args.plan_root.resolve(strict=True)
    release = plan_root / "github_release/plan_anchor_final20_v2"
    snapshot = release / "snapshot"
    legacy_core = plan_root / "core"
    scenario = _resolve_scenario(snapshot, args.scenario)
    plan = _resolve_plan(release, scenario, args.plan)
    checkpoint = (args.checkpoint or snapshot / "models/local_avoid_ppo_v13_gate_aligned_s3_best.pt").resolve(strict=True)
    max_steps = args.max_steps or _max_steps(scenario)
    output = (args.output_root / args.run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(snapshot / "core"), str(snapshot), str(legacy_core)]
    old_cwd = Path.cwd()
    restore_lifecycle = None
    os.chdir(snapshot / "core")
    try:
        import policies.red.skills.evasion_ppo as evasion_module

        if args.implementation == "skill":
            evasion_module.LocalAvoidRuntimePolicy = _SkillEvasionAdapter
        else:
            class OriginalFactory(_OriginalEvasionAdapter):
                def __init__(self, max_steps: int = 3000):
                    super().__init__(snapshot=snapshot, max_steps=max_steps)
            evasion_module.LocalAvoidRuntimePolicy = OriginalFactory

        spec = importlib.util.spec_from_file_location(
            "_local_avoid_plan_anchor_main", snapshot / "core/main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load Plan Anchor entry")
        plan_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plan_main)
        restore_lifecycle = _install_lifecycle_bridge()
        original_summary = plan_main.RunSummary

        def labeled_summary(*summary_args, **summary_kwargs):
            policies = dict(summary_kwargs.get("policies") or {})
            policies["red"] = f"planspec_v2:{plan.stem}"
            policies["red_motion"] = f"local_avoid_v13_{args.implementation}"
            summary_kwargs["policies"] = policies
            return original_summary(*summary_args, **summary_kwargs)

        plan_main.RunSummary = labeled_summary
        os.environ.update({
            "BLUE_POLICY": args.blue_policy,
            "BLUE_INTERCEPTOR_RATIO": str(args.blue_interceptor_ratio),
            "RED_POLICY": "r12_topformer",
            "RED_MOTION_POLICY": "local_avoid_s3",
            "EVASION_MODEL": str(checkpoint),
            "LOCAL_AVOID_GATE_MODE": "mission_v12",
            "BLUE_POLICY_SEED": str(args.seed),
            "RED_POLICY_SEED": str(args.seed),
            "SIMULATION_SEED": str(args.seed),
            "SEARCH_ROWS": "4",
            "SEARCH_COLS": "8",
            "PLAN_EARLY_STOP": "1",
        })
        core_args = list(args.core_args)
        if core_args[:1] == ["--"]:
            core_args = core_args[1:]
        sys.argv = ["main.py", "--scenario", str(scenario),
                    "--plan-spec", str(plan), "--output-dir", str(output),
                    "--max-steps", str(max_steps), "--total-rounds", str(args.total_rounds),
                    "--render-mode", args.render_mode, *core_args]
        plan_main.main()
    finally:
        if restore_lifecycle is not None:
            restore_lifecycle()
        for instance in _SkillEvasionAdapter.instances:
            instance.close()
        _SkillEvasionAdapter.instances.clear()
        os.chdir(old_cwd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--plan", default="A00_balanced")
    parser.add_argument("--implementation", choices=("original", "skill"), default="skill")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--blue-interceptor-ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--total-rounds", type=int, default=1)
    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument("core_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"PLAN_ANCHOR_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
