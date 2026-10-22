"""Run the H-missile composite below the published PlanSpec-v2 planner."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from .runtime_adapter import (
    HSingleManeuverRuntimePolicy,
    MissileTypeDispatchRuntimePolicy,
    default_profile_path,
)


BEST_B0_PLANS = {
    "E01": "A03_deploy_compact",
    "E02": "A10_l_screen",
    "E03": "A08_formation_dense",
    "M01": "A06_wave_rapid",
    "M02": "A04_deploy_wide",
    "M03": "A08_formation_dense",
    "H01": "A06_wave_rapid",
    "H02": "A01_deploy_forward",
    "H03": "A05_wave_mass",
}


def _environment_core(root: Path) -> Path:
    legacy = root / "core"
    if (legacy / "envengine").is_dir():
        return legacy
    if (root / "envengine").is_dir():
        return root
    raise ValueError(f"Environment root has no envengine package: {root}")


class _PreloadedPolicyProxy:
    def __init__(self, delegate, *, max_steps: int, checkpoint: Path) -> None:
        self._delegate = delegate
        self.max_steps = int(max_steps)
        self._checkpoint = checkpoint.resolve()

    def load(self, path: str | Path) -> None:
        if Path(path).resolve() != self._checkpoint:
            raise ValueError("preloaded local policy checkpoint changed")

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def _resolve_scenario(environment_root: Path, value: str) -> Path:
    bundled = environment_root / "scenarios/cases" / value / "scenario.json"
    supplied = Path(value).expanduser()
    path = bundled if bundled.is_file() else supplied
    if not path.is_file():
        raise ValueError(f"Unknown official scenario: {value}")
    return path.resolve()


def _resolve_plan(release: Path, scenario: Path, value: str) -> Path:
    plan_name = BEST_B0_PLANS.get(scenario.parent.name) if value == "auto" else value
    if plan_name is None:
        raise ValueError(f"No automatic B0 plan for scenario {scenario.parent.name}")
    supplied = Path(plan_name).expanduser()
    path = supplied if supplied.is_file() else release / "plans" / scenario.parent.name / f"{plan_name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown PlanSpec: {plan_name}")
    return path.resolve()


def _max_steps(scenario: Path) -> int:
    metadata = scenario.with_name("case_info.json")
    return int(json.loads(metadata.read_text(encoding="utf-8"))["max_steps"])


def _flatten_wave_releases(
    source: Path,
    destination: Path,
    *,
    concentrate_public_salvos: bool = False,
) -> Path:
    """Compile unavailable delayed launches into official auto-launch timing."""
    tokens = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(tokens, list):
        raise ValueError("PlanSpec must be a JSON token list")
    changed = False
    for token in tokens:
        if isinstance(token, dict) and token.get("op") == "WAVE":
            if token.get("release_step") != 0:
                token["release_step"] = 0
                changed = True
    if concentrate_public_salvos:
        public_assignments = [
            token
            for token in tokens
            if isinstance(token, dict)
            and token.get("op") == "ASSIGN"
            and token.get("goal_kind") == "public_target"
            and token.get("role") in ("H", "M")
        ]
        h_total = sum(
            int(token.get("count", 0))
            for token in public_assignments
            if token.get("role") == "H"
        )
        m_total = sum(
            int(token.get("count", 0))
            for token in public_assignments
            if token.get("role") == "M"
        )
        targets: list[tuple[int, int]] = []
        for token in public_assignments:
            target = (int(token["target_id"]), int(token["target_type"]))
            if target not in targets:
                targets.append(target)
        primary = [target for target in targets if target[1] == 9400]
        if h_total and targets and m_total and primary:
            tokens = [token for token in tokens if token not in public_assignments]
            h_base, h_extra = divmod(h_total, len(targets))
            replacements = []
            for index, (target_id, target_type) in enumerate(targets):
                count = h_base + (1 if index < h_extra else 0)
                if count:
                    replacements.append(
                        {
                            "count": count,
                            "goal_kind": "public_target",
                            "op": "ASSIGN",
                            "role": "H",
                            "target_id": target_id,
                            "target_type": target_type,
                            "wave_id": 1,
                        }
                    )
            m_base, m_extra = divmod(m_total, len(primary))
            for index, (target_id, target_type) in enumerate(primary):
                count = m_base + (1 if index < m_extra else 0)
                if count:
                    replacements.append(
                        {
                            "count": count,
                            "goal_kind": "public_target",
                            "op": "ASSIGN",
                            "role": "M",
                            "target_id": target_id,
                            "target_type": target_type,
                            "wave_id": 1,
                        }
                    )
            used_wave_ids = {
                int(token["wave_id"])
                for token in tokens
                if isinstance(token, dict)
                and token.get("op") == "ASSIGN"
                and "wave_id" in token
            }
            used_wave_ids.add(1)
            tokens = [
                token
                for token in tokens
                if not (
                    isinstance(token, dict)
                    and token.get("op") == "WAVE"
                    and int(token["wave_id"]) not in used_wave_ids
                )
            ]
            wave_index = next(
                index
                for index, token in enumerate(tokens)
                if token.get("op") == "WAVE" and int(token["wave_id"]) == 1
            )
            tokens[wave_index + 1 : wave_index + 1] = replacements
            changed = True
    if not changed:
        return source
    destination.write_text(
        json.dumps(tokens, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return destination


def _shift_wave_releases(source: Path, destination: Path, delay_steps: int) -> Path:
    if type(delay_steps) is not int or delay_steps < 0:
        raise ValueError("native wave delay must be a non-negative integer")
    if delay_steps == 0:
        return source
    tokens = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(tokens, list):
        raise ValueError("PlanSpec must be a JSON token list")
    for token in tokens:
        if isinstance(token, dict) and token.get("op") == "WAVE":
            token["release_step"] = int(token["release_step"]) + delay_steps
    destination.write_text(
        json.dumps(tokens, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return destination


def _configured_dispatch_policy(
    *,
    mode: str,
    profile: Path,
    local_factory,
    local_pulse_steps: int,
    joint_rule_profile: str,
    joint_release_window_steps: int,
    joint_max_members: int,
    mission_replan_mode: str,
):
    class ConfiguredDispatchPolicy(MissileTypeDispatchRuntimePolicy):
        def __init__(self, *, max_steps: int = 1000) -> None:
            super().__init__(
                mode=mode,
                max_steps=max_steps,
                h_profile=profile,
                local_factory=local_factory,
                local_pulse_steps=local_pulse_steps,
                joint_rule_profile=joint_rule_profile,
                joint_release_window_steps=joint_release_window_steps,
                joint_max_members=joint_max_members,
                mission_replan_mode=mission_replan_mode,
            )

    ConfiguredDispatchPolicy.__name__ = "ConfiguredDispatchPolicy"
    return ConfiguredDispatchPolicy


def _install_motion_module(policy_class) -> tuple[ModuleType, object]:
    """Replace only the snapshot's local runtime constructor for this process."""
    module = importlib.import_module("policies.red.skills.evasion_ppo")
    previous = module.LocalAvoidRuntimePolicy
    module.LocalAvoidRuntimePolicy = policy_class
    return module, previous


def _restore_motion_module(module: ModuleType, previous: object) -> None:
    module.LocalAvoidRuntimePolicy = previous


def _install_exact_motion_clock() -> tuple[ModuleType, int]:
    """Let the composite, rather than the unified agent, own action duration."""
    module = importlib.import_module("user_agents.unified_missile_agent")
    previous = int(module.MOTION_PULSE_STEPS)
    module.MOTION_PULSE_STEPS = 0
    return module, previous


def _install_physical_wave_launch():
    """Process-local bridge from command 200 to the native Launch method."""
    from envengine.common import SimmerCommandType

    modules = (
        importlib.import_module(
            "envengine.simulator.simlulator_impl.CompCruiseMissileHSimulator"
        ),
        importlib.import_module(
            "envengine.simulator.simlulator_impl.CompCruiseMissileMSimulator"
        ),
        importlib.import_module(
            "envengine.simulator.simlulator_impl.CompCruiseMissileLSimulator"
        ),
    )
    specifications = (
        (modules[0].CompCruiseMissileHSimulator, False),
        (modules[1].CompCruiseMissileMSimulator, False),
        (modules[2].CompCruiseMissileLSimulator, True),
    )
    originals = []

    for module, (simulator_class, launch_uses_time) in zip(modules, specifications):
        original_auto_launch = simulator_class._auto_launch
        original_command_received = simulator_class.command_received
        originals.append(
            (simulator_class, original_auto_launch, original_command_received)
        )

        def no_auto_launch(self) -> None:
            return None

        def command_received(
            self,
            command,
            *,
            _original=original_command_received,
            _module=module,
            _launch_uses_time=launch_uses_time,
        ) -> None:
            if command.commandTypeId != SimmerCommandType.MISSILE_LAUNCH:
                _original(self, command)
                return
            already_launched = self.launch >= 0 if _launch_uses_time else self.launch != 0
            if already_launched:
                return
            target = command.commandAttributes["target"]
            target_lla = _module.Vector3D(
                float(target["x"]),
                float(target["y"]),
                float(target.get("z", 0.0)),
            )
            self.model.Launch(target_lla * _module.ISimulator.MODEL_SCALE)
            self.launch = self.sim_time if _launch_uses_time else 1

        simulator_class._auto_launch = no_auto_launch
        simulator_class.command_received = command_received

    def restore() -> None:
        for simulator_class, auto_launch, command_received in originals:
            simulator_class._auto_launch = auto_launch
            simulator_class.command_received = command_received

    return restore


def _retarget_auto_launched(commands):
    """Translate the planner's launch row into the current engine's target ABI.

    Current official H/M/L simulators launch during reset and no longer accept
    command type 200.  They do accept CHANGE_TARGET, so the PlanSpec goal is
    preserved without changing simulator code or relaunching an entity.
    """
    values = np.asarray(commands, dtype=np.float64)
    if values.size == 0:
        return values
    original_shape = values.shape
    rows = values.reshape(1, -1) if values.ndim == 1 else values.copy()
    rows[rows[:, 0] == 1.0, 0] = 2.0
    return rows.reshape(original_shape)


def _install_lifecycle_bridge(*, auto_launch_retarget: bool):
    from envengine.agent_manager.agent_manager import AgentManager
    from envengine.environment.training_env import TrainingEnv
    from policies.red.learning.plan_skill import DeterministicPlanPolicy

    original_collect = AgentManager.collect_actions_from_agents
    original_set_shared_policy = TrainingEnv.set_shared_learning_policy
    original_reconcile = DeterministicPlanPolicy.reconcile_episode_telemetry
    original_select_decision = DeterministicPlanPolicy.select_decision
    added_policy_methods: list[str] = []
    added_environment_methods: list[str] = []

    def begin_environment_step(policy, full_observation):
        policy.begin_observation_phase(full_observation)

    def end_environment_step(policy, observation):
        entities = (observation or {}).get("entities", {})
        for agent in getattr(policy, "_agents", ()):
            entity = entities.get(int(agent.entity_id))
            dead = entity is None or (
                float(entity.get("health", 0.0)) <= 0.0
                or not bool(entity.get("isVisible", False))
            )
            if dead and int(agent.launch_step) >= 0:
                agent.finalize_death()
                dispatcher = getattr(policy, "_mission_dispatcher", None)
                observe_terminal = getattr(dispatcher, "observe_terminal", None)
                if callable(observe_terminal):
                    observe_terminal(agent)
                agent.latest_observation = None

    def finish_environment_step(policy):
        policy.finish_commit_phase()

    for name, method in (
        ("begin_environment_step", begin_environment_step),
        ("end_environment_step", end_environment_step),
        ("finish_environment_step", finish_environment_step),
    ):
        if not hasattr(DeterministicPlanPolicy, name):
            setattr(DeterministicPlanPolicy, name, method)
            added_policy_methods.append(name)

    if (
        not hasattr(TrainingEnv, "_get_init_ship_observation")
        and hasattr(TrainingEnv, "get_public_targets_observation")
    ):
        TrainingEnv._get_init_ship_observation = (
            TrainingEnv.get_public_targets_observation
        )
        added_environment_methods.append("_get_init_ship_observation")

    if not hasattr(TrainingEnv, "build_episode_telemetry"):
        def build_episode_telemetry(_environment, score):
            return {
                "engine_lifecycle_available": False,
                "engine_lifecycle_events": [],
                "score": dict(score),
            }

        TrainingEnv.build_episode_telemetry = build_episode_telemetry
        added_environment_methods.append("build_episode_telemetry")

    def set_shared_policy_compat(environment, policy):
        if isinstance(policy, DeterministicPlanPolicy):
            environment._shared_learning_policy = policy
            return
        original_set_shared_policy(environment, policy)

    TrainingEnv.set_shared_learning_policy = set_shared_policy_compat

    def reconcile_if_available(policy, telemetry):
        if not telemetry.get("engine_lifecycle_available", True):
            policy._engine_reconciled = False
            policy._engine_launch_ids = set()
            return
        original_reconcile(policy, telemetry)

    DeterministicPlanPolicy.reconcile_episode_telemetry = reconcile_if_available

    def select_with_mission_filter(policy, **kwargs):
        decision = original_select_decision(policy, **kwargs)
        filter_decision = getattr(policy, "_mission_replan_filter", None)
        if callable(filter_decision):
            return filter_decision(
                plan_policy=policy,
                agent_id=int(kwargs["agent_id"]),
                decision=decision,
            )
        return decision

    DeterministicPlanPolicy.select_decision = select_with_mission_filter

    def collect_phased(self, agent_observations):
        plan_agents = [
            agent
            for agent in self._agents.values()
            if isinstance(getattr(agent, "policy", None), DeterministicPlanPolicy)
        ]
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
        shared_skills = {
            id(agent.evasion_skill): agent.evasion_skill
            for agent in plan_agents
            if getattr(agent, "evasion_skill", None) is not None
        }
        for skill in shared_skills.values():
            prepare = getattr(skill, "prepare_decision_step", None)
            if callable(prepare):
                prepare(
                    step=step,
                    decision_order=order,
                    agent_observations=agent_observations,
                    agents_by_id=self._agents,
                    bindings=policy.bindings,
                    entity_types=policy.entity_types,
                    plan_policy=policy,
                )
            filter_decision = getattr(skill, "filter_plan_decision", None)
            if callable(filter_decision):
                policy._mission_replan_filter = filter_decision
                policy._mission_dispatcher = skill
        actions = []
        records = []
        for agent_id in order:
            agent = self._agents.get(agent_id)
            if agent is None or agent_id not in agent_observations:
                continue
            commands = agent.get_action()
            if auto_launch_retarget:
                commands = _retarget_auto_launched(commands)
            actions.append(commands)
            records.append(SimpleNamespace(agent_id=agent_id, step=step, commands=commands))
        policy.end_decision_phase(step)
        satellite_commands, _receipt = policy.settle_satellite_requests()
        if satellite_commands:
            actions.append(np.asarray(satellite_commands, dtype=np.float64))
        batch = SimpleNamespace(records=tuple(records))
        policy.observe_decision_batch(batch)
        dispatcher = getattr(policy, "_mission_dispatcher", None)
        observe_commands = getattr(dispatcher, "observe_command_batch", None)
        if callable(observe_commands):
            observe_commands(tuple(records), policy)
        return actions

    AgentManager.collect_actions_from_agents = collect_phased

    def restore() -> None:
        AgentManager.collect_actions_from_agents = original_collect
        TrainingEnv.set_shared_learning_policy = original_set_shared_policy
        DeterministicPlanPolicy.reconcile_episode_telemetry = original_reconcile
        DeterministicPlanPolicy.select_decision = original_select_decision
        if hasattr(DeterministicPlanPolicy, "_mission_replan_filter"):
            delattr(DeterministicPlanPolicy, "_mission_replan_filter")
        for name in added_policy_methods:
            delattr(DeterministicPlanPolicy, name)
        for name in added_environment_methods:
            delattr(TrainingEnv, name)

    return restore


def run(args: argparse.Namespace) -> None:
    environment_root = args.environment_root.resolve(strict=True)
    environment_core = _environment_core(environment_root)
    plan_root = args.plan_root.resolve(strict=True)
    release = plan_root / "github_release/plan_anchor_final20_v2"
    snapshot = release / "snapshot"
    scenario = _resolve_scenario(environment_root, args.scenario)
    source_plan = _resolve_plan(release, scenario, args.plan)
    profile = (args.profile or default_profile_path()).resolve(strict=True)
    local_checkpoint = (
        args.local_checkpoint
        or snapshot / "models/local_avoid_ppo_v13_gate_aligned_s3_best.pt"
    ).resolve(strict=True)
    if args.local_pulse_steps < 1:
        raise ValueError("--local-pulse-steps must be a positive integer")
    if args.native_wave_start_delay < 0:
        raise ValueError("--native-wave-start-delay must be non-negative")
    max_steps = args.max_steps or _max_steps(scenario)
    output = (args.output_root / args.run_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*/summary.json")):
        raise ValueError(f"Run ID already has a completed summary: {args.run_id}")
    patched_waves = args.launch_mode == "patched-waves"
    native_waves = args.launch_mode == "native-waves"
    auto_launch_mode = args.launch_mode == "official-auto"
    auto_launch_retarget = args.auto_launch_retarget and auto_launch_mode
    auto_launch_salvos = args.auto_launch_salvos and auto_launch_mode
    if auto_launch_retarget:
        plan = _flatten_wave_releases(
            source_plan,
            output / "effective_plan.json",
            concentrate_public_salvos=auto_launch_salvos,
        )
    elif native_waves and args.native_wave_start_delay:
        plan = _shift_wave_releases(
            source_plan,
            output / "effective_plan.json",
            args.native_wave_start_delay,
        )
    else:
        plan = source_plan

    sys.dont_write_bytecode = True
    inserted_paths = list(dict.fromkeys(
        [
            str(snapshot / "core"),
            str(snapshot),
            str(environment_core),
            str(environment_root),
        ]
    ))
    sys.path[:0] = inserted_paths
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    restore_lifecycle = None
    motion_parent = None
    previous_motion_module = None
    motion_clock_module = None
    previous_motion_clock = None
    restore_wave_launch = None
    environment_updates = {
        "BLUE_POLICY": args.blue_policy,
        "BLUE_INTERCEPTOR_RATIO": str(args.blue_interceptor_ratio),
        "RED_POLICY": "r12_topformer",
        "RED_MOTION_POLICY": "local_avoid_s3",
        "EVASION_MODEL": str(local_checkpoint),
        "LOCAL_AVOID_GATE_MODE": args.local_gate_mode,
        "BLUE_POLICY_SEED": str(args.seed),
        "RED_POLICY_SEED": str(args.seed),
        "SIMULATION_SEED": str(args.seed),
        "SEARCH_ROWS": "4",
        "SEARCH_COLS": "8",
        "PLAN_EARLY_STOP": "1",
        "H_SINGLE_MANEUVER_TRACE": str(output / "maneuver_decisions.jsonl"),
        "THREE_MISSILE_JOINT_TRACE": str(output / "joint_decisions.jsonl"),
        "L_SEARCH_COORDINATION_TRACE": str(output / "l_search_status.jsonl"),
        "L_RESPONSE_PACKAGE_TRACE": str(output / "l_response_status.jsonl"),
        "MISSION_REPLAN_TRACE": str(output / "mission_replan.jsonl"),
        "M_STRIKE_PACKAGE_TRACE": str(output / "m_strike_status.jsonl"),
    }
    old_environment = {name: os.environ.get(name) for name in environment_updates}
    os.environ.update(environment_updates)
    os.chdir(snapshot / "core")
    try:
        evasion_module = importlib.import_module("policies.red.skills.evasion_ppo")
        if args.local_runtime_root is not None:
            from local_avoid_v13_skill.checkpoint import (
                EXPECTED_CHECKPOINT_SHA256,
                file_sha256,
            )
            from local_avoid_v13_skill.original import (
                build_external_policy,
                build_original_policy,
            )

            builder = (
                build_original_policy
                if file_sha256(local_checkpoint) == EXPECTED_CHECKPOINT_SHA256
                else build_external_policy
            )
            preloaded_policy = builder(
                args.local_runtime_root.resolve(strict=True),
                local_checkpoint,
                max_steps=max_steps,
            )

            def local_factory(*, max_steps: int):
                return _PreloadedPolicyProxy(
                    preloaded_policy,
                    max_steps=max_steps,
                    checkpoint=local_checkpoint,
                )
        else:
            local_factory = evasion_module.LocalAvoidRuntimePolicy
        dispatch_policy = _configured_dispatch_policy(
            mode=args.lower_policy,
            profile=profile,
            local_factory=local_factory,
            local_pulse_steps=args.local_pulse_steps,
            joint_rule_profile=args.joint_rule_profile,
            joint_release_window_steps=args.joint_release_window_steps,
            joint_max_members=args.joint_max_members,
            mission_replan_mode=args.mission_replan,
        )
        motion_parent, previous_motion_module = _install_motion_module(
            dispatch_policy
        )
        spec = importlib.util.spec_from_file_location(
            "_h_single_maneuver_plan_main", snapshot / "core/main.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the PlanSpec entry")
        plan_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plan_main)
        if patched_waves:
            restore_wave_launch = _install_physical_wave_launch()
        motion_clock_module, previous_motion_clock = _install_exact_motion_clock()
        restore_lifecycle = _install_lifecycle_bridge(
            auto_launch_retarget=auto_launch_retarget
        )
        original_summary = plan_main.RunSummary

        def labeled_summary(*summary_args, **summary_kwargs):
            policies = dict(summary_kwargs.get("policies") or {})
            suffix = ":patched_waves" if patched_waves else ""
            if native_waves:
                suffix = ":native_waves"
                if args.native_wave_start_delay:
                    suffix += f":delay_{args.native_wave_start_delay}"
            if auto_launch_retarget:
                suffix += ":auto_launch_retarget"
            if auto_launch_retarget and auto_launch_salvos:
                suffix += ":public_salvos"
            policies["red"] = f"planspec_v2:{source_plan.stem}{suffix}"
            policies["red_motion"] = (
                "hybrid_h_rule_ml_local_v13"
                if args.lower_policy == "hybrid"
                else (
                    "joint_h_three_missile_v13"
                    if args.lower_policy == "joint-h-v13"
                    else (
                        "joint_l_local"
                        if args.lower_policy == "joint-l-local"
                        else (
                            "joint_hl_local"
                            if args.lower_policy == "joint-hl-local"
                            else (
                                "joint_m_local"
                                if args.lower_policy == "joint-m-local"
                                else (
                                    "joint_hml_local"
                                    if args.lower_policy == "joint-hml-local"
                                    else "local_v13_all"
                                )
                            )
                        )
                    )
                )
            )
            policies["local_gate_mode"] = args.local_gate_mode
            policies["local_checkpoint"] = str(local_checkpoint)
            if args.local_runtime_root is not None:
                policies["local_runtime_root"] = str(
                    args.local_runtime_root.resolve()
                )
            if args.lower_policy == "joint-h-v13":
                policies["joint_rule_profile"] = args.joint_rule_profile
            if args.mission_replan != "off":
                policies["mission_replan"] = args.mission_replan
            summary_kwargs["policies"] = policies
            return original_summary(*summary_args, **summary_kwargs)

        plan_main.RunSummary = labeled_summary
        core_args = list(args.core_args)
        if core_args[:1] == ["--"]:
            core_args = core_args[1:]
        sys.argv = [
            "main.py",
            "--scenario",
            str(scenario),
            "--plan-spec",
            str(plan),
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
        plan_main.main()
    finally:
        if restore_lifecycle is not None:
            restore_lifecycle()
        if restore_wave_launch is not None:
            restore_wave_launch()
        for instance in MissileTypeDispatchRuntimePolicy.instances:
            instance.close()
        MissileTypeDispatchRuntimePolicy.instances.clear()
        for instance in HSingleManeuverRuntimePolicy.instances:
            instance.close()
        HSingleManeuverRuntimePolicy.instances.clear()
        if motion_clock_module is not None and previous_motion_clock is not None:
            motion_clock_module.MOTION_PULSE_STEPS = previous_motion_clock
        if motion_parent is not None:
            _restore_motion_module(motion_parent, previous_motion_module)
        os.chdir(old_cwd)
        sys.argv = old_argv
        del sys.path[: len(inserted_paths)]
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--plan", default="auto")
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--lower-policy",
        choices=(
            "hybrid",
            "local-all",
            "joint-h-v13",
            "joint-l-local",
            "joint-hl-local",
            "joint-m-local",
            "joint-hml-local",
        ),
        default="hybrid",
        help="hybrid: H rule; local-all: all local policy; joint-h-v13: 2-6 H packages coordinate over the local policy",
    )
    parser.add_argument("--local-checkpoint", type=Path)
    parser.add_argument("--local-runtime-root", type=Path)
    parser.add_argument("--local-gate-mode", default="mission_v12")
    parser.add_argument("--local-pulse-steps", type=int, default=10)
    parser.add_argument(
        "--joint-rule-profile",
        choices=("safe", "experimental", "mission-aware"),
        default="safe",
    )
    parser.add_argument("--joint-release-window-steps", type=int, default=10)
    parser.add_argument("--joint-max-members", type=int, default=6)
    parser.add_argument(
        "--mission-replan",
        choices=("off", "l-response", "all-packages"),
        default="off",
    )
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--blue-interceptor-ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--launch-mode",
        choices=("official-auto", "patched-waves", "native-waves"),
        default="official-auto",
        help="official-auto preserves current auto-launch; patched-waves installs a command-200 bridge; native-waves uses an older environment's own command-gated launch",
    )
    parser.add_argument(
        "--native-wave-start-delay",
        type=int,
        default=0,
        help="shift every source PlanSpec wave for old engines whose step-0 clock is not initialized",
    )
    parser.add_argument(
        "--auto-launch-retarget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="translate PlanSpec launch rows to CHANGE_TARGET for current official auto-launch simulators",
    )
    parser.add_argument(
        "--auto-launch-salvos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="concentrate H across public targets and M across 9400 targets for B0 saturation",
    )
    parser.add_argument("--render-mode", choices=("none", "human"), default="none")
    parser.add_argument("core_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"H_SINGLE_MANEUVER_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
