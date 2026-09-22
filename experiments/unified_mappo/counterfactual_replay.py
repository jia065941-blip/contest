"""Deterministic subprocess orchestration for trajectory counterfactuals.

The factual episode is the sole on-policy rollout.  This module launches
read-only probe episodes from the same seeds/checkpoint, verifies that every
requested intervention was applied, and converts measured target-window
returns into the target-conditioned credit objects consumed by MAPPO.

The simulator has no serializable native snapshot API. Each probe reconstructs
the factual prefix once, then uses an OS copy-on-write fork at the requested
decision timestep: the child keeps the factual joint action and the parent
applies NULL to the requested entity. No state/RNG/action hash comparison is
required; the manifest records the actual branch and intervention.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from experiments.unified_mappo.trajectory_credit import (
    CreditCandidate,
    TargetRewardEvent,
    TrajectoryCreditResult,
    assign_trajectory_credit,
)


SCHEMA_VERSION = 1
TRAJECTORY_REWARD_MODE = "weighted_damage_trajectory_counterfactual"
DEFAULT_PYTHON = Path("/opt/conda/envs/competition/bin/python")
DEFAULT_MAIN = Path("core/main.py")


class ReplayInputError(ValueError):
    """Raised before launch when factual replay metadata is incomplete."""


class ReplayExecutionError(RuntimeError):
    """Raised when a counterfactual subprocess fails or emits bad output."""


class ReplayValidationError(RuntimeError):
    """Raised when a probe did not apply exactly its requested interventions."""


@dataclass(frozen=True, slots=True)
class ReplayIntervention:
    """A typed ``do(a_i <- NULL)`` operation."""

    entity_id: Any
    agent_id: Any
    timestep: int
    decision_type: str
    null_action: str

    @property
    def key(self) -> tuple[Any, int, str]:
        return (self.entity_id, self.timestep, self.decision_type)


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """One deterministic probe; a joint request may contain several NULLs."""

    request_id: str
    kind: str
    interventions: tuple[ReplayIntervention, ...]
    stop_step: int
    target_ids: tuple[Any, ...]
    factual_event_ids: tuple[Any, ...]

    @property
    def branch_timestep(self) -> int:
        return min(item.timestep for item in self.interventions)

    def to_spec(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "kind": self.kind,
            "reward_mode": TRAJECTORY_REWARD_MODE,
            "stop_step": self.stop_step,
            "target_ids": list(self.target_ids),
            "factual_event_ids": list(self.factual_event_ids),
            "interventions": [asdict(item) for item in self.interventions],
        }


@dataclass(frozen=True, slots=True)
class ReplayLaunch:
    """Fully resolved subprocess invocation passed to an injectable runner."""

    request: ReplayRequest
    command: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Path
    spec_path: Path
    result_path: Path
    stdout_path: Path
    stderr_path: Path
    timeout_seconds: float


class ReplayRunner(Protocol):
    def __call__(self, launch: ReplayLaunch) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FactualTargetRewardEvent:
    """Simulator-produced positive target reward awaiting causal assignment."""

    event_id: Any
    target_id: Any
    timestep: int
    reward: float
    candidates: tuple[CreditCandidate, ...]


@dataclass(frozen=True, slots=True)
class CounterfactualReplayOutcome:
    """Credit result and durable audit produced by all accepted probes."""

    credit: TrajectoryCreditResult
    manifest: Mapping[str, Any]
    manifest_path: Path
    single_replay_count: int
    joint_replay_count: int


class SubprocessReplayRunner:
    """Default probe runner used outside unit tests."""

    def __call__(self, launch: ReplayLaunch) -> Mapping[str, Any]:
        if launch.result_path.exists():
            raise ReplayExecutionError(
                f"refusing stale counterfactual result: {launch.result_path}"
            )
        completed = subprocess.run(
            list(launch.command),
            cwd=str(launch.cwd),
            env=dict(launch.environment),
            capture_output=True,
            text=True,
            timeout=launch.timeout_seconds,
            check=False,
        )
        launch.stdout_path.write_text(completed.stdout, encoding="utf-8")
        launch.stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise ReplayExecutionError(
                f"counterfactual probe {launch.request.request_id} exited "
                f"with code {completed.returncode}; see {launch.stderr_path}"
            )
        if not launch.result_path.is_file():
            raise ReplayExecutionError(
                f"counterfactual probe did not write {launch.result_path}"
            )
        try:
            payload = json.loads(launch.result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayExecutionError(
                f"invalid counterfactual result {launch.result_path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ReplayExecutionError("counterfactual result must be a JSON object")
        return payload


def run_counterfactual_replays(
    *,
    factual_events: Sequence[FactualTargetRewardEvent | Mapping[str, Any]],
    factual_target_rewards: Mapping[Any, Any] | Sequence[Any],
    scenario: str | Path,
    max_steps: int,
    frozen_checkpoint: str | Path,
    output_dir: str | Path,
    gamma: float,
    base_env: Mapping[str, str] | None = None,
    runner: ReplayRunner | None = None,
    python_executable: str | Path = DEFAULT_PYTHON,
    main_script: str | Path = DEFAULT_MAIN,
    timeout_seconds: float = 7200.0,
    epsilon: float = 1e-12,
) -> CounterfactualReplayOutcome:
    """Measure true counterfactuals and assign one-time trajectory credit.

    Single-removal probes are deduplicated by exactly
    ``(entity_id, tau, decision_type)`` and run far enough to serve every event
    that uses that decision.  A joint probe is launched only for an all-zero
    single-removal anomaly with at least one search/intercept/satellite or
    mixed-category candidate. Direct-damage-only redundancy uses measured-damage
    fallback in :func:`assign_trajectory_credit`.
    """

    events = tuple(_coerce_event(item) for item in factual_events)
    _validate_inputs(events, max_steps=max_steps, gamma=gamma, epsilon=epsilon)
    checkpoint = Path(frozen_checkpoint).resolve()
    scenario_path = Path(scenario).resolve()
    root = Path(output_dir).resolve()
    specs_dir = root / "counterfactual_replays" / "specs"
    results_dir = root / "counterfactual_replays" / "results"
    logs_dir = root / "counterfactual_replays" / "logs"
    for directory in (root, specs_dir, results_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "counterfactual_replay_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "reward_mode": TRAJECTORY_REWARD_MODE,
        "scenario": str(scenario_path),
        "max_steps": int(max_steps),
        "frozen_checkpoint": str(checkpoint),
        "gamma": float(gamma),
        "requests": [],
        "event_windows": [],
        "errors": [],
    }
    _write_json(manifest_path, manifest)

    effective_runner: ReplayRunner = runner or SubprocessReplayRunner()
    main_path = Path(main_script)
    if not main_path.is_absolute():
        cwd = Path(__file__).resolve().parents[2]
        main_path = cwd / main_path
    else:
        cwd = main_path.parent.parent

    try:
        single_requests, candidate_request_keys = _build_single_requests(events)
        single_results: dict[
            tuple[Any, int, str], Mapping[str, Any]
        ] = {}
        if runner is None:
            results_by_id = _run_request_batch(
                requests=single_requests,
                batch_name="single",
                base_env=base_env,
                scenario=scenario_path,
                checkpoint=checkpoint,
                max_steps=max_steps,
                output_root=root,
                specs_dir=specs_dir,
                results_dir=results_dir,
                logs_dir=logs_dir,
                python_executable=Path(python_executable),
                main_path=main_path,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
        else:
            results_by_id = {
                request.request_id: _run_request(
                    request=request,
                    runner=runner,
                    base_env=base_env,
                    scenario=scenario_path,
                    checkpoint=checkpoint,
                    max_steps=max_steps,
                    output_root=root,
                    specs_dir=specs_dir,
                    results_dir=results_dir,
                    logs_dir=logs_dir,
                    python_executable=Path(python_executable),
                    main_path=main_path,
                    cwd=cwd,
                    timeout_seconds=timeout_seconds,
                )
                for request in single_requests
            }
        for request in single_requests:
            result = results_by_id[request.request_id]
            single_results[request.interventions[0].key] = result
            manifest["requests"].append(
                _request_audit(request, result)
            )

        prepared: dict[Any, dict[str, Any]] = {}
        joint_groups: dict[
            tuple[tuple[Any, int, str], ...], dict[str, Any]
        ] = {}
        for event in events:
            raw = _measure_single_event(
                event,
                factual_target_rewards=factual_target_rewards,
                single_results=single_results,
                candidate_request_keys=candidate_request_keys,
                epsilon=epsilon,
            )
            prepared[event.event_id] = raw
            if raw["all_single_deltas_zero"] and not raw["direct_damage_only"]:
                joint_key = tuple(
                    sorted(
                        (candidate_request_keys[(event.event_id, candidate.entity_id)]
                         for candidate in event.candidates),
                        key=_stable_key,
                    )
                )
                group = joint_groups.setdefault(
                    joint_key,
                    {"events": [], "stop_step": 0, "target_ids": set()},
                )
                group["events"].append(event)
                group["stop_step"] = max(group["stop_step"], event.timestep)
                group["target_ids"].add(event.target_id)

        joint_results: dict[
            tuple[tuple[Any, int, str], ...], Mapping[str, Any]
        ] = {}
        joint_requests: dict[
            tuple[tuple[Any, int, str], ...], ReplayRequest
        ] = {}
        for joint_key, group in joint_groups.items():
            interventions = tuple(
                _intervention_for_key(
                    key,
                    events=events,
                    candidate_request_keys=candidate_request_keys,
                )
                for key in joint_key
            )
            joint_requests[joint_key] = _make_request(
                kind="joint",
                interventions=interventions,
                stop_step=group["stop_step"],
                target_ids=tuple(
                    sorted(group["target_ids"], key=_stable_key)
                ),
                factual_event_ids=tuple(
                    item.event_id for item in group["events"]
                ),
            )
        if runner is None:
            joint_by_id = _run_request_batch(
                requests=tuple(joint_requests.values()),
                batch_name="joint",
                base_env=base_env,
                scenario=scenario_path,
                checkpoint=checkpoint,
                max_steps=max_steps,
                output_root=root,
                specs_dir=specs_dir,
                results_dir=results_dir,
                logs_dir=logs_dir,
                python_executable=Path(python_executable),
                main_path=main_path,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
        else:
            joint_by_id = {
                request.request_id: _run_request(
                    request=request,
                    runner=runner,
                    base_env=base_env,
                    scenario=scenario_path,
                    checkpoint=checkpoint,
                    max_steps=max_steps,
                    output_root=root,
                    specs_dir=specs_dir,
                    results_dir=results_dir,
                    logs_dir=logs_dir,
                    python_executable=Path(python_executable),
                    main_path=main_path,
                    cwd=cwd,
                    timeout_seconds=timeout_seconds,
                )
                for request in joint_requests.values()
            }
        for joint_key, request in joint_requests.items():
            result = joint_by_id[request.request_id]
            joint_results[joint_key] = result
            manifest["requests"].append(
                _request_audit(request, result)
            )

        credit_events: list[TargetRewardEvent] = []
        for event in events:
            raw = prepared[event.event_id]
            joint_return: float | None = None
            if raw["all_single_deltas_zero"] and not raw["direct_damage_only"]:
                joint_key = tuple(
                    sorted(
                        (candidate_request_keys[(event.event_id, candidate.entity_id)]
                         for candidate in event.candidates),
                        key=_stable_key,
                    )
                )
                joint_rewards = _extract_target_rewards(joint_results[joint_key])
                joint_start = min(candidate.tau for candidate in event.candidates)
                joint_raw_return = target_window_return(
                    joint_rewards,
                    target_id=event.target_id,
                    start_exclusive=joint_start - 1,
                    end_inclusive=event.timestep,
                )
                joint_delta = max(raw["common_factual_return"] - joint_raw_return, 0.0)
                # Express all returns against the common factual baseline.  For
                # equal-tau candidates this is the raw simulator return.  For
                # mixed taus it preserves each measured F_i-CF_i exactly while
                # satisfying TargetRewardEvent's single-baseline data model.
                joint_return = raw["common_factual_return"] - joint_delta
                raw["joint_raw_counterfactual_return"] = joint_raw_return
                raw["joint_delta"] = joint_delta

            credit_events.append(
                TargetRewardEvent(
                    event_id=event.event_id,
                    target_id=event.target_id,
                    timestep=event.timestep,
                    reward=event.reward,
                    factual_return=raw["common_factual_return"],
                    candidates=event.candidates,
                    counterfactual_returns=raw["effective_counterfactual_returns"],
                    joint_counterfactual_return=joint_return,
                )
            )
            manifest["event_windows"].append(_json_safe(raw))

        credit = assign_trajectory_credit(
            credit_events,
            gamma=gamma,
            epsilon=epsilon,
        )
        manifest["status"] = "complete"
        manifest["single_replay_count"] = len(single_requests)
        manifest["joint_replay_count"] = len(joint_groups)
        manifest["credit_validation"] = _json_safe(asdict(credit.validation))
        manifest["backfilled_rewards"] = [
            {"agent_id": key[0], "timestep": key[1], "reward": value}
            for key, value in credit.backfilled_rewards.items()
        ]
        _write_json(manifest_path, manifest)
        return CounterfactualReplayOutcome(
            credit=credit,
            manifest=MappingProxyType(manifest),
            manifest_path=manifest_path,
            single_replay_count=len(single_requests),
            joint_replay_count=len(joint_groups),
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        _write_json(manifest_path, manifest)
        raise


def target_window_return(
    per_step_target_rewards: Mapping[Any, Any] | Sequence[Any],
    *,
    target_id: Any,
    start_exclusive: int,
    end_inclusive: int,
) -> float:
    """Return ``sum(r[j, u] for start_exclusive < u <= end_inclusive)``."""

    if start_exclusive > end_inclusive:
        raise ReplayInputError("window start cannot be later than window end")
    rewards = _normalize_target_rewards(per_step_target_rewards)
    values: list[float] = []
    for timestep in range(start_exclusive + 1, end_inclusive + 1):
        row = rewards.get(timestep, {})
        value = _lookup_id(row, target_id, default=0.0)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ReplayInputError(
                f"non-numeric target reward at step {timestep}: {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise ReplayInputError(
                f"non-finite target reward at step {timestep}: {number!r}"
            )
        values.append(number)
    return math.fsum(values)


def _coerce_event(
    value: FactualTargetRewardEvent | Mapping[str, Any],
) -> FactualTargetRewardEvent:
    if isinstance(value, FactualTargetRewardEvent):
        return value
    if not isinstance(value, Mapping):
        raise ReplayInputError("factual events must be objects")
    required = {"event_id", "target_id", "timestep", "reward", "candidates"}
    missing = required - set(value)
    if missing:
        raise ReplayInputError(f"factual event missing fields: {sorted(missing)}")
    candidates: list[CreditCandidate] = []
    for raw in value["candidates"]:
        if isinstance(raw, CreditCandidate):
            candidates.append(raw)
            continue
        if not isinstance(raw, Mapping):
            raise ReplayInputError("event candidates must be objects")
        try:
            candidates.append(
                CreditCandidate(
                    entity_id=raw["entity_id"],
                    agent_id=raw["agent_id"],
                    tau=int(raw["tau"]),
                    decision_type=str(raw["decision_type"]),
                    categories=frozenset(raw["categories"]),
                    direct_damage=float(raw.get("direct_damage", 0.0)),
                )
            )
        except KeyError as exc:
            raise ReplayInputError(
                f"candidate missing required field {exc.args[0]!r}"
            ) from exc
    return FactualTargetRewardEvent(
        event_id=value["event_id"],
        target_id=value["target_id"],
        timestep=int(value["timestep"]),
        reward=float(value["reward"]),
        candidates=tuple(candidates),
    )


def _validate_inputs(
    events: tuple[FactualTargetRewardEvent, ...],
    *,
    max_steps: int,
    gamma: float,
    epsilon: float,
) -> None:
    if isinstance(max_steps, bool) or int(max_steps) <= 0:
        raise ReplayInputError("max_steps must be positive")
    if not (math.isfinite(float(gamma)) and 0.0 < float(gamma) <= 1.0):
        raise ReplayInputError("gamma must be in (0, 1]")
    if not (math.isfinite(float(epsilon)) and float(epsilon) > 0.0):
        raise ReplayInputError("epsilon must be positive")
    seen: set[Any] = set()
    for event in events:
        if event.event_id in seen:
            raise ReplayInputError(f"duplicate factual event_id {event.event_id!r}")
        seen.add(event.event_id)
        if event.timestep < 1 or event.timestep > max_steps:
            raise ReplayInputError(
                f"event {event.event_id!r} timestep outside episode"
            )
        if not (math.isfinite(event.reward) and event.reward > 0.0):
            raise ReplayInputError("factual target rewards must be positive")
        if not event.candidates:
            raise ReplayInputError(
                f"event {event.event_id!r} has no causal candidates"
            )
        entity_ids = [candidate.entity_id for candidate in event.candidates]
        if len(set(entity_ids)) != len(entity_ids):
            raise ReplayInputError(
                f"event {event.event_id!r} repeats a candidate entity"
            )
        if any(candidate.tau < 1 for candidate in event.candidates):
            raise ReplayInputError(
                f"event {event.event_id!r} has a non-1-based causal tau"
            )
        if any(candidate.tau > event.timestep for candidate in event.candidates):
            raise ReplayInputError(
                f"event {event.event_id!r} requires tau <= reward timestep"
            )


def _build_single_requests(
    events: tuple[FactualTargetRewardEvent, ...],
) -> tuple[list[ReplayRequest], dict[tuple[Any, Any], tuple[Any, int, str]]]:
    groups: dict[tuple[Any, int, str], dict[str, Any]] = {}
    event_candidate_keys: dict[tuple[Any, Any], tuple[Any, int, str]] = {}
    for event in events:
        for candidate in event.candidates:
            key = (candidate.entity_id, candidate.tau, candidate.decision_type)
            event_candidate_keys[(event.event_id, candidate.entity_id)] = key
            group = groups.setdefault(
                key,
                {
                    "candidate": candidate,
                    "stop_step": 0,
                    "target_ids": set(),
                    "event_ids": [],
                },
            )
            if group["candidate"].agent_id != candidate.agent_id:
                raise ReplayInputError(
                    f"deduplicated intervention {key!r} has conflicting agent IDs"
                )
            group["stop_step"] = max(group["stop_step"], event.timestep)
            group["target_ids"].add(event.target_id)
            group["event_ids"].append(event.event_id)

    requests: list[ReplayRequest] = []
    for key in sorted(groups, key=_stable_key):
        group = groups[key]
        candidate = group["candidate"]
        intervention = ReplayIntervention(
            entity_id=candidate.entity_id,
            agent_id=candidate.agent_id,
            timestep=candidate.tau,
            decision_type=candidate.decision_type,
            null_action=_null_action(candidate.decision_type),
        )
        requests.append(
            _make_request(
                kind="single",
                interventions=(intervention,),
                stop_step=group["stop_step"],
                target_ids=tuple(sorted(group["target_ids"], key=_stable_key)),
                factual_event_ids=tuple(group["event_ids"]),
            )
        )
    return requests, event_candidate_keys


def _make_request(
    *,
    kind: str,
    interventions: tuple[ReplayIntervention, ...],
    stop_step: int,
    target_ids: tuple[Any, ...],
    factual_event_ids: tuple[Any, ...],
) -> ReplayRequest:
    if not interventions:
        raise ReplayInputError("a replay request requires an intervention")
    ordered = tuple(sorted(interventions, key=lambda item: _stable_key(item.key)))
    identity = {
        "kind": kind,
        "interventions": [asdict(item) for item in ordered],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    request_id = f"{kind}-{digest}"
    return ReplayRequest(
        request_id=request_id,
        kind=kind,
        interventions=ordered,
        stop_step=int(stop_step),
        target_ids=target_ids,
        factual_event_ids=factual_event_ids,
    )


def _intervention_for_key(
    key: tuple[Any, int, str],
    *,
    events: tuple[FactualTargetRewardEvent, ...],
    candidate_request_keys: Mapping[tuple[Any, Any], tuple[Any, int, str]],
) -> ReplayIntervention:
    for event in events:
        for candidate in event.candidates:
            if candidate_request_keys[(event.event_id, candidate.entity_id)] == key:
                return ReplayIntervention(
                    entity_id=candidate.entity_id,
                    agent_id=candidate.agent_id,
                    timestep=candidate.tau,
                    decision_type=candidate.decision_type,
                    null_action=_null_action(candidate.decision_type),
                )
    raise ReplayInputError(f"unknown intervention key {key!r}")


def _measure_single_event(
    event: FactualTargetRewardEvent,
    *,
    factual_target_rewards: Mapping[Any, Any] | Sequence[Any],
    single_results: Mapping[tuple[Any, int, str], Mapping[str, Any]],
    candidate_request_keys: Mapping[tuple[Any, Any], tuple[Any, int, str]],
    epsilon: float,
) -> dict[str, Any]:
    raw_factual: dict[Any, float] = {}
    raw_counterfactual: dict[Any, float] = {}
    deltas: dict[Any, float] = {}
    for candidate in event.candidates:
        key = candidate_request_keys[(event.event_id, candidate.entity_id)]
        factual = target_window_return(
            factual_target_rewards,
            target_id=event.target_id,
            start_exclusive=candidate.tau - 1,
            end_inclusive=event.timestep,
        )
        counterfactual = target_window_return(
            _extract_target_rewards(single_results[key]),
            target_id=event.target_id,
            start_exclusive=candidate.tau - 1,
            end_inclusive=event.timestep,
        )
        raw_factual[candidate.entity_id] = factual
        raw_counterfactual[candidate.entity_id] = counterfactual
        deltas[candidate.entity_id] = max(factual - counterfactual, 0.0)

    common_start = min(candidate.tau for candidate in event.candidates)
    common_factual = target_window_return(
        factual_target_rewards,
        target_id=event.target_id,
        start_exclusive=common_start - 1,
        end_inclusive=event.timestep,
    )
    effective_counterfactual = {
        entity_id: common_factual - delta
        for entity_id, delta in deltas.items()
    }
    return {
        "event_id": event.event_id,
        "target_id": event.target_id,
        "timestep": event.timestep,
        "reward": event.reward,
        "candidate_taus": {
            candidate.entity_id: candidate.tau for candidate in event.candidates
        },
        "raw_factual_returns": raw_factual,
        "raw_counterfactual_returns": raw_counterfactual,
        "deltas": deltas,
        "common_window_start_exclusive": common_start - 1,
        "common_factual_return": common_factual,
        "effective_counterfactual_returns": effective_counterfactual,
        "all_single_deltas_zero": all(value <= epsilon for value in deltas.values()),
        "direct_damage_only": all(
            candidate.categories == frozenset({"damage"})
            for candidate in event.candidates
        ),
    }


def _run_request_batch(
    *,
    requests: Sequence[ReplayRequest],
    batch_name: str,
    base_env: Mapping[str, str] | None,
    scenario: Path,
    checkpoint: Path,
    max_steps: int,
    output_root: Path,
    specs_dir: Path,
    results_dir: Path,
    logs_dir: Path,
    python_executable: Path,
    main_path: Path,
    cwd: Path,
    timeout_seconds: float,
) -> dict[str, Mapping[str, Any]]:
    if not requests:
        return {}
    resolved_requests = tuple(requests)
    batch_spec_path = specs_dir / f"batch-{batch_name}.json"
    batch_result_path = results_dir / f"batch-{batch_name}.json"
    request_specs = []
    result_paths: dict[str, Path] = {}
    for request in resolved_requests:
        spec_path = specs_dir / f"{request.request_id}.json"
        result_path = results_dir / f"{request.request_id}.json"
        if result_path.exists():
            raise ReplayExecutionError(
                f"refusing stale counterfactual result: {result_path}"
            )
        spec = request.to_spec()
        spec["result_path"] = str(result_path)
        _write_json(spec_path, spec)
        request_specs.append(spec)
        result_paths[request.request_id] = result_path
    _write_json(
        batch_spec_path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{batch_name}_shared_prefix",
            "requests": request_specs,
        },
    )
    stop_step = max(request.stop_step for request in resolved_requests)
    environment = _sanitized_environment(
        base_env,
        checkpoint=checkpoint,
        spec_path=batch_spec_path,
        result_path=batch_result_path,
        stop_step=stop_step,
    )
    environment.pop("RED_CF_SPEC", None)
    environment.pop("RED_CF_RESULT", None)
    environment.update({
        "RED_CF_BATCH_SPEC": str(batch_spec_path),
        "RED_CF_BATCH_RESULT": str(batch_result_path),
        "RED_CF_STOP_STEP": str(stop_step),
    })
    child_output = (
        output_root
        / "counterfactual_replays"
        / "episodes"
        / f"batch_{batch_name}"
    )
    child_output.mkdir(parents=True, exist_ok=True)
    command = (
        str(python_executable),
        str(main_path),
        "--scenario",
        str(scenario),
        "--total-rounds",
        "1",
        "--max-steps",
        str(max_steps),
        "--render-mode",
        "none",
        "--output-dir",
        str(child_output),
    )
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
        check=False,
    )
    (logs_dir / f"batch-{batch_name}.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (logs_dir / f"batch-{batch_name}.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ReplayExecutionError(
            f"counterfactual batch {batch_name} exited with code "
            f"{completed.returncode}"
        )
    if not batch_result_path.is_file():
        raise ReplayExecutionError(
            f"counterfactual batch did not write {batch_result_path}"
        )
    results: dict[str, Mapping[str, Any]] = {}
    for request in resolved_requests:
        result_path = result_paths[request.request_id]
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_replay_result(request, payload)
        results[request.request_id] = payload
    return results


def _run_request(
    *,
    request: ReplayRequest,
    runner: ReplayRunner,
    base_env: Mapping[str, str] | None,
    scenario: Path,
    checkpoint: Path,
    max_steps: int,
    output_root: Path,
    specs_dir: Path,
    results_dir: Path,
    logs_dir: Path,
    python_executable: Path,
    main_path: Path,
    cwd: Path,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    if request.stop_step > max_steps:
        raise ReplayInputError("probe stop step exceeds factual max_steps")
    spec_path = specs_dir / f"{request.request_id}.json"
    result_path = results_dir / f"{request.request_id}.json"
    child_output = output_root / "counterfactual_replays" / "episodes" / request.request_id
    child_output.mkdir(parents=True, exist_ok=True)
    _write_json(spec_path, request.to_spec())
    environment = _sanitized_environment(
        base_env,
        checkpoint=checkpoint,
        spec_path=spec_path,
        result_path=result_path,
        stop_step=request.stop_step,
    )
    command = (
        str(python_executable),
        str(main_path),
        "--scenario",
        str(scenario),
        "--total-rounds",
        "1",
        "--max-steps",
        str(max_steps),
        "--render-mode",
        "none",
        "--output-dir",
        str(child_output),
    )
    launch = ReplayLaunch(
        request=request,
        command=command,
        environment=MappingProxyType(environment),
        cwd=cwd,
        spec_path=spec_path,
        result_path=result_path,
        stdout_path=logs_dir / f"{request.request_id}.stdout.log",
        stderr_path=logs_dir / f"{request.request_id}.stderr.log",
        timeout_seconds=float(timeout_seconds),
    )
    result = runner(launch)
    _validate_replay_result(request, result)
    return result


def _sanitized_environment(
    base_env: Mapping[str, str] | None,
    *,
    checkpoint: Path,
    spec_path: Path,
    result_path: Path,
    stop_step: int,
) -> dict[str, str]:
    environment = {
        str(key): str(value)
        for key, value in (os.environ if base_env is None else base_env).items()
    }
    for key in tuple(environment):
        if key.startswith("RED_CF_") or key in {
            "RED_TRAJECTORY_CF_PROBE",
            "RED_LEARNING_MODEL",
            "RED_LEARNING_TRAIN",
            "RED_REWARD_MODE",
            "RED_UNIFIED_SCORE_ONLY",
        }:
            environment.pop(key, None)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "RED_TRAJECTORY_CF_PROBE": "1",
            "RED_CF_SPEC": str(spec_path),
            "RED_CF_RESULT": str(result_path),
            "RED_CF_STOP_STEP": str(stop_step),
            "RED_LEARNING_TRAIN": "0",
            "RED_LEARNING_MODEL": str(checkpoint),
            "RED_REWARD_MODE": TRAJECTORY_REWARD_MODE,
            "RED_UNIFIED_DEVICE": "cpu",
        }
    )
    return environment


def _validate_replay_result(
    request: ReplayRequest,
    result: Mapping[str, Any],
) -> None:
    if not isinstance(result, Mapping):
        raise ReplayExecutionError("replay runner must return a mapping")
    if result.get("success") is False or result.get("status") in {"failed", "error"}:
        raise ReplayExecutionError(
            f"probe {request.request_id} reported failure: {result.get('error')}"
        )
    if "target_rewards" not in result:
        raise ReplayExecutionError(
            "probe result is missing target_rewards"
        )
    counterfactual_rewards = _normalize_target_rewards(
        result["target_rewards"]
    )
    if int(result.get("branch_timestep", -1)) != request.branch_timestep:
        raise ReplayValidationError(
            f"probe {request.request_id} branched at the wrong timestep"
        )
    try:
        executed_until = int(result["executed_until"])
        terminated = bool(result["terminated"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayValidationError(
            "probe result lacks counterfactual completion metadata"
        ) from exc
    if executed_until < request.branch_timestep:
        raise ReplayValidationError(
            "probe ended before its branch timestep"
        )
    if not terminated and executed_until < request.stop_step:
        raise ReplayValidationError(
            "probe did not cover the requested window"
        )
    missing_steps = [
        step
        for step in range(1, executed_until + 1)
        if step not in counterfactual_rewards
    ]
    if missing_steps:
        raise ReplayValidationError(
            "probe target rewards omit executed steps: "
            f"{missing_steps[:5]}"
        )

    expected_interventions = {item.key for item in request.interventions}
    observed_interventions = _observed_interventions(result)
    if observed_interventions is None:
        if result.get("all_interventions_applied") is not True:
            raise ReplayValidationError(
                f"probe {request.request_id} did not prove intervention application"
            )
    elif observed_interventions != expected_interventions:
        raise ReplayValidationError(
            f"probe {request.request_id} applied interventions "
            f"{sorted(observed_interventions, key=_stable_key)!r}, expected "
            f"{sorted(expected_interventions, key=_stable_key)!r}"
        )


def _observed_interventions(
    result: Mapping[str, Any],
) -> set[tuple[Any, int, str]] | None:
    raw = result.get("applied_interventions")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ReplayValidationError("applied_interventions must be a sequence")
    observed: set[tuple[Any, int, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReplayValidationError("applied intervention records must be objects")
        try:
            observed.add(
                (
                    item["entity_id"],
                    int(item.get("timestep", item.get("tau"))),
                    str(item["decision_type"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayValidationError(
                f"malformed applied intervention record: {item!r}"
            ) from exc
    return observed


def _extract_target_rewards(result: Mapping[str, Any]) -> Mapping[Any, Any] | Sequence[Any]:
    value = result.get("target_rewards")
    if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (str, bytes)):
        raise ReplayExecutionError("probe target_rewards has an invalid shape")
    return value


def _normalize_target_rewards(
    raw: Mapping[Any, Any] | Sequence[Any],
) -> dict[int, Mapping[Any, Any]]:
    normalized: dict[int, Mapping[Any, Any]] = {}
    if isinstance(raw, Mapping):
        iterable = raw.items()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        records: list[tuple[Any, Any]] = []
        for index, row in enumerate(raw):
            if isinstance(row, Mapping) and "timestep" in row:
                payload = row.get("target_rewards", row.get("rewards"))
                records.append((row["timestep"], payload))
            else:
                records.append((index, row))
        iterable = records
    else:
        raise ReplayInputError("target rewards must be a mapping or sequence")
    for timestep_raw, row in iterable:
        try:
            timestep = int(timestep_raw)
        except (TypeError, ValueError) as exc:
            raise ReplayInputError(f"invalid target reward timestep {timestep_raw!r}") from exc
        if row is None:
            row = {}
        if not isinstance(row, Mapping):
            raise ReplayInputError(
                f"target rewards at step {timestep} must be an object"
            )
        normalized[timestep] = row
    return normalized


def _lookup_id(mapping: Mapping[Any, Any], key: Any, *, default: Any) -> Any:
    if key in mapping:
        return mapping[key]
    string_key = str(key)
    if string_key in mapping:
        return mapping[string_key]
    return default


def _null_action(decision_type: str) -> str:
    normalized = decision_type.strip().upper()
    if normalized == "LAUNCH":
        return "WAIT"
    if normalized == "RETARGET":
        return "KEEP"
    if normalized == "TARGET_SELECT":
        return "KEEP"
    if normalized in {"SEARCH", "REGION_SEARCH", "AREA_SEARCH"}:
        return "CANCEL_SEARCH"
    if normalized in {"MANEUVER", "INTERCEPT", "INTERCEPT_MANEUVER"}:
        return "ZERO_MANEUVER"
    if normalized == "SATELLITE_REQUEST":
        return "NO_SATELLITE_REQUEST"
    raise ReplayInputError(
        f"decision type {decision_type!r} has no defined NULL action"
    )


def _request_audit(
    request: ReplayRequest,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "request": _json_safe(request.to_spec()),
        "applied_interventions": _json_safe(result.get("applied_interventions")),
        "branch": {
            "method": "os_fork_copy_on_write",
            "timestep": result.get("branch_timestep"),
            "counterfactual_executed_until": result.get("executed_until"),
            "shared_factual_prefix": True,
            "hash_validation": False,
        },
        "accepted": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _stable_key(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=True)


__all__ = [
    "CounterfactualReplayOutcome",
    "DEFAULT_MAIN",
    "DEFAULT_PYTHON",
    "FactualTargetRewardEvent",
    "ReplayExecutionError",
    "ReplayInputError",
    "ReplayIntervention",
    "ReplayLaunch",
    "ReplayRequest",
    "ReplayRunner",
    "ReplayValidationError",
    "SCHEMA_VERSION",
    "SubprocessReplayRunner",
    "TRAJECTORY_REWARD_MODE",
    "run_counterfactual_replays",
    "target_window_return",
]
