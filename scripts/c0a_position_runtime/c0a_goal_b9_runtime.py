"""Position-only state capture and symmetric frozen-teacher branches.

Called by the opt-in RED_C0A_B9_REQUEST hook in core/main.py. Every candidate
forks the same CPU-only environment/blue/teacher snapshot. Historical red
commands are used only up to and including the decision boundary; after that
the frozen teacher recomputes commands from each branch's own observations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import traceback
from types import MethodType

import numpy as np
import torch

from c0a_goal_b9_objective import sample_candidates


SEARCH_TARGET_ID = -100


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def physical_state(observation):
    return {str(key): {field: value.get(field) for field in
                      ("type", "health", "isVisible", "position", "velocity")}
            for key, value in observation["entities"].items()}


def state_sha(observation):
    return hashlib.sha256(json.dumps(physical_state(observation), sort_keys=True,
                                    default=float).encode()).hexdigest()


def entity(observation, entity_id):
    entities = observation["entities"]
    return entities.get(entity_id, entities.get(str(entity_id), {}))


def objective_health(observation, initial):
    # The environment retains destroyed entities in observations. An absent
    # scored entity is an error, never silently treated as a successful kill.
    result = {}
    for target_id in initial:
        item = entity(observation, target_id)
        if "health" not in item:
            raise RuntimeError(f"missing scored objective {target_id}")
        result[target_id] = min(initial[target_id], max(0., float(item["health"])))
    return result


def step_reward(before, after, initial, weights):
    total_weight = sum(weights.values())
    return sum(weights[target] / total_weight *
               max(0., before[target] - after[target]) / initial[target]
               for target in weights if initial[target] > 0)


def locked_teacher_actions(actions, executor, locked):
    if not locked:
        return actions, 0
    kept = [action for action in actions if not
            (int(action.get("executor_id", -1)) == executor and
             int(action.get("commandType_id", -1)) in {200, 3014})]
    return kept, len(actions) - len(kept)


def warm_teacher_on_replayed_state(env):
    # Reconstruct reports, tracks, schedules, launch/maneuver history. Calling
    # only sync_prefix loses reports from platforms that die in the prefix,
    # which can prevent the teacher from ever replanning. Preserve public
    # environment randomness so state capture and replay remain identical.
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        return env._generate_actions_from_agents()
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)


def sync_teacher_commands(commander, actions):
    # RedCommander uses .targets; the legacy student sync helper looks only
    # at UnifiedMAPPOCommander's .catalogue_targets and cannot do this part.
    for action in actions:
        command = int(action.get("commandType_id", -1))
        if command not in {200, 3014} or "target" not in action:
            continue
        executor = int(action["executor_id"])
        coordinates = action["target"]
        if not commander.targets:
            continue
        target = min(commander.targets, key=lambda row: math.hypot(
            float(coordinates["x"]) - row.position.lon,
            float(coordinates["y"]) - row.position.lat))
        if math.hypot(float(coordinates["x"]) - target.position.lon,
                      float(coordinates["y"]) - target.position.lat) > 1e-4:
            continue
        target_id = int(target.entity_id)
        previous = commander.target_by_platform.get(executor)
        if previous != target_id:
            if previous is not None:
                commander.assigned_by_target[previous] = max(0, commander.assigned_by_target.get(previous, 0) - 1)
            commander.assigned_by_target[target_id] = commander.assigned_by_target.get(target_id, 0) + 1
        commander.target_by_platform[executor] = target_id
        if command == 200:
            commander.pending.pop(executor, None)


def frozen_teacher_action(policy, observation, action_mask):
    # MAPPOSharedPolicy.select_action also returns argmax(logits) when frozen.
    # Its Categorical/log_prob construction has no effect on that result.
    observation = torch.as_tensor(observation, dtype=torch.float32,
                                  device=policy.device).unsqueeze(0)
    mask = torch.as_tensor(action_mask, dtype=torch.bool,
                           device=policy.device).unsqueeze(0)
    with torch.no_grad():
        logits = policy.network.actor_forward(observation)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return int(logits.argmax(-1).item())


def run_request(env, commander, learning_policy, run_summary, args,
                sync_prefix, sync_reservations):
    request = json.loads(Path(os.environ["RED_C0A_B9_REQUEST"]).read_text())
    torch.set_num_threads(1)
    trace = json.loads(Path(request["trace"]).read_text())
    boundary = int(request["timestep"])
    executor = int(request["executor_id"])
    if request["mode"] == "branches":
        commander.fail_fast_agent_errors = True
        learning_policy.network.eval()
        for parameter in learning_policy.network.parameters():
            parameter.requires_grad_(False)
        if learning_policy.training:
            raise RuntimeError("continuation teacher must not train")
        learning_policy.select_action = MethodType(frozen_teacher_action, learning_policy)
    else:
        learning_policy.trainer.model.eval()
        for parameter in learning_policy.trainer.model.parameters():
            parameter.requires_grad_(False)
    env.red_model_deploy()
    env.red_model_deploy_from_native(trace["deployment_actions"])
    boundary_row = None
    for row in trace["steps"]:
        step = int(row["step"])
        if step == boundary:
            boundary_row = row
            break
        if step > boundary:
            break
        if request["mode"] == "branches":
            warm_teacher_on_replayed_state(env)
        observation, _, done, _ = env.step(native_actions=row["actions"])
        sync_prefix(env, commander, [row])
        if request["mode"] == "branches":
            sync_teacher_commands(commander, row["actions"])
        run_summary.update(step, observation)
        if done:
            raise RuntimeError("episode ended before requested decision boundary")
    if boundary_row is None:
        raise RuntimeError("decision boundary absent from teacher trace")
    observation = env._get_observation()
    physical_sha = state_sha(observation)
    boundary_commands = [action for action in boundary_row["actions"] if
                         int(action.get("executor_id", -1)) == executor and
                         int(action.get("commandType_id", -1)) == 200]
    if len(boundary_commands) != 1:
        raise RuntimeError("expected exactly one frozen launch at decision boundary")

    capture_path = Path(request["capture_path"])
    if request["mode"] == "capture":
        sync_reservations(env, commander, boundary_row["actions"],
                          excluded_entity_ids={executor})
        unit = {"unit_id": f"s{boundary}_e{executor}_c200", "timestep": boundary,
                "executor_id": executor, "command_type": 200}
        env.prepare_native_handoff_actions(boundary_row["actions"], [unit], "position")
        sample = learning_policy._pending_motion[executor]
        if not sample.mask_initial_position:
            raise RuntimeError("position head inactive at selected launch")
        encoded = sample.observation.detach().cpu().unsqueeze(0)
        features = sample.target_features.detach().cpu().unsqueeze(0)
        valid = sample.target_valid_mask.detach().cpu().bool().unsqueeze(0)
        model = learning_policy.trainer.model
        with torch.no_grad():
            mean = model.distribution_parameters(encoded, target_features=features,
                target_valid_mask=valid)["initial_mean"][0].cpu()
            log_std = model.initial_log_std.detach().cpu().clone()
        bounds = learning_policy._deployment.get(executor, {}).get("deploy_bounds")
        if bounds is None:
            raise RuntimeError("missing legal deployment bounds")
        bounds = list(map(float, bounds))
        generator = torch.Generator().manual_seed(int(request["sampling_seed"]))
        raw = mean + log_std.exp() * torch.randn(mean.shape, generator=generator)
        xy = raw.tanh()
        lon = bounds[0] + (float(xy[0])+1)*.5*(bounds[1]-bounds[0])
        lat = bounds[2] + (float(xy[1])+1)*.5*(bounds[3]-bounds[2])
        own_position = entity(observation, executor)["position"]
        altitude = float(own_position.get("z", own_position.get("alt", 0.)))
        deploy = {"executor_id":executor,"commandType_id":3008,
                  "lla":{"x":lon,"y":lat,"z":altitude},"data_type":"aiAction"}
        n = 3 if request.get("smoke_duplicate_teacher", False) else 2
        command = boundary_commands[0]
        target = learning_policy.teacher_command_target_id(command, entity_type=int(entity(observation,executor)["type"]))
        if target is None:
            raise RuntimeError("teacher target cannot be resolved")
        target = int(target)
        slot = learning_policy._target_slot_ids.index(target)
        item = {"seed":request["seed"],"start_state_id":request["start_state_id"],
            "timestep":boundary,"executor_id":executor,"observation":encoded,
            "target_features":features,"target_valid_mask":valid,
            "old_initial_mean":mean,"old_initial_log_std":log_std,
            "sampled_initial_raw":raw,"sampled_initial_xy":xy,"deploy_bounds":bounds,
            "candidate_position_commands":[deploy,None]+([None] if n==3 else []),
            "candidate_target_indices":torch.tensor([slot]*n),
            "candidate_target_ids":torch.tensor([target]*n),
            "candidate_coordinates":torch.tensor([[command["target"]["x"],command["target"]["y"]]]*n,dtype=torch.float64),
            "candidate_roles":["student_sampled_position","teacher_position"]+(["teacher_identity_repeat"] if n==3 else []),
            "native_unlocked":[True]*n,"snapshot_model_sha256":request["snapshot_model_sha256"],
            "physical_state_sha256":physical_sha,"sampling_seed":request["sampling_seed"],
            "boundary_source":"public_launch_stratified_entity_type"}
        torch.save(item,capture_path)
        save_json(capture_path.with_suffix(".json"), {key:value.tolist() if isinstance(value,torch.Tensor) else value
            for key,value in item.items() if key not in {"observation","target_features","target_valid_mask"}})
        return

    if request["mode"] != "branches":
        raise ValueError(request["mode"])
    commander.fail_fast_agent_errors = True
    item = torch.load(capture_path, map_location="cpu", weights_only=True)
    expected_physical_sha = item.get("physical_state_sha256")
    if expected_physical_sha is not None and physical_sha != expected_physical_sha:
        raise RuntimeError("student capture / teacher branch physical-state mismatch")
    item["physical_state_sha256"] = physical_sha
    warm_teacher_on_replayed_state(env)
    sync_prefix(env, commander, [boundary_row])
    sync_teacher_commands(commander, boundary_row["actions"])
    # All candidates, including top1, are run by this identical branch path.
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    rng_sha = hashlib.sha256(pickle.dumps((python_rng, numpy_rng, torch_rng.tolist()))).hexdigest()
    initial = {int(row["id"]): float(row["initial_health"])
               for row in trace["summary"]["objectives"]}
    weights = {int(key): float(value) for key, value in
               trace["summary"]["score"]["objective_weights"].items()}
    initial = {key: initial[key] for key in weights}
    start_health = objective_health(observation, initial)
    result_dir = Path(request["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    identity_audit = bool(request.get("identity_audit", False))
    if identity_audit:
        from c0a_goal_b9_identity import fingerprint, rng_fingerprint, StepTrace
        snapshot_fingerprint = fingerprint((env, commander, learning_policy, run_summary))
        save_json(result_dir / "common_snapshot.json", {
            "mechanism": "os.fork copy-on-write of same live process, including native simulator memory",
            "python_object_graph": snapshot_fingerprint,
            "native_state_equality_basis": "inherited process memory; opaque native internals are not serialized",
            "public_rng_sha256": rng_sha,
        })
    children = {}
    errors = []

    def reap():
        pid, status = os.wait()
        index = children.pop(pid)
        if os.waitstatus_to_exitcode(status) != 0:
            errors.append(index)

    sys.stdout.flush()
    sys.stderr.flush()
    for candidate_index, target in enumerate(item["candidate_target_ids"].tolist()):
        while len(children) >= int(request["branch_workers"]):
            reap()
        pid = os.fork()
        if pid:
            children[pid] = candidate_index
            continue
        try:
            with (result_dir / f"branch_{candidate_index:02d}.log").open("w") as log:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                # Python reseeds after fork. Restore all public RNG states.
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(torch_rng)
                if state_sha(env._get_observation()) != physical_sha:
                    raise RuntimeError("forked candidate did not retain common physical state")
                restored_rng_sha = hashlib.sha256(pickle.dumps((
                    random.getstate(), np.random.get_state(), torch.get_rng_state().tolist()))).hexdigest()
                if restored_rng_sha != rng_sha:
                    raise RuntimeError("forked candidate did not retain common public RNG state")
                if identity_audit:
                    child_fingerprint = fingerprint((env, commander, learning_policy, run_summary))
                    if child_fingerprint != snapshot_fingerprint:
                        raise RuntimeError("forked full Python object graph differs from parent")
                    step_trace = StepTrace(result_dir / f"branch_{candidate_index:02d}_steps.jsonl")
                current_health = dict(start_health)
                total_return = 0.
                # SEARCH is a one-boundary route choice in C0a_goal when
                # search-option chaining is disabled. It is launched toward
                # the student's sampled search coordinate and immediately
                # returned to the frozen teacher. Physical attack targets
                # retain the b9 contract and stay locked until termination.
                native_unlocked = bool(item.get("native_unlocked", [False] * len(item["candidate_target_ids"]))[candidate_index])
                locked = int(target) != SEARCH_TARGET_ID and not native_unlocked
                unlock_step = boundary if not locked else None
                suppressed = 0
                teacher_steps = 0
                non_target_boundary_changes = 0
                for step in range(boundary, int(args.max_steps) + 1):
                    if step == boundary:
                        if locked:
                            previous_target = commander.target_by_platform.get(executor)
                            if previous_target != int(target):
                                if previous_target is not None:
                                    commander.assigned_by_target[previous_target] = max(
                                        0, commander.assigned_by_target.get(previous_target, 0) - 1)
                                commander.assigned_by_target[int(target)] = commander.assigned_by_target.get(int(target), 0) + 1
                            commander.target_by_platform[executor] = int(target)
                        actions = copy.deepcopy(boundary_row["actions"])
                        position_command = item["candidate_position_commands"][candidate_index]
                        if position_command is not None:
                            # Only the chosen entity's initial position is changed.
                            actions = [action for action in actions if not (
                                int(action.get("executor_id",-1)) == executor and
                                int(action.get("commandType_id",-1)) == 3008)]
                            actions.insert(0,copy.deepcopy(position_command))
                        strip_position = lambda commands: [action for action in commands if not (
                            int(action.get("executor_id",-1)) == executor and
                            int(action.get("commandType_id",-1)) == 3008)]
                        if strip_position(actions) != strip_position(boundary_row["actions"]):
                            raise RuntimeError("non-position boundary actions changed")
                    else:
                        if locked:
                            commander.target_by_platform[executor] = int(target)
                        actions = env._generate_actions_from_agents()
                        teacher_steps += 1
                        actions, removed = locked_teacher_actions(actions, executor, locked)
                        suppressed += removed
                        if locked:
                            commander.target_by_platform[executor] = int(target)
                    if identity_audit and step == boundary and item["candidate_position_commands"][candidate_index] is None and actions != boundary_row["actions"]:
                        raise RuntimeError("same-target audit changed exact boundary actions")
                    observation, _, done, _ = env.step(native_actions=actions)
                    sync_prefix(env, commander, [{"step": step, "actions": actions}])
                    sync_teacher_commands(commander, actions)
                    run_summary.update(step, observation)
                    next_health = objective_health(observation, initial)
                    total_return += step_reward(current_health, next_health, initial, weights)
                    current_health = next_health
                    own = entity(observation, executor)
                    if locked and (float(own.get("health", 0)) <= 0 or
                                   not own.get("isVisible", False) or
                                   current_health[int(target)] <= 0):
                        locked = False
                        unlock_step = step
                    if identity_audit:
                        step_trace.add(step, actions, observation, commander,
                                       current_health, total_return, locked, suppressed)
                    if step % 500 == 0:
                        save_json(result_dir / f"branch_{candidate_index:02d}_progress.json", {
                            "step": step, "target_id": int(target),
                            "official_return_so_far": total_return,
                            "teacher_launched": len(commander.launched_ids),
                            "teacher_report_count": len(commander.reports)})
                    if done:
                        break
                telescoped = step_reward(start_health, current_health, initial, weights)
                if abs(total_return - telescoped) > 1e-9:
                    raise RuntimeError("official step rewards do not telescope")
                summary = run_summary.build(observation,
                    termination_reason="environment_done" if done else "time_limit",
                    red_launched=len(commander.launched_ids))
                audit_fields = {}
                if identity_audit:
                    audit_fields = {
                        "identity_trace": step_trace.close(),
                        "common_snapshot_python_sha256": snapshot_fingerprint["sha256"],
                        "native_unlocked": native_unlocked,
                        "exact_boundary_actions_equal": item["candidate_position_commands"][candidate_index] is None,
                    }
                save_json(result_dir / f"branch_{candidate_index:02d}.json", {
                    **audit_fields,
                    "candidate_index": candidate_index, "target_id": int(target),
                    "position_command": item["candidate_position_commands"][candidate_index],
                    "non_position_boundary_changes": 0,
                    "target_index": int(item["candidate_target_indices"][candidate_index]),
                    "official_joint_return": total_return, "score": summary["score"]["score"],
                    "end_step": step, "teacher_closed_loop_steps": teacher_steps,
                    "locked_target_id": int(target) if int(target) != SEARCH_TARGET_ID else None,
                    "option_termination_step": unlock_step,
                    "suppressed_target_change_commands": suppressed,
                    "non_target_boundary_changes": non_target_boundary_changes,
                    "physical_state_sha256": physical_sha, "public_rng_sha256": rng_sha,
                    "suffix_semantics": (
                        "frozen_teacher_closed_loop_native_unlocked" if native_unlocked else
                        "frozen_teacher_closed_loop_target_locked"
                        if int(target) != SEARCH_TARGET_ID
                        else "frozen_teacher_closed_loop_search_then_release"
                    ),
                    "initial_objective_health": start_health,
                    "final_objective_health": current_health,
                    "objective_weights": weights, "summary": summary})
                sys.stdout.flush()
                sys.stderr.flush()
            os._exit(0)
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
    while children:
        reap()
    if errors:
        raise RuntimeError(f"candidate branches failed: {errors}; see {result_dir}")
    branches = [json.loads((result_dir / f"branch_{i:02d}.json").read_text())
                for i in range(len(item["candidate_target_ids"]))]
    item["official_joint_returns"] = torch.tensor(
        [row["official_joint_return"] for row in branches], dtype=torch.float64)
    item["branch_results"] = branches
    item["return_source"] = "sum_tau_to_T_all_formal_objective_weighted_health_deltas"
    item["public_rng_sha256"] = rng_sha
    torch.save(item, Path(request["item_path"]))
    save_json(result_dir / "summary.json", {
        "status": "complete", "seed": item["seed"],
        "candidate_target_ids": item["candidate_target_ids"].tolist(),
        "returns": item["official_joint_returns"].tolist(),
        "span": float(item["official_joint_returns"].max() - item["official_joint_returns"].min()),
        "physical_state_sha256": physical_sha, "public_rng_sha256": rng_sha})
