#!/usr/bin/env python3
"""b9: fresh candidates, elite CE + full-legal KL, persistent target-only Adam."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

from train_c0a_goal_b8_advantage import (
    SIM_ROOT, TARGET_PREFIXES, file_sha256, load_model, model_sha256,
    target_logits, write_json,
)
from c0a_goal_b9_objective import elite_labels, goal_loss
from tools.train_start_state_option_curriculum import (
    TEACHER_MODEL, ordered_trajectories, scenario_entity_types,
)


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--informative-states", type=int, default=24)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--branch-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_catalogues(manifest):
    entity_types = scenario_entity_types(Path(manifest["scenario"]))
    course, all_launches = [], []
    for row in ordered_trajectories(manifest):
        if int(row["seed"]) >= 1000:
            raise ValueError("formal evaluation seeds must not enter training")
        trace = json.loads(Path(row["trace"]).read_text())
        anchor = row["damage_anchors"][str(row["assigned_target_id"])]
        for step_row in trace["steps"]:
            for action in step_row["actions"]:
                executor = int(action.get("executor_id", -1))
                if (int(action.get("commandType_id", -1)) != 200 or
                        entity_types.get(executor) not in {21000, 21001}):
                    continue
                item = {"trace": row["trace"], "seed": int(row["seed"]),
                        "timestep": int(step_row["step"]), "executor_id": executor,
                        "start_state_id": f"seed{row['seed']}:step{step_row['step']}:entity{executor}"}
                all_launches.append({**item, "boundary_source": "all_launches"})
                if (item["timestep"] == int(anchor["decision_step"]) and
                        executor == int(anchor["attacking_entity_id"])):
                    course.append({**item, "boundary_source": "damage_anchors"})
    if not course or not all_launches:
        raise ValueError("missing course / launch boundary catalogue")
    return {"damage_anchors": course, "all_launches": all_launches}


def subprocess_environment(request, mode, snapshot):
    environment = os.environ.copy()
    # Prevent unrelated parent experiment flags from changing this protocol.
    for name in tuple(environment):
        if name.startswith(("RED_", "BLUE_", "SIMULATION_")):
            environment.pop(name)
    environment.update({
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(SIM_ROOT),
                                       environment.get("PYTHONPATH", ""))),
        "BLUE_POLICY": "b0_fixed_ratio_random", "BLUE_POLICY_SEED": str(request["seed"]),
        "SIMULATION_SEED": str(request["seed"]),
        "RED_POLICY_SEED": str(request["seed"]),
        "RED_POLICY": "r12_unified_mappo" if mode == "capture" else "r9_hierarchical_learning",
        "RED_MOTION_POLICY": "unified_mappo" if mode == "capture" else "mappo",
        "RED_LEARNING_MODEL": str(snapshot if mode == "capture" else TEACHER_MODEL),
        "RED_LEARNING_TRAIN": "1" if mode == "capture" else "0",
        "RED_REWARD_MODE": "weighted_damage_individual",
        "RED_UNIFIED_DEVICE": "cpu", "RED_NATIVE_ROLLOUT_DEVICE": "cpu",
        "RED_NATIVE_SNAPSHOT_GUIDANCE": "1",
        "RED_UNIFIED_DYNAMIC_LIFECYCLE": "1", "RED_UNIFIED_TEMPORAL_ATTACK_OPTIONS": "1",
        "RED_UNIFIED_ATTACK_OPTION_MIN_DWELL_STEPS": "60",
        "RED_UNIFIED_FORCE_DETERMINISTIC_ACTOR": "1",
    })
    environment["LD_LIBRARY_PATH"] = os.pathsep.join((
        str(SIM_ROOT / "core/envengine/simulator/models/HXDMissileModel"),
        str(Path(sys.prefix) / "lib"), environment.get("LD_LIBRARY_PATH", "")))
    return environment


def collect_state(args, manifest, boundary, snapshot, source_sha, cycle, index):
    output = args.output_dir / f"round_{cycle:02d}" / "states" / f"state_{index:04d}"
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and (output / "item.pt").is_file():
        item = torch.load(output / "item.pt", map_location="cpu", weights_only=True)
        if (item["snapshot_model_sha256"] != source_sha or
                item["start_state_id"] != boundary["start_state_id"] or
                item["sampling_seed"] != args.seed + cycle * 100000 + index):
            raise RuntimeError("resume item differs from current batch contract")
        item.update({"collection_seconds": 0., "item_path": str(output / "item.pt"),
                     "resumed_completed_item": True})
        return item
    request = {**boundary, "k": args.k, "rho": args.rho,
               "sampling_seed": args.seed + cycle * 100000 + index,
               "snapshot_model_sha256": source_sha,
               "capture_path": str((output / "capture.pt").resolve()),
               "item_path": str((output / "item.pt").resolve()),
               "result_dir": str((output / "branches").resolve()),
               "branch_workers": args.branch_workers}
    start = time.monotonic()
    for mode in ("capture", "branches"):
        request["mode"] = mode
        request_path = output / f"{mode}_request.json"
        write_json(request_path, request)
        environment = subprocess_environment(request, mode, snapshot)
        environment["RED_C0A_B9_REQUEST"] = str(request_path.resolve())
        with (output / f"{mode}.log").open("w") as log:
            process = subprocess.run([
                sys.executable, str(SIM_ROOT / "core/main.py"),
                "--scenario", manifest["scenario"], "--output-dir", str(output / f"sim_{mode}"),
                "--max-steps", "3000", "--total-rounds", "1", "--render-mode", "none",
                "--disable-log-color"], cwd=SIM_ROOT / "core", env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError(f"{mode} failed: {output / (mode + '.log')}")
        if mode == "capture":
            captured = torch.load(output / "capture.pt", map_location="cpu", weights_only=True)
            if len(captured["candidate_target_indices"]) == 1:
                captured.update({"official_joint_returns": None, "skip_reason": "singleton",
                                 "collection_seconds": time.monotonic() - start,
                                 "item_path": str(output / "item.pt")})
                torch.save(captured, output / "item.pt")
                return captured
    item = torch.load(output / "item.pt", map_location="cpu", weights_only=True)
    item["collection_seconds"] = time.monotonic() - start
    item["item_path"] = str(output / "item.pt")
    return item


def is_informative(item, epsilon):
    if len(item["candidate_target_indices"]) < 2:
        return False
    returns = item["official_joint_returns"]
    return len(returns) >= 2 and float(returns.max() - returns.min()) > epsilon


def validate_item(item, args):
    indices = item["candidate_target_indices"]
    ids = item["candidate_target_ids"]
    valid = item["target_valid_mask"][0]
    if (len(indices) != min(args.k, int(valid.sum())) or
            len(set(indices.tolist())) != len(indices) or
            len(set(ids.tolist())) != len(ids) or not valid[indices].all()):
        raise RuntimeError("candidate legality/cardinality/uniqueness mismatch")
    top = int(item["old_logits"].masked_fill(~valid, -torch.inf).argmax())
    if int(indices[0]) != top:
        raise RuntimeError("candidate set omitted the old-policy top1")
    if int(valid.sum()) > args.k:
        import math
        n_uniform = max(1, math.ceil(args.rho * (args.k - 1)))
        if item["candidate_roles"] != ["top1"] + ["uniform"] * n_uniform + ["policy"] * (args.k - 1 - n_uniform):
            raise RuntimeError("candidate exploration quota mismatch")
    if len(indices) == 1:
        if item["official_joint_returns"] is not None or item.get("skip_reason") != "singleton":
            raise RuntimeError("singleton must be explicitly marked unevaluated")
        return
    if not torch.isfinite(item["official_joint_returns"]).all():
        raise RuntimeError("nonfinite formal returns")
    branches = item["branch_results"]
    if len(branches) != len(indices):
        raise RuntimeError("missing candidate branch")
    for index, branch in enumerate(branches):
        if (branch["target_id"] != int(ids[index]) or branch["target_index"] != int(indices[index]) or
                branch["physical_state_sha256"] != item["physical_state_sha256"] or
                branch["public_rng_sha256"] != item["public_rng_sha256"] or
                branch["non_target_boundary_changes"] != 0 or
                branch["suffix_semantics"] != "frozen_teacher_closed_loop_target_locked" or
                branch["teacher_closed_loop_steps"] != branch["end_step"] - item["timestep"]):
            raise RuntimeError("candidate branch violates same-state closed-loop contract")
        initial = {int(row["id"]): float(row["initial_health"]) for row in branch["summary"]["objectives"]}
        weights = {int(key): float(value) for key, value in branch["objective_weights"].items()}
        before = {int(key): float(value) for key, value in branch["initial_objective_health"].items()}
        after = {int(key): float(value) for key, value in branch["final_objective_health"].items()}
        reward = sum(weights[key] / sum(weights.values()) * max(0., before[key] - after[key]) / initial[key]
                     for key in weights if initial[key] > 0)
        if (abs(reward - branch["official_joint_return"]) > 1e-9 or
                abs(reward - float(item["official_joint_returns"][index])) > 1e-9):
            raise RuntimeError("candidate label differs from formal joint suffix reward")


def stack_items(items, device):
    maximum = max(len(item["candidate_target_indices"]) for item in items)
    indices = torch.zeros((len(items), maximum), dtype=torch.long)
    returns = torch.zeros((len(items), maximum), dtype=torch.float64)
    mask = torch.zeros((len(items), maximum), dtype=torch.bool)
    for index, item in enumerate(items):
        n = len(item["candidate_target_indices"])
        indices[index, :n] = item["candidate_target_indices"]
        returns[index, :n] = item["official_joint_returns"]
        mask[index, :n] = True
    return {"observation": torch.cat([item["observation"] for item in items]).to(device),
            "features": torch.cat([item["target_features"] for item in items]).to(device),
            "valid": torch.cat([item["target_valid_mask"] for item in items]).to(device),
            "indices": indices.to(device), "returns": returns.to(device),
            "candidate_mask": mask.to(device)}


def evaluate(model, anchor_logits, data, args):
    with torch.no_grad():
        current = target_logits(model, data["observation"], data["features"], data["valid"])
        loss, detail = goal_loss(current, anchor_logits, data["valid"], data["indices"],
                                 data["returns"], data["candidate_mask"], args.epsilon,
                                 args.xi, args.kl_coef)
        mask = detail["informative"]
        elite = detail["elite"]
        probability = detail["candidate_probabilities"]
        best = data["returns"].masked_fill(~data["candidate_mask"], -torch.inf).max(-1).values
        regret = (best - (probability * data["returns"]).sum(-1)) / detail["span"].clamp_min(args.epsilon)
        top = current.argmax(-1)
        hits = ((data["indices"] == top[:, None]) & elite).any(-1)
        tv = .5 * (current.softmax(-1) - anchor_logits.softmax(-1)).abs().sum(-1)
        return {"loss": float(loss), "cross_entropy": float(detail["cross_entropy"]),
                "kl_old_to_current": float(detail["kl"]),
                "elite_hit": float(hits[mask].float().mean()),
                "elite_probability": float((probability * elite).sum(-1)[mask].mean()),
                "normalized_candidate_regret": float(regret[mask].mean()),
                "mean_tv_from_round_source": float(tv[mask].mean())}


def train(args):
    args.output_dir = args.output_dir.resolve()
    args.source_checkpoint = args.source_checkpoint.resolve()
    args.manifest = args.manifest.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("output directory is not empty; refusing to overwrite a run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.informative_states < 4 or args.informative_states % 4:
        raise ValueError("informative-states must be a positive multiple of 4")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    catalogues = build_catalogues(manifest)
    source, config, model = load_model(args.source_checkpoint, args.device)
    initial_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    trainable = []
    trainable_names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TARGET_PREFIXES))
        if parameter.requires_grad:
            trainable.append(parameter)
            trainable_names.append(name)
    if any(not any(name.startswith(prefix) for name in trainable_names) for prefix in TARGET_PREFIXES):
        raise RuntimeError("missing target Transformer module")
    model.eval()
    # Created exactly once. Never load source['optimizer'], scheduler or PPO state.
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.update({"source_sha256": file_sha256(args.source_checkpoint),
                     "code_sha256": {str(path): file_sha256(path) for path in (
                         Path(__file__), Path(__file__).with_name("c0a_goal_b9_runtime.py"),
                         Path(__file__).with_name("c0a_goal_b9_objective.py"),
                         SIM_ROOT / "core/main.py")},
                     "teacher_checkpoint": str(TEACHER_MODEL), "teacher_sha256": file_sha256(TEACHER_MODEL),
                     "fresh_optimizer_initial_state_count": len(optimizer.state),
                     "trainable_names": trainable_names,
                     "catalogue_counts": {key: len(value) for key, value in catalogues.items()},
                     "teacher_continuation": "closed_loop_per_branch_target_locked",
                     "gradient_clipping": False, "ppo": False, "scheduler": False,
                     "curriculum_fraction_informative": 0.75})
    history = []
    total_raw = 0
    if args.resume:
        recorded = json.loads((args.output_dir / "protocol.json").read_text())
        for name in ("source_sha256", "teacher_sha256", "code_sha256", "rounds", "steps",
                     "informative_states", "learning_rate", "k", "rho", "xi", "epsilon", "kl_coef", "seed"):
            if protocol[name] != recorded[name]:
                raise RuntimeError(f"resume protocol changed: {name}")
        history_path = args.output_dir / "training_history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text())
        if history:
            resumed = torch.load(history[-1]["checkpoint"], map_location=args.device, weights_only=True)
            if resumed["continuation"]["stage"] != "C0a_goal_b9":
                raise RuntimeError("only this b9 job's target Adam may be resumed")
            model.load_state_dict(resumed["model"], strict=True)
            optimizer.load_state_dict(resumed["optimizer"])
            total_raw = sum(row["raw_states"] for row in history)
            checkpoint = resumed
    else:
        write_json(args.output_dir / "protocol.json", protocol)
    start = time.monotonic()
    for cycle in range(len(history) + 1, args.rounds + 1):
        round_dir = args.output_dir / f"round_{cycle:02d}"
        round_dir.mkdir(exist_ok=args.resume)
        anchor = copy.deepcopy(model).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        source_sha = model_sha256(anchor.state_dict())
        snapshot = round_dir / "frozen_goal_policy.pt"
        torch.save({"algorithm": source["algorithm"], "config": asdict(config),
                    "model": {key: value.cpu() for key, value in anchor.state_dict().items()},
                    "update_count": 0, "transition_count": 0,
                    "continuation": {"role": "frozen_candidate_and_kl_anchor",
                                     "snapshot_model_sha256": source_sha}}, snapshot)
        targets = {"damage_anchors": args.informative_states * 3 // 4,
                   "all_launches": args.informative_states // 4}
        pools = copy.deepcopy(catalogues)
        rng = random.Random(args.seed + cycle)
        for pool in pools.values():
            rng.shuffle(pool)
        if args.runtime_check_only:
            item = collect_state(args, manifest, pools["damage_anchors"][0], snapshot, source_sha, cycle, 0)
            validate_item(item, args)
            write_json(args.output_dir / "runtime_check.json", {
                "status": "passed", "item_path": item["item_path"],
                "candidate_count": len(item["candidate_target_ids"]),
                "informative": is_informative(item, args.epsilon),
                "seconds": item["collection_seconds"]})
            return
        items, raw, used = [], [], set()
        counts = dict.fromkeys(targets, 0)
        while len(items) < args.informative_states:
            pending = []
            for group, needed in targets.items():
                for _ in range(needed - counts[group]):
                    while pools[group] and pools[group][-1]["start_state_id"] in used:
                        pools[group].pop()
                    if not pools[group]:
                        raise RuntimeError(f"insufficient distinct informative states in {group}")
                    row = pools[group].pop()
                    used.add(row["start_state_id"])
                    pending.append(row)
            write_json(args.output_dir / "progress.json", {
                "status": "collecting", "round": cycle, "total_rounds": args.rounds,
                "informative_counts": counts, "raw_states_this_round": len(raw),
                "pending": len(pending), "completed_adam_steps": (cycle - 1) * args.steps})
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(collect_state, args, manifest, row, snapshot,
                                       source_sha, cycle, len(raw) + index)
                           for index, row in enumerate(pending)]
                for future in as_completed(futures):
                    item = future.result()
                    validate_item(item, args)
                    raw.append(item)
                    informative = is_informative(item, args.epsilon)
                    if informative:
                        items.append(item)
                        counts[item["boundary_source"]] += 1
                    write_json(args.output_dir / "progress.json", {
                        "status": "collecting", "round": cycle, "total_rounds": args.rounds,
                        "informative_counts": counts, "raw_states_this_round": len(raw),
                        "completed_adam_steps": (cycle - 1) * args.steps})
                    print(json.dumps({"event": "state_complete", "round": cycle,
                                      "raw": len(raw), "informative": len(items),
                                      "seed": item["seed"], "seconds": item["collection_seconds"]}), flush=True)
        total_raw += len(raw)
        items.sort(key=lambda item: item["start_state_id"])
        raw.sort(key=lambda item: item["item_path"])
        if len(items) != args.informative_states or counts != targets:
            raise RuntimeError("effective-state count / distribution mismatch")
        torch.save({"stage": "C0a_goal", "round": cycle, "snapshot_model_sha256": source_sha,
                    "items": items, "raw_item_paths": [item["item_path"] for item in raw]},
                   round_dir / "candidate_batch.pt")
        data = stack_items(items, args.device)
        with torch.no_grad():
            old_logits = target_logits(anchor, data["observation"], data["features"], data["valid"])
            captured_logits = torch.stack([item["old_logits"] for item in items]).to(args.device)
            if not torch.allclose(old_logits[data["valid"]], captured_logits[data["valid"]], atol=3e-5, rtol=1e-5):
                raise RuntimeError("sampling policy differs from the frozen KL anchor")
        baseline = evaluate(model, old_logits, data, args)
        steps = []
        for local_step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = target_logits(model, data["observation"], data["features"], data["valid"])
            loss, _ = goal_loss(logits, old_logits, data["valid"], data["indices"],
                                data["returns"], data["candidate_mask"], args.epsilon, args.xi, args.kl_coef)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("unexpected empty/nonfinite training loss")
            loss.backward()
            if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in trainable):
                raise RuntimeError("missing or nonfinite target gradient")
            optimizer.step()
            metrics = evaluate(model, old_logits, data, args)
            steps.append({"step": local_step, "global_step": (cycle - 1) * args.steps + local_step, **metrics})
            print(json.dumps({"event": "update", "round": cycle, **steps[-1]}), flush=True)
        frozen_changes = [name for name, value in model.state_dict().items()
                          if not name.startswith(TARGET_PREFIXES) and
                          not torch.equal(value.cpu(), initial_weights[name])]
        if frozen_changes:
            raise RuntimeError(f"frozen parameters changed: {frozen_changes}")
        if model_sha256(anchor.state_dict()) != source_sha:
            raise RuntimeError("frozen anchor changed during the round")
        adam_steps = sorted({int(state["step"]) for state in optimizer.state.values()})
        if adam_steps != [cycle * args.steps]:
            raise RuntimeError(f"Adam state lifecycle mismatch: {adam_steps}")
        checkpoint = {"algorithm": source["algorithm"], "config": asdict(config),
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "update_count": cycle * args.steps, "transition_count": cycle * args.informative_states,
                      "continuation": {"stage": "C0a_goal_b9", "source_checkpoint": str(args.source_checkpoint),
                                       "round": cycle, "optimizer_created_once": True,
                                       "old_ppo_optimizer_loaded": False},
                      "metrics": steps[-1]}
        checkpoint_path = round_dir / "working.pt"
        torch.save(checkpoint, checkpoint_path)
        record = {"round": cycle, "raw_states": len(raw), "informative_states": len(items),
                  "groups": counts, "snapshot_model_sha256": source_sha,
                  "output_model_sha256": model_sha256(model.state_dict()),
                  "baseline": baseline, "final": steps[-1], "steps": steps,
                  "optimizer_steps": adam_steps, "changed_frozen_tensors": frozen_changes,
                  "checkpoint": str(checkpoint_path.resolve())}
        history.append(record)
        write_json(round_dir / "training_summary.json", record)
        write_json(args.output_dir / "training_history.json", history)
        write_json(args.output_dir / "progress.json", {"status": "round_complete", "round": cycle,
                   "completed_adam_steps": cycle * args.steps, "total_raw_states": total_raw})
    final_dir = args.output_dir / "checkpoints"
    final_dir.mkdir(exist_ok=args.resume)
    final_path = final_dir / "working.pt"
    torch.save(checkpoint, final_path)
    summary = {"status": "complete", "rounds": args.rounds, "adam_steps": args.rounds * args.steps,
               "informative_states": args.rounds * args.informative_states, "raw_states": total_raw,
               "output_checkpoint": str(final_path.resolve()), "output_sha256": file_sha256(final_path),
               "frozen_tensor_changes": [], "elapsed_seconds": time.monotonic() - start,
               "promotion_evaluation_performed": False}
    write_json(args.output_dir / "training_summary.json", summary)
    write_json(args.output_dir / "progress.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    try:
        train(arguments)
    except BaseException as error:
        if arguments.output_dir.is_dir():
            write_json(arguments.output_dir / "failure.json", {"status": "failed", "error": repr(error)})
        raise
