#!/usr/bin/env python3
"""Four-seed E01 same-target identity and native-unlocked control diagnostic."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from evaluate_c0a_goal_b9_closedloop import build_jobs, file_sha256, TARGET_SLOT_IDS
from train_c0a_goal_b9 import SIM_ROOT, subprocess_environment, write_json, TEACHER_MODEL

BASE = Path(__file__).resolve().parents[1] / "competition-platform-refine-logs/target_transformer_main_20260914"


def compare(left, right):
    a = [json.loads(s) for s in Path(left["identity_trace"]["path"]).read_text().splitlines()]
    b = [json.loads(s) for s in Path(right["identity_trace"]["path"]).read_text().splitlines()]
    keys = ["actions_sha256", "full_observation_sha256", "teacher_state_sha256", "public_rng_sha256", "objective_health", "official_return"]
    first = {k: next((x["step"] for x, y in zip(a,b) if x[k] != y[k]), None) for k in keys}
    return {
        "compared_steps": min(len(a),len(b)), "same_length": len(a)==len(b),
        "first_divergence_step": first,
        "all_compared_fields_equal": len(a)==len(b) and all(v is None for v in first.values()),
        "score_delta": left["score"]-right["score"],
        "return_delta": left["official_joint_return"]-right["official_joint_return"],
        "final_objective_health_equal": left["final_objective_health"]==right["final_objective_health"],
    }


def run(args, manifest, job):
    out = Path(job["output"])
    out.mkdir(parents=True, exist_ok=True)
    target = int(job["assigned_target_id"])
    slot = TARGET_SLOT_IDS.index(target)
    item = {
        "seed": job["seed"], "start_state_id": job["start_state_id"],
        "candidate_target_indices": torch.tensor([slot]*3),
        "candidate_target_ids": torch.tensor([target]*3),
        "candidate_coordinates": torch.tensor([job["teacher_coordinate"]]*3, dtype=torch.float64),
        "candidate_roles": ["student_forced_teacher_target", "teacher_same_b9_control", "teacher_native_unlocked"],
        "native_unlocked": [False, False, True],
        "physical_state_sha256": None,
    }
    torch.save(item, out/"branch_input.pt")
    request = {
        "mode": "branches", "trace": job["trace"], "seed": job["seed"],
        "start_state_id": job["start_state_id"], "timestep": job["timestep"],
        "executor_id": job["executor_id"], "capture_path": str(out/"branch_input.pt"),
        "item_path": str(out/"branch_output.pt"), "result_dir": str(out/"branches"),
        "branch_workers": 3, "identity_audit": True,
    }
    write_json(out/"request.json", request)
    env = subprocess_environment(request, "branches", args.checkpoint)
    env["RED_C0A_B9_REQUEST"] = str(out/"request.json")
    command = [sys.executable, str(SIM_ROOT/"core/main.py"), "--scenario", manifest["scenario"], "--output-dir", str(out/"sim"), "--max-steps", str(args.max_steps), "--total-rounds", "1", "--render-mode", "none", "--disable-log-color"]
    write_json(out/"launch.json", {"command": command, "cwd": str(SIM_ROOT/"core"), "environment": {k:v for k,v in env.items() if k.startswith(("RED_", "BLUE_", "SIMULATION_"))}})
    start = time.monotonic()
    with (out/"run.log").open("w") as log:
        proc = subprocess.run(command, cwd=SIM_ROOT/"core", env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"seed {job['seed']} failed: {out/'run.log'}")
    branches = [json.loads((out/f"branches/branch_{i:02d}.json").read_text()) for i in range(3)]
    result = {
        "seed": job["seed"], "target_id": target, "boundary": job["timestep"], "executor_id": job["executor_id"],
        "original_b9_target_id": job["student_target_id"],
        "scores": [b["score"] for b in branches], "returns": [b["official_joint_return"] for b in branches],
        "suppressed_commands": [b["suppressed_target_change_commands"] for b in branches],
        "option_termination_steps": [b["option_termination_step"] for b in branches],
        "same_control": compare(branches[0], branches[1]),
        "native_unlocked_control": compare(branches[0], branches[2]),
        "same_snapshot_hash": len({b["common_snapshot_python_sha256"] for b in branches})==1,
        "exact_boundary_actions_equal": all(b["exact_boundary_actions_equal"] for b in branches),
        "elapsed_seconds": time.monotonic()-start,
    }
    write_json(out/"result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=3000)
    args = p.parse_args()
    source = BASE/"validations/c0a_goal_b9_64_closedloop_20260919/protocol.json"
    old = json.loads(source.read_text())
    args.old_progress, args.manifest, args.checkpoint = (Path(old[k]) for k in ("old_progress", "manifest", "checkpoint"))
    args.only_seed = None
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest, jobs = build_jobs(args)
    if Path(manifest["scenario"]).parent.name != "E01":
        raise ValueError("expected E01 scenario")
    code = [Path(__file__), Path(__file__).with_name("c0a_goal_b9_runtime.py"), Path(__file__).with_name("c0a_goal_b9_identity.py"), SIM_ROOT/"core/main.py"]
    protocol = {
        "evaluation_type": "simulation_only", "purpose": "same-target branch identity, not target prediction quality",
        "scenario": manifest["scenario"], "seeds": [j["seed"] for j in jobs], "seed_selection": "first four entries of pre-existing 64-seed evaluation order; no outcome selection",
        "max_steps": args.max_steps, "b9_checkpoint": str(args.checkpoint), "teacher_checkpoint": str(TEACHER_MODEL),
        "b9_sha256": file_sha256(args.checkpoint), "teacher_sha256": file_sha256(TEACHER_MODEL),
        "source_protocol": str(source), "manifest": str(args.manifest), "old_progress": str(args.old_progress),
        "intervention": "override completed b9 target decision with exact teacher target coordinate in float64; target head bypassed",
        "branches": ["forced student target, b9 lock", "teacher target, identical b9 lock", "teacher target, native unlocked continuation"],
        "common_state": "one prefork live CPU env/blue/teacher/native-memory snapshot; no independent reconstruction per branch",
        "rng": "restore public Python/NumPy/Torch states after fork; private RNGs and native state inherited",
        "identity_fields": ["boundary actions exact equality", "full observation every step", "actions every step", "teacher state every step", "public RNG every step", "objective health", "official return", "end step", "score"],
        "scope_limit": "diagnoses b9 closed-loop branch runner; does not certify legacy r12 student handoff or fresh native teacher prefix",
        "hashes": {str(x.resolve()): file_sha256(x) for x in code+[args.manifest, args.old_progress, Path(manifest["scenario"])]+[Path(j["trace"]) for j in jobs]},
    }
    write_json(args.output_dir/"protocol.json", protocol)
    results=[]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(run,args,manifest,j) for j in jobs]
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            results.sort(key=lambda r: protocol["seeds"].index(r["seed"]))
            write_json(args.output_dir/"results.json", results)
            write_json(args.output_dir/"progress.json", {"completed":len(results),"total":len(jobs),"status":"complete" if len(results)==len(jobs) else "running"})
            print(json.dumps(result), flush=True)
    write_json(args.output_dir/"summary.json", {
        "seeds": protocol["seeds"], "completed": len(results),
        "same_control_all_identical": all(r["same_control"]["all_compared_fields_equal"] and r["same_control"]["score_delta"]==0 and r["same_snapshot_hash"] and r["exact_boundary_actions_equal"] for r in results),
        "native_unlocked_divergent_seeds": [r["seed"] for r in results if not r["native_unlocked_control"]["all_compared_fields_equal"]],
        "results": results,
    })

if __name__ == "__main__":
    main()
