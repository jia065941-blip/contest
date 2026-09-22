"""Independent, fixed q08 game scoring; never resumes or mutates R12 training."""
from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from experiments.paos_feasibility import run_pipeline as shared
from policies.red.learning.sc4_policy import load_sc4_checkpoint
from scenarios.cases.reward import calculate_reward, load_reward_policy

ROOT = shared.ROOT
VERSION = ROOT / "refine-logs/game_q08_score_20260905_113515"
FIXED = ROOT / "refine-logs/game_q08_score"
OUT = VERSION / "run"
OLD = ROOT / "refine-logs/sc4_straight_20260905_010943/run_20260905_031420"
TOP = OLD / "checkpoints/q08.json"
BOTTOM = ROOT / "refine-logs/map1_feasibility/robust_20260903_203053/models/ppo_e01_robust_selected.pt"
TOP_SHA = "9f4a7375cbeafce23f246429e9151d6fc6bc67c7ed6c6f77408cda637368df37"
BOTTOM_SHA = "265f8916c71024343a48addb5fe714f30b13cf765569a69ffda04ca6767ded9f"
THETA_SHA = "abc453025ff464a11ded083ca03e9e1accb8cfdc07876931ebbad101b5ba3def"
PYTHON = Path("/opt/conda/envs/competition/bin/python")
MAX_STEPS = 1200
SEEDS = tuple(shared.SeedTuple(42000001 + i, 20260904, 20260906) for i in range(3))
POLICIES = {"red": "r11_paos", "red_motion": "ppo_custom", "blue": "b0_fixed_ratio_random"}


def require(condition, message):
    if not condition:
        raise shared.ContractError(message)


def request_for(index):
    return {
        "schema_version": "r11_paos_request_v1", "kind": "rollout",
        "evaluation_stage": "final",  # Runtime inference mode, NOT R12 gate success.
        "run_id": f"game_q08_score_20260905_113515_f{index + 1}",
        "cycle": None, "direction": None, "sign": None, "radius": None,
        "direction_vector": None, "seed_tuple": SEEDS[index].to_dict(),
        "center_sha256": THETA_SHA, "candidate_sha256": THETA_SHA,
        "candidate_file_sha256": TOP_SHA, "radius_grid": [.25, .5, 1., 2., 4.],
        "candidate_box": [-4., 4.], "preview_tv_range": [.05, .20],
        "expected": {"legal_state_hash": None, "preview_vector": None, "preview_hash": None},
    }


def runtime():
    return shared.SubprocessRuntime(argparse.Namespace(
        python_executable=PYTHON, scenario="easy/E01", blue_policy=POLICIES["blue"],
        red_policy=POLICIES["red"], red_motion_policy=POLICIES["red_motion"],
        bottom_model=BOTTOM, output_dir=OUT, max_steps=MAX_STEPS,
        process_timeout_seconds=600,
    ))


def validate_terminal(summary):
    steps = summary.get("steps_executed")
    reason = summary.get("termination_reason")
    require(type(steps) is int and 1 <= steps <= MAX_STEPS, "invalid terminal steps")
    require(reason in {"environment_done", "time_limit"}, "invalid terminal reason")
    require(reason != "time_limit" or steps == MAX_STEPS, "truncated time-limit episode")
    return {"episode_complete": True, "steps_executed": steps, "termination_reason": reason}


def recompute_score(summary):
    """Recompute from final game state using the official judge implementation."""
    policy = load_reward_policy(ROOT / "scenarios/cases/easy/E01/scenario.json")
    require(policy is not None and policy.max_steps == MAX_STEPS, "judge policy mismatch")
    objectives = summary["objectives"]
    ids = [row["id"] for row in objectives]
    require(len(ids) == len(set(ids)) and set(ids) == set(policy.objective_ids), "objective identity mismatch")
    weights = dict(policy.objective_weights)
    health, timing = {}, {}
    for row in objectives:
        entity = row["id"]
        value = float(row["final_health"])
        require(math.isfinite(value), "nonfinite final health")
        require(row["weight"] == weights[entity], "objective weight mismatch")
        health[entity] = value
        step = row["destroyed_step"]
        if value <= 0:
            require(type(step) is int and 0 <= step <= summary["steps_executed"], "invalid completion timing")
            timing[entity] = step
        else:
            require(step is None, "live objective has completion timing")
    expected = calculate_reward(policy=policy, target_health=health, destruction_steps=timing).to_dict()
    for key in ("score", "raw_score", "K", "T"):
        actual = float(summary["score"][key])
        require(math.isfinite(actual) and math.isclose(actual, expected[key], abs_tol=1e-9, rel_tol=0), f"official recomputation mismatch: {key}")
    require(summary["score"]["completed"] is expected["completed"], "game completion mismatch")
    return expected


def validate_summary(summary, request):
    terminal = validate_terminal(summary)
    require(all(summary["policies"].get(k) == v for k, v in POLICIES.items()), "policy mismatch")
    audit = shared.validate_rollout_summary(
        summary, request=request, expected_bottom_sha256=BOTTOM_SHA,
        expected_candidate_file_sha256=TOP_SHA, raw_atol=1e-9, expected_input_dim=90,
    )
    require(audit["action_contract_pass"], f"runtime audit failed: {audit['failed_checks']}")
    score = recompute_score(summary)
    return {**terminal, "score": score["score"], "raw_score": score["raw_score"],
            "K": score["K"], "T": score["T"], "game_completed": score["completed"],
            "official_recomputation": score, "runtime_checks": audit["checks"]}


def track(state):
    shared.atomic_json(OUT / "state.json", state)
    lines = ["# q08 独立游戏评分执行状态", "", f"状态：{state['status']}；更新：{shared.now()}", "",
             "| Run | 状态 |", "|---|---|"]
    for name in ("G0", "F1", "F2", "F3", "REPORT"):
        status = "完成" if name in state["completed"] else ("进行中" if state.get("active") == name else "未运行")
        lines.append(f"| {name} | {status} |")
    if state.get("error"):
        lines += ["", f"工程错误：{state['error']}"]
    lines += ["", "原 R12 失败记录保持不变；仅评估 q08，不进行训练、基线或消融。", ""]
    for directory in (VERSION, FIXED):
        (directory / "EXPERIMENT_TRACKER.md").write_text("\n".join(lines), encoding="utf-8")


def sanity():
    require(not (OUT / "state.json").exists(), "existing run state: no overwrite or implicit rerun")
    require(shared.sha256(TOP) == TOP_SHA and shared.sha256(BOTTOM) == BOTTOM_SHA, "frozen model hash mismatch")
    load_sc4_checkpoint(TOP)
    require(shared.read_json(TOP)["theta_sha256"] == THETA_SHA, "theta identity mismatch")
    require(shared.read_json(TOP)["metadata"]["context"]["stage"] == "q08", "not q08")
    review = shared.read_json(VERSION / "code_review_gate.json")
    require(review.get("verdict") == "GO", "fresh code review has not approved deployment")
    for name in ("experiments/game_score_eval.py", "tests/test_game_score_eval.py"):
        require(review["source_sha256"].get(name) == shared.sha256(ROOT / name), "reviewed code changed")
    result = subprocess.run([str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-p", "test_game_score_eval.py", "-v"], cwd=ROOT, capture_output=True, text=True)
    (OUT / "sanity_tests.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    require(result.returncode == 0, "score wrapper unit tests failed; see sanity_tests.log")
    commands = []
    for index, seeds in enumerate(SEEDS):
        request = request_for(index)
        path = OUT / "requests" / f"F{index + 1}.json"
        shared.atomic_json(path, request)
        commands.append(runtime().command(mode="rollout", request_path=path, checkpoint=TOP, seeds=seeds, run_id=request["run_id"]))
    shared.atomic_json(OUT / "config.json", {
        "schema_version": "independent_q08_score_v1", "checkpoint": str(TOP),
        "top_sha256": TOP_SHA, "bottom_sha256": BOTTOM_SHA, "theta_sha256": THETA_SHA,
        "seed_tuples": [s.to_dict() for s in SEEDS], "max_steps": MAX_STEPS,
        "commands": commands, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "python_executable": str(PYTHON), "training": False, "old_r12_status_unchanged": True,
    })
    sources = [ROOT / name for name in shared.DEFAULT_SOURCE_PATHS]
    for directory in shared.DEFAULT_SOURCE_DIRS:
        sources.extend((ROOT / directory).rglob("*.py"))
    sources += [Path(__file__), ROOT / "tests/test_game_score_eval.py", TOP, BOTTOM,
                OLD / "terminal_result.json", VERSION / "EXPERIMENT_PLAN.md",
                VERSION / "code_review_gate.json", OUT / "config.json"]
    sources.extend((OUT / "requests").glob("*.json"))
    shared.atomic_json(OUT / "source_manifest.json", shared.build_source_manifest(sources))
    track({"status": "sanity_passed", "completed": ["G0"], "active": None, "created_at": shared.now()})
    print("G0 PASSED: no simulator episodes executed", flush=True)


def report(records, state, elapsed):
    scores = [row["score"] for row in records]
    result = {"schema_version": "independent_q08_score_v1", "candidate": "q08",
              "status": "evaluation_complete", "complete_episodes": len(records),
              "mean_score": statistics.fmean(scores), "min_score": min(scores),
              "max_score": max(scores), "population_std": statistics.pstdev(scores),
              "elapsed_seconds": elapsed, "episodes": records,
              "top_sha256": TOP_SHA, "bottom_sha256": BOTTOM_SHA,
              "source_manifest_sha256": shared.read_json(OUT / "source_manifest.json")["manifest_sha256"],
              "old_r12_failure_preserved": True, "unseen_test_claim": False}
    shared.atomic_json(OUT / "results.json", result)
    shared.atomic_csv(OUT / "scores.csv", ["run", "blue_seed", "red_seed", "simulation_seed", "score", "raw_score", "K", "T", "episode_complete", "game_completed", "steps_executed", "termination_reason"], records)
    lines = ["# q08 独立完整游戏评分结果", "", "完成预先固定的 3 个完整回合；训练 0 步，无基线、消融或额外试验。", "",
             "| 回合 | Blue seed | 官方分数 /100 | 步数 | 终止原因 | 全部游戏目标完成 |", "|---|---:|---:|---:|---|---|"]
    for row in records:
        lines.append(f"| {row['run']} | {row['blue_seed']} | {row['score']:.9f} | {row['steps_executed']} | {row['termination_reason']} | {row['game_completed']} |")
    lines += ["", f"平均分：{result['mean_score']:.9f}；范围：{min(scores):.9f}–{max(scores):.9f}；总体标准差：{result['population_std']:.9f}。",
              f"评估用时：{elapsed:.1f} 秒。每局正常退出、唯一 FINAL_SUMMARY、完整终止、官方重算和模型/种子/源码身份检查均通过。", "",
              "结论：当前 q08 + 冻结 PPO 可以完成 E01 的正式完整回合评分。三个种子并非未见数据；不能据此主张泛化。",
              "原 R12 的 q16 动作门失败没有被更改，本结果不表示 q16 成功；game_completed 不是回合完整性指标。", "",
              "原始证据：本版本 run/results.json、run/scores.csv、run/summaries/、run/logs/、run/source_manifest.json。", ""]
    for directory in (VERSION, FIXED):
        (directory / "EXPERIMENT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    state.update(status="complete", active=None, finished_at=shared.now())
    state["completed"].append("REPORT")
    track(state)
    outputs = [p for p in VERSION.rglob("*") if p.is_file() and p.name not in {"MANIFEST.md", "output_manifest.json", ".pipeline.lock"}]
    manifest = shared.build_source_manifest(outputs)
    shared.atomic_json(VERSION / "output_manifest.json", manifest)
    entries = ["# 输出文件清单", "", "| 文件 | SHA-256 |", "|---|---|"]
    entries += [f"| {entry['path']} | {entry['sha256']} |" for entry in manifest["entries"]]
    (VERSION / "MANIFEST.md").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"COMPLETE mean_score={result['mean_score']:.9f} scores={scores}", flush=True)


def evaluate(lock):
    state = shared.read_json(OUT / "state.json")
    require(state["status"] == "sanity_passed", "evaluation already attempted; no automatic repeat")
    source = shared.read_json(OUT / "source_manifest.json")
    shared.verify_source_manifest(source)
    state.update(status="running", started_at=shared.now())
    track(state)
    started = time.monotonic()
    records = []
    try:
        for index, seeds in enumerate(SEEDS):
            name = f"F{index + 1}"
            state.update(active=name, active_pid=None)
            track(state)
            shared.verify_source_manifest(source)
            require(time.monotonic() - started < 1800, "total walltime budget exceeded")
            request_path = OUT / "requests" / f"{name}.json"
            request = shared.read_json(request_path)
            require(request == request_for(index), "request changed")
            log_path = OUT / "logs" / f"{name}.log"
            require(not log_path.exists(), "refusing to overwrite an attempted episode")
            def on_start(pid):
                state["active_pid"] = pid
                track(state)
            print(f"START {name}: {seeds.to_dict()}", flush=True)
            summary = runtime().execute(mode="rollout", request_path=request_path,
                checkpoint=TOP, seeds=seeds, run_id=request["run_id"],
                log_path=log_path, lock_fd=lock.fileno(), on_start=on_start)
            summary_path = OUT / "summaries" / f"{name}.json"
            shared.atomic_json(summary_path, summary)
            shared.verify_source_manifest(source)
            record = {"run": name, "run_id": request["run_id"], "blue_seed": seeds.blue,
                      "red_seed": seeds.red, "simulation_seed": seeds.simulation,
                      "process_exit_code": 0, "final_summary_count": 1,
                      "summary_sha256": shared.sha256(summary_path),
                      "log_sha256": shared.sha256(log_path), **validate_summary(summary, request)}
            shared.atomic_json(OUT / "records" / f"{name}.json", record)
            records.append(record)
            state["completed"].append(name)
            state.update(active=None, active_pid=None)
            track(state)
            print(f"DONE {name}: score={record['score']:.9f} steps={record['steps_executed']}", flush=True)
        shared.verify_source_manifest(source)
        report(records, state, time.monotonic() - started)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", failed_at=shared.now())
        track(state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sanity", "evaluate"), required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    require(Path(sys.executable).resolve() == PYTHON.resolve(), "use the pinned competition Python")
    with shared.acquire_run_lock(OUT) as lock:
        sanity() if args.phase == "sanity" else evaluate(lock)


if __name__ == "__main__":
    main()
