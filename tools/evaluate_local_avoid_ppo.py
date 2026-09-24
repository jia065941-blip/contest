"""Evaluate the local avoid checkpoint against a straight-flight baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_ppo import LocalAvoidPPO, evaluate_policy


def parse_stages(value: str) -> tuple[int, ...]:
    try:
        stages = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("stages must be comma-separated integers") from error
    if not stages or any(stage not in (0, 1, 2, 3) for stage in stages):
        raise argparse.ArgumentTypeError("stages must only contain 0, 1, 2, or 3")
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stages",
        type=parse_stages,
        default=(0, 1, 2),
        help="comma-separated curriculum stages to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/local_avoid_ppo/evaluation.json",
    )
    args = parser.parse_args()
    policy, environment_config, payload = LocalAvoidPPO.load(
        args.model, device=args.device
    )
    report = {
        "model": str(args.model.resolve()),
        "checkpoint_stage": int(payload.get("stage", -1)),
        "episodes_per_stage": args.episodes,
        "stages": {},
    }
    for stage in args.stages:
        report["stages"][str(stage)] = {
            "ppo": evaluate_policy(
                policy,
                environment_config,
                stage=stage,
                episodes=args.episodes,
                seed=args.seed + stage * 10_000,
            ),
            "straight": evaluate_policy(
                policy,
                environment_config,
                stage=stage,
                episodes=args.episodes,
                seed=args.seed + stage * 10_000,
                straight_baseline=True,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("LOCAL_AVOID_EVALUATION " + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
