from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.unified_mappo.counterfactual_replay import (
    ReplayExecutionError,
    ReplayValidationError,
    TRAJECTORY_REWARD_MODE,
    run_counterfactual_replays,
    target_window_return,
)
from experiments.unified_mappo.trajectory_credit import CreditStrategy



def candidate(
    entity_id: str,
    *,
    category: str = "damage",
    direct_damage: float = 1.0,
    tau: int = 1,
    decision_type: str = "LAUNCH",
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "agent_id": f"agent-{entity_id}",
        "tau": tau,
        "decision_type": decision_type,
        "categories": [category],
        "direct_damage": direct_damage if category == "damage" else 0.0,
    }


def event(
    event_id: str,
    timestep: int,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "target_id": 54,
        "timestep": timestep,
        "reward": 0.1,
        "candidates": candidates,
    }


def accepted_result(launch, target_rewards, *, factual_target_rewards=None):
    factual_rewards = (
        target_rewards if factual_target_rewards is None else factual_target_rewards
    )
    return {
        "status": "complete",
        "target_rewards": target_rewards,
        "factual_target_rewards": factual_rewards,
        "branch_timestep": launch.request.branch_timestep,
        "executed_until": launch.request.stop_step,
        "terminated": False,
        "factual_executed_until": launch.request.stop_step,
        "factual_terminated": False,
        "applied_interventions": [
            {
                "entity_id": item.entity_id,
                "timestep": item.timestep,
                "decision_type": item.decision_type,
            }
            for item in launch.request.interventions
        ],
    }

def run_one_probe(directory, runner):
    return run_counterfactual_replays(
        factual_events=(event("e1", 2, [candidate("A")]),),
        factual_target_rewards={1: {54: 0.0}, 2: {54: 0.1}},
        scenario="scenario.json",
        max_steps=2,
        frozen_checkpoint="frozen.pt",
        output_dir=directory,
        gamma=0.99,
        runner=runner,
    )



class CounterfactualReplayTests(unittest.TestCase):
    def test_target_window_is_open_at_tau_and_closed_at_event(self) -> None:
        rewards = {
            0: {54: 100.0},
            1: {54: 1.0},
            2: {54: 2.0},
            3: {54: 4.0},
            4: {54: 8.0},
        }
        self.assertEqual(
            target_window_return(
                rewards,
                target_id=54,
                start_exclusive=1,
                end_inclusive=3,
            ),
            6.0,
        )

    def test_single_replays_are_deduplicated_by_entity_tau_and_type(self) -> None:
        launches = []

        def runner(launch):
            launches.append(launch)
            return accepted_result(
                launch,
                {1: {54: 0.0}, 2: {54: 0.0}, 3: {54: 0.0}},
            )

        with tempfile.TemporaryDirectory() as directory:
            outcome = run_counterfactual_replays(
                factual_events=(
                    event("e1", 2, [candidate("A")]),
                    event("e2", 3, [candidate("A")]),
                ),
                factual_target_rewards={
                    1: {54: 0.0},
                    2: {54: 0.1},
                    3: {54: 0.2},
                },
                scenario="scenario.json",
                max_steps=3,
                frozen_checkpoint="frozen.pt",
                output_dir=directory,
                gamma=0.99,
                base_env={"PATH": "/bin", "RED_CF_OLD": "stale"},
                runner=runner,
            )

            self.assertEqual(len(launches), 1)
            self.assertEqual(outcome.single_replay_count, 1)
            self.assertEqual(outcome.joint_replay_count, 0)
            self.assertEqual(launches[0].request.branch_timestep, 1)
            self.assertEqual(launches[0].request.stop_step, 3)
            self.assertNotIn("expected_fingerprints", launches[0].request.to_spec())
            max_steps_index = launches[0].command.index("--max-steps") + 1
            self.assertEqual(launches[0].command[max_steps_index], "3")
            self.assertEqual(launches[0].environment["RED_CF_STOP_STEP"], "3")
            self.assertEqual(
                launches[0].environment["RED_REWARD_MODE"],
                TRAJECTORY_REWARD_MODE,
            )
            self.assertEqual(launches[0].environment["RED_LEARNING_TRAIN"], "0")
            self.assertNotIn("RED_CF_OLD", launches[0].environment)
            self.assertTrue(outcome.credit.validation.all_invariants_hold)
            manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            branch_audit = manifest["requests"][0]["branch"]
            self.assertFalse(branch_audit["hash_validation"])
            self.assertNotIn("state_hash", branch_audit)
            self.assertNotIn("rng_hash", branch_audit)
            self.assertNotIn("joint_action_hash", branch_audit)

    def test_missing_applied_intervention_fails_and_records_manifest(self) -> None:
        def runner(launch):
            result = accepted_result(launch, {1: {54: 0.0}, 2: {54: 0.0}})
            result.pop("applied_interventions")
            return result

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ReplayValidationError):
                run_counterfactual_replays(
                    factual_events=(event("e1", 2, [candidate("A")]),),
                    factual_target_rewards={1: {54: 0.0}, 2: {54: 0.1}},
                    scenario="scenario.json",
                    max_steps=2,
                    frozen_checkpoint="frozen.pt",
                    output_dir=directory,
                    gamma=0.99,
                    runner=runner,
                )

            manifest = json.loads(
                (Path(directory) / "counterfactual_replay_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["errors"][0]["type"], "ReplayValidationError")
    def test_missing_paired_factual_sibling_is_rejected(self) -> None:
        def runner(launch):
            result = accepted_result(
                launch,
                {1: {54: 0.0}, 2: {54: 0.0}},
            )
            result.pop("factual_target_rewards")
            return result

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ReplayExecutionError,
                "factual tau-fork sibling",
            ):
                run_one_probe(directory, runner)

    def test_wrong_branch_timestep_is_rejected(self) -> None:
        def runner(launch):
            result = accepted_result(
                launch,
                {1: {54: 0.0}, 2: {54: 0.0}},
            )
            result["branch_timestep"] = launch.request.branch_timestep + 1
            return result

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ReplayValidationError,
                "wrong timestep",
            ):
                run_one_probe(directory, runner)

    def test_incomplete_reward_rows_are_rejected_for_both_branches(self) -> None:
        for field in ("target_rewards", "factual_target_rewards"):
            with self.subTest(field=field):
                def runner(launch):
                    rewards = {1: {54: 0.0}, 2: {54: 0.0}}
                    result = accepted_result(
                        launch,
                        dict(rewards),
                        factual_target_rewards=dict(rewards),
                    )
                    result[field].pop(2)
                    return result

                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(
                        ReplayValidationError,
                        "omit executed steps",
                    ):
                        run_one_probe(directory, runner)

    def test_mixed_all_zero_singles_trigger_one_joint_removal(self) -> None:
        launches = []

        def runner(launch):
            launches.append(launch)
            rewards = {
                1: {54: 0.0},
                2: {54: 0.0 if launch.request.kind == "joint" else 1.0},
            }
            return accepted_result(launch, rewards)

        with tempfile.TemporaryDirectory() as directory:
            outcome = run_counterfactual_replays(
                factual_events=(
                    event(
                        "redundant",
                        2,
                        [
                            candidate("A", category="damage", direct_damage=10.0),
                            candidate(
                                "B",
                                category="search",
                                direct_damage=0.0,
                                decision_type="SEARCH",
                            ),
                        ],
                    ),
                ),
                factual_target_rewards={1: {54: 0.0}, 2: {54: 1.0}},
                scenario="scenario.json",
                max_steps=2,
                frozen_checkpoint="frozen.pt",
                output_dir=directory,
                gamma=1.0,
                runner=runner,
            )

            self.assertEqual([item.request.kind for item in launches].count("single"), 2)
            self.assertEqual([item.request.kind for item in launches].count("joint"), 1)
            self.assertEqual(outcome.single_replay_count, 2)
            self.assertEqual(outcome.joint_replay_count, 1)
            assignment = outcome.credit.assignments[0]
            self.assertEqual(
                assignment.strategy,
                CreditStrategy.JOINT_REMOVAL_EQUAL_SPLIT,
            )
            self.assertAlmostEqual(assignment.alpha_for_entity("A"), 0.5)
            self.assertAlmostEqual(assignment.alpha_for_entity("B"), 0.5)


if __name__ == "__main__":
    unittest.main()
