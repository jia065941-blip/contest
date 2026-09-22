from __future__ import annotations

import unittest

from experiments.unified_mappo.trajectory_credit import (
    ALL_ZERO_DELTA_REASON,
    CreditCandidate,
    CreditStrategy,
    DuplicateEventError,
    TargetRewardEvent,
    UnresolvedCreditError,
    assign_target_reward_event,
    assign_trajectory_credit,
)


def candidate(
    entity_id: str,
    *,
    agent_id: str | None = None,
    tau: int = 20,
    decision_type: str = "LAUNCH",
    categories: frozenset[str] = frozenset({"damage"}),
    direct_damage: float = 0.0,
) -> CreditCandidate:
    return CreditCandidate(
        agent_id=entity_id if agent_id is None else agent_id,
        entity_id=entity_id,
        tau=tau,
        decision_type=decision_type,
        categories=categories,
        direct_damage=direct_damage,
    )


def event(
    event_id: str,
    candidates: tuple[CreditCandidate, ...],
    counterfactual_returns: dict[str, float],
    *,
    timestep: int = 47,
    reward: float = 0.0571,
    factual_return: float = 1.0,
    joint_counterfactual_return: float | None = None,
) -> TargetRewardEvent:
    return TargetRewardEvent(
        event_id=event_id,
        target_id="target-54",
        timestep=timestep,
        reward=reward,
        factual_return=factual_return,
        candidates=candidates,
        counterfactual_returns=counterfactual_returns,
        joint_counterfactual_return=joint_counterfactual_return,
    )


class TrajectoryCreditTests(unittest.TestCase):
    def test_tau20_t47_backfill_uses_gamma_power_27(self) -> None:
        source = candidate("entity-A", tau=20)
        result = assign_trajectory_credit(
            (event("hit-54", (source,), {"entity-A": 0.0}),),
            gamma=0.99,
        )

        allocation = result.assignments[0].candidate_credits[0]
        expected = (0.99**27) * 0.0571
        self.assertAlmostEqual(allocation.delta, 1.0)
        self.assertAlmostEqual(allocation.alpha, 1.0)
        self.assertAlmostEqual(allocation.credit, 0.0571)
        self.assertAlmostEqual(allocation.backfill_multiplier, 0.99**27)
        self.assertAlmostEqual(allocation.backfilled_reward, expected)
        self.assertAlmostEqual(result.reward_for("entity-A", 20), expected)
        self.assertAlmostEqual(result.reward_for("entity-A", 47), 0.0)
        self.assertTrue(result.validation.all_invariants_hold)
        self.assertAlmostEqual(result.validation.discounted_return_error, 0.0)

    def test_non_candidate_has_implicit_zero_alpha_and_credit(self) -> None:
        source = candidate("entity-A")
        assignment = assign_target_reward_event(
            event("only-A", (source,), {"entity-A": 0.25}),
            gamma=0.99,
        )

        self.assertIsNone(assignment.allocation_for_entity("entity-B"))
        self.assertEqual(assignment.alpha_for_entity("entity-B"), 0.0)
        self.assertEqual(assignment.credit_for_entity("entity-B"), 0.0)
        self.assertEqual(len(assignment.candidate_credits), 1)

    def test_direct_damage_redundancy_uses_measured_damage(self) -> None:
        first = candidate("A", direct_damage=10.0)
        second = candidate("B", direct_damage=30.0)
        assignment = assign_target_reward_event(
            event(
                "redundant-damage",
                (first, second),
                {"A": 1.0, "B": 1.0},
                reward=0.8,
            ),
            gamma=1.0,
        )

        self.assertEqual(
            assignment.strategy,
            CreditStrategy.DIRECT_DAMAGE_REDUNDANCY,
        )
        self.assertAlmostEqual(assignment.alpha_for_entity("A"), 0.25)
        self.assertAlmostEqual(assignment.alpha_for_entity("B"), 0.75)
        self.assertAlmostEqual(assignment.credit_for_entity("A"), 0.2)
        self.assertAlmostEqual(assignment.credit_for_entity("B"), 0.6)
        self.assertIsNotNone(assignment.anomaly)
        assert assignment.anomaly is not None
        self.assertEqual(assignment.anomaly.reason, ALL_ZERO_DELTA_REASON)
        self.assertEqual(
            assignment.anomaly.resolution_strategy,
            CreditStrategy.DIRECT_DAMAGE_REDUNDANCY,
        )

    def test_joint_removal_redundancy_is_explicit_equal_split(self) -> None:
        launcher = candidate("A", direct_damage=10.0)
        searcher = candidate(
            "B",
            tau=12,
            decision_type="SEARCH",
            categories=frozenset({"search"}),
        )
        assignment = assign_target_reward_event(
            event(
                "joint-redundancy",
                (launcher, searcher),
                {"A": 1.0, "B": 1.0},
                reward=0.4,
                joint_counterfactual_return=0.2,
            ),
            gamma=0.99,
        )

        self.assertEqual(
            assignment.strategy,
            CreditStrategy.JOINT_REMOVAL_EQUAL_SPLIT,
        )
        self.assertAlmostEqual(assignment.alpha_for_entity("A"), 0.5)
        self.assertAlmostEqual(assignment.alpha_for_entity("B"), 0.5)
        self.assertAlmostEqual(assignment.credit_for_entity("A"), 0.2)
        self.assertAlmostEqual(assignment.credit_for_entity("B"), 0.2)
        self.assertIsNotNone(assignment.anomaly)
        assert assignment.anomaly is not None
        self.assertAlmostEqual(assignment.anomaly.joint_delta or 0.0, 0.8)
        self.assertEqual(
            assignment.anomaly.resolution_strategy,
            CreditStrategy.JOINT_REMOVAL_EQUAL_SPLIT,
        )

    def test_unresolved_all_zero_anomaly_raises_with_audit_record(self) -> None:
        launcher = candidate("A", direct_damage=10.0)
        searcher = candidate(
            "B",
            decision_type="SEARCH",
            categories=frozenset({"search"}),
        )
        unresolved_event = event(
            "unresolved",
            (launcher, searcher),
            {"A": 1.0, "B": 1.0},
            joint_counterfactual_return=1.0,
        )

        with self.assertRaises(UnresolvedCreditError) as caught:
            assign_target_reward_event(unresolved_event, gamma=0.99)

        self.assertEqual(caught.exception.anomaly.event_id, "unresolved")
        self.assertEqual(caught.exception.anomaly.reason, ALL_ZERO_DELTA_REASON)
        self.assertEqual(caught.exception.anomaly.joint_delta, 0.0)
        self.assertIsNone(caught.exception.anomaly.resolution_strategy)

    def test_duplicate_event_id_is_rejected_before_recording_twice(self) -> None:
        source = candidate("A")
        first = event("duplicate", (source,), {"A": 0.0})
        second = event(
            "duplicate",
            (source,),
            {"A": 0.5},
            reward=0.1,
        )

        with self.assertRaises(DuplicateEventError):
            assign_trajectory_credit((first, second), gamma=0.99)

    def test_batch_validates_reward_and_gamma_one_conservation(self) -> None:
        first = candidate("A", tau=20)
        second = candidate("B", tau=21)
        result = assign_trajectory_credit(
            (
                event(
                    "event-1",
                    (first, second),
                    {"A": 0.0, "B": 0.5},
                    reward=0.3,
                ),
                event(
                    "event-2",
                    (first,),
                    {"A": 0.2},
                    timestep=50,
                    reward=0.7,
                ),
            ),
            gamma=1.0,
        )

        report = result.validation
        self.assertEqual(report.event_count, 2)
        self.assertEqual(report.unique_event_count, 2)
        self.assertTrue(report.event_ids_unique)
        self.assertTrue(report.every_reward_recorded_once)
        self.assertAlmostEqual(report.original_reward_total, 1.0)
        self.assertAlmostEqual(report.allocated_credit_total, 1.0)
        self.assertAlmostEqual(report.gamma_one_original_return, 1.0)
        self.assertAlmostEqual(report.gamma_one_backfilled_return, 1.0)
        self.assertAlmostEqual(report.gamma_one_conservation_error, 0.0)
        self.assertAlmostEqual(report.discounted_return_error, 0.0)


if __name__ == "__main__":
    unittest.main()
