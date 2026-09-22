from __future__ import annotations

import unittest

from experiments.unified_mappo.causal_candidates import (
    CausalCandidateError,
    build_factual_target_events,
)


def decision(
    event_id: int,
    event_type: str,
    *,
    entity_id: int,
    step: int,
    target_id: int | None = None,
    agent_id: int | None = None,
    action: int | None = None,
    search_source: int | None = None,
) -> dict:
    row = {
        "event_id": event_id,
        "event_type": event_type,
        "entity_id": entity_id,
        "step": step,
    }
    if target_id is not None:
        row["target_id"] = target_id
    if agent_id is not None:
        row["agent_id"] = agent_id
    if action is not None:
        row["action"] = action
    if search_source is not None:
        row["search_source"] = search_source
    return row


def direct_hit(
    event_id: int,
    *,
    attacker_id: int,
    target_id: int,
    step: int,
    damage: float,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": "direct_hit",
        "attacking_entity_id": attacker_id,
        "target_entity_id": target_id,
        "step": step,
        "actual_damage": damage,
    }


def interceptor_launch(
    event_id: int,
    *,
    red_entity_id: int,
    step: int,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": "interceptor_launch",
        "intercepted_red_entity_id": red_entity_id,
        "step": step,
    }


class CausalCandidateTests(unittest.TestCase):
    def test_builds_damage_search_and_intercept_union_with_strict_taus(self) -> None:
        decisions = (
            decision(1, "SEARCH", entity_id=100, step=3),
            decision(
                2,
                "LAUNCH",
                entity_id=200,
                step=8,
                target_id=54,
                search_source=100,
            ),
            decision(3, "RETARGET", entity_id=300, step=9, target_id=54),
            decision(4, "LAUNCH", entity_id=400, step=4, target_id=54),
            decision(5, "MANEUVER", entity_id=400, step=10, action=0),
            decision(6, "MANEUVER", entity_id=400, step=11, action=-1),
            # These entities have ledger activity but no causal evidence for target 54.
            decision(7, "LAUNCH", entity_id=500, step=7, target_id=99),
            decision(8, "SEARCH", entity_id=600, step=2),
            decision(9, "MANEUVER", entity_id=700, step=12, action=1),
        )
        simulator_events = (
            direct_hit(10, attacker_id=200, target_id=54, step=20, damage=5.0),
            direct_hit(11, attacker_id=300, target_id=54, step=20, damage=15.0),
            interceptor_launch(12, red_entity_id=400, step=14),
            # A launch against another target does not establish intercept credit here.
            interceptor_launch(13, red_entity_id=500, step=15),
            # An interception at the reward step is too late to be causal.
            interceptor_launch(14, red_entity_id=700, step=20),
        )

        events = build_factual_target_events(
            episode_id="round-2",
            timestep=20,
            target_rewards={54: 0.25, 55: 0.0, 56: -0.1},
            decision_events=decisions,
            simulator_events=simulator_events,
            entity_to_agent={
                100: 10,
                200: 20,
                300: 30,
                400: 40,
                500: 50,
                600: 60,
                700: 70,
            },
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event_id"], "round-2:20:54")
        self.assertEqual(event["target_id"], 54)
        self.assertEqual(event["timestep"], 20)
        self.assertAlmostEqual(event["reward"], 0.25)
        by_entity = {row["entity_id"]: row for row in event["candidates"]}
        self.assertEqual(set(by_entity), {100, 200, 300, 400})

        self.assertEqual(by_entity[100]["agent_id"], 10)
        self.assertEqual(by_entity[100]["tau"], 3)
        self.assertEqual(by_entity[100]["decision_type"], "SEARCH")
        self.assertEqual(by_entity[100]["categories"], ["search"])

        self.assertEqual(by_entity[200]["tau"], 8)
        self.assertEqual(by_entity[200]["decision_type"], "LAUNCH")
        self.assertEqual(by_entity[200]["categories"], ["damage"])
        self.assertAlmostEqual(by_entity[200]["direct_damage"], 5.0)

        self.assertEqual(by_entity[300]["tau"], 9)
        self.assertEqual(by_entity[300]["decision_type"], "RETARGET")
        self.assertEqual(by_entity[300]["categories"], ["damage"])
        self.assertAlmostEqual(by_entity[300]["direct_damage"], 15.0)

        # The zero maneuver is ignored; the latest nonzero causal maneuver wins.
        self.assertEqual(by_entity[400]["tau"], 11)
        self.assertEqual(by_entity[400]["decision_type"], "MANEUVER")
        self.assertEqual(by_entity[400]["categories"], ["intercept"])
        self.assertAlmostEqual(by_entity[400]["direct_damage"], 0.0)

        # Non-candidates are absent, which makes their downstream alpha implicitly zero.
        for excluded in (500, 600, 700, 999):
            self.assertNotIn(excluded, by_entity)
        self.assertTrue(all(row["tau"] < event["timestep"] for row in by_entity.values()))

    def test_same_entity_categories_are_merged_and_latest_tau_is_retained(self) -> None:
        decisions = (
            decision(1, "LAUNCH", entity_id=200, step=5, target_id=54),
            decision(2, "MANEUVER", entity_id=200, step=12, action=1),
        )
        events = build_factual_target_events(
            episode_id=1,
            timestep=20,
            target_rewards={54: 0.1},
            decision_events=decisions,
            simulator_events=(
                direct_hit(3, attacker_id=200, target_id=54, step=20, damage=7.0),
                interceptor_launch(4, red_entity_id=200, step=15),
            ),
            entity_to_agent={200: 2},
        )

        self.assertEqual(len(events[0]["candidates"]), 1)
        candidate = events[0]["candidates"][0]
        self.assertEqual(candidate["categories"], ["damage", "intercept"])
        self.assertEqual(candidate["tau"], 12)
        self.assertEqual(candidate["decision_type"], "MANEUVER")
        self.assertAlmostEqual(candidate["direct_damage"], 7.0)

    def test_target_select_keep_is_a_temporal_option_damage_anchor(self) -> None:
        events = build_factual_target_events(
            episode_id=1,
            timestep=20,
            target_rewards={54: 0.1},
            decision_events=(
                decision(1, "TARGET_SELECT", entity_id=200, step=8, target_id=54),
            ),
            simulator_events=(
                direct_hit(2, attacker_id=200, target_id=54, step=20, damage=4.0),
            ),
            entity_to_agent={200: 2},
        )

        candidate = events[0]["candidates"][0]
        self.assertEqual(candidate["decision_type"], "TARGET_SELECT")
        self.assertEqual(candidate["tau"], 8)
        self.assertEqual(candidate["categories"], ["damage"])

    def test_positive_reward_requires_direct_hit_and_damage_anchor_evidence(self) -> None:
        cases = (
            (
                "no direct hit",
                (),
                (),
                "no direct-hit evidence",
            ),
            (
                "zero damage is not evidence",
                (),
                (direct_hit(1, attacker_id=200, target_id=54, step=20, damage=0.0),),
                "no direct-hit evidence",
            ),
            (
                "hit without launch or retarget",
                (),
                (direct_hit(1, attacker_id=200, target_id=54, step=20, damage=4.0),),
                "lacks LAUNCH/RETARGET anchor",
            ),
        )
        for label, decisions, simulator_events, expected in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(CausalCandidateError, expected):
                    build_factual_target_events(
                        episode_id=1,
                        timestep=20,
                        target_rewards={54: 0.1},
                        decision_events=decisions,
                        simulator_events=simulator_events,
                        entity_to_agent={200: 2},
                    )

    def test_cited_search_source_requires_prior_search_anchor(self) -> None:
        with self.assertRaisesRegex(CausalCandidateError, "without a SEARCH anchor"):
            build_factual_target_events(
                episode_id=1,
                timestep=20,
                target_rewards={54: 0.1},
                decision_events=(
                    decision(
                        1,
                        "LAUNCH",
                        entity_id=200,
                        step=8,
                        target_id=54,
                        search_source=100,
                    ),
                ),
                simulator_events=(
                    direct_hit(2, attacker_id=200, target_id=54, step=20, damage=4.0),
                ),
                entity_to_agent={100: 1, 200: 2},
            )

    def test_negative_tau_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            CausalCandidateError,
            r"invalid causal tau=-1 for reward t=20",
        ):
            build_factual_target_events(
                episode_id=1,
                timestep=20,
                target_rewards={54: 0.1},
                decision_events=(
                    decision(1, "LAUNCH", entity_id=200, step=-1, target_id=54),
                ),
                simulator_events=(
                    direct_hit(2, attacker_id=200, target_id=54, step=20, damage=4.0),
                ),
                entity_to_agent={200: 2},
            )

    def test_reward_step_decision_cannot_be_used_as_a_causal_anchor(self) -> None:
        with self.assertRaisesRegex(CausalCandidateError, "lacks LAUNCH/RETARGET anchor"):
            build_factual_target_events(
                episode_id=1,
                timestep=20,
                target_rewards={54: 0.1},
                decision_events=(
                    decision(1, "LAUNCH", entity_id=200, step=20, target_id=54),
                ),
                simulator_events=(
                    direct_hit(2, attacker_id=200, target_id=54, step=20, damage=4.0),
                ),
                entity_to_agent={200: 2},
            )


if __name__ == "__main__":
    unittest.main()
