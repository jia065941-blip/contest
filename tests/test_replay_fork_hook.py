from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from policies.red.learning.unified_mappo_policy import UnifiedMAPPOSharedPolicy


def launch_policy_fixture() -> UnifiedMAPPOSharedPolicy:
    policy = UnifiedMAPPOSharedPolicy.__new__(UnifiedMAPPOSharedPolicy)
    policy.replay_probe = True
    policy.trainer = SimpleNamespace(device=torch.device("cpu"))
    policy._replay_spec = {
        "kind": "single",
        "interventions": [
            {
                "entity_id": 7,
                "agent_id": 3,
                "timestep": 1,
                "decision_type": "LAUNCH",
                "null_action": "WAIT",
            }
        ],
    }
    policy._prepared_replay_step = 1
    policy._replay_branch_role = "prefix"
    policy._replay_factual_pid = None
    policy._prepared_replay_semantics = {
        7: {
            "agent_id": 3,
            "kind": "launch",
            "current_target_id": None,
            "selected_target_id": 54,
            "selected_index": 0,
            "maneuver_active": False,
            "action_has_maneuver": True,
            "raw_maneuver": 1,
        }
    }
    policy._prepared_lifecycle = {
        7: {
            "kind": "launch",
            "target_index": 0,
            "target_id": 54,
            "initial_lon": 120.0,
            "initial_lat": 30.0,
        }
    }
    policy._prepared_maneuvers = {7: 2}
    policy._applied_interventions = []
    policy._targets = [
        {"entity_id": 54},
        {"entity_id": 95},
        {"entity_id": -100},
    ]
    return policy


class ReplayForkHookTests(unittest.TestCase):
    @patch(
        "policies.red.learning.unified_mappo_policy.torch.get_num_threads",
        return_value=1,
    )
    @patch("policies.red.learning.unified_mappo_policy.os.fork", return_value=0)
    def test_factual_child_keeps_prepared_joint_action(
        self,
        _fork,
        _threads,
    ) -> None:
        policy = launch_policy_fixture()

        policy.fork_and_apply_replay_interventions()

        self.assertEqual(policy.replay_branch_role, "factual")
        self.assertIsNone(policy._replay_spec)
        self.assertEqual(policy._prepared_lifecycle[7]["kind"], "launch")
        self.assertEqual(policy._prepared_maneuvers[7], 2)
        self.assertEqual(policy.applied_interventions, ())

    @patch(
        "policies.red.learning.unified_mappo_policy.torch.get_num_threads",
        return_value=1,
    )
    @patch("policies.red.learning.unified_mappo_policy.os.fork", return_value=4321)
    def test_counterfactual_parent_applies_only_requested_null(
        self,
        _fork,
        _threads,
    ) -> None:
        policy = launch_policy_fixture()
        untouched_lifecycle = {
            "kind": "keep",
            "target_index": 1,
            "target_id": 95,
            "initial_lon": 0.0,
            "initial_lat": 0.0,
        }
        policy._prepared_lifecycle[8] = dict(untouched_lifecycle)
        policy._prepared_maneuvers[8] = 0

        policy.fork_and_apply_replay_interventions()

        self.assertEqual(policy.replay_branch_role, "counterfactual")
        self.assertEqual(policy.replay_factual_pid, 4321)
        self.assertEqual(policy._prepared_lifecycle[7]["kind"], "wait")
        self.assertEqual(policy._prepared_lifecycle[7]["initial_lon"], 0.0)
        self.assertEqual(policy._prepared_lifecycle[7]["initial_lat"], 0.0)
        self.assertNotIn(7, policy._prepared_maneuvers)
        self.assertEqual(policy._prepared_lifecycle[8], untouched_lifecycle)
        self.assertEqual(policy._prepared_maneuvers[8], 0)
        self.assertEqual(
            policy.applied_interventions,
            (
                {
                    "entity_id": 7,
                    "agent_id": 3,
                    "timestep": 1,
                    "decision_type": "LAUNCH",
                    "null_action": "WAIT",
                    "application": "cow_tau_null_replacement",
                },
            ),
        )

    def test_retarget_null_restores_keep_and_current_target(self) -> None:
        policy = launch_policy_fixture()
        policy._replay_spec["interventions"][0].update({
            "decision_type": "RETARGET",
            "null_action": "KEEP",
        })
        policy._prepared_replay_semantics[7].update({
            "kind": "retarget",
            "current_target_id": 54,
            "selected_target_id": 95,
            "selected_index": 1,
            "maneuver_active": True,
        })
        policy._prepared_lifecycle[7].update({
            "kind": "retarget",
            "target_index": 1,
            "target_id": 95,
            "initial_lon": 0.0,
            "initial_lat": 0.0,
        })
        with patch(
            "policies.red.learning.unified_mappo_policy.torch.get_num_threads",
            return_value=1,
        ), patch(
            "policies.red.learning.unified_mappo_policy.os.fork",
            return_value=4321,
        ):
            policy.fork_and_apply_replay_interventions()

        self.assertEqual(policy._prepared_lifecycle[7]["kind"], "keep")
        self.assertEqual(policy._prepared_lifecycle[7]["target_id"], 54)
        self.assertEqual(policy._prepared_lifecycle[7]["target_index"], 0)

    def test_search_null_cancels_prelaunch_search(self) -> None:
        policy = launch_policy_fixture()
        policy._replay_spec["interventions"][0].update({
            "decision_type": "SEARCH",
            "null_action": "CANCEL_SEARCH",
        })
        policy._prepared_replay_semantics[7].update({
            "kind": "launch",
            "current_target_id": 54,
            "selected_target_id": -100,
            "selected_index": 2,
        })
        policy._prepared_lifecycle[7].update({
            "target_index": 2,
            "target_id": -100,
        })
        with patch(
            "policies.red.learning.unified_mappo_policy.torch.get_num_threads",
            return_value=1,
        ), patch(
            "policies.red.learning.unified_mappo_policy.os.fork",
            return_value=4321,
        ):
            policy.fork_and_apply_replay_interventions()

        self.assertEqual(policy._prepared_lifecycle[7]["kind"], "wait")
        self.assertEqual(policy._prepared_lifecycle[7]["target_id"], 54)
        self.assertNotIn(7, policy._prepared_maneuvers)

    def test_maneuver_null_sets_zero_index_without_changing_lifecycle(self) -> None:
        policy = launch_policy_fixture()
        policy._replay_spec["interventions"][0].update({
            "decision_type": "MANEUVER",
            "null_action": "ZERO_MANEUVER",
        })
        policy._prepared_replay_semantics[7].update({
            "kind": "keep",
            "maneuver_active": True,
            "raw_maneuver": 1,
        })
        policy._prepared_lifecycle[7]["kind"] = "keep"
        original_lifecycle = dict(policy._prepared_lifecycle[7])
        with patch(
            "policies.red.learning.unified_mappo_policy.torch.get_num_threads",
            return_value=1,
        ), patch(
            "policies.red.learning.unified_mappo_policy.os.fork",
            return_value=4321,
        ):
            policy.fork_and_apply_replay_interventions()

        self.assertEqual(policy._prepared_lifecycle[7], original_lifecycle)
        self.assertEqual(policy._prepared_maneuvers[7], 1)



if __name__ == "__main__":
    unittest.main()
