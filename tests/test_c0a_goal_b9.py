"""Numerical and sampling contract checks for the b9 written protocol."""
from pathlib import Path
import sys
import unittest
import random
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from c0a_goal_b9_objective import elite_labels, goal_loss, sample_candidates
from c0a_goal_b9_runtime import (locked_teacher_actions, step_reward,
                                sync_teacher_commands, warm_teacher_on_replayed_state,
                                frozen_teacher_action)


class GoalProtocolTests(unittest.TestCase):
    def test_frozen_teacher_fast_action_is_exactly_equal(self):
        sys.path.insert(0, "/home/ubuntu/yuanlei/cz/competition-platform-env")
        from policies.red.learning.mappo_policy import MAPPOSharedPolicy
        layer = torch.nn.Linear(90, 3)
        policy = SimpleNamespace(training=False, device=torch.device("cpu"),
                                 network=SimpleNamespace(actor_forward=layer))
        for _ in range(100):
            observation = np.random.randn(90).astype(np.float32)
            mask = np.random.rand(3) > .5
            mask[0] = True
            self.assertEqual(MAPPOSharedPolicy.select_action(policy, observation, mask),
                             frozen_teacher_action(policy, observation, mask))

    def test_teacher_prefix_warmup_preserves_public_random_stream(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        def generate():
            random.random()
            np.random.rand()
            torch.rand(1)
            return ["updated_internal_reports"]
        result = warm_teacher_on_replayed_state(SimpleNamespace(_generate_actions_from_agents=generate))
        self.assertEqual(result, ["updated_internal_reports"])
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_equal(np.random.get_state(), before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_teacher_target_and_pending_launch_sync(self):
        target = SimpleNamespace(entity_id=51, position=SimpleNamespace(lon=120., lat=25.))
        commander = SimpleNamespace(targets=[target], target_by_platform={7: 52},
                                    assigned_by_target={52: 2, 51: 0}, pending={7: [1]})
        sync_teacher_commands(commander, [{"executor_id": 7, "commandType_id": 200,
                                           "target": {"x": 120., "y": 25.}}])
        self.assertEqual(commander.target_by_platform[7], 51)
        self.assertEqual(commander.assigned_by_target, {52: 1, 51: 1})
        self.assertNotIn(7, commander.pending)

    def test_small_sets_are_fully_enumerated(self):
        for n in range(1, 11):
            valid = torch.arange(25) < n
            indices, _ = sample_candidates(torch.arange(25).float(), valid)
            self.assertEqual(set(indices), set(range(n)))
            self.assertEqual(indices[0], n - 1)

    def test_top1_uniform_policy_slots_unique_and_legal(self):
        logits = torch.arange(25).float()
        valid = torch.ones(25, dtype=torch.bool)
        valid[[2, 10, 24]] = False
        seen_uniform = set()
        for seed in range(100):
            indices, roles = sample_candidates(logits, valid,
                generator=torch.Generator().manual_seed(seed))
            self.assertEqual(indices[0], 23)
            self.assertEqual(len(set(indices)), 10)
            self.assertTrue(valid[indices].all())
            self.assertEqual(roles.count("uniform"), 2)
            self.assertEqual(roles.count("policy"), 7)
            seen_uniform.update(indices[1:3])
        self.assertIn(0, seen_uniform)  # A low-policy-probability target is explored.

    def test_elite_ties_epsilon_and_padding(self):
        values = torch.tensor([[0., .5, 1., .91], [1., 1., 1., 1.],
                               [2., 99., 99., 99.], [0., 5e-7, 0., 0.]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1],
                             [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
        q, info, _, _ = elite_labels(values, mask)
        torch.testing.assert_close(q[0], torch.tensor([0., 0., .5, .5]))
        self.assertEqual(info.tolist(), [True, False, False, False])

    def test_loss_matches_formula_and_full_legal_kl(self):
        logits = torch.tensor([[.4, -.2, .8, 1.1]], requires_grad=True)
        old = torch.tensor([[.2, -.1, .1, -.4]], requires_grad=True)
        valid = torch.ones_like(logits, dtype=torch.bool)
        indices = torch.tensor([[0, 2]])
        returns = torch.tensor([[0., 1.]])
        mask = torch.ones_like(indices, dtype=torch.bool)
        loss, _ = goal_loss(logits, old, valid, indices, returns, mask, kl_coef=.2)
        expected_ce = -logits[:, [0, 2]].log_softmax(-1)[0, 1]
        expected_kl = (old.detach().softmax(-1) *
                       (old.detach().log_softmax(-1) - logits.log_softmax(-1))).sum()
        torch.testing.assert_close(loss, expected_ce + .2 * expected_kl)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertNotEqual(float(logits.grad[0, 3]), 0.)  # Legal noncandidate contributes to KL.

    def test_state_equal_weighting_and_no_factual_preference(self):
        logits = torch.zeros((2, 3), requires_grad=True)
        valid = torch.ones_like(logits, dtype=torch.bool)
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        values = torch.tensor([[0., 1., .5], [100., 0., 50.]])
        loss, _ = goal_loss(logits, logits.detach(), valid, indices, values, valid)
        loss.backward()
        torch.testing.assert_close(logits.grad[0, 1], logits.grad[1, 0])
        self.assertLess(float(logits.grad[1, 0]), 0.)  # First candidate may itself be best.

    def test_masked_slots_have_finite_gradients(self):
        logits = torch.zeros((2, 4), requires_grad=True)
        valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        indices = torch.tensor([[0, 1, 0], [0, 1, 2]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        returns = torch.tensor([[0., 1., 100.], [1., .95, 0.]])
        loss, _ = goal_loss(logits, logits.detach(), valid, indices, returns, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[~valid] == 0).all())

    def test_uninformative_batch_skips_update(self):
        logits = torch.zeros((1, 2))
        mask = torch.ones((1, 2), dtype=torch.bool)
        loss, _ = goal_loss(logits, logits, mask, torch.tensor([[0, 1]]),
                            torch.ones((1, 2)), mask)
        self.assertIsNone(loss)

    def test_formal_joint_reward_and_lock_release(self):
        reward = step_reward({51: 100., 52: 50.}, {51: 50., 52: 0.},
                             {51: 100., 52: 100.}, {51: 3., 52: 1.})
        self.assertAlmostEqual(reward, .5)
        commands = [{"executor_id": 7, "commandType_id": 3014},
                    {"executor_id": 7, "commandType_id": 3007},
                    {"executor_id": 8, "commandType_id": 3014}]
        kept, n = locked_teacher_actions(commands, 7, True)
        self.assertEqual(n, 1)
        self.assertEqual(kept, commands[1:])
        self.assertEqual(locked_teacher_actions(commands, 7, False), (commands, 0))


if __name__ == "__main__":
    unittest.main()
