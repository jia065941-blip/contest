"""Evidence checks: hidden state changes and equal-score trajectory divergence."""
from pathlib import Path
from types import SimpleNamespace
from collections import deque
import random
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from c0a_goal_b9_identity import fingerprint

class IdentityEvidenceTests(unittest.TestCase):
    def test_hidden_queue_and_private_rng_are_observed(self):
        obj = SimpleNamespace(observation={'health': 100}, pending=deque([1]), rng=random.Random(42))
        before = fingerprint(obj)
        self.assertEqual(before, fingerprint(obj))
        obj.pending.append(2)
        after = fingerprint(obj)
        self.assertNotEqual(before['sha256'], after['sha256'])
        obj.rng.random()
        self.assertNotEqual(after['sha256'], fingerprint(obj)['sha256'])

    def test_slots_and_cycles_are_observed(self):
        class SlotState:
            __slots__ = ('value', 'owner')
        obj = SlotState()
        obj.value, obj.owner = 1, obj
        before = fingerprint(obj)
        obj.value = 2
        self.assertNotEqual(before['sha256'], fingerprint(obj)['sha256'])
        self.assertEqual(before['opaque_types'], [])

    def test_complete_observation_includes_detection_objects(self):
        obj = {'entities': {1: {'health': 100, 'detect': SimpleNamespace(track=1)}}}
        before = fingerprint(obj)
        obj['entities'][1]['detect'].track = 2
        self.assertNotEqual(before['sha256'], fingerprint(obj)['sha256'])

if __name__ == '__main__':
    unittest.main()
