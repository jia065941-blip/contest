"""Replace only Motion initialization; preserve baseline action and goal plumbing."""
from collections import Counter
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent


def install():
    import base_agents
    import torch
    from r3_runtime.local_runtime import LocalAvoidRuntimePolicy

    class R3Motion(base_agents.Motion):
        def __init__(self, max_steps):
            gate = os.environ['R3_GATE_MODE']
            if gate not in ('none', 'mission_v12'):
                raise ValueError('Invalid R3 gate')
            os.environ['LOCAL_AVOID_GATE_MODE'] = gate
            torch.set_num_threads(1)
            self.model_path = HERE / 'model.pt'
            self.policy = LocalAvoidRuntimePolicy(max_steps=max_steps)
            self.policy.load(str(self.model_path))
            assert self.policy.model.algorithm_name == 'local_avoid_set_attention_ppo_v1'
            assert self.policy.model.config.actor_obs_dim == 85
            assert self.policy.gate_mode == gate
            assert not self.policy.training and not self.policy._rollout_mode
            assert self.policy._trace_path is None
            self.calls, self.nonzero, self.audit = Counter(), Counter(), Counter()
            self.last_goal = {}
            encode = self.policy.encode_motion_observation
            mask = self.policy.prepare_action_mask

            def counted_encode(obs, **kw):
                result = encode(obs, **kw)
                assert result.shape == (85,)
                self.audit['encoded_85'] += 1
                goal = kw['goal_position']
                key, point = kw['agent_id'], (goal.lon, goal.lat)
                if key in self.last_goal and point != self.last_goal[key]:
                    self.audit['goal_coordinate_changes'] += 1
                self.last_goal[key] = point
                return result

            def counted_mask(obs, allowed):
                result = mask(obs, allowed)
                self.audit['mask_unrestricted' if result.all() else 'mask_restricted'] += 1
                return result

            self.policy.encode_motion_observation = counted_encode
            self.policy.prepare_action_mask = counted_mask

    base_agents.Motion = R3Motion
