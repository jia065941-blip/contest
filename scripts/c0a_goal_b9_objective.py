"""Exact candidate sampling and state-equal elite objective for C0a_goal b9."""
from __future__ import annotations

import math
import torch


def sample_candidates(logits, valid, k=10, rho=0.2, generator=None):
    if k < 2 or not 0 <= rho <= 1:
        raise ValueError("K >= 2 and 0 <= rho <= 1 required")
    valid = valid.bool()
    legal = valid.nonzero(as_tuple=False).flatten()
    if not len(legal) or not torch.isfinite(logits[valid]).all():
        raise ValueError("empty legal set or nonfinite legal logits")
    top = int(logits.masked_fill(~valid, -torch.inf).argmax())
    if len(legal) <= k:
        indices = [top] + [int(i) for i in legal if int(i) != top]
        return indices, ["top1"] + ["enumerated"] * (len(indices) - 1)
    remaining = legal[legal != top]
    n_uniform = max(1, math.ceil(rho * (k - 1)))
    permutation = torch.randperm(len(remaining), generator=generator)
    uniform = remaining[permutation[:n_uniform]]
    remaining = remaining[permutation[n_uniform:]]
    n_policy = k - 1 - n_uniform
    # Float64 reduces underflow; exact zero support is handled explicitly.
    chosen_policy = []
    for _ in range(n_policy):
        probabilities = torch.softmax(logits[remaining].double(), -1)
        choice = int(torch.multinomial(probabilities, 1, generator=generator))
        chosen_policy.append(int(remaining[choice]))
        remaining = torch.cat((remaining[:choice], remaining[choice + 1:]))
    return ([top] + uniform.tolist() + chosen_policy,
            ["top1"] + ["uniform"] * n_uniform + ["policy"] * n_policy)


def elite_labels(returns, candidate_mask, epsilon=1e-6, xi=0.1):
    if epsilon < 0 or not 0 <= xi < 1:
        raise ValueError("epsilon >= 0 and 0 <= xi < 1 required")
    hi = returns.masked_fill(~candidate_mask, -torch.inf).max(-1).values
    lo = returns.masked_fill(~candidate_mask, torch.inf).min(-1).values
    span = hi - lo
    informative = (candidate_mask.sum(-1) >= 2) & (span > epsilon)
    elite = candidate_mask & (returns >= (hi - xi * span).unsqueeze(-1))
    q = elite.to(returns.dtype) / elite.sum(-1, keepdim=True).clamp_min(1)
    return q, informative, elite, span


def goal_loss(logits, old_logits, valid, indices, returns, candidate_mask,
              epsilon=1e-6, xi=0.1, kl_coef=0.01):
    q, informative, elite, span = elite_labels(returns, candidate_mask, epsilon, xi)
    if not bool(informative.any()):
        return None, {"informative": informative, "elite": elite, "span": span}
    candidate_logits = logits.gather(1, indices).masked_fill(~candidate_mask, -torch.inf)
    candidate_logp = torch.log_softmax(candidate_logits, -1)
    safe_candidate_logp = candidate_logp.masked_fill(~candidate_mask, 0)
    ce = -(q.detach() * safe_candidate_logp).sum(-1)
    logp = torch.log_softmax(logits.masked_fill(~valid, -torch.inf), -1)
    old_logp = torch.log_softmax(old_logits.detach().masked_fill(~valid, -torch.inf), -1)
    old_p = old_logp.exp()
    # Avoid 0 * (-inf - -inf) in masked slots, including the backward pass.
    kl = (old_p * (old_logp.masked_fill(~valid, 0) - logp.masked_fill(~valid, 0))).sum(-1)
    loss = (ce + kl_coef * kl)[informative].mean()
    return loss, {"informative": informative, "elite": elite, "span": span,
                  "cross_entropy": ce[informative].mean(), "kl": kl[informative].mean(),
                  "candidate_probabilities": candidate_logp.exp(), "q": q}
