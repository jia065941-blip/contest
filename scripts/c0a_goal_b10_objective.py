"""Full-legal expected official-return regret plus frozen-reference KL."""
from __future__ import annotations
import torch


def regret_loss(logits, old_logits, valid, indices, returns, candidate_mask,
                kl_coef=0.01):
    """Every row is one equally weighted state with all legal returns evaluated.

    No elite target distribution, candidate renormalization, span normalization,
    or gradients through labels/reference logits. KL is old || current.
    Equal-return and singleton states remain in the N-state average.
    """
    if kl_coef < 0:
        raise ValueError("kl_coef must be nonnegative")
    if logits.ndim != 2 or old_logits.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError("logits/reference/legal-mask shape mismatch")
    if indices.ndim != 2 or indices.shape != returns.shape or indices.shape != candidate_mask.shape or indices.shape[0] != logits.shape[0]:
        raise ValueError("candidate shape mismatch")
    if logits.shape[0] == 0:
        raise ValueError("empty batch")
    valid, candidate_mask = valid.bool(), candidate_mask.bool()
    if not valid.any(-1).all():
        raise ValueError("empty legal set")
    if not torch.isfinite(logits[valid]).all() or not torch.isfinite(old_logits[valid]).all():
        raise ValueError("nonfinite legal logits")
    if not torch.isfinite(returns[candidate_mask]).all():
        raise ValueError("nonfinite evaluated returns")
    active = indices[candidate_mask]
    if (active < 0).any() or (active >= logits.shape[1]).any():
        raise ValueError("candidate index out of bounds")
    safe_indices = indices.masked_fill(~candidate_mask, 0)
    coverage = torch.zeros_like(logits, dtype=torch.long)
    coverage.scatter_add_(1, safe_indices, candidate_mask.long())
    if not torch.equal(coverage, valid.long()):
        raise ValueError("regret requires every legal target exactly once; missing/duplicate/illegal candidate")
    labels = returns.detach().masked_fill(~candidate_mask, 0)
    full_returns = torch.zeros(logits.shape, dtype=labels.dtype, device=logits.device)
    full_returns.scatter_add_(1, safe_indices, labels)
    maximum = full_returns.masked_fill(~valid, -torch.inf).max(-1).values
    gaps = (maximum[:, None] - full_returns).masked_fill(~valid, 0).detach()
    logp = logits.masked_fill(~valid, -torch.inf).log_softmax(-1)
    old_logp = old_logits.detach().masked_fill(~valid, -torch.inf).log_softmax(-1)
    probabilities = logp.exp()
    state_regret = (probabilities * gaps).sum(-1)
    state_kl = (old_logp.exp() * (old_logp.masked_fill(~valid, 0) - logp.masked_fill(~valid, 0))).sum(-1)
    loss = (state_regret + kl_coef * state_kl).mean()
    return loss, {
        "regret": state_regret.mean(), "kl": state_kl.mean(),
        "state_regret": state_regret, "state_kl": state_kl,
        "probabilities": probabilities, "full_returns": full_returns,
        "maximum_returns": maximum,
        "expected_return": (probabilities * full_returns).sum(-1).mean(),
        "top1_return": full_returns.gather(1, logits.masked_fill(~valid, -torch.inf).argmax(-1)[:, None]).mean(),
    }
