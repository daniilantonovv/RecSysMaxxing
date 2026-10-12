"""Is a difference between two recommenders more than noise between users?"""

from __future__ import annotations

import torch
from torch import Tensor

RESAMPLES_PER_PASS = 100  # bounds the memory of resampling many users


def paired_bootstrap(before: Tensor, after: Tensor, num_resamples: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean per-user change (after - before) with a 95% bootstrap interval over users.

    before, after: float [U], the same quantity for the same users under two recommenders. Users for whom
    either value is NaN are left out. The change is significant at the 5% level when the interval excludes 0.
    """
    defined = ~(before.isnan() | after.isnan())
    before, after = before[defined].double(), after[defined].double()
    change = after - before
    if len(change) == 0:
        return {name: float("nan") for name in ("before", "after", "change", "change_ci_low", "change_ci_high")} | {"num_users": 0.0}
    generator = torch.Generator().manual_seed(seed)
    resampled_means = []
    for _ in range(0, num_resamples, RESAMPLES_PER_PASS):
        users = torch.randint(len(change), (RESAMPLES_PER_PASS, len(change)), generator=generator)
        resampled_means.append(change[users].mean(dim=1))
    interval = torch.quantile(torch.cat(resampled_means), torch.tensor([0.025, 0.975], dtype=torch.double))
    return {
        "before": before.mean().item(),
        "after": after.mean().item(),
        "change": change.mean().item(),
        "change_ci_low": interval[0].item(),
        "change_ci_high": interval[1].item(),
        "num_users": float(len(change)),
    }
