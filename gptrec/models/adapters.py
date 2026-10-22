"""What fine-tuning may change in the recommender, and operating points reached in weight space without training.

Fine-tuning can be held to one score offset per item, the same for every history, or to rank-r updates of the
transformer's projections (LoRA, Hu et al. 2022), which change how a history is read but no item's embedding.
Moving the weights along the line from the base recommender to a fine-tuned one, or past it, gives operating
points without further training (WiSE-FT, Wortsman et al. 2022; WARP, Ramé et al. 2024; ExPO, Zheng et al. 2024).
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gptrec.models.policy import NextItemPolicy


class Adaptation(str, Enum):
    """What fine-tuning may change in the recommender."""

    FULL = "full"  # every weight
    ITEM_OFFSETS = "item_offsets"  # one score offset per item, the same for every history: a re-weighting of the catalogue
    LOW_RANK = "low_rank"  # rank-r updates of the transformer's projections; the item embeddings stay as they are


class LowRankUpdate(nn.Module):
    """A frozen projection plus a trainable update of rank r. The update starts at zero, so the module starts as the projection."""

    def __init__(self, projection: nn.Module, rank: int) -> None:
        super().__init__()
        in_features, out_features = projection_shape(projection)
        self.projection = projection.requires_grad_(False)
        self.down = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.up = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs) + F.linear(F.linear(inputs, self.down), self.up)

    @torch.no_grad()
    def merged(self) -> nn.Module:
        """The projection with the update folded into its weight."""
        update = (self.up @ self.down).to(self.projection.weight.dtype)
        self.projection.weight += update if isinstance(self.projection, nn.Linear) else update.T
        return self.projection


def projection_shape(projection: nn.Module) -> tuple[int, int]:
    """(input features, output features) of a linear layer, or of GPT-2's Conv1D, which stores its weight as [in, out]."""
    if isinstance(projection, nn.Linear):
        return projection.in_features, projection.out_features
    in_features, out_features = projection.weight.shape
    return in_features, out_features


def add_low_rank_updates(module: nn.Module, rank: int) -> nn.Module:
    """Freezes `module` and gives every projection inside it a trainable update of rank `rank`. Returns `module`."""
    module.requires_grad_(False)
    projections = [(parent, name, child) for parent in module.modules() for name, child in parent.named_children() if _is_projection(child)]
    for parent, name, child in projections:
        setattr(parent, name, LowRankUpdate(child, rank))
    return module


def merge_low_rank_updates(module: nn.Module) -> nn.Module:
    """Folds every low-rank update inside `module` into its projection, which restores the module's own structure. Returns `module`."""
    updates = [(parent, name, child) for parent in module.modules() for name, child in parent.named_children() if isinstance(child, LowRankUpdate)]
    for parent, name, child in updates:
        setattr(parent, name, child.merged())
    return module


def with_item_offsets(policy: NextItemPolicy, offsets: Tensor | None = None) -> NextItemPolicy:
    """A copy of `policy` that adds offsets[j] (float [V+1]) to the score of every item j: zero offsets unless given."""
    shifted = NextItemPolicy(replace(policy.config, item_offsets=True))
    missing, unexpected = shifted.load_state_dict(policy.state_dict(), strict=False)
    if unexpected or set(missing) - {"item_offsets"}:
        raise ValueError(f"the recommender's weights do not fit its configuration: missing {missing}, unexpected {unexpected}")
    if offsets is not None:
        with torch.no_grad():
            shifted.item_offsets += offsets.to(shifted.item_offsets)
    return shifted.to(policy.item_embedding.weight.device).eval()


def adaptable_copy(policy: NextItemPolicy, adaptation: Adaptation, rank: int = 1) -> NextItemPolicy:
    """A copy of `policy` in which exactly the parameters that `adaptation` allows to change require gradients."""
    adaptation = Adaptation(adaptation)
    if adaptation is Adaptation.FULL:
        return copy.deepcopy(policy).requires_grad_(True)
    if adaptation is Adaptation.ITEM_OFFSETS:
        adapted = with_item_offsets(policy).requires_grad_(False)
        adapted.item_offsets.requires_grad_(True)
        return adapted
    adapted = copy.deepcopy(policy).requires_grad_(False)
    add_low_rank_updates(adapted.backbone, rank)
    return adapted


def interpolated(base: NextItemPolicy, finetuned: NextItemPolicy, factor: float) -> NextItemPolicy:
    """The recommender with weights theta_0 + factor * (theta - theta_0): 0 gives `base`, 1 `finetuned`, above 1 extrapolates past it."""
    if base.config != finetuned.config:
        raise ValueError("only recommenders of the same configuration can be interpolated")
    base_weights = base.state_dict()
    blended = copy.deepcopy(finetuned)
    blended.load_state_dict({name: base_weights[name] + factor * (weight - base_weights[name]) for name, weight in finetuned.state_dict().items()})
    return blended.eval()


def _is_projection(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or type(module).__name__ == "Conv1D"
