"""Lists predicted from the real history alone: one head per position (multi-token prediction, Gloeckle et al. 2024).

A generated list feeds every chosen item back as if the user had consumed it, which most were not. Head k here
predicts the item k events ahead from the real history, so no position of the list rests on an unconfirmed one.
The heads sit on the frozen base recommender; head 1 is the recommender itself.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gptrec.data.batches import EventBatch, HistoryBatch
from gptrec.data.log import InteractionLog
from gptrec.models.policy import NextItemPolicy
from gptrec.models.sequence_recommender import SequenceRecommender
from gptrec.rollout import GenerationConfig, constrain_logits
from gptrec.training.precision import seed_everything
from gptrec.training.supervised import Metrics, OptimizationConfig, fit_on_train_period


class MultiTokenHeads(nn.Module):
    """Heads 2..K: each maps the recommender's state to the state from which its item k events ahead is scored.

    Each starts as the identity, that is as a copy of head 1, the recommender's own next-item head.
    """

    def __init__(self, hidden_size: int, horizon: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.transforms = nn.ModuleList(_identity(hidden_size) for _ in range(horizon - 1))

    def logits(self, recommender: SequenceRecommender, hidden: Tensor, offset: int) -> Tensor:
        """Scores of the item `offset` events ahead (1: the next one) from recommender states. [..., hidden] -> [..., V+1]"""
        return recommender.item_logits(hidden if offset == 1 else self.transforms[offset - 2](hidden))


def train_multi_token_heads(
    recommender: NextItemPolicy,
    log: InteractionLog,
    horizon: int,
    config: OptimizationConfig,
    window_stride: int,
    device: torch.device,
    max_targets_per_head: int | None = None,
) -> tuple[MultiTokenHeads, list[Metrics]]:
    """Trains heads 2..`horizon` on the train period's windows; the recommender stays as it is.

    max_targets_per_head: positions per batch that each head learns from, drawn at random; bounds the memory of
    scoring a large catalogue K - 1 times per position. None: every position.
    """
    seed_everything(config.seed)
    recommender = recommender.to(device).eval().requires_grad_(False)
    heads = MultiTokenHeads(recommender.config.transformer.hidden_size, horizon).to(device)
    loss = partial(multi_token_loss, recommender=recommender, max_targets_per_head=max_targets_per_head)
    window_length = recommender.config.transformer.max_positions
    history = fit_on_train_period(heads, log, loss, lambda batch: batch.is_target, window_length, window_stride, config, device)
    return heads.eval().requires_grad_(False), history


def multi_token_loss(heads: MultiTokenHeads, batch: EventBatch, recommender: NextItemPolicy, max_targets_per_head: int | None = None) -> Tensor:
    """Mean over heads 2..K of the cross-entropy of the target item that many events ahead of each position."""
    with torch.no_grad():
        hidden = recommender.hidden_states(batch.items)
    losses = []
    for offset in range(2, min(heads.horizon, batch.items.shape[1] - 1) + 1):
        selected = batch.is_target[:, offset:]
        if selected.any():
            states, targets = hidden[:, :-offset][selected], batch.items[:, offset:][selected]
            if max_targets_per_head is not None and len(targets) > max_targets_per_head:
                kept = torch.randperm(len(targets), device=targets.device)[:max_targets_per_head]
                states, targets = states[kept], targets[kept]
            losses.append(F.cross_entropy(heads.logits(recommender, states, offset), targets))
    if not losses:
        return sum(parameter.sum() for parameter in heads.parameters()) * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def multi_token_lists(recommender: NextItemPolicy, heads: MultiTokenHeads, history: HistoryBatch, generation: GenerationConfig) -> Tensor:
    """For each history, position k of the list is head k's best item among those the list constraints still allow. long [B, K]"""
    state = recommender.hidden_states(history.items)[torch.arange(len(history), device=history.items.device), history.lengths - 1]
    chosen = history.items[:, :0]
    for offset in range(1, heads.horizon + 1):
        logits = constrain_logits(heads.logits(recommender, state, offset), history.items, chosen, generation)
        chosen = torch.cat([chosen, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return chosen


def save_multi_token_heads(heads: MultiTokenHeads, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden_size": heads.hidden_size, "horizon": heads.horizon, "weights": heads.state_dict()}, path)


def load_multi_token_heads(path: str | Path, device: torch.device | str = "cpu") -> MultiTokenHeads:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    heads = MultiTokenHeads(checkpoint["hidden_size"], checkpoint["horizon"])
    heads.load_state_dict(checkpoint["weights"])
    return heads.to(device).eval().requires_grad_(False)


def _identity(size: int) -> nn.Linear:
    layer = nn.Linear(size, size)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(size))
        layer.bias.zero_()
    return layer
