"""The reward model: predicts how a user responds to an item, given the user's history."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from gptrec.data.log import PAD_ITEM
from gptrec.models.backbone import EMBEDDING_INIT_STD, CausalTransformer, TransformerConfig, new_item_embedding
from gptrec.tensors import TensorGroup

# The played ratio given for an item whose response nobody knows, e.g. a generated item. Only a reward model
# trained with hidden responses (`RewardModelConfig.unknown_feedback`) can read it.
UNKNOWN_FEEDBACK = -1.0


@dataclass
class Feedback(TensorGroup):
    """A user's response to items, one value per event. All tensors share one shape.

    Holds observed values, predicted expectations in [0, 1], or raw logits - the producer says which.
    """

    played_ratio: Tensor  # fraction of the item consumed
    session_continues: Tensor  # the user went on to consume another item in the same session
    returns: Tensor  # the user came back for another session within the return window
    is_chosen: Tensor  # the user took the item up at all: true of every logged event, so only a prediction can be below 1
    relative_ratio: Tensor  # the played ratio as a quantile among everyone who consumed the item: above 0.5 is more than usual
    liked: Tensor  # the user liked the item
    disliked: Tensor  # the user disliked the item
    adopts: Tensor  # the item was new to the user, and the user came back to it later


@dataclass(frozen=True)
class RewardModelConfig:
    num_items: int
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    num_ratio_buckets: int = 10
    unknown_feedback: bool = False  # one more input value: "the response to this item is not known"

    @classmethod
    def from_dict(cls, values: dict) -> RewardModelConfig:
        return cls(
            num_items=values["num_items"],
            transformer=TransformerConfig(**values["transformer"]),
            num_ratio_buckets=values["num_ratio_buckets"],
            unknown_feedback=values.get("unknown_feedback", False),
        )


class RewardModel(nn.Module):
    """Causal transformer over (item, feedback) events.

    The token at position t carries item t together with the feedback on item t - 1. The output at
    position t has therefore seen the candidate item t and the full history before it, but not the
    response to item t - which is exactly what it predicts.

    Three heads describe the response to an item the user consumed: how much of it, whether the session
    went on, whether the user came back. Logged events are all items somebody chose, so those heads say
    nothing about an item this user would never pick. The fourth head does: it tells the item the user
    chose from another item put in its place. The fifth predicts the played ratio relative to what is usual
    for the item, a response that no item earns just by being short or easy to finish. Three more describe
    explicit and lasting reactions where the log records them: a like, a dislike, and coming back to a new item.
    """

    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.transformer.hidden_size
        self.item_embedding = new_item_embedding(config.num_items, hidden_size, PAD_ITEM)
        # One input value per played-ratio bucket, one for "no previous event", and one for an unknown response where it is learned.
        self.previous_ratio_embedding = nn.Embedding(config.num_ratio_buckets + 1 + int(config.unknown_feedback), hidden_size)
        nn.init.normal_(self.previous_ratio_embedding.weight, std=EMBEDDING_INIT_STD)
        self.backbone = CausalTransformer(config.transformer)
        self.feedback_heads = nn.Linear(hidden_size, 5)
        # Created last, so that the parameters above start as they did before these heads existed.
        self.reaction_heads = nn.Linear(hidden_size, 3)

    def forward(self, items: Tensor, played_ratio: Tensor) -> Feedback:
        """Feedback logits for every position. items: long [B, L], played_ratio: float [B, L] -> [B, L] tensors.

        played_ratio[:, t] is read only by positions after t, so the last column may hold anything.
        """
        token_embeddings = self.item_embedding(items) + self.previous_ratio_embedding(self._previous_ratio_bucket(played_ratio))
        states = self.backbone(token_embeddings)
        played_ratio_logits, session_continues, returns, is_chosen, relative_ratio = self.feedback_heads(states).float().unbind(dim=-1)
        liked, disliked, adopts = self.reaction_heads(states).float().unbind(dim=-1)
        return Feedback(played_ratio_logits, session_continues, returns, is_chosen, relative_ratio, liked, disliked, adopts)

    def predict(self, items: Tensor, played_ratio: Tensor) -> Feedback:
        """Expected feedback in [0, 1] for every position."""
        return self.forward(items, played_ratio).map(torch.sigmoid)

    def _previous_ratio_bucket(self, played_ratio: Tensor) -> Tensor:
        num_buckets = self.config.num_ratio_buckets
        bucket = (played_ratio * num_buckets).long().clamp(0, num_buckets - 1)
        if self.config.unknown_feedback:
            bucket = bucket.masked_fill(played_ratio == UNKNOWN_FEEDBACK, num_buckets + 1)
        no_previous_event = torch.full_like(bucket[:, :1], num_buckets)
        return torch.cat([no_previous_event, bucket[:, :-1]], dim=1)
