"""The recommender: GPT-2 over item tokens, trained to predict the next item."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from gptrec.data.log import PAD_ITEM
from gptrec.models.backbone import CausalTransformer, TransformerConfig, new_item_embedding
from gptrec.models.sequence_recommender import MASKED_LOGIT, SequenceRecommender


@dataclass(frozen=True)
class PolicyConfig:
    num_items: int
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    item_offsets: bool = False  # one learned score offset per item, the same for every history

    @classmethod
    def from_dict(cls, values: dict) -> PolicyConfig:
        return cls(
            num_items=values["num_items"],
            transformer=TransformerConfig(**values["transformer"]),
            item_offsets=values.get("item_offsets", False),
        )


class NextItemPolicy(SequenceRecommender):
    """pi(next item | items so far). Input and output share one item embedding table, as in GPT-2.

    With `config.item_offsets` every item's score gets an offset of its own, added whatever the history: all that
    a fine-tuning which may only re-weight the catalogue can change.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = new_item_embedding(config.num_items, config.transformer.hidden_size, PAD_ITEM)
        self.backbone = CausalTransformer(config.transformer)
        self.register_parameter("item_offsets", nn.Parameter(torch.zeros(config.num_items + 1)) if config.item_offsets else None)

    def hidden_states(self, items: Tensor) -> Tensor:
        return self.backbone(self.item_embedding(items))

    def item_logits(self, hidden: Tensor) -> Tensor:
        logits = super().item_logits(hidden)
        if self.item_offsets is None:
            return logits
        logits = logits + self.item_offsets
        logits[..., PAD_ITEM] = MASKED_LOGIT
        return logits
