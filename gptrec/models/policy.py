"""The recommender: GPT-2 over item tokens, trained to predict the next item."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from gptrec.data.log import PAD_ITEM
from gptrec.models.backbone import CausalTransformer, TransformerConfig, new_item_embedding
from gptrec.models.sequence_recommender import SequenceRecommender


@dataclass(frozen=True)
class PolicyConfig:
    num_items: int
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    @classmethod
    def from_dict(cls, values: dict) -> PolicyConfig:
        return cls(num_items=values["num_items"], transformer=TransformerConfig(**values["transformer"]))


class NextItemPolicy(SequenceRecommender):
    """pi(next item | items so far). Input and output share one item embedding table, as in GPT-2."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = new_item_embedding(config.num_items, config.transformer.hidden_size, PAD_ITEM)
        self.backbone = CausalTransformer(config.transformer)

    def hidden_states(self, items: Tensor) -> Tensor:
        return self.backbone(self.item_embedding(items))
