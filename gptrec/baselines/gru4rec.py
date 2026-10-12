"""GRU4Rec: a recurrent next-item model (Hidasi et al., 2016), here trained with the full softmax like the GPT recommender."""

from __future__ import annotations

from torch import Tensor, nn

from gptrec.data.log import PAD_ITEM
from gptrec.models.backbone import TransformerConfig, new_item_embedding
from gptrec.models.sequence_recommender import SequenceRecommender


class GRU4Rec(SequenceRecommender):
    """Uses the width, depth and dropout of `config`; a GRU needs neither attention heads nor positions."""

    def __init__(self, num_items: int, config: TransformerConfig) -> None:
        super().__init__()
        self.item_embedding = new_item_embedding(num_items, config.hidden_size, PAD_ITEM)
        self.input_dropout = nn.Dropout(config.dropout)
        dropout_between_layers = config.dropout if config.num_layers > 1 else 0.0
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=config.num_layers, batch_first=True, dropout=dropout_between_layers)

    def hidden_states(self, items: Tensor) -> Tensor:
        states, _ = self.gru(self.input_dropout(self.item_embedding(items)))
        return states
