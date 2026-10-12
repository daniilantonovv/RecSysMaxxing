"""All baselines, fitted on the same train period and the same windows as the GPT recommender."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple

import torch
from torch import Tensor, nn

from gptrec.baselines.bert4rec import BERT4Rec, cloze_loss, last_item_loss, one_target_per_window
from gptrec.baselines.counting import MarkovChainRecommender, PopularityRecommender, ReplayRecommender
from gptrec.baselines.gru4rec import GRU4Rec
from gptrec.baselines.sasrec import SASRec, sampled_binary_cross_entropy
from gptrec.data.batches import EventBatch
from gptrec.data.log import InteractionLog
from gptrec.models.backbone import TransformerConfig
from gptrec.models.sequence_recommender import NextItemScorer
from gptrec.training.precision import seed_everything
from gptrec.training.supervised import LossFunction, Metrics, OptimizationConfig, Validation, fit_on_train_period, next_item_loss, predicts_target


@dataclass(frozen=True)
class BaselineConfig:
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)  # shared by the neural baselines
    sasrec_negatives: int = 1  # sampled negatives per target, 1 in the SASRec paper
    bert4rec_mask_probability: float = 0.2
    markov_chain_successors: int = 200  # successors remembered per item


class _Recipe(NamedTuple):
    """How one neural baseline is built and trained."""

    model_class: Callable[[int, TransformerConfig], nn.Module]
    training_loss: LossFunction
    validation_loss: LossFunction
    scored_positions: Callable[[EventBatch], Tensor]


def fit_baselines(
    log: InteractionLog,
    transformer: TransformerConfig,
    config: BaselineConfig,
    window_stride: int,
    device: torch.device,
    checkpoint_directory: Path,
    extra_validation: Validation | None = None,
) -> tuple[dict[str, NextItemScorer], dict[str, list[Metrics]]]:
    """Fits every baseline on the train period.

    The neural baselines share the GPT recommender's width, depth, dropout and sequence length. Their weights
    are kept in `checkpoint_directory` and loaded instead of retrained when present. `extra_validation` adds
    metrics to each training epoch's record, e.g. a ranking metric that is comparable between models whose
    losses are not. Returns the baselines by name, and the training history of those trained in this call.
    """
    sasrec_loss = partial(sampled_binary_cross_entropy, num_negatives=config.sasrec_negatives)
    recipes = {
        "GRU4Rec": _Recipe(GRU4Rec, next_item_loss, next_item_loss, predicts_target),
        "SASRec": _Recipe(SASRec, sasrec_loss, sasrec_loss, predicts_target),
        "BERT4Rec": _Recipe(BERT4Rec, partial(cloze_loss, mask_probability=config.bert4rec_mask_probability), last_item_loss, one_target_per_window),
    }
    baselines: dict[str, NextItemScorer] = {
        "most popular": PopularityRecommender(log.items, device),
        "replay history": ReplayRecommender(log.items, device),
        "Markov chain": MarkovChainRecommender(log, device, config.markov_chain_successors),
    }
    histories: dict[str, list[Metrics]] = {}
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    for name, recipe in recipes.items():
        seed_everything(config.optimization.seed)
        model = recipe.model_class(log.num_items, transformer).to(device)
        checkpoint = checkpoint_directory / f"{name}.pt"
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        else:
            print(f"training {name}")
            histories[name] = fit_on_train_period(
                model,
                log,
                recipe.training_loss,
                recipe.scored_positions,
                transformer.max_positions,
                window_stride,
                config.optimization,
                device,
                validation_loss=recipe.validation_loss,
                extra_validation=extra_validation,
            )
            torch.save(model.state_dict(), checkpoint)
        baselines[name] = model.eval().requires_grad_(False)
    return baselines, histories
