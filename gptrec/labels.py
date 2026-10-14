"""Names of the reported quantities in words, for figure titles and for the tables of the write-up.

The reports name every quantity by a path such as "next_k/liked/ndcg@10" or "predicted/heldout/played_ratio".
This module has no dependencies, so the scripts that typeset the reports can use it without the training stack.
"""

from __future__ import annotations

LIST_KINDS = {"top_k": "ranked lists", "next_k": "generated lists"}
TARGET_SETS = {
    "all": "all next items",
    "liked": "liked items",
    "skipped": "skipped items",
    "above_usual": "items consumed above usual",
    "below_usual": "items consumed below usual",
    "repeat": "items from the history",
    "new": "items not in the history",
}
RANKING_METRICS = {"ndcg": "NDCG", "recall": "Recall", "map": "MAP", "hit_rate": "HitRate"}
REWARD_MODELS = {"train": "training reward model", "heldout": "held-out reward model"}
SIGNALS = {
    "played_ratio": "played ratio",
    "played_ratio_for_other_user": "played ratio if shown to another user",
    "personal_played_ratio": "personal part of the played ratio",
    "relative_ratio": "played ratio relative to the item's usual",
    "personal_relative_ratio": "personal part of the relative played ratio",
    "session_continues": "probability that the session continues",
    "returns": "probability that the user returns",
    "is_chosen": "probability that the user chooses the item",
    "chosen_played_ratio": "played ratio weighted by choice",
    "played_seconds": "seconds played",
}
NAMED = {
    "num_users": "users",
    "profile/catalog_coverage": "catalogue coverage",
    "profile/item_gini": "Gini coefficient of item exposure",
    "profile/top_item_user_share": "share of users who get the most frequent item",
    "profile/mean_user_pair_overlap": "overlap between the lists of two users",
    "profile/intra_list_distance": "mean distance between two items of a list",
    "profile/duplicate_rate": "share of repeated items in a list",
    "profile/history_repeat_rate": "share of recommendations from the history",
    "profile/mean_popularity": "mean popularity of recommended items",
    "profile/head_item_share": "share of recommendations from the most popular 1% of items",
    "profile/mean_duration_seconds": "mean duration of recommended items, seconds",
    "profile/mean_item_played_ratio": "mean usual played ratio of recommended items",
    "logged/hits_per_list": "recommended items the user went on to consume, per list",
    "logged/played_ratio_of_hits": "logged played ratio of the items the user went on to consume",
    "logged/liked_share_of_hits": "share of those items the user liked",
    "logged/relative_ratio_of_hits": "logged relative played ratio of those items",
    "logged/played_ratio_per_item": "logged played ratio per recommended item",
    "logged/relative_surplus_per_item": "logged played ratio above the item's usual, per recommended item",
    "logged/served_hits_per_list": "recommended items a recommender later served, per list",
    "logged/played_ratio_of_served_hits": "logged played ratio of the items a recommender later served",
    "training/kl_to_reference": "KL to the base recommender, nats per item",
    "training/anchor_weight": "anchor weight at the end of fine-tuning",
    "training/clip_fraction": "share of actions beyond the clipping threshold",
    "training/value_explained_variance": "variance of returns explained by the value model",
    "policy_entropy": "entropy of the policy, nats",
    "in_true_continuation": "probability that the item is among the user's next items",
    "in_history": "probability that the item is in the history",
    "popularity": "popularity of the item",
    "duration_seconds": "duration of the item, seconds",
    "played_ratio_disagreement/train_vs_heldout": "disagreement of the two reward models on the played ratio",
    "reward": "reward per item",
    "kl_to_reference": "KL to the base recommender, nats per item",
    "reference_kl_weight": "anchor weight",
    "entropy": "entropy of the policy, nats",
    "distinct_action_share": "share of distinct items among those sampled",
    "logged_hit_rate": "share of sampled items among the user's next items",
    "history_repeat_rate": "share of sampled items from the history",
    "mean_action_popularity": "mean popularity of sampled items",
    "mean_action_duration_seconds": "mean duration of sampled items, seconds",
    "gradient_norm": "gradient norm",
    "clip_fraction": "share of actions beyond the clipping threshold",
    "value_explained_variance": "variance of returns explained by the value model",
    "train_loss": "training loss",
    "validation_loss": "validation loss",
}


def readable(metric: str) -> str:
    """The reported name of a quantity in words: "next_k/liked/ndcg@10" -> "NDCG@10, liked items (generated lists)".

    A name this module does not know is returned with its underscores turned into spaces.
    """
    qualifiers = []
    if metric.startswith("validation/"):
        metric, qualifiers = metric.removeprefix("validation/"), ["validation requests"]
    list_kind, _, quantity = metric.partition("/")
    if list_kind in LIST_KINDS:
        metric, qualifiers = quantity, [LIST_KINDS[list_kind], *qualifiers]
    label = _quantity(metric)
    return f"{label} ({', '.join(qualifiers)})" if qualifiers else label


def _quantity(metric: str) -> str:
    if metric in NAMED:
        return NAMED[metric]
    parts = metric.split("/")
    if len(parts) == 3 and parts[0] == "predicted":
        return _predicted(signal=parts[2], reward_model=parts[1])
    if len(parts) == 2 and parts[0].startswith("predicted_"):
        return _predicted(signal=parts[0].removeprefix("predicted_"), reward_model=parts[1])
    if len(parts) == 2 and parts[0] in TARGET_SETS and "@" in parts[1]:
        name, _, cutoff = parts[1].partition("@")
        if name in RANKING_METRICS:
            return f"{RANKING_METRICS[name]}@{cutoff}, {TARGET_SETS[parts[0]]}"
    return metric.replace("_", " ")


def _predicted(signal: str, reward_model: str) -> str:
    return f"predicted {SIGNALS.get(signal, signal.replace('_', ' '))}, {REWARD_MODELS.get(reward_model, reward_model)}"
