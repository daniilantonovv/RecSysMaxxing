"""Saving and restoring models together with the configuration needed to rebuild them."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from gptrec.models.policy import NextItemPolicy, PolicyConfig
from gptrec.models.reward_model import RewardModel, RewardModelConfig


def save_model(model: NextItemPolicy | RewardModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "weights": model.state_dict()}, path)


def load_policy(path: str | Path, device: torch.device | str = "cpu") -> NextItemPolicy:
    """The saved recommender, in evaluation mode."""
    checkpoint = _read(path)
    policy = NextItemPolicy(PolicyConfig.from_dict(checkpoint["config"]))
    policy.load_state_dict(checkpoint["weights"])
    return policy.to(device).eval()


def load_reward_model(path: str | Path, device: torch.device | str = "cpu") -> RewardModel:
    """The saved reward model, frozen and in evaluation mode."""
    checkpoint = _read(path)
    reward_model = RewardModel(RewardModelConfig.from_dict(checkpoint["config"]))
    reward_model.load_state_dict(checkpoint["weights"])
    return reward_model.to(device).eval().requires_grad_(False)


def _read(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True)
