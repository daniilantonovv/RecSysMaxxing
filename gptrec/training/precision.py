"""Device, numeric precision and reproducibility helpers shared by every training and evaluation loop."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

FIRST_GPU_GENERATION_WITH_BFLOAT16 = 8  # CUDA compute capability of Ampere; older GPUs such as the T4 only emulate bfloat16


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixed_precision_dtype(device: torch.device) -> torch.dtype | None:
    """The reduced precision the device computes natively: bfloat16 on recent GPUs, float16 on older ones, none on a CPU."""
    if device.type != "cuda":
        return None
    gpu_generation, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if gpu_generation >= FIRST_GPU_GENERATION_WITH_BFLOAT16 else torch.float16


def autocast(device: torch.device, enabled: bool = True) -> torch.autocast:
    """Mixed-precision context for the device; a no-op on a CPU or when disabled."""
    dtype = mixed_precision_dtype(device)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and dtype is not None)


def gradient_scaler(device: torch.device, enabled: bool = True) -> torch.amp.GradScaler:
    """Loss scaling, which float16 needs to keep small gradients from underflowing; a pass-through at any other precision."""
    return torch.amp.GradScaler(device.type, enabled=enabled and mixed_precision_dtype(device) is torch.float16)


def clipped_optimizer_step(
    loss: Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    parameters: Iterable[nn.Parameter],
    max_grad_norm: float,
) -> Tensor:
    """Backpropagates `loss`, clips the gradient norm and updates the parameters. Returns the norm before clipping."""
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return gradient_norm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
