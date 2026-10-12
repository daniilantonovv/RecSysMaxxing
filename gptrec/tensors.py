"""A base for dataclasses that bundle tensors describing the same batch."""

from __future__ import annotations

from dataclasses import fields
from typing import Callable, Sequence, TypeVar

import torch
from torch import Tensor

Group = TypeVar("Group", bound="TensorGroup")


class TensorGroup:
    """Dataclass mixin: every field is a tensor, or another TensorGroup, and all share the leading dimension."""

    def map(self: Group, function: Callable[[Tensor], Tensor]) -> Group:
        """A new group with `function` applied to every tensor."""
        return type(self)(**{name: _map(value, function) for name, value in self._fields().items()})

    def to(self: Group, device: torch.device | str) -> Group:
        return self.map(lambda tensor: tensor.to(device))

    def cpu(self: Group) -> Group:
        return self.to("cpu")

    def __getitem__(self: Group, index) -> Group:
        return self.map(lambda tensor: tensor[index])

    def __len__(self) -> int:
        return len(next(iter(self._fields().values())))

    @classmethod
    def concatenate(cls: type[Group], parts: Sequence[Group], dim: int = 0) -> Group:
        return cls._combine(parts, lambda tensors: torch.cat(tensors, dim))

    @classmethod
    def stack(cls: type[Group], parts: Sequence[Group], dim: int = 0) -> Group:
        return cls._combine(parts, lambda tensors: torch.stack(tensors, dim))

    @classmethod
    def _combine(cls: type[Group], parts: Sequence[Group], combine: Callable[[list[Tensor]], Tensor]) -> Group:
        combined = {}
        for field in fields(cls):
            values = [getattr(part, field.name) for part in parts]
            is_nested = isinstance(values[0], TensorGroup)
            combined[field.name] = type(values[0])._combine(values, combine) if is_nested else combine(values)
        return cls(**combined)

    def _fields(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _map(value, function: Callable[[Tensor], Tensor]):
    return value.map(function) if isinstance(value, TensorGroup) else function(value)
