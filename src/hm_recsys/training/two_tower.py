from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NegativeSamplingStrategy = Literal[
    "in_batch",
    "random",
    "popularity_weighted",
    "same_category_hard",
    "mixed",
]
ALLOWED_NEGATIVE_SAMPLING_STRATEGIES = frozenset(
    {"in_batch", "random", "popularity_weighted", "same_category_hard", "mixed"}
)


@dataclass(frozen=True)
class TwoTowerTrainingConfig:
    """Validated contract for future two-tower retrieval experiments."""

    embedding_dim: int
    negative_sampling: NegativeSamplingStrategy
    batch_size: int
    epochs: int
    seed: int
    item_embedding_provider: str | None = None
    image_embedding_provider: str | None = None
    text_embedding_provider: str | None = None

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.negative_sampling not in ALLOWED_NEGATIVE_SAMPLING_STRATEGIES:
            raise ValueError(f"unsupported negative_sampling: {self.negative_sampling!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
