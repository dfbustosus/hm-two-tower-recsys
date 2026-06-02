from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from hm_recsys.core.ids import is_article_id
from hm_recsys.embeddings.contracts import ArticleEmbeddingRecord, EmbeddingVector

DistanceMetric = Literal["cosine", "dot", "l2"]
ALLOWED_DISTANCE_METRICS = frozenset({"cosine", "dot", "l2"})


@dataclass(frozen=True)
class IndexSearchResult:
    article_id: str
    score: float

    def __post_init__(self) -> None:
        if not is_article_id(self.article_id):
            raise ValueError(f"invalid article_id: {self.article_id!r}")


@dataclass(frozen=True)
class VectorIndexConfig:
    name: str
    metric: DistanceMetric
    dimension: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("index name must not be empty")
        if self.metric not in ALLOWED_DISTANCE_METRICS:
            raise ValueError(f"unsupported distance metric: {self.metric!r}")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")


@runtime_checkable
class VectorIndex(Protocol):
    """Interface for exact or approximate nearest-neighbor indexes."""

    @property
    def config(self) -> VectorIndexConfig: ...

    def build(self, records: Iterable[ArticleEmbeddingRecord]) -> None: ...

    def query(self, vector: EmbeddingVector, top_k: int) -> tuple[IndexSearchResult, ...]: ...


IndexBuilder = Callable[[VectorIndexConfig], VectorIndex]


class VectorIndexFactory:
    """Registry-backed factory for interchangeable vector indexes."""

    def __init__(self) -> None:
        self._builders: dict[str, IndexBuilder] = {}

    def register(self, name: str, builder: IndexBuilder) -> None:
        if not name:
            raise ValueError("index provider name must not be empty")
        if name in self._builders:
            raise ValueError(f"index provider already registered: {name}")
        self._builders[name] = builder

    def create(self, provider_name: str, config: VectorIndexConfig) -> VectorIndex:
        try:
            return self._builders[provider_name](config)
        except KeyError as exc:
            available = ", ".join(self.available_provider_names()) or "<none>"
            message = f"unknown index provider {provider_name!r}; available: {available}"
            raise KeyError(message) from exc

    def available_provider_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))
