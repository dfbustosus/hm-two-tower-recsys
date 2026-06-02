"""Vector index contracts for dense retrieval candidate generation."""

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
    """Single vector-index search result.

    Attributes:
        article_id: Retrieved H&M article identifier.
        score: Provider-specific similarity or distance score.
    """

    article_id: str
    score: float

    def __post_init__(self) -> None:
        """Validate the returned article identifier.

        Raises:
            ValueError: If ``article_id`` is malformed.
        """

        if not is_article_id(self.article_id):
            raise ValueError(f"invalid article_id: {self.article_id!r}")


@dataclass(frozen=True)
class VectorIndexConfig:
    """Runtime configuration for a vector index provider.

    Attributes:
        name: Logical index name used in artifacts and reports.
        metric: Distance or similarity metric used by the provider.
        dimension: Expected vector dimensionality.
    """

    name: str
    metric: DistanceMetric
    dimension: int

    def __post_init__(self) -> None:
        """Validate index configuration values.

        Raises:
            ValueError: If the name, metric, or dimension is invalid.
        """

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
    def config(self) -> VectorIndexConfig:
        """Return the immutable configuration used by this index."""
        ...

    def build(self, records: Iterable[ArticleEmbeddingRecord]) -> None:
        """Build or refresh the index from article embedding records.

        Args:
            records: Embedding records tied to exact article IDs.
        """
        ...

    def query(self, vector: EmbeddingVector, top_k: int) -> tuple[IndexSearchResult, ...]:
        """Retrieve nearest article embeddings for one query vector.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            Ranked search results from the index.
        """
        ...


IndexBuilder = Callable[[VectorIndexConfig], VectorIndex]


class VectorIndexFactory:
    """Registry-backed factory for interchangeable vector indexes."""

    def __init__(self) -> None:
        """Initialize an empty vector-index provider registry."""

        self._builders: dict[str, IndexBuilder] = {}

    def register(self, name: str, builder: IndexBuilder) -> None:
        """Register an index builder under a provider name.

        Args:
            name: Provider name used later in ``create``.
            builder: Callable that accepts ``VectorIndexConfig`` and returns an
                index instance.

        Raises:
            ValueError: If ``name`` is empty or already registered.
        """

        if not name:
            raise ValueError("index provider name must not be empty")
        if name in self._builders:
            raise ValueError(f"index provider already registered: {name}")
        self._builders[name] = builder

    def create(self, provider_name: str, config: VectorIndexConfig) -> VectorIndex:
        """Create a vector index from a registered provider.

        Args:
            provider_name: Registered provider name.
            config: Runtime index configuration passed to the builder.

        Returns:
            Constructed vector index.

        Raises:
            KeyError: If the provider name is unknown.
        """

        try:
            return self._builders[provider_name](config)
        except KeyError as exc:
            available = ", ".join(self.available_provider_names()) or "<none>"
            message = f"unknown index provider {provider_name!r}; available: {available}"
            raise KeyError(message) from exc

    def available_provider_names(self) -> tuple[str, ...]:
        """Return registered index provider names in deterministic order.

        Returns:
            Sorted provider names.
        """

        return tuple(sorted(self._builders))
