"""Embedding provider contracts for article content retrieval features."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from hm_recsys.core.ids import is_article_id

EmbeddingVector = tuple[float, ...]


@dataclass(frozen=True)
class ArticleEmbeddingInput:
    """Article content available to an embedding provider.

    Attributes:
        article_id: H&M article identifier being embedded.
        text_fields: Named article text fields, such as product name and detail
            description.
        image_path: Optional local path to the article image.
    """

    article_id: str
    text_fields: Mapping[str, str]
    image_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate the article identifier after initialization.

        Raises:
            ValueError: If ``article_id`` is malformed.
        """

        if not is_article_id(self.article_id):
            raise ValueError(f"invalid article_id: {self.article_id!r}")


@dataclass(frozen=True)
class ArticleEmbeddingRecord:
    """Versioned embedding tied to the exact article ID mapping.

    Attributes:
        article_id: H&M article identifier represented by the vector.
        vector: Dense embedding values in provider-defined coordinate space.
        provider_name: Name of the provider that generated the vector.
    """

    article_id: str
    vector: EmbeddingVector
    provider_name: str

    def __post_init__(self) -> None:
        """Validate embedding identity, vector shape, and provider metadata.

        Raises:
            ValueError: If the article ID, vector, or provider name is invalid.
        """

        if not is_article_id(self.article_id):
            raise ValueError(f"invalid article_id: {self.article_id!r}")
        if not self.vector:
            raise ValueError("embedding vector must not be empty")
        if not self.provider_name:
            raise ValueError("provider_name must not be empty")

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality.

        Returns:
            Number of floats in ``vector``.
        """

        return len(self.vector)


@runtime_checkable
class ArticleEmbeddingProvider(Protocol):
    """Provider interface for local or external article embedding systems."""

    @property
    def name(self) -> str:
        """Return the stable provider registry name."""
        ...

    @property
    def dimension(self) -> int:
        """Return the dimensionality emitted by this provider."""
        ...

    def embed_articles(
        self, articles: Iterable[ArticleEmbeddingInput]
    ) -> Iterator[ArticleEmbeddingRecord]:
        """Embed article inputs into dense vectors.

        Args:
            articles: Article content records to embed.

        Yields:
            One embedding record per successfully embedded article.
        """
        ...


ProviderBuilder = Callable[[], ArticleEmbeddingProvider]


class EmbeddingProviderFactory:
    """Registry-backed factory for embedding providers."""

    def __init__(self) -> None:
        """Initialize an empty embedding provider registry."""

        self._builders: dict[str, ProviderBuilder] = {}

    def register(self, name: str, builder: ProviderBuilder) -> None:
        """Register a provider builder under a stable name.

        Args:
            name: Provider name used later in ``create``.
            builder: Zero-argument callable that constructs the provider.

        Raises:
            ValueError: If the name is empty or already registered.
        """

        if not name:
            raise ValueError("provider name must not be empty")
        if name in self._builders:
            raise ValueError(f"embedding provider already registered: {name}")
        self._builders[name] = builder

    def create(self, name: str) -> ArticleEmbeddingProvider:
        """Construct a registered embedding provider.

        Args:
            name: Provider name to instantiate.

        Returns:
            Provider instance created by the registered builder.

        Raises:
            KeyError: If ``name`` is unknown.
            ValueError: If the builder returns a provider with a mismatched name.
        """

        try:
            provider = self._builders[name]()
        except KeyError as exc:
            available = ", ".join(self.available_provider_names()) or "<none>"
            raise KeyError(f"unknown embedding provider {name!r}; available: {available}") from exc
        if provider.name != name:
            raise ValueError(
                f"provider builder for {name!r} returned provider named {provider.name!r}"
            )
        return provider

    def available_provider_names(self) -> tuple[str, ...]:
        """Return registered provider names in deterministic order.

        Returns:
            Sorted provider names.
        """

        return tuple(sorted(self._builders))
