from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from hm_recsys.core.ids import is_article_id

EmbeddingVector = tuple[float, ...]


@dataclass(frozen=True)
class ArticleEmbeddingInput:
    """Article content available to an embedding provider."""

    article_id: str
    text_fields: Mapping[str, str]
    image_path: Path | None = None

    def __post_init__(self) -> None:
        if not is_article_id(self.article_id):
            raise ValueError(f"invalid article_id: {self.article_id!r}")


@dataclass(frozen=True)
class ArticleEmbeddingRecord:
    """Versioned embedding tied to the exact article ID mapping."""

    article_id: str
    vector: EmbeddingVector
    provider_name: str

    def __post_init__(self) -> None:
        if not is_article_id(self.article_id):
            raise ValueError(f"invalid article_id: {self.article_id!r}")
        if not self.vector:
            raise ValueError("embedding vector must not be empty")
        if not self.provider_name:
            raise ValueError("provider_name must not be empty")

    @property
    def dimension(self) -> int:
        return len(self.vector)


@runtime_checkable
class ArticleEmbeddingProvider(Protocol):
    """Provider interface for local or external article embedding systems."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_articles(
        self, articles: Iterable[ArticleEmbeddingInput]
    ) -> Iterator[ArticleEmbeddingRecord]: ...


ProviderBuilder = Callable[[], ArticleEmbeddingProvider]


class EmbeddingProviderFactory:
    """Registry-backed factory for embedding providers."""

    def __init__(self) -> None:
        self._builders: dict[str, ProviderBuilder] = {}

    def register(self, name: str, builder: ProviderBuilder) -> None:
        if not name:
            raise ValueError("provider name must not be empty")
        if name in self._builders:
            raise ValueError(f"embedding provider already registered: {name}")
        self._builders[name] = builder

    def create(self, name: str) -> ArticleEmbeddingProvider:
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
        return tuple(sorted(self._builders))
