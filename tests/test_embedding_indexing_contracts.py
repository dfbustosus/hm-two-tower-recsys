from collections.abc import Iterable, Iterator
from typing import Any, cast

import pytest

from hm_recsys.embeddings.contracts import (
    ArticleEmbeddingInput,
    ArticleEmbeddingProvider,
    ArticleEmbeddingRecord,
    EmbeddingProviderFactory,
)
from hm_recsys.indexing.contracts import (
    IndexSearchResult,
    VectorIndex,
    VectorIndexConfig,
    VectorIndexFactory,
)
from hm_recsys.training.two_tower import TwoTowerTrainingConfig

ARTICLE_ID = "0000000001"


class DeterministicTestEmbeddingProvider:
    @property
    def name(self) -> str:
        return "deterministic-test"

    @property
    def dimension(self) -> int:
        return 2

    def embed_articles(
        self, articles: Iterable[ArticleEmbeddingInput]
    ) -> Iterator[ArticleEmbeddingRecord]:
        for article in articles:
            yield ArticleEmbeddingRecord(
                article_id=article.article_id,
                vector=(1.0, 0.0),
                provider_name=self.name,
            )


class InMemoryTestIndex:
    def __init__(self, config: VectorIndexConfig) -> None:
        self._config = config
        self._records: list[ArticleEmbeddingRecord] = []

    @property
    def config(self) -> VectorIndexConfig:
        return self._config

    def build(self, records: Iterable[ArticleEmbeddingRecord]) -> None:
        self._records = list(records)

    def query(self, vector: tuple[float, ...], top_k: int) -> tuple[IndexSearchResult, ...]:
        del vector
        return tuple(IndexSearchResult(record.article_id, 1.0) for record in self._records[:top_k])


def test_embedding_provider_factory_registers_and_creates_provider() -> None:
    factory = EmbeddingProviderFactory()
    factory.register("deterministic-test", DeterministicTestEmbeddingProvider)

    provider = factory.create("deterministic-test")
    records = tuple(
        provider.embed_articles(
            [ArticleEmbeddingInput(article_id=ARTICLE_ID, text_fields={"prod_name": "shirt"})]
        )
    )

    assert isinstance(provider, ArticleEmbeddingProvider)
    assert factory.available_provider_names() == ("deterministic-test",)
    assert records[0].article_id == ARTICLE_ID
    assert records[0].dimension == 2


def test_embedding_provider_factory_rejects_unknown_provider() -> None:
    factory = EmbeddingProviderFactory()

    with pytest.raises(KeyError, match="unknown embedding provider"):
        factory.create("missing")


def test_vector_index_factory_registers_and_creates_index() -> None:
    factory = VectorIndexFactory()
    factory.register("in-memory-test", InMemoryTestIndex)
    config = VectorIndexConfig(name="article-index", metric="cosine", dimension=2)

    index = factory.create("in-memory-test", config)
    index.build([ArticleEmbeddingRecord(ARTICLE_ID, (1.0, 0.0), "deterministic-test")])

    assert isinstance(index, VectorIndex)
    assert index.query((1.0, 0.0), top_k=1) == (IndexSearchResult(ARTICLE_ID, 1.0),)


def test_two_tower_training_config_validates_positive_dimensions() -> None:
    config = TwoTowerTrainingConfig(
        embedding_dim=128,
        negative_sampling="in_batch",
        batch_size=1024,
        epochs=3,
        seed=42,
        image_embedding_provider="clip-image-v1",
        text_embedding_provider="clip-text-v1",
    )

    assert config.embedding_dim == 128


def test_two_tower_training_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TwoTowerTrainingConfig(
            embedding_dim=128,
            negative_sampling="in_batch",
            batch_size=0,
            epochs=3,
            seed=42,
        )


def test_vector_index_config_rejects_invalid_metric() -> None:
    with pytest.raises(ValueError, match="unsupported distance metric"):
        VectorIndexConfig(name="article-index", metric=cast(Any, "bad"), dimension=2)


def test_two_tower_training_config_rejects_invalid_negative_sampling() -> None:
    with pytest.raises(ValueError, match="unsupported negative_sampling"):
        TwoTowerTrainingConfig(
            embedding_dim=128,
            negative_sampling=cast(Any, "bad"),
            batch_size=1024,
            epochs=3,
            seed=42,
        )
