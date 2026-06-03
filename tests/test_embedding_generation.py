import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from hm_recsys.embeddings.cache_manifest import read_article_embedding_cache_manifest
from hm_recsys.embeddings.contracts import (
    ArticleEmbeddingInput,
    ArticleEmbeddingProvider,
    ArticleEmbeddingRecord,
)
from hm_recsys.embeddings.generation import (
    ArticleEmbeddingCacheWriteConfig,
    write_article_embedding_cache_from_content_export,
)


def test_write_article_embedding_cache_from_content_export(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content_path = tmp_path / "article_content.csv"
    embeddings_path = tmp_path / "models" / "embeddings.jsonl"
    mapping_path = tmp_path / "models" / "mapping.csv"
    manifest_path = tmp_path / "models" / "manifest.json"
    content_path.write_text(
        "article_id,combined_text,image_relative_path,image_exists,prod_name\n"
        "0108775015,shirt,images/010/0108775015.jpg,true,shirt\n"
        "0201234567,trousers,,false,trousers\n",
        encoding="utf-8",
    )

    summary = write_article_embedding_cache_from_content_export(
        FakeProvider(),
        raw_data_dir=raw_dir,
        article_content_path=content_path,
        embeddings_path=embeddings_path,
        article_mapping_path=mapping_path,
        manifest_path=manifest_path,
        config=_config(max_articles=None),
    )

    assert summary.article_count == 2
    assert summary.embedding_count == 2
    assert summary.missing_embedding_count == 0
    assert summary.skipped_missing_image_count == 0
    rows = [json.loads(line) for line in embeddings_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"article_id": "0108775015", "vector": [1.0, 0.0]},
        {"article_id": "0201234567", "vector": [0.0, 1.0]},
    ]
    assert mapping_path.read_text(encoding="utf-8").splitlines() == [
        "embedding_index,article_id",
        "0,0108775015",
        "1,0201234567",
    ]
    manifest = read_article_embedding_cache_manifest(manifest_path)
    assert manifest.provider_name == "fake-provider"
    assert manifest.embedding_count == 2
    assert manifest.vector_format == "jsonl"


def test_write_article_embedding_cache_skips_missing_images_for_image_kind(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content_path = tmp_path / "article_content.csv"
    content_path.write_text(
        "article_id,combined_text,image_relative_path,image_exists,prod_name\n"
        "0108775015,shirt,images/010/0108775015.jpg,true,shirt\n"
        "0201234567,trousers,,false,trousers\n",
        encoding="utf-8",
    )

    summary = write_article_embedding_cache_from_content_export(
        FakeProvider(),
        raw_data_dir=raw_dir,
        article_content_path=content_path,
        embeddings_path=tmp_path / "embeddings.jsonl",
        article_mapping_path=tmp_path / "mapping.csv",
        manifest_path=tmp_path / "manifest.json",
        config=_config(embedding_kind="image"),
    )

    assert summary.article_count == 2
    assert summary.embedding_count == 1
    assert summary.missing_embedding_count == 1
    assert summary.skipped_missing_image_count == 1


def test_write_article_embedding_cache_counts_provider_skips_and_max_articles(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content_path = tmp_path / "article_content.csv"
    content_path.write_text(
        "article_id,combined_text,image_relative_path,image_exists,prod_name\n"
        "0108775015,shirt,,false,shirt\n"
        "0201234567,trousers,,false,trousers\n"
        "0300000000,dress,,false,dress\n",
        encoding="utf-8",
    )

    summary = write_article_embedding_cache_from_content_export(
        FakeProvider(skip_article_id="0201234567"),
        raw_data_dir=raw_dir,
        article_content_path=content_path,
        embeddings_path=tmp_path / "embeddings.jsonl",
        article_mapping_path=tmp_path / "mapping.csv",
        manifest_path=tmp_path / "manifest.json",
        config=_config(max_articles=2),
    )

    assert summary.article_count == 2
    assert summary.embedding_count == 1
    assert summary.missing_embedding_count == 1
    assert summary.max_articles == 2


def test_embedding_cache_writer_rejects_provider_contract_violations(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content_path = tmp_path / "article_content.csv"
    content_path.write_text(
        "article_id,combined_text,image_relative_path,image_exists,prod_name\n"
        "0108775015,shirt,,false,shirt\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provider_name"):
        write_article_embedding_cache_from_content_export(
            FakeProvider(provider_name="wrong-provider", emitted_provider_name="other-provider"),
            raw_data_dir=raw_dir,
            article_content_path=content_path,
            embeddings_path=tmp_path / "embeddings.jsonl",
            article_mapping_path=tmp_path / "mapping.csv",
            manifest_path=tmp_path / "manifest.json",
            config=_config(),
        )
    with pytest.raises(ValueError, match="dimension"):
        write_article_embedding_cache_from_content_export(
            FakeProvider(vector=(1.0,)),
            raw_data_dir=raw_dir,
            article_content_path=content_path,
            embeddings_path=tmp_path / "embeddings.jsonl",
            article_mapping_path=tmp_path / "mapping.csv",
            manifest_path=tmp_path / "manifest.json",
            config=_config(),
        )


def test_embedding_cache_write_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="provider_model_id"):
        _config(provider_model_id="")
    with pytest.raises(ValueError, match="batch_size"):
        _config(batch_size=0)
    with pytest.raises(ValueError, match="max_articles"):
        _config(max_articles=0)


class FakeProvider(ArticleEmbeddingProvider):
    def __init__(
        self,
        *,
        provider_name: str = "fake-provider",
        emitted_provider_name: str | None = None,
        vector: tuple[float, ...] | None = None,
        skip_article_id: str | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._emitted_provider_name = emitted_provider_name or provider_name
        self._vector = vector
        self._skip_article_id = skip_article_id

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def dimension(self) -> int:
        return 2

    def embed_articles(
        self, articles: Iterable[ArticleEmbeddingInput]
    ) -> Iterator[ArticleEmbeddingRecord]:
        for article in articles:
            if article.article_id == self._skip_article_id:
                continue
            vector = self._vector or _vector_for_article(article.article_id)
            yield ArticleEmbeddingRecord(article.article_id, vector, self._emitted_provider_name)


def _vector_for_article(article_id: str) -> tuple[float, float]:
    if article_id == "0108775015":
        return (1.0, 0.0)
    return (0.0, 1.0)


def _config(**overrides: object) -> ArticleEmbeddingCacheWriteConfig:
    kwargs = {
        "provider_model_id": "fake/model",
        "provider_model_revision": "test",
        "embedding_kind": "multimodal",
        "preprocessing": "fake preprocessing",
        "license": "fake license",
        "batch_size": 2,
        "max_articles": None,
    }
    kwargs.update(overrides)
    return ArticleEmbeddingCacheWriteConfig(**kwargs)  # type: ignore[arg-type]
