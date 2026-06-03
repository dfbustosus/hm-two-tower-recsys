import csv
import json
from pathlib import Path
from typing import Any

import pytest

from hm_recsys.embeddings.cache_io import (
    ARTICLE_EMBEDDING_MAPPING_HEADER,
    load_article_embedding_cache,
    load_article_embedding_mapping,
    validate_article_embedding_records,
)
from hm_recsys.embeddings.cache_manifest import ArticleEmbeddingCacheManifest
from hm_recsys.embeddings.contracts import ArticleEmbeddingRecord


def test_load_article_embedding_cache_from_jsonl(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "embeddings.jsonl"
    _write_jsonl_embeddings(
        embeddings_path,
        ("0108775015", (1.0, 0.0)),
        ("0201234567", (0.0, 1.0)),
    )
    manifest = _manifest(
        tmp_path,
        vector_format="jsonl",
        embeddings_path=embeddings_path,
        article_count=2,
        embedding_count=2,
        missing_embedding_count=0,
    )

    records = load_article_embedding_cache(manifest)

    assert records == (
        ArticleEmbeddingRecord("0108775015", (1.0, 0.0), "fashionclip"),
        ArticleEmbeddingRecord("0201234567", (0.0, 1.0), "fashionclip"),
    )


def test_load_article_embedding_cache_from_csv(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "embeddings.csv"
    with embeddings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("article_id", "v0", "v1"))
        writer.writerow(("0108775015", "1", "0"))
        writer.writerow(("0201234567", "0", "1"))
    manifest = _manifest(
        tmp_path,
        vector_format="csv",
        embeddings_path=embeddings_path,
        article_count=2,
        embedding_count=2,
        missing_embedding_count=0,
    )

    records = load_article_embedding_cache(manifest)

    assert records[0].article_id == "0108775015"
    assert records[0].vector == (1.0, 0.0)
    assert records[1].article_id == "0201234567"
    assert records[1].vector == (0.0, 1.0)


def test_load_article_embedding_mapping_validates_contiguous_indices(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ARTICLE_EMBEDDING_MAPPING_HEADER)
        writer.writerow(("0", "0108775015"))
        writer.writerow(("1", "0201234567"))

    assert load_article_embedding_mapping(mapping_path) == ("0108775015", "0201234567")

    bad_mapping_path = tmp_path / "bad_mapping.csv"
    with bad_mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ARTICLE_EMBEDDING_MAPPING_HEADER)
        writer.writerow(("1", "0108775015"))
    with pytest.raises(ValueError, match="contiguous"):
        load_article_embedding_mapping(bad_mapping_path)


def test_embedding_cache_loader_rejects_count_dimension_and_duplicate_errors(
    tmp_path: Path,
) -> None:
    embeddings_path = tmp_path / "embeddings.jsonl"
    _write_jsonl_embeddings(
        embeddings_path,
        ("0108775015", (1.0, 0.0)),
        ("0108775015", (0.0, 1.0)),
    )
    duplicate_manifest = _manifest(
        tmp_path,
        vector_format="jsonl",
        embeddings_path=embeddings_path,
        article_count=2,
        embedding_count=2,
        missing_embedding_count=0,
    )
    with pytest.raises(ValueError, match="duplicate article_id"):
        load_article_embedding_cache(duplicate_manifest)

    validate_manifest = _manifest(
        tmp_path,
        vector_format="jsonl",
        embeddings_path=embeddings_path,
        article_count=1,
        embedding_count=1,
        missing_embedding_count=0,
    )
    with pytest.raises(ValueError, match="dimension"):
        validate_article_embedding_records(
            (ArticleEmbeddingRecord("0201234567", (1.0,), "fashionclip"),),
            validate_manifest,
        )
    with pytest.raises(ValueError, match="provider"):
        validate_article_embedding_records(
            (ArticleEmbeddingRecord("0201234567", (1.0, 0.0), "openclip"),),
            validate_manifest,
        )
    with pytest.raises(ValueError, match="expected 1"):
        validate_article_embedding_records((), validate_manifest)


def test_embedding_cache_loader_rejects_malformed_jsonl_payload(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "embeddings.jsonl"
    embeddings_path.write_text('{"article_id":"0108775015","vector":[true,0]}\n', encoding="utf-8")
    manifest = _manifest(
        tmp_path,
        vector_format="jsonl",
        embeddings_path=embeddings_path,
        article_count=1,
        embedding_count=1,
        missing_embedding_count=0,
    )

    with pytest.raises(ValueError, match="boolean"):
        load_article_embedding_cache(manifest)


def _write_jsonl_embeddings(path: Path, *rows: tuple[str, tuple[float, ...]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for article_id, vector in rows:
            handle.write(json.dumps({"article_id": article_id, "vector": vector}) + "\n")


def _manifest(
    tmp_path: Path,
    *,
    vector_format: str,
    embeddings_path: Path,
    article_count: int,
    embedding_count: int,
    missing_embedding_count: int,
) -> ArticleEmbeddingCacheManifest:
    kwargs: dict[str, Any] = {
        "generated_at_utc": "2026-06-03T00:00:00+00:00",
        "provider_name": "fashionclip",
        "provider_model_id": "patrickjohncyh/fashion-clip",
        "provider_model_revision": "main",
        "embedding_kind": "multimodal",
        "dimension": 2,
        "distance_metric": "cosine",
        "normalized": True,
        "vector_format": vector_format,
        "dtype": "float32",
        "article_count": article_count,
        "embedding_count": embedding_count,
        "missing_embedding_count": missing_embedding_count,
        "source_article_content_path": str(tmp_path / "article_content.csv"),
        "source_image_inventory_path": str(tmp_path / "image_inventory.csv"),
        "embeddings_path": str(embeddings_path),
        "article_mapping_path": str(tmp_path / "mapping.csv"),
        "preprocessing": "test preprocessing",
        "license": "test license",
        "manifest_path": None,
    }
    return ArticleEmbeddingCacheManifest(**kwargs)
