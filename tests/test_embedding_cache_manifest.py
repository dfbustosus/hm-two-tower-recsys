import json
from pathlib import Path
from typing import Any, cast

import pytest

from hm_recsys.embeddings.cache_manifest import (
    ArticleEmbeddingCacheManifest,
    build_article_embedding_cache_manifest,
    read_article_embedding_cache_manifest,
    write_article_embedding_cache_manifest,
)


def test_build_article_embedding_cache_manifest_resolves_paths_and_counts(tmp_path: Path) -> None:
    manifest = build_article_embedding_cache_manifest(
        provider_name="fashionclip",
        provider_model_id="patrickjohncyh/fashion-clip",
        provider_model_revision="main",
        embedding_kind="multimodal",
        dimension=512,
        distance_metric="cosine",
        normalized=True,
        vector_format="npy",
        dtype="float32",
        article_count=10,
        embedding_count=8,
        source_article_content_path=tmp_path / "article_content.csv",
        source_image_inventory_path=tmp_path / "article_image_inventory.csv",
        embeddings_path=tmp_path / "embeddings.npy",
        article_mapping_path=tmp_path / "article_ids.csv",
        preprocessing="FashionCLIP processor defaults; missing images skipped",
        license="Model license documented in provider card",
    )

    assert manifest.provider_name == "fashionclip"
    assert manifest.embedding_kind == "multimodal"
    assert manifest.dimension == 512
    assert manifest.missing_embedding_count == 2
    assert manifest.source_article_content_path == str((tmp_path / "article_content.csv").resolve())
    assert manifest.source_image_inventory_path == str(
        (tmp_path / "article_image_inventory.csv").resolve()
    )
    assert manifest.manifest_path is None


def test_write_article_embedding_cache_manifest_writes_json(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    manifest_path = tmp_path / "models" / "manifest.json"

    written = write_article_embedding_cache_manifest(manifest, manifest_path)

    assert written.manifest_path == str(manifest_path.resolve())
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["provider_name"] == "openclip"
    assert report["manifest_path"] == str(manifest_path.resolve())
    assert read_article_embedding_cache_manifest(manifest_path) == written


def test_article_embedding_cache_manifest_rejects_invalid_schema(tmp_path: Path) -> None:
    valid_kwargs = _valid_manifest_kwargs(tmp_path)

    with pytest.raises(ValueError, match="provider_name"):
        ArticleEmbeddingCacheManifest(**{**valid_kwargs, "provider_name": ""})
    with pytest.raises(ValueError, match="dimension"):
        ArticleEmbeddingCacheManifest(**{**valid_kwargs, "dimension": 0})
    with pytest.raises(ValueError, match="embedding_kind"):
        ArticleEmbeddingCacheManifest(**{**valid_kwargs, "embedding_kind": cast(Any, "audio")})
    with pytest.raises(ValueError, match="distance_metric"):
        ArticleEmbeddingCacheManifest(**{**valid_kwargs, "distance_metric": cast(Any, "bad")})
    with pytest.raises(ValueError, match="vector_format"):
        ArticleEmbeddingCacheManifest(**{**valid_kwargs, "vector_format": cast(Any, "feather")})
    with pytest.raises(ValueError, match="must equal article_count"):
        ArticleEmbeddingCacheManifest(**{**valid_kwargs, "embedding_count": 8})


def _valid_manifest(tmp_path: Path) -> ArticleEmbeddingCacheManifest:
    return ArticleEmbeddingCacheManifest(**_valid_manifest_kwargs(tmp_path))


def _valid_manifest_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "generated_at_utc": "2026-06-03T00:00:00+00:00",
        "provider_name": "openclip",
        "provider_model_id": "ViT-B-32/laion2b_s34b_b79k",
        "provider_model_revision": "open_clip_torch_2.x",
        "embedding_kind": "image",
        "dimension": 512,
        "distance_metric": "cosine",
        "normalized": True,
        "vector_format": "npy",
        "dtype": "float32",
        "article_count": 10,
        "embedding_count": 9,
        "missing_embedding_count": 1,
        "source_article_content_path": str(tmp_path / "article_content.csv"),
        "source_image_inventory_path": str(tmp_path / "article_image_inventory.csv"),
        "embeddings_path": str(tmp_path / "embeddings.npy"),
        "article_mapping_path": str(tmp_path / "article_ids.csv"),
        "preprocessing": "OpenCLIP image preprocessing defaults",
        "license": "Open-source model license documented upstream",
        "manifest_path": None,
    }
