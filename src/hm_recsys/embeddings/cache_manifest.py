"""Versioned manifest contracts for cached article embeddings.

Embedding matrices produced by open-source providers such as FashionCLIP,
OpenCLIP, SigLIP, or SentenceTransformers belong under ignored local storage.
This module captures the metadata needed to reproduce and safely join those
matrices back to exact H&M article IDs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from hm_recsys.indexing.contracts import ALLOWED_DISTANCE_METRICS, DistanceMetric

EmbeddingCacheKind = Literal["image", "text", "multimodal", "item"]
EmbeddingVectorFormat = Literal["npy", "npz", "parquet", "csv", "jsonl"]
ALLOWED_EMBEDDING_CACHE_KINDS = frozenset({"image", "text", "multimodal", "item"})
ALLOWED_VECTOR_FORMATS = frozenset({"npy", "npz", "parquet", "csv", "jsonl"})


@dataclass(frozen=True)
class ArticleEmbeddingCacheManifest:
    """Reproducibility manifest for an article embedding cache.

    Attributes:
        generated_at_utc: UTC timestamp for manifest generation.
        provider_name: Stable provider family name, e.g. ``fashionclip`` or
            ``openclip``.
        provider_model_id: Exact open-source model identifier.
        provider_model_revision: Model revision, checkpoint hash, or release tag.
        embedding_kind: Whether embeddings represent image, text, multimodal, or
            trained item-tower vectors.
        dimension: Dense vector dimensionality.
        distance_metric: Retrieval metric intended for the vectors.
        normalized: Whether vectors were L2-normalized before persistence.
        vector_format: File format used for the embedding matrix.
        dtype: Numeric dtype used for stored vectors.
        article_count: Number of article IDs in the source content universe.
        embedding_count: Number of vectors successfully written.
        missing_embedding_count: Number of source articles without a vector.
        source_article_content_path: Article-content export used as provider
            input.
        source_image_inventory_path: Optional image inventory used for coverage
            diagnostics.
        embeddings_path: Local path to the embedding matrix.
        article_mapping_path: Local path mapping embedding rows to article IDs.
        preprocessing: Human-readable preprocessing/version notes.
        license: Provider/model license note.
        manifest_path: Path to this JSON manifest after writing.
    """

    generated_at_utc: str
    provider_name: str
    provider_model_id: str
    provider_model_revision: str
    embedding_kind: EmbeddingCacheKind
    dimension: int
    distance_metric: DistanceMetric
    normalized: bool
    vector_format: EmbeddingVectorFormat
    dtype: str
    article_count: int
    embedding_count: int
    missing_embedding_count: int
    source_article_content_path: str
    source_image_inventory_path: str | None
    embeddings_path: str
    article_mapping_path: str
    preprocessing: str
    license: str
    manifest_path: str | None = None

    def __post_init__(self) -> None:
        """Validate manifest identity, vector schema, and counts."""

        if not self.provider_name:
            raise ValueError("provider_name must not be empty")
        if not self.provider_model_id:
            raise ValueError("provider_model_id must not be empty")
        if not self.provider_model_revision:
            raise ValueError("provider_model_revision must not be empty")
        if self.embedding_kind not in ALLOWED_EMBEDDING_CACHE_KINDS:
            raise ValueError(f"unsupported embedding_kind: {self.embedding_kind!r}")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.distance_metric not in ALLOWED_DISTANCE_METRICS:
            raise ValueError(f"unsupported distance_metric: {self.distance_metric!r}")
        if self.vector_format not in ALLOWED_VECTOR_FORMATS:
            raise ValueError(f"unsupported vector_format: {self.vector_format!r}")
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        if self.article_count < 0:
            raise ValueError("article_count must be non-negative")
        if self.embedding_count < 0:
            raise ValueError("embedding_count must be non-negative")
        if self.missing_embedding_count < 0:
            raise ValueError("missing_embedding_count must be non-negative")
        if self.embedding_count + self.missing_embedding_count != self.article_count:
            raise ValueError(
                "embedding_count plus missing_embedding_count must equal article_count"
            )
        if not self.source_article_content_path:
            raise ValueError("source_article_content_path must not be empty")
        if not self.embeddings_path:
            raise ValueError("embeddings_path must not be empty")
        if not self.article_mapping_path:
            raise ValueError("article_mapping_path must not be empty")
        if not self.preprocessing:
            raise ValueError("preprocessing must not be empty")
        if not self.license:
            raise ValueError("license must not be empty")


def build_article_embedding_cache_manifest(
    *,
    provider_name: str,
    provider_model_id: str,
    provider_model_revision: str,
    embedding_kind: EmbeddingCacheKind,
    dimension: int,
    distance_metric: DistanceMetric,
    normalized: bool,
    vector_format: EmbeddingVectorFormat,
    dtype: str,
    article_count: int,
    embedding_count: int,
    source_article_content_path: Path | str,
    embeddings_path: Path | str,
    article_mapping_path: Path | str,
    preprocessing: str,
    license: str,
    source_image_inventory_path: Path | str | None = None,
) -> ArticleEmbeddingCacheManifest:
    """Build a validated article embedding cache manifest.

    The helper computes ``missing_embedding_count`` from source and written
    counts and resolves paths for reproducibility.
    """

    missing_embedding_count = article_count - embedding_count
    return ArticleEmbeddingCacheManifest(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        provider_name=provider_name,
        provider_model_id=provider_model_id,
        provider_model_revision=provider_model_revision,
        embedding_kind=embedding_kind,
        dimension=dimension,
        distance_metric=distance_metric,
        normalized=normalized,
        vector_format=vector_format,
        dtype=dtype,
        article_count=article_count,
        embedding_count=embedding_count,
        missing_embedding_count=missing_embedding_count,
        source_article_content_path=str(Path(source_article_content_path).expanduser().resolve()),
        source_image_inventory_path=(
            str(Path(source_image_inventory_path).expanduser().resolve())
            if source_image_inventory_path is not None
            else None
        ),
        embeddings_path=str(Path(embeddings_path).expanduser().resolve()),
        article_mapping_path=str(Path(article_mapping_path).expanduser().resolve()),
        preprocessing=preprocessing,
        license=license,
    )


def write_article_embedding_cache_manifest(
    manifest: ArticleEmbeddingCacheManifest,
    manifest_path: Path | str,
) -> ArticleEmbeddingCacheManifest:
    """Write an article embedding cache manifest JSON file."""

    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_with_path = replace(manifest, manifest_path=str(resolved_manifest_path))
    with resolved_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(manifest_with_path), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_with_path
