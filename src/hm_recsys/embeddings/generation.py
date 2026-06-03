"""Generate versioned article embedding caches from provider outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from hm_recsys.embeddings.article_content import iter_article_content_records_from_csv
from hm_recsys.embeddings.cache_io import ARTICLE_EMBEDDING_MAPPING_HEADER
from hm_recsys.embeddings.cache_manifest import (
    ArticleEmbeddingCacheManifest,
    EmbeddingCacheKind,
    build_article_embedding_cache_manifest,
    write_article_embedding_cache_manifest,
)
from hm_recsys.embeddings.contracts import (
    ArticleEmbeddingInput,
    ArticleEmbeddingProvider,
    ArticleEmbeddingRecord,
)
from hm_recsys.indexing.contracts import DistanceMetric
from hm_recsys.indexing.exact import l2_normalize


@dataclass(frozen=True)
class ArticleEmbeddingCacheWriteConfig:
    """Configuration for writing an article embedding cache.

    Attributes mirror manifest fields plus operational controls.  The writer
    persists JSONL vectors by default to avoid requiring NumPy in the base
    package.
    """

    provider_model_id: str
    provider_model_revision: str
    embedding_kind: EmbeddingCacheKind
    preprocessing: str
    license: str
    distance_metric: DistanceMetric = "cosine"
    normalized: bool = True
    dtype: str = "float32"
    batch_size: int = 32
    max_articles: int | None = None
    source_image_inventory_path: Path | str | None = None

    def __post_init__(self) -> None:
        """Validate cache-write configuration."""

        if not self.provider_model_id:
            raise ValueError("provider_model_id must not be empty")
        if not self.provider_model_revision:
            raise ValueError("provider_model_revision must not be empty")
        if not self.preprocessing:
            raise ValueError("preprocessing must not be empty")
        if not self.license:
            raise ValueError("license must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_articles is not None and self.max_articles <= 0:
            raise ValueError("max_articles must be positive when provided")


@dataclass(frozen=True)
class ArticleEmbeddingCacheWriteSummary:
    """Summary for an article embedding cache generation run."""

    generated_at_utc: str
    provider_name: str
    provider_model_id: str
    provider_model_revision: str
    embedding_kind: str
    source_article_content_path: str
    embeddings_path: str
    article_mapping_path: str
    manifest_path: str
    article_count: int
    embedding_count: int
    missing_embedding_count: int
    skipped_missing_image_count: int
    batch_size: int
    max_articles: int | None


def write_article_embedding_cache_from_content_export(
    provider: ArticleEmbeddingProvider,
    *,
    raw_data_dir: Path | str,
    article_content_path: Path | str,
    embeddings_path: Path | str,
    article_mapping_path: Path | str,
    manifest_path: Path | str,
    config: ArticleEmbeddingCacheWriteConfig,
) -> ArticleEmbeddingCacheWriteSummary:
    """Write JSONL article embeddings, mapping CSV, and manifest.

    Args:
        provider: Embedding provider used to produce vectors.
        raw_data_dir: H&M raw-data directory for resolving relative image paths.
        article_content_path: Encoder-ready article content CSV.
        embeddings_path: Destination JSONL vector cache.
        article_mapping_path: Destination row-index-to-article-ID mapping CSV.
        manifest_path: Destination manifest JSON.
        config: Reproducibility and batching configuration.

    Returns:
        Cache generation summary.

    Raises:
        ValueError: If provider output violates the cache contract.
    """

    resolved_raw_data_dir = Path(raw_data_dir).expanduser().resolve()
    resolved_article_content_path = Path(article_content_path).expanduser().resolve()
    resolved_embeddings_path = Path(embeddings_path).expanduser().resolve()
    resolved_article_mapping_path = Path(article_mapping_path).expanduser().resolve()
    resolved_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_article_mapping_path.parent.mkdir(parents=True, exist_ok=True)

    article_count = 0
    embedding_count = 0
    skipped_missing_image_count = 0
    seen_article_ids: set[str] = set()
    with (
        resolved_embeddings_path.open("w", encoding="utf-8") as embeddings_handle,
        resolved_article_mapping_path.open("w", encoding="utf-8", newline="") as mapping_handle,
    ):
        mapping_writer = csv.writer(mapping_handle)
        mapping_writer.writerow(ARTICLE_EMBEDDING_MAPPING_HEADER)
        pending_batch: list[ArticleEmbeddingInput] = []
        for input_record in _iter_provider_inputs(
            raw_data_dir=resolved_raw_data_dir,
            article_content_path=resolved_article_content_path,
            max_articles=config.max_articles,
        ):
            article_count += 1
            if config.embedding_kind == "image" and input_record.image_path is None:
                skipped_missing_image_count += 1
                continue
            pending_batch.append(input_record)
            if len(pending_batch) == config.batch_size:
                embedding_count = _write_provider_batch(
                    provider=provider,
                    batch=pending_batch,
                    embeddings_handle=embeddings_handle,
                    mapping_writer=mapping_writer,
                    embedding_count=embedding_count,
                    seen_article_ids=seen_article_ids,
                    normalize=config.normalized,
                )
                pending_batch = []
        if pending_batch:
            embedding_count = _write_provider_batch(
                provider=provider,
                batch=pending_batch,
                embeddings_handle=embeddings_handle,
                mapping_writer=mapping_writer,
                embedding_count=embedding_count,
                seen_article_ids=seen_article_ids,
                normalize=config.normalized,
            )

    manifest = build_article_embedding_cache_manifest(
        provider_name=provider.name,
        provider_model_id=config.provider_model_id,
        provider_model_revision=config.provider_model_revision,
        embedding_kind=config.embedding_kind,
        dimension=provider.dimension,
        distance_metric=config.distance_metric,
        normalized=config.normalized,
        vector_format="jsonl",
        dtype=config.dtype,
        article_count=article_count,
        embedding_count=embedding_count,
        source_article_content_path=resolved_article_content_path,
        source_image_inventory_path=config.source_image_inventory_path,
        embeddings_path=resolved_embeddings_path,
        article_mapping_path=resolved_article_mapping_path,
        preprocessing=config.preprocessing,
        license=config.license,
    )
    written_manifest = write_article_embedding_cache_manifest(manifest, manifest_path)
    return _summary_from_manifest(
        written_manifest,
        skipped_missing_image_count=skipped_missing_image_count,
        batch_size=config.batch_size,
        max_articles=config.max_articles,
    )


def _iter_provider_inputs(
    *,
    raw_data_dir: Path,
    article_content_path: Path,
    max_articles: int | None,
) -> Iterable[ArticleEmbeddingInput]:
    for yielded, content_record in enumerate(
        iter_article_content_records_from_csv(article_content_path),
        start=1,
    ):
        if max_articles is not None and yielded > max_articles:
            break
        yield content_record.to_embedding_input(raw_data_dir)


def _write_provider_batch(
    *,
    provider: ArticleEmbeddingProvider,
    batch: list[ArticleEmbeddingInput],
    embeddings_handle: TextIO,
    mapping_writer: Any,
    embedding_count: int,
    seen_article_ids: set[str],
    normalize: bool,
) -> int:
    batch_article_ids = {article.article_id for article in batch}
    for record in provider.embed_articles(batch):
        _validate_provider_record(record, provider, batch_article_ids, seen_article_ids)
        vector = l2_normalize(record.vector) if normalize else record.vector
        embeddings_handle.write(json.dumps({"article_id": record.article_id, "vector": vector}))
        embeddings_handle.write("\n")
        mapping_writer.writerow((str(embedding_count), record.article_id))
        seen_article_ids.add(record.article_id)
        embedding_count += 1
    return embedding_count


def _validate_provider_record(
    record: ArticleEmbeddingRecord,
    provider: ArticleEmbeddingProvider,
    batch_article_ids: set[str],
    seen_article_ids: set[str],
) -> None:
    if record.provider_name != provider.name:
        raise ValueError(
            f"provider emitted record with provider_name {record.provider_name!r}, "
            f"expected {provider.name!r}"
        )
    if record.dimension != provider.dimension:
        raise ValueError(
            f"provider emitted dimension {record.dimension}, expected {provider.dimension}"
        )
    if record.article_id not in batch_article_ids:
        raise ValueError(f"provider emitted unexpected article_id {record.article_id!r}")
    if record.article_id in seen_article_ids:
        raise ValueError(f"provider emitted duplicate article_id {record.article_id!r}")


def _summary_from_manifest(
    manifest: ArticleEmbeddingCacheManifest,
    *,
    skipped_missing_image_count: int,
    batch_size: int,
    max_articles: int | None,
) -> ArticleEmbeddingCacheWriteSummary:
    return ArticleEmbeddingCacheWriteSummary(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        provider_name=manifest.provider_name,
        provider_model_id=manifest.provider_model_id,
        provider_model_revision=manifest.provider_model_revision,
        embedding_kind=manifest.embedding_kind,
        source_article_content_path=manifest.source_article_content_path,
        embeddings_path=manifest.embeddings_path,
        article_mapping_path=manifest.article_mapping_path,
        manifest_path=manifest.manifest_path or "",
        article_count=manifest.article_count,
        embedding_count=manifest.embedding_count,
        missing_embedding_count=manifest.missing_embedding_count,
        skipped_missing_image_count=skipped_missing_image_count,
        batch_size=batch_size,
        max_articles=max_articles,
    )
