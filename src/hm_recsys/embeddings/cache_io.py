"""Load cached article embeddings from versioned manifests.

The base project stays lightweight: JSONL and CSV embedding caches are loaded
with the standard library, while NumPy caches are supported only when ``numpy``
is installed in the local environment that generated the embeddings.
"""

from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from hm_recsys.core.ids import is_article_id
from hm_recsys.embeddings.cache_manifest import ArticleEmbeddingCacheManifest
from hm_recsys.embeddings.contracts import ArticleEmbeddingRecord

ARTICLE_EMBEDDING_MAPPING_HEADER = ("embedding_index", "article_id")


def load_article_embedding_cache(
    manifest: ArticleEmbeddingCacheManifest,
) -> tuple[ArticleEmbeddingRecord, ...]:
    """Load article embedding records described by a manifest.

    Args:
        manifest: Versioned embedding-cache manifest.

    Returns:
        Embedding records tied to exact article IDs.

    Raises:
        ValueError: If files, vector dimensions, article mappings, or counts do
            not match the manifest.
        ImportError: If loading a NumPy cache without ``numpy`` installed.
    """

    if manifest.vector_format == "jsonl":
        records = tuple(_iter_jsonl_embedding_records(manifest))
    elif manifest.vector_format == "csv":
        records = tuple(_iter_csv_embedding_records(manifest))
    elif manifest.vector_format == "npy":
        records = tuple(_iter_npy_embedding_records(manifest))
    else:
        raise ValueError(
            f"vector_format {manifest.vector_format!r} is not loadable by the "
            "lightweight cache reader"
        )
    validate_article_embedding_records(records, manifest)
    return records


def validate_article_embedding_records(
    records: Iterable[ArticleEmbeddingRecord],
    manifest: ArticleEmbeddingCacheManifest,
) -> None:
    """Validate loaded embedding records against a cache manifest.

    Args:
        records: Loaded article embedding records.
        manifest: Manifest whose counts and schema should match the records.

    Raises:
        ValueError: If count, duplicate ID, provider, or dimension checks fail.
    """

    seen_article_ids: set[str] = set()
    count = 0
    for record in records:
        count += 1
        if record.provider_name != manifest.provider_name:
            raise ValueError(
                f"embedding provider {record.provider_name!r} does not match manifest "
                f"provider {manifest.provider_name!r}"
            )
        if record.dimension != manifest.dimension:
            raise ValueError(
                f"article {record.article_id} dimension {record.dimension} does not match "
                f"manifest dimension {manifest.dimension}"
            )
        if record.article_id in seen_article_ids:
            raise ValueError(f"duplicate article_id in embedding cache: {record.article_id!r}")
        seen_article_ids.add(record.article_id)

    if count != manifest.embedding_count:
        raise ValueError(
            f"loaded {count} embeddings, expected {manifest.embedding_count} from manifest"
        )


def load_article_embedding_mapping(mapping_path: Path | str) -> tuple[str, ...]:
    """Load a row-index-to-article-ID mapping CSV.

    Args:
        mapping_path: CSV path with header ``embedding_index,article_id``.

    Returns:
        Article IDs ordered by embedding row index.

    Raises:
        ValueError: If the header, indices, IDs, or duplicates are invalid.
    """

    resolved_path = Path(mapping_path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != ARTICLE_EMBEDDING_MAPPING_HEADER:
            raise ValueError(
                f"{resolved_path} header must be {ARTICLE_EMBEDDING_MAPPING_HEADER}, got {header}"
            )
        article_ids_by_index: list[str] = []
        seen_article_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            index = _parse_non_negative_int(row.get("embedding_index", ""), line_number)
            article_id = row.get("article_id", "")
            if not is_article_id(article_id):
                raise ValueError(f"line {line_number}: invalid article_id {article_id!r}")
            if article_id in seen_article_ids:
                raise ValueError(f"line {line_number}: duplicate article_id {article_id!r}")
            if index != len(article_ids_by_index):
                raise ValueError(
                    f"line {line_number}: embedding_index {index} is not contiguous from zero"
                )
            seen_article_ids.add(article_id)
            article_ids_by_index.append(article_id)
    return tuple(article_ids_by_index)


def _iter_jsonl_embedding_records(
    manifest: ArticleEmbeddingCacheManifest,
) -> Iterator[ArticleEmbeddingRecord]:
    embeddings_path = Path(manifest.embeddings_path).expanduser().resolve()
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            payload = json.loads(stripped_line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"line {line_number}: expected JSON object")
            yield _embedding_record_from_payload(payload, manifest, line_number=line_number)


def _iter_csv_embedding_records(
    manifest: ArticleEmbeddingCacheManifest,
) -> Iterator[ArticleEmbeddingRecord]:
    embeddings_path = Path(manifest.embeddings_path).expanduser().resolve()
    with embeddings_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if len(header) != manifest.dimension + 1 or header[0] != "article_id":
            raise ValueError(
                f"{embeddings_path} header must be article_id plus "
                f"{manifest.dimension} vector columns"
            )
        for line_number, row in enumerate(reader, start=2):
            vector = tuple(
                _parse_float(row[column], line_number=line_number) for column in header[1:]
            )
            yield ArticleEmbeddingRecord(
                article_id=row["article_id"],
                vector=vector,
                provider_name=manifest.provider_name,
            )


def _iter_npy_embedding_records(
    manifest: ArticleEmbeddingCacheManifest,
) -> Iterator[ArticleEmbeddingRecord]:
    numpy = importlib.import_module("numpy")
    article_ids = load_article_embedding_mapping(manifest.article_mapping_path)
    embeddings = numpy.load(manifest.embeddings_path)
    if len(getattr(embeddings, "shape", ())) != 2:
        raise ValueError("NumPy embedding cache must be a two-dimensional matrix")
    rows, columns = embeddings.shape
    if rows != len(article_ids):
        raise ValueError(f"NumPy rows {rows} do not match mapping rows {len(article_ids)}")
    if columns != manifest.dimension:
        raise ValueError(
            f"NumPy columns {columns} do not match manifest dimension {manifest.dimension}"
        )
    for article_id, vector in zip(article_ids, embeddings, strict=True):
        yield ArticleEmbeddingRecord(
            article_id=article_id,
            vector=tuple(float(value) for value in vector.tolist()),
            provider_name=manifest.provider_name,
        )


def _embedding_record_from_payload(
    payload: Mapping[str, Any],
    manifest: ArticleEmbeddingCacheManifest,
    *,
    line_number: int,
) -> ArticleEmbeddingRecord:
    article_id = payload.get("article_id")
    vector_value = payload.get("vector")
    if not isinstance(article_id, str):
        raise ValueError(f"line {line_number}: article_id must be a string")
    if not isinstance(vector_value, list | tuple):
        raise ValueError(f"line {line_number}: vector must be a list of floats")
    vector = tuple(_coerce_float(value, line_number=line_number) for value in vector_value)
    return ArticleEmbeddingRecord(
        article_id=article_id,
        vector=vector,
        provider_name=manifest.provider_name,
    )


def _parse_non_negative_int(value: str, line_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid embedding_index {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"line {line_number}: embedding_index must be non-negative")
    return parsed


def _parse_float(value: str, *, line_number: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid vector value {value!r}") from exc


def _coerce_float(value: object, *, line_number: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"line {line_number}: vector values must be numeric, got boolean")
    if not isinstance(value, int | float):
        raise ValueError(f"line {line_number}: vector values must be numeric")
    return float(value)
