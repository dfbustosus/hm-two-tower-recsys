"""Article content export for text/image embedding providers.

This module prepares deterministic article-level inputs for open-source
encoders such as FashionCLIP, OpenCLIP, or SigLIP.  It does not load image
pixels and does not depend on any heavy ML framework; provider-specific jobs can
consume the exported CSV and write versioned embedding caches separately.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from hm_recsys.core.ids import is_article_id
from hm_recsys.data.io import CsvValueError, TransactionEvent, iter_csv_rows
from hm_recsys.embeddings.contracts import ArticleEmbeddingInput
from hm_recsys.embeddings.image_inventory import build_article_image_inventory

DEFAULT_ARTICLE_TEXT_FIELD_NAMES = (
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
)
ARTICLE_CONTENT_FIXED_COLUMNS = (
    "article_id",
    "combined_text",
    "image_relative_path",
    "image_exists",
)
ARTICLE_CONTENT_EXPORT_HEADER = (
    *ARTICLE_CONTENT_FIXED_COLUMNS,
    *DEFAULT_ARTICLE_TEXT_FIELD_NAMES,
)


@dataclass(frozen=True)
class ArticleContentRecord:
    """Article text fields and local image-path availability for embedding.

    Attributes:
        article_id: H&M article identifier preserved as a string.
        text_fields: Ordered text fields to pass to text/multimodal encoders.
        combined_text: Deterministic normalized text prompt built from
            ``text_fields``.
        image_relative_path: Canonical image path relative to the raw H&M data
            directory, or an empty string when no local image exists.
        image_exists: Whether the canonical image is available locally.
    """

    article_id: str
    text_fields: Mapping[str, str]
    combined_text: str
    image_relative_path: str
    image_exists: bool

    def __post_init__(self) -> None:
        """Validate record identity and image-path consistency."""

        if not is_article_id(self.article_id):
            raise ValueError(f"invalid article_id: {self.article_id!r}")
        if self.image_exists and not self.image_relative_path:
            raise ValueError("image_relative_path must be set when image_exists is true")

    def to_embedding_input(self, raw_data_dir: Path | str) -> ArticleEmbeddingInput:
        """Convert this record to the provider-facing embedding input contract.

        Args:
            raw_data_dir: H&M raw-data directory used to resolve local image paths.

        Returns:
            ``ArticleEmbeddingInput`` with an absolute image path when available.
        """

        raw_dir = Path(raw_data_dir).expanduser().resolve()
        image_path = raw_dir / self.image_relative_path if self.image_exists else None
        return ArticleEmbeddingInput(
            article_id=self.article_id,
            text_fields=self.text_fields,
            image_path=image_path,
        )


@dataclass(frozen=True)
class ArticleContentExportSummary:
    """Summary metadata for article content export artifacts."""

    generated_at_utc: str
    raw_data_dir: str
    articles_path: str
    output_path: str
    report_path: str | None
    text_field_names: tuple[str, ...]
    article_count: int
    records_written: int
    image_available_count: int
    image_missing_count: int
    empty_combined_text_count: int
    missing_image_examples: tuple[str, ...]
    empty_text_examples: tuple[str, ...]
    max_articles: int | None = None
    priority_article_count: int | None = None


def iter_article_content_records(
    raw_data_dir: Path | str,
    *,
    text_field_names: Sequence[str] = DEFAULT_ARTICLE_TEXT_FIELD_NAMES,
    max_examples: int = 10,
    article_id_order: Sequence[str] | None = None,
    max_articles: int | None = None,
) -> Iterable[ArticleContentRecord]:
    """Yield deterministic article content records for embedding providers.

    Args:
        raw_data_dir: Directory containing H&M ``articles.csv`` and optional
            ``images/`` subdirectory.
        text_field_names: Article CSV text/metadata columns included in the
            provider input and combined prompt.
        max_examples: Maximum issue examples retained internally by the image
            inventory scan. This does not affect yielded records.
        article_id_order: Optional priority order used before falling back to
            ``articles.csv`` order for articles not present in the priority list.
        max_articles: Optional maximum number of records to yield after applying
            the priority order.

    Yields:
        Article content records preserving string IDs. Without ``article_id_order``
        records follow article-file order. With a priority order, known priority
        IDs are emitted first, then remaining articles in file order.

    Raises:
        ValueError: If text fields are empty or numeric limits are invalid.
        CsvSchemaError: If required article columns are missing.
        CsvValueError: If an article ID is malformed.
    """

    field_names = _validate_text_field_names(text_field_names)
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    if max_articles is not None and max_articles <= 0:
        raise ValueError("max_articles must be positive when provided")
    raw_dir = Path(raw_data_dir).expanduser().resolve()
    records = tuple(
        _iter_article_content_records_in_articles_order(raw_dir, field_names, max_examples)
    )
    ordered_records = _apply_article_priority_order(records, article_id_order)
    for yielded, record in enumerate(ordered_records, start=1):
        if max_articles is not None and yielded > max_articles:
            break
        yield record


def build_article_popularity_priority(
    transactions: Iterable[TransactionEvent],
    *,
    cutoff: date | None = None,
    lookback_days: int | None = None,
) -> tuple[str, ...]:
    """Rank article IDs by leakage-safe transaction popularity.

    This is intended for bounded embedding-cache experiments: if only N article
    embeddings can be generated locally, prioritize items that were actually
    visible before the validation cutoff instead of taking the first N rows from
    ``articles.csv``.  Validation target-window transactions must never be passed
    through this ranker, so ``cutoff`` is exclusive when provided.

    Args:
        transactions: Transaction events preserving string article IDs.
        cutoff: Optional exclusive cutoff. Events on or after this date are
            ignored.
        lookback_days: Optional lookback ending at ``cutoff``. Requires
            ``cutoff`` so the window is auditable.

    Returns:
        Article IDs sorted by descending purchase count, most recent pre-cutoff
        purchase date, then article ID for deterministic ties.

    Raises:
        ValueError: If ``lookback_days`` is invalid or supplied without cutoff.
    """

    if lookback_days is not None and lookback_days <= 0:
        raise ValueError("lookback_days must be positive when provided")
    if lookback_days is not None and cutoff is None:
        raise ValueError("cutoff is required when lookback_days is provided")

    window_start = cutoff - timedelta(days=lookback_days) if cutoff and lookback_days else None
    counts: Counter[str] = Counter()
    latest_dates: dict[str, date] = {}
    for event in transactions:
        if cutoff is not None and event.t_dat >= cutoff:
            continue
        if window_start is not None and event.t_dat < window_start:
            continue
        counts[event.article_id] += 1
        previous_latest = latest_dates.get(event.article_id)
        if previous_latest is None or event.t_dat > previous_latest:
            latest_dates[event.article_id] = event.t_dat

    return tuple(
        article_id
        for article_id, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], -latest_dates[item[0]].toordinal(), item[0]),
        )
    )


def _iter_article_content_records_in_articles_order(
    raw_dir: Path,
    field_names: Sequence[str],
    max_examples: int,
) -> Iterable[ArticleContentRecord]:
    """Yield content records in raw ``articles.csv`` order."""

    articles_path = raw_dir / "articles.csv"
    image_inventory = build_article_image_inventory(raw_dir, max_examples=max_examples)
    image_by_article_id = {record.article_id: record for record in image_inventory.records}
    required_columns = ("article_id", *field_names)
    for line_number, row in enumerate(iter_csv_rows(articles_path, required_columns), start=2):
        article_id = row["article_id"]
        if not is_article_id(article_id):
            raise CsvValueError(f"line {line_number}: invalid article_id {article_id!r}")
        text_fields = {
            field_name: _normalize_text_value(row[field_name]) for field_name in field_names
        }
        image_record = image_by_article_id[article_id]
        yield ArticleContentRecord(
            article_id=article_id,
            text_fields=text_fields,
            combined_text=combine_article_text_fields(text_fields),
            image_relative_path=image_record.image_relative_path,
            image_exists=image_record.image_exists,
        )


def _apply_article_priority_order(
    records: Sequence[ArticleContentRecord],
    article_id_order: Sequence[str] | None,
) -> tuple[ArticleContentRecord, ...]:
    """Apply an optional article priority order while retaining all records."""

    if article_id_order is None:
        return tuple(records)

    records_by_article_id = {record.article_id: record for record in records}
    ordered: list[ArticleContentRecord] = []
    seen: set[str] = set()
    for article_id in article_id_order:
        if article_id in seen:
            continue
        seen.add(article_id)
        record = records_by_article_id.get(article_id)
        if record is not None:
            ordered.append(record)
    ordered.extend(record for record in records if record.article_id not in seen)
    return tuple(ordered)


def combine_article_text_fields(text_fields: Mapping[str, str]) -> str:
    """Build a deterministic encoder prompt from article text fields.

    Args:
        text_fields: Ordered field-name/value mapping.

    Returns:
        Normalized prompt such as ``"prod name: Shirt | colour group name: Blue"``.
    """

    segments: list[str] = []
    for field_name, raw_value in text_fields.items():
        value = _normalize_text_value(raw_value)
        if not value:
            continue
        label = field_name.replace("_", " ")
        segments.append(f"{label}: {value}")
    return " | ".join(segments)


def write_article_content_export(
    raw_data_dir: Path | str,
    *,
    output_path: Path | str,
    report_path: Path | str,
    text_field_names: Sequence[str] = DEFAULT_ARTICLE_TEXT_FIELD_NAMES,
    max_examples: int = 10,
    article_id_order: Sequence[str] | None = None,
    max_articles: int | None = None,
) -> ArticleContentExportSummary:
    """Write article content CSV plus a JSON summary report.

    Args:
        raw_data_dir: Directory containing H&M ``articles.csv`` and optional
            ``images/``.
        output_path: Destination CSV path.
        report_path: Destination JSON report path.
        text_field_names: Article CSV columns to include in encoder inputs.
        max_examples: Maximum missing-image/empty-text examples retained.
        article_id_order: Optional priority order emitted before article-file
            order, useful for bounded popularity-focused embedding caches.
        max_articles: Optional maximum number of records to write after applying
            the priority order.

    Returns:
        Export summary with resolved artifact paths and coverage counts.
    """

    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    if max_articles is not None and max_articles <= 0:
        raise ValueError("max_articles must be positive when provided")
    field_names = _validate_text_field_names(text_field_names)
    raw_dir = Path(raw_data_dir).expanduser().resolve()
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_report_path = Path(report_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    article_count = 0
    image_available_count = 0
    empty_combined_text_count = 0
    missing_image_examples: list[str] = []
    empty_text_examples: list[str] = []
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("article_id", "combined_text", "image_relative_path", "image_exists", *field_names)
        )
        for record in iter_article_content_records(
            raw_dir,
            text_field_names=field_names,
            max_examples=max_examples,
            article_id_order=article_id_order,
            max_articles=max_articles,
        ):
            article_count += 1
            if record.image_exists:
                image_available_count += 1
            elif len(missing_image_examples) < max_examples:
                missing_image_examples.append(record.article_id)
            if not record.combined_text:
                empty_combined_text_count += 1
                if len(empty_text_examples) < max_examples:
                    empty_text_examples.append(record.article_id)
            writer.writerow(article_content_record_to_row(record, field_names))

    summary = ArticleContentExportSummary(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_data_dir=str(raw_dir),
        articles_path=str(raw_dir / "articles.csv"),
        output_path=str(resolved_output_path),
        report_path=None,
        text_field_names=tuple(field_names),
        article_count=article_count,
        records_written=article_count,
        image_available_count=image_available_count,
        image_missing_count=article_count - image_available_count,
        empty_combined_text_count=empty_combined_text_count,
        missing_image_examples=tuple(missing_image_examples),
        empty_text_examples=tuple(empty_text_examples),
        max_articles=max_articles,
        priority_article_count=(len(article_id_order) if article_id_order is not None else None),
    )
    return write_article_content_export_report(summary, resolved_report_path)


def article_content_record_to_row(
    record: ArticleContentRecord,
    text_field_names: Sequence[str] = DEFAULT_ARTICLE_TEXT_FIELD_NAMES,
) -> tuple[str, ...]:
    """Convert a content record to the article-content CSV row contract."""

    field_names = _validate_text_field_names(text_field_names)
    return (
        record.article_id,
        record.combined_text,
        record.image_relative_path,
        str(record.image_exists).lower(),
        *(record.text_fields.get(field_name, "") for field_name in field_names),
    )


def write_article_content_export_report(
    summary: ArticleContentExportSummary,
    report_path: Path | str,
) -> ArticleContentExportSummary:
    """Write an article-content JSON summary report."""

    resolved_report_path = Path(report_path).expanduser().resolve()
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_with_report = replace(summary, report_path=str(resolved_report_path))
    with resolved_report_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary_with_report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary_with_report


def iter_article_content_records_from_csv(
    content_path: Path | str,
) -> Iterable[ArticleContentRecord]:
    """Yield article content records from an exported article-content CSV.

    Args:
        content_path: CSV path written by ``write_article_content_export``.

    Yields:
        Article content records preserving the CSV row order.

    Raises:
        ValueError: If the header, IDs, booleans, or image-path consistency are
            invalid.
    """

    resolved_path = Path(content_path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if len(header) < len(ARTICLE_CONTENT_FIXED_COLUMNS):
            raise ValueError(f"{resolved_path} is missing article-content columns")
        fixed_header = header[: len(ARTICLE_CONTENT_FIXED_COLUMNS)]
        if fixed_header != ARTICLE_CONTENT_FIXED_COLUMNS:
            raise ValueError(
                f"{resolved_path} fixed columns must be {ARTICLE_CONTENT_FIXED_COLUMNS}, "
                f"got {fixed_header}"
            )
        text_field_names = header[len(ARTICLE_CONTENT_FIXED_COLUMNS) :]
        _validate_text_field_names(text_field_names)
        for line_number, row in enumerate(reader, start=2):
            article_id = row["article_id"]
            if not is_article_id(article_id):
                raise ValueError(f"line {line_number}: invalid article_id {article_id!r}")
            image_exists = _parse_bool(row["image_exists"], line_number=line_number)
            text_fields = {
                field_name: _normalize_text_value(row[field_name])
                for field_name in text_field_names
            }
            yield ArticleContentRecord(
                article_id=article_id,
                text_fields=text_fields,
                combined_text=_normalize_text_value(row["combined_text"]),
                image_relative_path=row["image_relative_path"],
                image_exists=image_exists,
            )


def _validate_text_field_names(text_field_names: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize requested article text field names."""

    if not text_field_names:
        raise ValueError("text_field_names must not be empty")
    normalized = tuple(text_field_names)
    if any(not field_name for field_name in normalized):
        raise ValueError("text_field_names must not contain empty values")
    if len(set(normalized)) != len(normalized):
        raise ValueError("text_field_names must not contain duplicates")
    return normalized


def _normalize_text_value(value: str) -> str:
    """Collapse whitespace and strip CSV text values for stable encoder inputs."""

    return " ".join(value.split())


def _parse_bool(value: str, *, line_number: int) -> bool:
    """Parse lowercase CSV boolean values."""

    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"line {line_number}: image_exists must be 'true' or 'false'")
