"""Ranker-ready candidate table export for H&M retrieval sources."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import (
    TemporalSplit,
    TemporalSplitSummary,
    summarize_temporal_split_with_labels,
)
from hm_recsys.retrieval.baselines import build_repeat_popularity_candidate_sources
from hm_recsys.retrieval.co_visitation import (
    DEFAULT_MAX_HISTORY_ITEMS,
    DEFAULT_MAX_NEIGHBORS_PER_ITEM,
    CoVisitationIndex,
    build_co_visitation_candidate_records,
    build_co_visitation_index,
)
from hm_recsys.retrieval.source_names import (
    ALL_TIME_POPULARITY_SOURCE,
    CO_VISITATION_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
)

CANDIDATE_EXPORT_HEADER = (
    "customer_id",
    "article_id",
    "source",
    "source_rank",
    "source_score",
)


@dataclass(frozen=True)
class CandidateRecord:
    """Single ranker-ready candidate-source record.

    Attributes:
        customer_id: H&M customer identifier.
        article_id: H&M article identifier.
        source: Retrieval source name.
        source_rank: One-based rank within that source for the customer.
        source_score: Source-specific numeric score. Ranked-only sources use
            reciprocal rank; co-visitation uses its aggregate item-item score.
    """

    customer_id: str
    article_id: str
    source: str
    source_rank: int
    source_score: float


@dataclass(frozen=True)
class CandidateExportSummary:
    """Summary metadata for a candidate-table export run.

    Attributes:
        generated_at_utc: UTC timestamp for the export run.
        cutoff: Validation cutoff date.
        validation_end_exclusive: Exclusive validation end date.
        horizon_days: Validation horizon in days.
        target_scope: Exported customer scope.
        max_target_customers: Optional deterministic customer cap used for smoke runs.
        target_customers: Number of customers exported.
        k: Maximum candidates per source per customer.
        popularity_lookback_days: Recent popularity lookback length.
        include_co_visitation: Whether co-visitation rows were exported.
        co_visitation_max_history_items: Co-visitation customer-history length.
        co_visitation_max_neighbors_per_item: Co-visitation neighbor cap per item.
        rows_written: Number of CSV data rows written.
        source_row_counts: Row counts by source.
        output_path: Candidate CSV path.
        runtime_seconds: Wall-clock runtime for export.
        split_summary: Temporal split row/customer/article counts.
    """

    generated_at_utc: str
    cutoff: str
    validation_end_exclusive: str
    horizon_days: int
    target_scope: str
    max_target_customers: int | None
    target_customers: int
    k: int
    popularity_lookback_days: int
    include_co_visitation: bool
    co_visitation_max_history_items: int
    co_visitation_max_neighbors_per_item: int
    rows_written: int
    source_row_counts: dict[str, int]
    output_path: str
    runtime_seconds: float
    split_summary: TemporalSplitSummary


def write_validation_candidate_export(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    split: TemporalSplit,
    submission_customer_ids: Iterable[str],
    output_path: Path | str,
    k: int = 12,
    popularity_lookback_days: int = 7,
    include_co_visitation: bool = True,
    co_visitation_max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    co_visitation_max_neighbors_per_item: int = DEFAULT_MAX_NEIGHBORS_PER_ITEM,
    max_target_customers: int | None = None,
) -> CandidateExportSummary:
    """Write ranker-ready candidates for validation-label customers.

    The export target is intentionally the labeled validation subset by default:
    this is the correct scope for training and evaluating a ranker without
    exploding global popularity rows across all submission customers.

    Args:
        transaction_iter_factory: Callable returning a fresh transaction iterable
            for each pass over the data.
        split: Temporal split defining train and validation windows.
        submission_customer_ids: Authoritative customer universe used to filter
            validation labels.
        output_path: Destination CSV path.
        k: Maximum candidates per source per customer.
        popularity_lookback_days: Recent popularity lookback length.
        include_co_visitation: Whether to export co-visitation source rows.
        co_visitation_max_history_items: Recent unique customer-history length
            used by co-visitation.
        co_visitation_max_neighbors_per_item: Maximum neighbors retained per item.
        max_target_customers: Optional deterministic cap for smoke runs.

    Returns:
        Export summary with row counts, source counts, split metadata, and path.

    Raises:
        ValueError: If numeric limits are invalid.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if popularity_lookback_days <= 0:
        raise ValueError("popularity_lookback_days must be positive")
    if max_target_customers is not None and max_target_customers <= 0:
        raise ValueError("max_target_customers must be positive when provided")

    started_at = perf_counter()
    validation_data = summarize_temporal_split_with_labels(transaction_iter_factory(), split)
    target_customer_ids = select_validation_label_customer_ids(
        validation_labels=validation_data.validation_labels,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=max_target_customers,
    )

    baseline_sources = build_repeat_popularity_candidate_sources(
        transactions=transaction_iter_factory(),
        split=split,
        target_customer_ids=target_customer_ids,
        k=k,
        popularity_lookback_days=popularity_lookback_days,
    )
    co_visitation_index = (
        build_co_visitation_index(
            transactions=transaction_iter_factory(),
            split=split,
            target_customer_ids=target_customer_ids,
            max_history_items=co_visitation_max_history_items,
            max_neighbors_per_item=co_visitation_max_neighbors_per_item,
        )
        if include_co_visitation
        else None
    )

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    source_row_counts: Counter[str] = Counter()
    rows_written = 0
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        for record in _iter_candidate_records(
            target_customer_ids=target_customer_ids,
            repeat_recommendations=baseline_sources.repeat_recommendations,
            recent_popularity=baseline_sources.recent_popularity[:k],
            all_time_popularity=baseline_sources.all_time_popularity[:k],
            co_visitation_index=co_visitation_index,
            k=k,
        ):
            writer.writerow(candidate_record_to_row(record))
            source_row_counts[record.source] += 1
            rows_written += 1

    return CandidateExportSummary(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoff=split.cutoff.isoformat(),
        validation_end_exclusive=split.validation_end.isoformat(),
        horizon_days=split.horizon_days,
        target_scope="validation_label_customers",
        max_target_customers=max_target_customers,
        target_customers=len(target_customer_ids),
        k=k,
        popularity_lookback_days=popularity_lookback_days,
        include_co_visitation=include_co_visitation,
        co_visitation_max_history_items=co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=co_visitation_max_neighbors_per_item,
        rows_written=rows_written,
        source_row_counts=dict(sorted(source_row_counts.items())),
        output_path=str(resolved_output_path),
        runtime_seconds=perf_counter() - started_at,
        split_summary=validation_data.summary,
    )


def select_validation_label_customer_ids(
    validation_labels: Mapping[str, Iterable[str]],
    submission_customer_ids: Iterable[str],
    max_target_customers: int | None = None,
) -> tuple[str, ...]:
    """Select deterministic validation-label customers inside submission scope.

    Args:
        validation_labels: Validation labels keyed by customer ID.
        submission_customer_ids: Authoritative submission customer universe.
        max_target_customers: Optional deterministic cap for smoke runs.

    Returns:
        Sorted customer IDs with validation labels and submission membership.

    Raises:
        ValueError: If ``max_target_customers`` is not positive when provided.
    """

    if max_target_customers is not None and max_target_customers <= 0:
        raise ValueError("max_target_customers must be positive when provided")
    submission_customer_set = set(submission_customer_ids)
    target_customer_ids = tuple(
        sorted(
            customer_id
            for customer_id in validation_labels
            if customer_id in submission_customer_set
        )
    )
    if max_target_customers is not None:
        return target_customer_ids[:max_target_customers]
    return target_customer_ids


def candidate_record_to_row(record: CandidateRecord) -> tuple[str, str, str, str, str]:
    """Convert a candidate record to CSV string fields.

    Args:
        record: Candidate record to serialize.

    Returns:
        Tuple matching ``CANDIDATE_EXPORT_HEADER``.
    """

    return (
        record.customer_id,
        record.article_id,
        record.source,
        str(record.source_rank),
        f"{record.source_score:.10g}",
    )


def candidate_export_summary_to_dict(summary: CandidateExportSummary) -> dict[str, Any]:
    """Convert a candidate export summary to JSON-serializable primitives.

    Args:
        summary: Summary object to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """

    return asdict(summary)


def write_candidate_export_summary(summary: CandidateExportSummary, path: Path | str) -> Path:
    """Write a candidate export summary as deterministic JSON.

    Args:
        summary: Summary object to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(candidate_export_summary_to_dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _iter_candidate_records(
    target_customer_ids: Sequence[str],
    repeat_recommendations: Mapping[str, tuple[str, ...]],
    recent_popularity: Sequence[str],
    all_time_popularity: Sequence[str],
    co_visitation_index: CoVisitationIndex | None,
    k: int,
) -> Iterable[CandidateRecord]:
    """Yield source-specific candidate records in deterministic source order."""

    for customer_id in target_customer_ids:
        yield from iter_candidate_records_for_customer(
            customer_id=customer_id,
            repeat_recommendations=repeat_recommendations,
            recent_popularity=recent_popularity,
            all_time_popularity=all_time_popularity,
            co_visitation_index=co_visitation_index,
            k=k,
        )


def iter_candidate_records_for_customer(
    customer_id: str,
    repeat_recommendations: Mapping[str, tuple[str, ...]],
    recent_popularity: Sequence[str],
    all_time_popularity: Sequence[str],
    co_visitation_index: CoVisitationIndex | None,
    k: int,
) -> Iterable[CandidateRecord]:
    """Yield ranker-ready source rows for one customer.

    Args:
        customer_id: Customer requiring candidate records.
        repeat_recommendations: Customer-specific repeat-purchase rankings.
        recent_popularity: Global recent-popularity ranking.
        all_time_popularity: Global all-time popularity ranking.
        co_visitation_index: Optional co-visitation index for item-item rows.
        k: Maximum candidates emitted per source.

    Yields:
        Source-specific candidate records in deterministic source order.
    """

    yield from _ranked_article_records(
        customer_id=customer_id,
        source=REPEAT_SOURCE,
        article_ids=repeat_recommendations.get(customer_id, ())[:k],
    )
    yield from _ranked_article_records(
        customer_id=customer_id,
        source=RECENT_POPULARITY_SOURCE,
        article_ids=recent_popularity[:k],
    )
    yield from _ranked_article_records(
        customer_id=customer_id,
        source=ALL_TIME_POPULARITY_SOURCE,
        article_ids=all_time_popularity[:k],
    )
    if co_visitation_index is not None:
        yield from (
            CandidateRecord(
                customer_id=customer_id,
                article_id=candidate.article_id,
                source=CO_VISITATION_SOURCE,
                source_rank=candidate.rank,
                source_score=candidate.score,
            )
            for candidate in build_co_visitation_candidate_records(
                co_visitation_index,
                customer_id,
                k=k,
            )
        )


def _ranked_article_records(
    customer_id: str,
    source: str,
    article_ids: Sequence[str],
) -> Iterable[CandidateRecord]:
    """Yield reciprocal-rank records for a ranked article sequence."""

    for rank, article_id in enumerate(article_ids, start=1):
        yield CandidateRecord(
            customer_id=customer_id,
            article_id=article_id,
            source=source,
            source_rank=rank,
            source_score=1.0 / rank,
        )
