"""Candidate-source diagnostics for leakage-safe H&M retrieval experiments."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import chain
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.metrics import mean_average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import (
    TemporalSplit,
    TemporalSplitSummary,
    summarize_temporal_split_with_labels,
)
from hm_recsys.retrieval.baselines import (
    BaselineCandidateSources,
    build_repeat_popularity_candidate_sources,
    merge_ranked_sources,
)

DEFAULT_EVALUATION_KS = (12, 50, 100)
REPEAT_SOURCE = "repeat"
RECENT_POPULARITY_SOURCE = "recent_popularity"
ALL_TIME_POPULARITY_SOURCE = "all_time_popularity"
REPEAT_POPULARITY_BLEND_SOURCE = "repeat_popularity_blend"


@dataclass(frozen=True)
class CandidateCountDistribution:
    """Distribution summary for per-customer candidate counts.

    Attributes:
        minimum: Minimum candidate count.
        p50: Median candidate count using nearest-rank percentile.
        p90: 90th percentile candidate count using nearest-rank percentile.
        p95: 95th percentile candidate count using nearest-rank percentile.
        p99: 99th percentile candidate count using nearest-rank percentile.
        maximum: Maximum candidate count.
        mean: Mean candidate count.
    """

    minimum: int
    p50: int
    p90: int
    p95: int
    p99: int
    maximum: int
    mean: float


@dataclass(frozen=True)
class HistoryBucketMetrics:
    """Source metrics for a customer-history bucket.

    Attributes:
        bucket: History bucket name.
        evaluated_customers: Number of labeled validation customers in the bucket.
        map_at_12: MAP@12 for customers in the bucket.
        recall_at_12: Recall@12 for customers in the bucket.
    """

    bucket: str
    evaluated_customers: int
    map_at_12: float
    recall_at_12: float


@dataclass(frozen=True)
class CandidateSourceDiagnostics:
    """Diagnostics for one candidate source or deterministic source blend.

    Attributes:
        source: Source name.
        target_customers: Number of customers requiring candidates.
        evaluated_customers: Number of target customers with validation labels.
        map_at_12: MAP@12 using the source's native ordering.
        recall_at_k: Mean recall at each configured cutoff.
        rows_with_candidates: Target-customer rows with at least one candidate.
        candidate_coverage: Share of target customers with at least one candidate.
        rows_with_full_candidate_count: Rows with at least ``max_k`` candidates.
        duplicate_candidate_rows: Rows whose candidate list contains duplicates.
        article_coverage: Number of distinct articles emitted by the source.
        candidate_count_distribution: Per-customer candidate count distribution.
        history_bucket_metrics: Metrics split by pre-cutoff customer history.
    """

    source: str
    target_customers: int
    evaluated_customers: int
    map_at_12: float
    recall_at_k: dict[str, float]
    rows_with_candidates: int
    candidate_coverage: float
    rows_with_full_candidate_count: int
    duplicate_candidate_rows: int
    article_coverage: int
    candidate_count_distribution: CandidateCountDistribution
    history_bucket_metrics: tuple[HistoryBucketMetrics, ...]


@dataclass(frozen=True)
class CandidateDiagnosticsReport:
    """Candidate-source diagnostics report for one temporal split.

    Attributes:
        generated_at_utc: UTC timestamp for the diagnostic run.
        cutoff: Validation cutoff date.
        validation_end_exclusive: Exclusive validation-window end date.
        horizon_days: Number of validation days.
        popularity_lookback_days: Recent popularity lookback length.
        evaluation_ks: Recall depths evaluated for every source.
        max_candidates_per_customer: Maximum candidate depth generated per source.
        target_customers: Number of customers requiring candidates.
        evaluated_customers: Number of target customers with validation labels.
        runtime_seconds: Wall-clock runtime for diagnostics.
        split_summary: Temporal split row/customer/article counts.
        sources: Source-specific diagnostics.
    """

    generated_at_utc: str
    cutoff: str
    validation_end_exclusive: str
    horizon_days: int
    popularity_lookback_days: int
    evaluation_ks: tuple[int, ...]
    max_candidates_per_customer: int
    target_customers: int
    evaluated_customers: int
    runtime_seconds: float
    split_summary: TemporalSplitSummary
    sources: tuple[CandidateSourceDiagnostics, ...]


def evaluate_baseline_candidate_diagnostics(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    split: TemporalSplit,
    target_customer_ids: Iterable[str],
    popularity_lookback_days: int = 7,
    evaluation_ks: Sequence[int] = DEFAULT_EVALUATION_KS,
) -> CandidateDiagnosticsReport:
    """Evaluate repeat and popularity candidate sources on a temporal split.

    Args:
        transaction_iter_factory: Callable returning a fresh transaction iterable
            for each pass over the data.
        split: Temporal split defining train and validation windows.
        target_customer_ids: Customer universe requiring candidates, normally from
            ``sample_submission.csv``.
        popularity_lookback_days: Number of pre-cutoff days used for recent
            popularity.
        evaluation_ks: Recall cutoffs to report for each source.

    Returns:
        Candidate diagnostics report for the configured source set.

    Raises:
        ValueError: If no recall cutoffs are provided or any cutoff is invalid.
    """

    normalized_ks = _normalize_evaluation_ks(evaluation_ks)
    max_k = max(normalized_ks)
    started_at = perf_counter()
    target_customer_tuple = tuple(target_customer_ids)
    target_customer_set = set(target_customer_tuple)

    validation_data = summarize_temporal_split_with_labels(transaction_iter_factory(), split)
    labels = {
        customer_id: articles
        for customer_id, articles in validation_data.validation_labels.items()
        if customer_id in target_customer_set
    }
    sources = build_repeat_popularity_candidate_sources(
        transactions=transaction_iter_factory(),
        split=split,
        target_customer_ids=target_customer_tuple,
        k=max_k,
        popularity_lookback_days=popularity_lookback_days,
    )
    popularity_backfill = merge_ranked_sources(
        (sources.recent_popularity, sources.all_time_popularity),
        k=max_k,
    )

    source_diagnostics = (
        _diagnose_repeat_source(
            sources=sources,
            target_customer_ids=target_customer_tuple,
            labels=labels,
            evaluation_ks=normalized_ks,
        ),
        _diagnose_global_source(
            source=RECENT_POPULARITY_SOURCE,
            candidates=sources.recent_popularity[:max_k],
            target_customer_ids=target_customer_tuple,
            labels=labels,
            sources=sources,
            evaluation_ks=normalized_ks,
        ),
        _diagnose_global_source(
            source=ALL_TIME_POPULARITY_SOURCE,
            candidates=sources.all_time_popularity[:max_k],
            target_customer_ids=target_customer_tuple,
            labels=labels,
            sources=sources,
            evaluation_ks=normalized_ks,
        ),
        _diagnose_blend_source(
            sources=sources,
            popularity_backfill=popularity_backfill,
            target_customer_ids=target_customer_tuple,
            labels=labels,
            evaluation_ks=normalized_ks,
        ),
    )
    return CandidateDiagnosticsReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoff=split.cutoff.isoformat(),
        validation_end_exclusive=split.validation_end.isoformat(),
        horizon_days=split.horizon_days,
        popularity_lookback_days=popularity_lookback_days,
        evaluation_ks=normalized_ks,
        max_candidates_per_customer=max_k,
        target_customers=len(target_customer_tuple),
        evaluated_customers=len(labels),
        runtime_seconds=perf_counter() - started_at,
        split_summary=validation_data.summary,
        sources=source_diagnostics,
    )


def candidate_diagnostics_report_to_dict(
    report: CandidateDiagnosticsReport,
) -> dict[str, Any]:
    """Convert a candidate diagnostics report to JSON-serializable primitives.

    Args:
        report: Report object to convert.

    Returns:
        Dictionary suitable for ``json.dumps``.
    """

    return asdict(report)


def write_candidate_diagnostics_report(
    report: CandidateDiagnosticsReport,
    path: Path | str,
) -> Path:
    """Write candidate diagnostics as deterministic JSON.

    Args:
        report: Diagnostics report to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(candidate_diagnostics_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _normalize_evaluation_ks(evaluation_ks: Sequence[int]) -> tuple[int, ...]:
    """Validate and sort recall cutoffs.

    Args:
        evaluation_ks: Candidate recall cutoffs.

    Returns:
        Sorted unique positive cutoffs.

    Raises:
        ValueError: If no positive cutoffs are supplied.
    """

    normalized = tuple(sorted(set(evaluation_ks)))
    if not normalized:
        raise ValueError("at least one evaluation cutoff is required")
    if any(k <= 0 for k in normalized):
        raise ValueError("evaluation cutoffs must be positive")
    return normalized


def _diagnose_repeat_source(
    sources: BaselineCandidateSources,
    target_customer_ids: Sequence[str],
    labels: Mapping[str, tuple[str, ...]],
    evaluation_ks: tuple[int, ...],
) -> CandidateSourceDiagnostics:
    """Build diagnostics for customer repeat-purchase candidates."""

    label_predictions = {
        customer_id: sources.repeat_recommendations.get(customer_id, ()) for customer_id in labels
    }
    candidate_counts = [
        len(sources.repeat_recommendations.get(customer_id, ()))
        for customer_id in target_customer_ids
    ]
    article_coverage = _article_coverage(sources.repeat_recommendations.values())
    duplicate_rows = _duplicate_rows(sources.repeat_recommendations.values())
    return _build_source_diagnostics(
        source=REPEAT_SOURCE,
        target_customers=len(target_customer_ids),
        labels=labels,
        label_predictions=label_predictions,
        candidate_counts=candidate_counts,
        article_coverage=article_coverage,
        duplicate_candidate_rows=duplicate_rows,
        evaluation_ks=evaluation_ks,
        history_counts=sources.customer_train_purchase_counts,
    )


def _diagnose_global_source(
    source: str,
    candidates: tuple[str, ...],
    target_customer_ids: Sequence[str],
    labels: Mapping[str, tuple[str, ...]],
    sources: BaselineCandidateSources,
    evaluation_ks: tuple[int, ...],
) -> CandidateSourceDiagnostics:
    """Build diagnostics for a global popularity candidate source."""

    candidate_count = len(candidates)
    label_predictions = dict.fromkeys(labels, candidates)
    return _build_source_diagnostics(
        source=source,
        target_customers=len(target_customer_ids),
        labels=labels,
        label_predictions=label_predictions,
        candidate_counts=_constant_counts(len(target_customer_ids), candidate_count),
        article_coverage=len(set(candidates)),
        duplicate_candidate_rows=(
            len(target_customer_ids) if len(set(candidates)) != len(candidates) else 0
        ),
        evaluation_ks=evaluation_ks,
        history_counts=sources.customer_train_purchase_counts,
    )


def _diagnose_blend_source(
    sources: BaselineCandidateSources,
    popularity_backfill: tuple[str, ...],
    target_customer_ids: Sequence[str],
    labels: Mapping[str, tuple[str, ...]],
    evaluation_ks: tuple[int, ...],
) -> CandidateSourceDiagnostics:
    """Build diagnostics for repeat candidates blended with popularity backfill."""

    max_k = max(evaluation_ks)
    label_predictions = {
        customer_id: merge_ranked_sources(
            (sources.repeat_recommendations.get(customer_id, ()), popularity_backfill),
            k=max_k,
        )
        for customer_id in labels
    }
    candidate_counts: Sequence[int]
    if len(popularity_backfill) >= max_k:
        candidate_counts = _constant_counts(len(target_customer_ids), max_k)
    else:
        candidate_counts = [
            len(
                merge_ranked_sources(
                    (sources.repeat_recommendations.get(customer_id, ()), popularity_backfill),
                    k=max_k,
                )
            )
            for customer_id in target_customer_ids
        ]
    article_sets = chain(
        sources.repeat_recommendations.values(),
        (popularity_backfill,),
    )
    return _build_source_diagnostics(
        source=REPEAT_POPULARITY_BLEND_SOURCE,
        target_customers=len(target_customer_ids),
        labels=labels,
        label_predictions=label_predictions,
        candidate_counts=candidate_counts,
        article_coverage=_article_coverage(article_sets),
        duplicate_candidate_rows=0,
        evaluation_ks=evaluation_ks,
        history_counts=sources.customer_train_purchase_counts,
    )


def _build_source_diagnostics(
    source: str,
    target_customers: int,
    labels: Mapping[str, tuple[str, ...]],
    label_predictions: Mapping[str, tuple[str, ...]],
    candidate_counts: Sequence[int],
    article_coverage: int,
    duplicate_candidate_rows: int,
    evaluation_ks: tuple[int, ...],
    history_counts: Mapping[str, int],
) -> CandidateSourceDiagnostics:
    """Assemble source diagnostics from counts and labeled predictions."""

    max_k = max(evaluation_ks)
    rows_with_candidates = sum(1 for count in candidate_counts if count > 0)
    rows_with_full_candidate_count = sum(1 for count in candidate_counts if count >= max_k)
    recall_by_k = {str(k): _mean_recall_at_k(labels, label_predictions, k=k) for k in evaluation_ks}
    return CandidateSourceDiagnostics(
        source=source,
        target_customers=target_customers,
        evaluated_customers=len(labels),
        map_at_12=mean_average_precision_at_k(labels, label_predictions, k=12),
        recall_at_k=recall_by_k,
        rows_with_candidates=rows_with_candidates,
        candidate_coverage=(rows_with_candidates / target_customers if target_customers else 0.0),
        rows_with_full_candidate_count=rows_with_full_candidate_count,
        duplicate_candidate_rows=duplicate_candidate_rows,
        article_coverage=article_coverage,
        candidate_count_distribution=_candidate_count_distribution(candidate_counts),
        history_bucket_metrics=_history_bucket_metrics(
            labels=labels,
            label_predictions=label_predictions,
            history_counts=history_counts,
        ),
    )


def _constant_counts(length: int, value: int) -> tuple[int, ...]:
    """Return a repeated count tuple for global candidate sources."""

    return (value,) * length


def _candidate_count_distribution(
    candidate_counts: Sequence[int],
) -> CandidateCountDistribution:
    """Summarize candidate count distribution with deterministic percentiles."""

    if not candidate_counts:
        return CandidateCountDistribution(0, 0, 0, 0, 0, 0, 0.0)
    sorted_counts = sorted(candidate_counts)
    return CandidateCountDistribution(
        minimum=sorted_counts[0],
        p50=_nearest_rank(sorted_counts, 50),
        p90=_nearest_rank(sorted_counts, 90),
        p95=_nearest_rank(sorted_counts, 95),
        p99=_nearest_rank(sorted_counts, 99),
        maximum=sorted_counts[-1],
        mean=sum(sorted_counts) / len(sorted_counts),
    )


def _nearest_rank(sorted_values: Sequence[int], percentile: int) -> int:
    """Return a nearest-rank percentile from pre-sorted integer values."""

    if not sorted_values:
        return 0
    index = max(0, ceil((percentile / 100) * len(sorted_values)) - 1)
    return sorted_values[min(index, len(sorted_values) - 1)]


def _history_bucket_metrics(
    labels: Mapping[str, tuple[str, ...]],
    label_predictions: Mapping[str, tuple[str, ...]],
    history_counts: Mapping[str, int],
) -> tuple[HistoryBucketMetrics, ...]:
    """Compute MAP@12 and recall@12 by pre-cutoff customer history bucket."""

    buckets: dict[str, dict[str, tuple[str, ...]]] = {
        "no_history": {},
        "sparse_1_2": {},
        "medium_3_10": {},
        "dense_11_plus": {},
    }
    for customer_id, actual in labels.items():
        bucket = _history_bucket(history_counts.get(customer_id, 0))
        buckets[bucket][customer_id] = actual
    metrics: list[HistoryBucketMetrics] = []
    for bucket, bucket_labels in buckets.items():
        bucket_predictions = {
            customer_id: label_predictions.get(customer_id, ()) for customer_id in bucket_labels
        }
        metrics.append(
            HistoryBucketMetrics(
                bucket=bucket,
                evaluated_customers=len(bucket_labels),
                map_at_12=mean_average_precision_at_k(bucket_labels, bucket_predictions, k=12),
                recall_at_12=_mean_recall_at_k(bucket_labels, bucket_predictions, k=12),
            )
        )
    return tuple(metrics)


def _history_bucket(purchase_count: int) -> str:
    """Return a named history bucket for a pre-cutoff purchase count."""

    if purchase_count <= 0:
        return "no_history"
    if purchase_count <= 2:
        return "sparse_1_2"
    if purchase_count <= 10:
        return "medium_3_10"
    return "dense_11_plus"


def _mean_recall_at_k(
    actual_by_customer: Mapping[str, tuple[str, ...]],
    predicted_by_customer: Mapping[str, tuple[str, ...]],
    k: int,
) -> float:
    """Compute mean recall@K over labeled customers."""

    scores = [
        recall_at_k(actual, predicted_by_customer.get(customer_id, ()), k=k)
        for customer_id, actual in actual_by_customer.items()
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _article_coverage(candidate_lists: Iterable[Iterable[str]]) -> int:
    """Count unique article IDs appearing in candidate lists."""

    return len({article_id for candidates in candidate_lists for article_id in candidates})


def _duplicate_rows(candidate_lists: Iterable[Iterable[str]]) -> int:
    """Count candidate rows containing duplicate article IDs."""

    duplicate_rows = 0
    for candidates in candidate_lists:
        candidate_tuple = tuple(candidates)
        if len(set(candidate_tuple)) != len(candidate_tuple):
            duplicate_rows += 1
    return duplicate_rows
