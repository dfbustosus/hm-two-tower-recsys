"""Diagnostics for cached article-embedding similarity retrieval."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from hm_recsys.data.io import TransactionEvent
from hm_recsys.embeddings.cache_io import load_article_embedding_cache
from hm_recsys.embeddings.cache_manifest import read_article_embedding_cache_manifest
from hm_recsys.evaluation.metrics import mean_average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import (
    TemporalSplit,
    TemporalSplitSummary,
    summarize_temporal_split_with_labels,
)
from hm_recsys.retrieval.candidate_diagnostics import CandidateCountDistribution
from hm_recsys.retrieval.candidate_export import select_validation_label_customer_ids
from hm_recsys.retrieval.co_visitation import DEFAULT_MAX_HISTORY_ITEMS
from hm_recsys.retrieval.content_similarity import (
    build_content_similarity_candidate_records,
    build_content_similarity_index,
    content_similarity_article_coverage,
)
from hm_recsys.retrieval.source_names import MULTIMODAL_SIMILARITY_SOURCE

DEFAULT_CONTENT_SIMILARITY_EVALUATION_KS = (12, 50, 100)


@dataclass(frozen=True)
class ContentSimilarityDiagnosticsReport:
    """Validation diagnostics for one cached content-similarity source."""

    generated_at_utc: str
    cutoff: str
    validation_end_exclusive: str
    horizon_days: int
    source_name: str
    manifest_path: str
    provider_name: str
    provider_model_id: str
    provider_model_revision: str
    embedding_kind: str
    embedding_count: int
    manifest_article_count: int
    target_customers: int
    evaluated_customers: int
    max_target_customers: int | None
    evaluation_ks: tuple[int, ...]
    max_history_items: int
    exclude_history: bool
    index_train_rows_used: int
    rows_with_embedding_history: int
    rows_with_candidates: int
    candidate_coverage: float
    article_coverage: int
    map_at_12: float
    recall_at_k: dict[str, float]
    candidate_count_distribution: CandidateCountDistribution
    runtime_seconds: float
    split_summary: TemporalSplitSummary


def evaluate_cached_content_similarity(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    split: TemporalSplit,
    submission_customer_ids: Iterable[str],
    manifest_path: Path | str,
    *,
    source_name: str = MULTIMODAL_SIMILARITY_SOURCE,
    evaluation_ks: Sequence[int] = DEFAULT_CONTENT_SIMILARITY_EVALUATION_KS,
    max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    exclude_history: bool = True,
    max_target_customers: int | None = None,
) -> ContentSimilarityDiagnosticsReport:
    """Evaluate a cached embedding source on leakage-safe validation labels."""

    normalized_ks = _normalize_evaluation_ks(evaluation_ks)
    if max_target_customers is not None and max_target_customers <= 0:
        raise ValueError("max_target_customers must be positive when provided")
    if not source_name:
        raise ValueError("source_name must not be empty")
    started_at = perf_counter()
    manifest = read_article_embedding_cache_manifest(manifest_path)
    embedding_records = load_article_embedding_cache(manifest)

    validation_data = summarize_temporal_split_with_labels(transaction_iter_factory(), split)
    target_customer_ids = select_validation_label_customer_ids(
        validation_labels=validation_data.validation_labels,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=max_target_customers,
    )
    labels = {
        customer_id: validation_data.validation_labels[customer_id]
        for customer_id in target_customer_ids
    }
    max_k = max(normalized_ks)

    index = build_content_similarity_index(
        transactions=transaction_iter_factory(),
        split=split,
        target_customer_ids=target_customer_ids,
        embedding_records=embedding_records,
        source_name=source_name,
        max_history_items=max_history_items,
        exclude_history=exclude_history,
    )
    predictions: dict[str, tuple[str, ...]] = {}
    candidate_counts: list[int] = []
    rows_with_candidates = 0
    for customer_id in target_customer_ids:
        candidates = build_content_similarity_candidate_records(index, customer_id, k=max_k)
        predicted = tuple(candidate.article_id for candidate in candidates)
        predictions[customer_id] = predicted
        candidate_counts.append(len(predicted))
        if predicted:
            rows_with_candidates += 1

    recall_by_k = {
        str(k): _mean_recall_at_k(labels=labels, predictions=predictions, k=k)
        for k in normalized_ks
    }
    target_count = len(target_customer_ids)
    return ContentSimilarityDiagnosticsReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoff=split.cutoff.isoformat(),
        validation_end_exclusive=split.validation_end.isoformat(),
        horizon_days=split.horizon_days,
        source_name=source_name,
        manifest_path=str(Path(manifest_path).expanduser().resolve()),
        provider_name=manifest.provider_name,
        provider_model_id=manifest.provider_model_id,
        provider_model_revision=manifest.provider_model_revision,
        embedding_kind=manifest.embedding_kind,
        embedding_count=len(embedding_records),
        manifest_article_count=manifest.article_count,
        target_customers=target_count,
        evaluated_customers=len(labels),
        max_target_customers=max_target_customers,
        evaluation_ks=normalized_ks,
        max_history_items=max_history_items,
        exclude_history=exclude_history,
        index_train_rows_used=index.train_rows_used,
        rows_with_embedding_history=len(index.customer_histories),
        rows_with_candidates=rows_with_candidates,
        candidate_coverage=rows_with_candidates / target_count if target_count else 0.0,
        article_coverage=content_similarity_article_coverage(index, target_customer_ids, k=max_k),
        map_at_12=mean_average_precision_at_k(labels, predictions, k=12),
        recall_at_k=recall_by_k,
        candidate_count_distribution=_candidate_count_distribution(candidate_counts),
        runtime_seconds=perf_counter() - started_at,
        split_summary=validation_data.summary,
    )


def write_content_similarity_diagnostics_report(
    report: ContentSimilarityDiagnosticsReport,
    report_path: Path | str,
) -> Path:
    """Write content-similarity diagnostics as JSON."""

    resolved_path = Path(report_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(_report_to_dict(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved_path


def _mean_recall_at_k(
    *,
    labels: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    k: int,
) -> float:
    recalls = [
        recall_at_k(actual, predictions.get(customer_id, ()), k=k)
        for customer_id, actual in labels.items()
    ]
    return sum(recalls) / len(recalls) if recalls else 0.0


def _normalize_evaluation_ks(evaluation_ks: Sequence[int]) -> tuple[int, ...]:
    if not evaluation_ks:
        raise ValueError("evaluation_ks must not be empty")
    normalized = tuple(sorted(set(evaluation_ks)))
    if any(k <= 0 for k in normalized):
        raise ValueError("all evaluation_ks must be positive")
    return normalized


def _candidate_count_distribution(counts: Sequence[int]) -> CandidateCountDistribution:
    if not counts:
        return CandidateCountDistribution(0, 0, 0, 0, 0, 0, 0.0)
    sorted_counts = sorted(counts)
    return CandidateCountDistribution(
        minimum=sorted_counts[0],
        p50=_nearest_rank_percentile(sorted_counts, 0.50),
        p90=_nearest_rank_percentile(sorted_counts, 0.90),
        p95=_nearest_rank_percentile(sorted_counts, 0.95),
        p99=_nearest_rank_percentile(sorted_counts, 0.99),
        maximum=sorted_counts[-1],
        mean=sum(sorted_counts) / len(sorted_counts),
    )


def _nearest_rank_percentile(sorted_counts: Sequence[int], percentile: float) -> int:
    if not sorted_counts:
        return 0
    index = max(0, min(len(sorted_counts) - 1, round(percentile * (len(sorted_counts) - 1))))
    return sorted_counts[index]


def _report_to_dict(report: ContentSimilarityDiagnosticsReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["split_summary"] = asdict(report.split_summary)
    payload["candidate_count_distribution"] = asdict(report.candidate_count_distribution)
    return payload
