from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
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


@dataclass
class ArticleStats:
    count: int = 0
    last_seen: date = date.min

    def update(self, seen_date: date) -> None:
        self.count += 1
        if seen_date > self.last_seen:
            self.last_seen = seen_date


@dataclass(frozen=True)
class BaselinePredictions:
    predictions: dict[str, tuple[str, ...]]
    repeat_recommendations: dict[str, tuple[str, ...]]
    recent_popularity: tuple[str, ...]
    all_time_popularity: tuple[str, ...]
    train_rows_used: int


@dataclass(frozen=True)
class BaselineDiagnostics:
    evaluated_customers: int
    customers_with_full_length_predictions: int
    prediction_coverage: float
    duplicate_prediction_rows: int
    average_prediction_count: float
    predicted_article_coverage: int
    repeat_source_customers: int
    recent_popularity_articles: int
    all_time_popularity_articles: int


@dataclass(frozen=True)
class BaselineEvaluationReport:
    generated_at_utc: str
    cutoff: str
    validation_end_exclusive: str
    horizon_days: int
    k: int
    popularity_lookback_days: int
    map_at_k: float
    recall_at_k: float
    runtime_seconds: float
    diagnostics: BaselineDiagnostics
    split_summary: TemporalSplitSummary


def build_repeat_popularity_baseline(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    target_customer_ids: Iterable[str],
    k: int = 12,
    popularity_lookback_days: int = 7,
) -> BaselinePredictions:
    if k <= 0:
        raise ValueError("k must be positive")
    if popularity_lookback_days <= 0:
        raise ValueError("popularity_lookback_days must be positive")

    target_customer_set = set(target_customer_ids)
    recent_start = split.cutoff - timedelta(days=popularity_lookback_days)
    recent_popularity_stats: dict[str, ArticleStats] = {}
    all_time_popularity_stats: dict[str, ArticleStats] = {}
    repeat_stats: dict[str, dict[str, ArticleStats]] = {}
    train_rows_used = 0

    for transaction in transactions:
        if transaction.t_dat >= split.cutoff:
            continue
        train_rows_used += 1
        _update_article_stats(all_time_popularity_stats, transaction.article_id, transaction.t_dat)
        if transaction.t_dat >= recent_start:
            _update_article_stats(
                recent_popularity_stats, transaction.article_id, transaction.t_dat
            )
        if transaction.customer_id in target_customer_set:
            customer_stats = repeat_stats.setdefault(transaction.customer_id, {})
            _update_article_stats(customer_stats, transaction.article_id, transaction.t_dat)

    recent_popularity = rank_article_stats(
        recent_popularity_stats, limit=max(k, len(recent_popularity_stats))
    )
    all_time_popularity = rank_article_stats(
        all_time_popularity_stats, limit=max(k, len(all_time_popularity_stats))
    )
    popularity_backfill = _merge_sources((recent_popularity, all_time_popularity), k=k)
    repeat_recommendations = {
        customer_id: rank_article_stats(article_stats, limit=k)
        for customer_id, article_stats in repeat_stats.items()
    }
    predictions = {
        customer_id: _merge_sources(
            (repeat_recommendations.get(customer_id, ()), popularity_backfill), k=k
        )
        for customer_id in sorted(target_customer_set)
    }
    return BaselinePredictions(
        predictions=predictions,
        repeat_recommendations=repeat_recommendations,
        recent_popularity=recent_popularity,
        all_time_popularity=all_time_popularity,
        train_rows_used=train_rows_used,
    )


def evaluate_repeat_popularity_baseline(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    split: TemporalSplit,
    k: int = 12,
    popularity_lookback_days: int = 7,
) -> BaselineEvaluationReport:
    started_at = perf_counter()
    validation_data = summarize_temporal_split_with_labels(transaction_iter_factory(), split)
    labels = validation_data.validation_labels
    baseline = build_repeat_popularity_baseline(
        transactions=transaction_iter_factory(),
        split=split,
        target_customer_ids=labels,
        k=k,
        popularity_lookback_days=popularity_lookback_days,
    )
    predictions = baseline.predictions
    map_score = mean_average_precision_at_k(labels, predictions, k=k)
    recall_score = _mean_recall_at_k(labels, predictions, k=k)
    diagnostics = build_baseline_diagnostics(
        baseline=baseline,
        evaluated_customer_count=len(labels),
        k=k,
    )
    return BaselineEvaluationReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoff=split.cutoff.isoformat(),
        validation_end_exclusive=split.validation_end.isoformat(),
        horizon_days=split.horizon_days,
        k=k,
        popularity_lookback_days=popularity_lookback_days,
        map_at_k=map_score,
        recall_at_k=recall_score,
        runtime_seconds=perf_counter() - started_at,
        diagnostics=diagnostics,
        split_summary=validation_data.summary,
    )


def rank_article_stats(stats: dict[str, ArticleStats], limit: int) -> tuple[str, ...]:
    ranked = sorted(
        stats.items(),
        key=lambda item: (-item[1].count, -item[1].last_seen.toordinal(), item[0]),
    )
    return tuple(article_id for article_id, _ in ranked[:limit])


def build_baseline_diagnostics(
    baseline: BaselinePredictions,
    evaluated_customer_count: int,
    k: int,
) -> BaselineDiagnostics:
    prediction_lengths = [len(predictions) for predictions in baseline.predictions.values()]
    duplicate_rows = sum(
        1
        for predictions in baseline.predictions.values()
        if len(set(predictions)) != len(predictions)
    )
    predicted_articles = {
        article_id for predictions in baseline.predictions.values() for article_id in predictions
    }
    full_length_rows = sum(1 for length in prediction_lengths if length == k)
    average_prediction_count = (
        sum(prediction_lengths) / len(prediction_lengths) if prediction_lengths else 0.0
    )
    return BaselineDiagnostics(
        evaluated_customers=evaluated_customer_count,
        customers_with_full_length_predictions=full_length_rows,
        prediction_coverage=(
            full_length_rows / evaluated_customer_count if evaluated_customer_count else 0.0
        ),
        duplicate_prediction_rows=duplicate_rows,
        average_prediction_count=average_prediction_count,
        predicted_article_coverage=len(predicted_articles),
        repeat_source_customers=len(baseline.repeat_recommendations),
        recent_popularity_articles=len(baseline.recent_popularity),
        all_time_popularity_articles=len(baseline.all_time_popularity),
    )


def baseline_evaluation_report_to_dict(report: BaselineEvaluationReport) -> dict[str, Any]:
    return asdict(report)


def write_baseline_evaluation_report(report: BaselineEvaluationReport, path: Path | str) -> Path:
    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(baseline_evaluation_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _update_article_stats(stats: dict[str, ArticleStats], article_id: str, seen_date: date) -> None:
    article_stats = stats.setdefault(article_id, ArticleStats())
    article_stats.update(seen_date)


def _merge_sources(sources: Iterable[Iterable[str]], k: int) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for article_id in source:
            if article_id in seen:
                continue
            merged.append(article_id)
            seen.add(article_id)
            if len(merged) == k:
                return tuple(merged)
    return tuple(merged)


def _mean_recall_at_k(
    actual_by_customer: dict[str, tuple[str, ...]],
    predicted_by_customer: dict[str, tuple[str, ...]],
    k: int,
) -> float:
    scores = [
        recall_at_k(actual, predicted_by_customer.get(customer_id, ()), k=k)
        for customer_id, actual in actual_by_customer.items()
    ]
    return sum(scores) / len(scores) if scores else 0.0
