"""Rolling-window validation reports for ranker baselines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hm_recsys.ranking.linear import LearnedLinearRankerReport, LinearRankerConfig
from hm_recsys.retrieval.co_visitation import (
    DEFAULT_MAX_HISTORY_ITEMS,
    DEFAULT_MAX_NEIGHBORS_PER_ITEM,
)


@dataclass(frozen=True)
class RollingRankerWindowResult:
    """One leakage-safe train/evaluation window in a rolling ranker report.

    Attributes:
        train_cutoff: Cutoff date for the previous label window used to train the
            learned ranker.
        train_validation_end_exclusive: Exclusive end of the training label window.
        evaluation_cutoff: Cutoff date for the evaluation label window.
        evaluation_end_exclusive: Exclusive end of the evaluation label window.
        training_candidate_path: Ranker-ready candidate CSV used for training.
        evaluation_candidate_path: Ranker-ready candidate CSV used for evaluation.
        training_candidate_rows: Source-specific training candidate rows read.
        training_unique_candidate_pairs: Unique training customer/article pairs.
        training_positive_pairs: Positive labeled pairs found in training candidates.
        training_negative_pairs: Negative training candidate pairs.
        training_positive_weight: Positive-class weight used by the learned ranker.
        final_average_loss: Weighted logistic loss in the final training epoch.
        evaluation_candidate_rows: Source-specific evaluation candidate rows read.
        evaluation_unique_candidate_pairs: Unique evaluation customer/article pairs.
        evaluated_customers: Evaluation customers with validation labels.
        source_order_map_at_k: MAP@K for repeat→recent-popularity→all-time-popularity
            source ordering on the same candidates.
        deterministic_map_at_k: MAP@K for the transparent deterministic ranker.
        learned_map_at_k: MAP@K for the learned linear ranker.
        source_order_recall_at_k: Recall@K for the source-order baseline.
        deterministic_recall_at_k: Recall@K for the deterministic ranker.
        learned_recall_at_k: Recall@K for the learned ranker.
        delta_learned_vs_deterministic_map_at_k: Learned MAP minus deterministic MAP.
        delta_learned_vs_source_order_map_at_k: Learned MAP minus source-order MAP.
        model_weights: Learned linear weights keyed by feature name.
    """

    train_cutoff: str
    train_validation_end_exclusive: str
    evaluation_cutoff: str
    evaluation_end_exclusive: str
    training_candidate_path: str
    evaluation_candidate_path: str
    training_candidate_rows: int
    training_unique_candidate_pairs: int
    training_positive_pairs: int
    training_negative_pairs: int
    training_positive_weight: float
    final_average_loss: float
    evaluation_candidate_rows: int
    evaluation_unique_candidate_pairs: int
    evaluated_customers: int
    source_order_map_at_k: float
    deterministic_map_at_k: float
    learned_map_at_k: float
    source_order_recall_at_k: float
    deterministic_recall_at_k: float
    learned_recall_at_k: float
    delta_learned_vs_deterministic_map_at_k: float
    delta_learned_vs_source_order_map_at_k: float
    model_weights: dict[str, float]


@dataclass(frozen=True)
class RollingRankerAggregateMetrics:
    """Aggregate metrics across rolling ranker windows.

    Attributes:
        mean_source_order_map_at_k: Mean source-order MAP@K across windows.
        mean_deterministic_map_at_k: Mean deterministic ranker MAP@K across windows.
        mean_learned_map_at_k: Mean learned ranker MAP@K across windows.
        mean_delta_learned_vs_deterministic_map_at_k: Mean learned-vs-deterministic
            MAP@K delta.
        min_delta_learned_vs_deterministic_map_at_k: Worst learned-vs-deterministic
            MAP@K delta across windows.
        max_delta_learned_vs_deterministic_map_at_k: Best learned-vs-deterministic
            MAP@K delta across windows.
        mean_delta_learned_vs_source_order_map_at_k: Mean learned-vs-source-order
            MAP@K delta.
        min_delta_learned_vs_source_order_map_at_k: Worst learned-vs-source-order
            MAP@K delta across windows.
        max_delta_learned_vs_source_order_map_at_k: Best learned-vs-source-order
            MAP@K delta across windows.
        windows_improved_vs_deterministic: Count of windows where learned MAP@K is
            strictly greater than deterministic MAP@K.
        windows_improved_vs_source_order: Count of windows where learned MAP@K is
            strictly greater than source-order MAP@K.
    """

    mean_source_order_map_at_k: float
    mean_deterministic_map_at_k: float
    mean_learned_map_at_k: float
    mean_delta_learned_vs_deterministic_map_at_k: float
    min_delta_learned_vs_deterministic_map_at_k: float
    max_delta_learned_vs_deterministic_map_at_k: float
    mean_delta_learned_vs_source_order_map_at_k: float
    min_delta_learned_vs_source_order_map_at_k: float
    max_delta_learned_vs_source_order_map_at_k: float
    windows_improved_vs_deterministic: int
    windows_improved_vs_source_order: int


@dataclass(frozen=True)
class RollingRankerValidationReport:
    """Leakage-safe rolling validation report for ranker baselines.

    Attributes:
        generated_at_utc: UTC timestamp for the report.
        cutoffs: Evaluation cutoffs included in the report.
        horizon_days: Label-window horizon in days.
        k: Recommendation depth for MAP and recall.
        candidate_k: Maximum candidates per source used to build ranker features.
        config: Learned linear ranker training configuration shared by windows.
        max_target_customers: Optional deterministic smoke-run customer cap.
        include_co_visitation: Whether co-visitation candidates were included.
        co_visitation_max_history_items: Recent unique items per customer used for
            co-visitation retrieval.
        co_visitation_max_neighbors_per_item: Neighbor cap per source article used
            for co-visitation retrieval.
        candidate_summary_paths: Candidate-export summary JSON artifacts written
            for this rolling run.
        window_count: Number of evaluation windows.
        aggregate: Aggregate metrics across windows.
        windows: Per-window train/evaluation metrics.
    """

    generated_at_utc: str
    cutoffs: tuple[str, ...]
    horizon_days: int
    k: int
    candidate_k: int
    config: LinearRankerConfig
    max_target_customers: int | None
    include_co_visitation: bool
    co_visitation_max_history_items: int
    co_visitation_max_neighbors_per_item: int
    candidate_summary_paths: tuple[str, ...]
    window_count: int
    aggregate: RollingRankerAggregateMetrics
    windows: tuple[RollingRankerWindowResult, ...]


def learned_report_to_rolling_window(
    report: LearnedLinearRankerReport,
) -> RollingRankerWindowResult:
    """Convert one learned-ranker report into a rolling-window summary.

    Args:
        report: Leakage-safe learned-ranker train/evaluate report.

    Returns:
        Compact per-window result suitable for a rolling validation report.
    """

    model_weights = dict(zip(report.model.feature_names, report.model.weights, strict=True))
    return RollingRankerWindowResult(
        train_cutoff=report.train_cutoff,
        train_validation_end_exclusive=report.train_validation_end_exclusive,
        evaluation_cutoff=report.evaluation_cutoff,
        evaluation_end_exclusive=report.evaluation_end_exclusive,
        training_candidate_path=report.training.candidate_path,
        evaluation_candidate_path=report.evaluation.candidate_path,
        training_candidate_rows=report.training.candidate_rows,
        training_unique_candidate_pairs=report.training.unique_candidate_pairs,
        training_positive_pairs=report.training.positive_pairs,
        training_negative_pairs=report.training.negative_pairs,
        training_positive_weight=report.training.positive_weight,
        final_average_loss=report.training.final_average_loss,
        evaluation_candidate_rows=report.evaluation.candidate_rows,
        evaluation_unique_candidate_pairs=report.evaluation.unique_candidate_pairs,
        evaluated_customers=report.evaluation.evaluated_customers,
        source_order_map_at_k=report.evaluation.baseline_map_at_k,
        deterministic_map_at_k=report.evaluation.deterministic_map_at_k,
        learned_map_at_k=report.evaluation.map_at_k,
        source_order_recall_at_k=report.evaluation.baseline_recall_at_k,
        deterministic_recall_at_k=report.evaluation.deterministic_recall_at_k,
        learned_recall_at_k=report.evaluation.recall_at_k,
        delta_learned_vs_deterministic_map_at_k=(report.evaluation.delta_vs_deterministic_map_at_k),
        delta_learned_vs_source_order_map_at_k=report.evaluation.delta_vs_baseline_map_at_k,
        model_weights=model_weights,
    )


def build_rolling_ranker_validation_report(
    learned_reports: Sequence[LearnedLinearRankerReport],
    *,
    max_target_customers: int | None = None,
    include_co_visitation: bool = True,
    co_visitation_max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    co_visitation_max_neighbors_per_item: int = DEFAULT_MAX_NEIGHBORS_PER_ITEM,
    candidate_summary_paths: Sequence[str] = (),
) -> RollingRankerValidationReport:
    """Build an aggregate report from per-window learned-ranker reports.

    Args:
        learned_reports: Per-window learned-ranker reports. Each report must use
            the same horizon, rank depth, candidate depth, and training config.
        max_target_customers: Optional deterministic smoke-run customer cap.
        include_co_visitation: Whether co-visitation candidates were included.
        co_visitation_max_history_items: Co-visitation customer-history length.
        co_visitation_max_neighbors_per_item: Co-visitation neighbor cap per item.
        candidate_summary_paths: Candidate-export summary JSON artifacts written
            during the rolling run.

    Returns:
        Rolling validation report with per-window and aggregate metrics.

    Raises:
        ValueError: If reports are missing, inconsistent, duplicated, unordered, or
            contain overlapping train/evaluation label windows.
    """

    reports = tuple(learned_reports)
    if not reports:
        raise ValueError("at least one learned ranker report is required")
    if max_target_customers is not None and max_target_customers <= 0:
        raise ValueError("max_target_customers must be positive when provided")

    _validate_rolling_reports_are_comparable(reports)
    windows = tuple(learned_report_to_rolling_window(report) for report in reports)
    aggregate = _aggregate_windows(windows)

    first_report = reports[0]
    return RollingRankerValidationReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoffs=tuple(window.evaluation_cutoff for window in windows),
        horizon_days=first_report.horizon_days,
        k=first_report.k,
        candidate_k=first_report.candidate_k,
        config=first_report.config,
        max_target_customers=max_target_customers,
        include_co_visitation=include_co_visitation,
        co_visitation_max_history_items=co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=co_visitation_max_neighbors_per_item,
        candidate_summary_paths=tuple(candidate_summary_paths),
        window_count=len(windows),
        aggregate=aggregate,
        windows=windows,
    )


def rolling_ranker_validation_report_to_dict(
    report: RollingRankerValidationReport,
) -> dict[str, Any]:
    """Convert a rolling ranker report to JSON-serializable primitives.

    Args:
        report: Rolling validation report to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """

    return asdict(report)


def write_rolling_ranker_validation_report(
    report: RollingRankerValidationReport,
    path: Path | str,
) -> Path:
    """Write a rolling ranker validation report as deterministic JSON.

    Args:
        report: Report object to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(rolling_ranker_validation_report_to_dict(report), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _validate_rolling_reports_are_comparable(
    reports: tuple[LearnedLinearRankerReport, ...],
) -> None:
    """Validate report consistency and leakage-safe window ordering."""

    first_report = reports[0]
    cutoffs = tuple(report.evaluation_cutoff for report in reports)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("rolling evaluation cutoffs must be unique")
    if tuple(sorted(cutoffs)) != cutoffs:
        raise ValueError("rolling evaluation cutoffs must be in ascending order")

    for report in reports:
        if report.horizon_days != first_report.horizon_days:
            raise ValueError("all rolling reports must use the same horizon_days")
        if report.k != first_report.k:
            raise ValueError("all rolling reports must use the same k")
        if report.candidate_k != first_report.candidate_k:
            raise ValueError("all rolling reports must use the same candidate_k")
        if report.config != first_report.config:
            raise ValueError("all rolling reports must use the same linear ranker config")
        if report.train_validation_end_exclusive > report.evaluation_cutoff:
            raise ValueError("training label window must end before the evaluation cutoff")


def _aggregate_windows(
    windows: tuple[RollingRankerWindowResult, ...],
) -> RollingRankerAggregateMetrics:
    """Compute aggregate metrics across rolling windows."""

    deterministic_deltas = tuple(
        window.delta_learned_vs_deterministic_map_at_k for window in windows
    )
    source_order_deltas = tuple(window.delta_learned_vs_source_order_map_at_k for window in windows)
    return RollingRankerAggregateMetrics(
        mean_source_order_map_at_k=_mean(tuple(window.source_order_map_at_k for window in windows)),
        mean_deterministic_map_at_k=_mean(
            tuple(window.deterministic_map_at_k for window in windows)
        ),
        mean_learned_map_at_k=_mean(tuple(window.learned_map_at_k for window in windows)),
        mean_delta_learned_vs_deterministic_map_at_k=_mean(deterministic_deltas),
        min_delta_learned_vs_deterministic_map_at_k=min(deterministic_deltas),
        max_delta_learned_vs_deterministic_map_at_k=max(deterministic_deltas),
        mean_delta_learned_vs_source_order_map_at_k=_mean(source_order_deltas),
        min_delta_learned_vs_source_order_map_at_k=min(source_order_deltas),
        max_delta_learned_vs_source_order_map_at_k=max(source_order_deltas),
        windows_improved_vs_deterministic=sum(1 for delta in deterministic_deltas if delta > 0.0),
        windows_improved_vs_source_order=sum(1 for delta in source_order_deltas if delta > 0.0),
    )


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty numeric sequence."""

    return sum(values) / len(values)
