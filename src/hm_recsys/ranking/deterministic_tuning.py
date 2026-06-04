"""Leakage-safe deterministic ranker weight tuning."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from hm_recsys.evaluation.metrics import mean_average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.deterministic import (
    DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    CandidateFeatures,
    DeterministicRankerWeights,
    aggregate_candidate_features,
    iter_candidate_records_from_csv,
    rank_candidates_by_customer,
)
from hm_recsys.retrieval.candidate_export import CandidateRecord


@dataclass(frozen=True)
class DeterministicRankerTuningGrid:
    """Small explicit grid for transparent deterministic-ranker weight tuning.

    The default grid intentionally tunes only the segment/metadata-affinity and
    generic aggregation weights that have shown bounded-validation signal. Core
    repeat, recent-popularity, all-time-popularity, and co-visitation weights stay
    anchored to the hand-built deterministic baseline unless a future experiment
    has a narrower hypothesis.

    Attributes:
        garment_group_popularity_presence_weights: Candidate additive weights for
            garment-group affinity candidates.
        garment_group_popularity_score_weights: Candidate score weights for
            garment-group affinity candidates.
        age_segment_popularity_presence_weights: Candidate additive weights for
            age-segment popularity candidates.
        age_segment_popularity_score_weights: Candidate score weights for
            age-segment popularity candidates.
        source_count_weights: Candidate weights for source-count agreement.
        best_rank_score_weights: Candidate weights for reciprocal best source rank.
    """

    garment_group_popularity_presence_weights: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    garment_group_popularity_score_weights: tuple[float, ...] = (0.10, 0.25, 0.40)
    age_segment_popularity_presence_weights: tuple[float, ...] = (0.0, 0.15, 0.30, 0.45)
    age_segment_popularity_score_weights: tuple[float, ...] = (0.0, 0.10, 0.20)
    source_count_weights: tuple[float, ...] = (0.0, 0.05, 0.10)
    best_rank_score_weights: tuple[float, ...] = (0.0, 0.05)

    def __post_init__(self) -> None:
        """Validate grid values.

        Raises:
            ValueError: If any weight list is empty or contains a negative value.
        """

        for field_name, values in asdict(self).items():
            if not values:
                raise ValueError(f"{field_name} must contain at least one value")
            if any(value < 0.0 for value in values):
                raise ValueError(f"{field_name} must not contain negative values")

    @property
    def trial_count(self) -> int:
        """Return the number of weight combinations in the grid."""

        count = 1
        for values in asdict(self).values():
            count *= len(values)
        return count

    def iter_weights(
        self,
        base_weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    ) -> Iterable[DeterministicRankerWeights]:
        """Yield deterministic ranker weights in stable grid order.

        Args:
            base_weights: Baseline weights for fields not tuned by this grid.

        Yields:
            Deterministic ranker weights with one grid combination applied.
        """

        for (
            garment_presence,
            garment_score,
            age_presence,
            age_score,
            source_count,
            best_rank,
        ) in product(
            self.garment_group_popularity_presence_weights,
            self.garment_group_popularity_score_weights,
            self.age_segment_popularity_presence_weights,
            self.age_segment_popularity_score_weights,
            self.source_count_weights,
            self.best_rank_score_weights,
        ):
            yield replace(
                base_weights,
                garment_group_popularity_presence_weight=garment_presence,
                garment_group_popularity_score_weight=garment_score,
                age_segment_popularity_presence_weight=age_presence,
                age_segment_popularity_score_weight=age_score,
                source_count_weight=source_count,
                best_rank_score_weight=best_rank,
            )


DEFAULT_DETERMINISTIC_RANKER_TUNING_GRID = DeterministicRankerTuningGrid()


@dataclass(frozen=True)
class DeterministicRankerTuningMetrics:
    """Metrics for one deterministic-ranker weight set on one candidate file.

    Attributes:
        candidate_path: Candidate CSV evaluated.
        candidate_rows: Source-specific candidate rows read.
        unique_candidate_pairs: Unique customer/article pairs after aggregation.
        evaluated_customers: Candidate-file customers with validation labels.
        map_at_k: MAP@K for ranked predictions.
        recall_at_k: Mean recall@K for ranked predictions.
        source_row_counts: Source-specific row counts.
    """

    candidate_path: str
    candidate_rows: int
    unique_candidate_pairs: int
    evaluated_customers: int
    map_at_k: float
    recall_at_k: float
    source_row_counts: dict[str, int]


@dataclass(frozen=True)
class DeterministicRankerTuningTrial:
    """One ranked training-window weight trial retained in the tuning report."""

    rank: int
    train_map_at_k: float
    train_recall_at_k: float
    weights: DeterministicRankerWeights


@dataclass(frozen=True)
class DeterministicRankerWeightSelection:
    """Weights selected on one labeled temporal window.

    Attributes:
        candidate_path: Candidate CSV used for weight selection.
        k: Recommendation depth used for tuning metrics.
        grid: Explicit tuning grid.
        trial_count: Number of grid trials evaluated.
        top_train_trials: Best trials by tuning-window MAP@K then recall@K.
        default_weights: Untuned deterministic ranker weights.
        selected_weights: Best weights selected on the tuning window.
        default_metrics: Untuned metrics on the tuning candidate file.
        selected_metrics: Selected-weight metrics on the tuning candidate file.
        delta_selected_vs_default_map_at_k: Tuning-window MAP delta.
        delta_selected_vs_default_recall_at_k: Tuning-window recall delta.
    """

    candidate_path: str
    k: int
    grid: DeterministicRankerTuningGrid
    trial_count: int
    top_train_trials: tuple[DeterministicRankerTuningTrial, ...]
    default_weights: DeterministicRankerWeights
    selected_weights: DeterministicRankerWeights
    default_metrics: DeterministicRankerTuningMetrics
    selected_metrics: DeterministicRankerTuningMetrics
    delta_selected_vs_default_map_at_k: float
    delta_selected_vs_default_recall_at_k: float


@dataclass(frozen=True)
class DeterministicRankerTuningReport:
    """Leakage-safe deterministic-ranker weight tuning report.

    Attributes:
        generated_at_utc: UTC timestamp for the report.
        train_cutoff: Cutoff of the previous label window used for tuning.
        train_validation_end_exclusive: Exclusive end of the tuning label window.
        evaluation_cutoff: Cutoff of the evaluation label window.
        evaluation_end_exclusive: Exclusive end of the evaluation label window.
        horizon_days: Label-window horizon in days.
        k: Recommendation depth for MAP/recall.
        candidate_k: Maximum candidates per source used to build ranker features.
        grid: Explicit tuning grid.
        trial_count: Total number of grid trials evaluated on the tuning window.
        top_train_trials: Best training-window trials by MAP@K then recall@K.
        default_weights: Untuned deterministic ranker weights.
        selected_weights: Best weights selected using only the training window.
        default_train: Untuned metrics on the tuning candidate file.
        selected_train: Selected-weight metrics on the tuning candidate file.
        default_evaluation: Untuned metrics on the evaluation candidate file.
        selected_evaluation: Selected-weight metrics on the evaluation candidate file.
        delta_selected_vs_default_map_at_k: Evaluation MAP delta.
        delta_selected_vs_default_recall_at_k: Evaluation recall delta.
    """

    generated_at_utc: str
    train_cutoff: str
    train_validation_end_exclusive: str
    evaluation_cutoff: str
    evaluation_end_exclusive: str
    horizon_days: int
    k: int
    candidate_k: int
    grid: DeterministicRankerTuningGrid
    trial_count: int
    top_train_trials: tuple[DeterministicRankerTuningTrial, ...]
    default_weights: DeterministicRankerWeights
    selected_weights: DeterministicRankerWeights
    default_train: DeterministicRankerTuningMetrics
    selected_train: DeterministicRankerTuningMetrics
    default_evaluation: DeterministicRankerTuningMetrics
    selected_evaluation: DeterministicRankerTuningMetrics
    delta_selected_vs_default_map_at_k: float
    delta_selected_vs_default_recall_at_k: float


def tune_deterministic_ranker_from_csv(
    train_candidate_path: Path | str,
    train_validation_labels: Mapping[str, Iterable[str]],
    train_split: TemporalSplit,
    evaluation_candidate_path: Path | str,
    evaluation_validation_labels: Mapping[str, Iterable[str]],
    evaluation_split: TemporalSplit,
    *,
    k: int = 12,
    candidate_k: int = 12,
    grid: DeterministicRankerTuningGrid = DEFAULT_DETERMINISTIC_RANKER_TUNING_GRID,
    default_weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    top_n: int = 10,
) -> DeterministicRankerTuningReport:
    """Tune deterministic weights on one window and evaluate on a later window.

    Args:
        train_candidate_path: Candidate CSV for the tuning label window.
        train_validation_labels: Tuning-window labels keyed by customer ID.
        train_split: Temporal split for the tuning label window.
        evaluation_candidate_path: Candidate CSV for the held-out evaluation window.
        evaluation_validation_labels: Evaluation-window labels keyed by customer ID.
        evaluation_split: Temporal split for the held-out evaluation window.
        k: MAP/recommendation depth.
        candidate_k: Maximum candidates per source used to create the candidate CSVs.
        grid: Weight grid searched using only the tuning window.
        default_weights: Untuned baseline weights for comparison.
        top_n: Number of best training-window trials retained in the report.

    Returns:
        Tuning report with selected weights and held-out evaluation metrics.

    Raises:
        ValueError: If limits are invalid or temporal windows overlap.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if train_split.validation_end > evaluation_split.cutoff:
        raise ValueError("training label window must end before the evaluation cutoff")

    train_data = _load_tuning_candidate_data(train_candidate_path, train_validation_labels)
    evaluation_data = _load_tuning_candidate_data(
        evaluation_candidate_path,
        evaluation_validation_labels,
    )
    default_train = _evaluate_weights(train_data, k=k, weights=default_weights)
    default_evaluation = _evaluate_weights(evaluation_data, k=k, weights=default_weights)
    selected_weights, top_trials = _select_best_weights(
        train_data,
        k=k,
        grid=grid,
        default_weights=default_weights,
        top_n=top_n,
    )
    selected_train = _evaluate_weights(train_data, k=k, weights=selected_weights)
    selected_evaluation = _evaluate_weights(evaluation_data, k=k, weights=selected_weights)

    return DeterministicRankerTuningReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        train_cutoff=train_split.cutoff.isoformat(),
        train_validation_end_exclusive=train_split.validation_end.isoformat(),
        evaluation_cutoff=evaluation_split.cutoff.isoformat(),
        evaluation_end_exclusive=evaluation_split.validation_end.isoformat(),
        horizon_days=evaluation_split.horizon_days,
        k=k,
        candidate_k=candidate_k,
        grid=grid,
        trial_count=grid.trial_count,
        top_train_trials=top_trials,
        default_weights=default_weights,
        selected_weights=selected_weights,
        default_train=default_train,
        selected_train=selected_train,
        default_evaluation=default_evaluation,
        selected_evaluation=selected_evaluation,
        delta_selected_vs_default_map_at_k=(
            selected_evaluation.map_at_k - default_evaluation.map_at_k
        ),
        delta_selected_vs_default_recall_at_k=(
            selected_evaluation.recall_at_k - default_evaluation.recall_at_k
        ),
    )


def select_deterministic_ranker_weights_from_csv(
    candidate_path: Path | str,
    validation_labels: Mapping[str, Iterable[str]],
    *,
    k: int = 12,
    grid: DeterministicRankerTuningGrid = DEFAULT_DETERMINISTIC_RANKER_TUNING_GRID,
    default_weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    top_n: int = 10,
) -> DeterministicRankerWeightSelection:
    """Select deterministic weights using one labeled temporal window.

    Args:
        candidate_path: Candidate CSV for the labeled tuning window.
        validation_labels: Labels keyed by customer ID for that same window.
        k: Recommendation depth used for MAP/recall.
        grid: Weight grid searched using only this labeled window.
        default_weights: Untuned baseline weights for comparison.
        top_n: Number of top trials retained in the report.

    Returns:
        Selected weights and tuning-window diagnostics.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    data = _load_tuning_candidate_data(candidate_path, validation_labels)
    default_metrics = _evaluate_weights(data, k=k, weights=default_weights)
    selected_weights, top_trials = _select_best_weights(
        data,
        k=k,
        grid=grid,
        default_weights=default_weights,
        top_n=top_n,
    )
    selected_metrics = _evaluate_weights(data, k=k, weights=selected_weights)
    return DeterministicRankerWeightSelection(
        candidate_path=data.candidate_path,
        k=k,
        grid=grid,
        trial_count=grid.trial_count,
        top_train_trials=top_trials,
        default_weights=default_weights,
        selected_weights=selected_weights,
        default_metrics=default_metrics,
        selected_metrics=selected_metrics,
        delta_selected_vs_default_map_at_k=selected_metrics.map_at_k - default_metrics.map_at_k,
        delta_selected_vs_default_recall_at_k=(
            selected_metrics.recall_at_k - default_metrics.recall_at_k
        ),
    )


def deterministic_ranker_tuning_report_to_dict(
    report: DeterministicRankerTuningReport,
) -> dict[str, Any]:
    """Convert a deterministic tuning report to JSON-serializable primitives."""

    return asdict(report)


def write_deterministic_ranker_tuning_report(
    report: DeterministicRankerTuningReport,
    path: Path | str,
) -> Path:
    """Write a deterministic-ranker tuning report as JSON.

    Args:
        report: Tuning report to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(deterministic_ranker_tuning_report_to_dict(report), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report_path


@dataclass(frozen=True)
class _TuningCandidateData:
    candidate_path: str
    features_by_customer: dict[str, dict[str, CandidateFeatures]]
    labels_for_candidate_customers: dict[str, tuple[str, ...]]
    candidate_rows: int
    unique_candidate_pairs: int
    source_row_counts: dict[str, int]


def _load_tuning_candidate_data(
    candidate_path: Path | str,
    validation_labels: Mapping[str, Iterable[str]],
) -> _TuningCandidateData:
    resolved_candidate_path = Path(candidate_path).expanduser().resolve()
    source_row_counts: Counter[str] = Counter()

    def counting_records() -> Iterable[CandidateRecord]:
        for record in iter_candidate_records_from_csv(resolved_candidate_path):
            source_row_counts[record.source] += 1
            yield record

    features_by_customer = aggregate_candidate_features(counting_records(), validation_labels)
    labels_for_candidate_customers = {
        customer_id: tuple(validation_labels[customer_id])
        for customer_id in features_by_customer
        if customer_id in validation_labels
    }
    unique_candidate_pairs = sum(
        len(article_features) for article_features in features_by_customer.values()
    )
    return _TuningCandidateData(
        candidate_path=str(resolved_candidate_path),
        features_by_customer=features_by_customer,
        labels_for_candidate_customers=labels_for_candidate_customers,
        candidate_rows=sum(source_row_counts.values()),
        unique_candidate_pairs=unique_candidate_pairs,
        source_row_counts=dict(sorted(source_row_counts.items())),
    )


def _select_best_weights(
    train_data: _TuningCandidateData,
    *,
    k: int,
    grid: DeterministicRankerTuningGrid,
    default_weights: DeterministicRankerWeights,
    top_n: int,
) -> tuple[DeterministicRankerWeights, tuple[DeterministicRankerTuningTrial, ...]]:
    scored_trials: list[tuple[float, float, int, DeterministicRankerWeights]] = []
    for index, weights in enumerate(grid.iter_weights(default_weights), start=1):
        metrics = _evaluate_weights(train_data, k=k, weights=weights)
        scored_trials.append((metrics.map_at_k, metrics.recall_at_k, index, weights))

    scored_trials.sort(key=lambda trial: (-trial[0], -trial[1], trial[2]))
    top_trials = tuple(
        DeterministicRankerTuningTrial(
            rank=rank,
            train_map_at_k=train_map,
            train_recall_at_k=train_recall,
            weights=weights,
        )
        for rank, (train_map, train_recall, _, weights) in enumerate(
            scored_trials[:top_n],
            start=1,
        )
    )
    return top_trials[0].weights, top_trials


def _evaluate_weights(
    data: _TuningCandidateData,
    *,
    k: int,
    weights: DeterministicRankerWeights,
) -> DeterministicRankerTuningMetrics:
    predictions = rank_candidates_by_customer(data.features_by_customer, k=k, weights=weights)
    return DeterministicRankerTuningMetrics(
        candidate_path=data.candidate_path,
        candidate_rows=data.candidate_rows,
        unique_candidate_pairs=data.unique_candidate_pairs,
        evaluated_customers=len(data.labels_for_candidate_customers),
        map_at_k=mean_average_precision_at_k(data.labels_for_candidate_customers, predictions, k=k),
        recall_at_k=_mean_recall_at_k(data.labels_for_candidate_customers, predictions, k=k),
        source_row_counts=data.source_row_counts,
    )


def _mean_recall_at_k(
    actual_by_customer: Mapping[str, Iterable[str]],
    predicted_by_customer: Mapping[str, Iterable[str]],
    k: int,
) -> float:
    scores = [
        recall_at_k(actual, predicted_by_customer.get(customer_id, ()), k=k)
        for customer_id, actual in actual_by_customer.items()
    ]
    return sum(scores) / len(scores) if scores else 0.0
