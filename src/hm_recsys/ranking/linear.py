"""Leakage-safe learned linear ranker for candidate-source features."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import exp, log1p
from pathlib import Path
from typing import Any

from hm_recsys.evaluation.metrics import mean_average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.deterministic import (
    CandidateFeatures,
    aggregate_candidate_features,
    build_source_order_baseline_predictions,
    iter_candidate_records_from_csv,
    rank_candidates_by_customer,
)
from hm_recsys.retrieval.candidate_export import CandidateRecord

LINEAR_FEATURE_NAMES = (
    "bias",
    "has_repeat",
    "repeat_score",
    "repeat_rank_reciprocal",
    "has_recent_popularity",
    "recent_popularity_score",
    "recent_popularity_rank_reciprocal",
    "has_all_time_popularity",
    "all_time_popularity_score",
    "all_time_popularity_rank_reciprocal",
    "has_co_visitation",
    "co_visitation_log_score",
    "co_visitation_rank_reciprocal",
    "has_age_segment_popularity",
    "age_segment_popularity_score",
    "age_segment_popularity_rank_reciprocal",
    "has_garment_group_popularity",
    "garment_group_popularity_score",
    "garment_group_popularity_rank_reciprocal",
    "has_content_similarity",
    "content_similarity_score",
    "content_similarity_rank_reciprocal",
    "has_two_tower_retrieval",
    "two_tower_retrieval_score",
    "two_tower_retrieval_rank_reciprocal",
    "source_count_scaled",
    "best_rank_reciprocal",
)


@dataclass(frozen=True)
class LinearRankerConfig:
    """Training configuration for the learned linear ranker.

    Attributes:
        epochs: Number of deterministic passes over aggregated training pairs.
        learning_rate: SGD step size.
        l2: L2 regularization strength applied to non-bias weights.
        positive_weight: Optional explicit positive-class weight. When ``None``,
            the trainer uses the negative/positive ratio capped by
            ``max_auto_positive_weight``.
        max_auto_positive_weight: Maximum automatically computed positive weight.
    """

    epochs: int = 3
    learning_rate: float = 0.01
    l2: float = 0.001
    positive_weight: float | None = None
    max_auto_positive_weight: float = 10.0

    def __post_init__(self) -> None:
        """Validate training hyperparameters.

        Raises:
            ValueError: If any numeric hyperparameter is invalid.
        """

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        if self.positive_weight is not None and self.positive_weight <= 0:
            raise ValueError("positive_weight must be positive when provided")
        if self.max_auto_positive_weight <= 0:
            raise ValueError("max_auto_positive_weight must be positive")


DEFAULT_LINEAR_RANKER_CONFIG = LinearRankerConfig()


@dataclass(frozen=True)
class LinearRankerModel:
    """Learned linear ranker model.

    Attributes:
        feature_names: Feature names matching the weight vector.
        weights: Learned model weights.
    """

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class LinearRankerTrainingSummary:
    """Training summary for the learned linear ranker.

    Attributes:
        candidate_path: Training candidate CSV path.
        candidate_rows: Source-specific candidate rows read.
        unique_candidate_pairs: Unique training customer/article pairs.
        positive_pairs: Positive labeled pairs present in candidates.
        negative_pairs: Negative candidate pairs.
        positive_weight: Positive-class weight used during SGD.
        epochs: Number of epochs completed.
        final_average_loss: Weighted logistic loss in the final epoch.
        source_row_counts: Training source row counts.
    """

    candidate_path: str
    candidate_rows: int
    unique_candidate_pairs: int
    positive_pairs: int
    negative_pairs: int
    positive_weight: float
    epochs: int
    final_average_loss: float
    source_row_counts: dict[str, int]


@dataclass(frozen=True)
class LinearRankerEvaluation:
    """Evaluation metrics for a learned linear ranker on one candidate file.

    Attributes:
        candidate_path: Evaluation candidate CSV path.
        candidate_rows: Source-specific candidate rows read.
        unique_candidate_pairs: Unique evaluation customer/article pairs.
        evaluated_customers: Customers with labels in the candidate file.
        map_at_k: Learned ranker MAP@K.
        recall_at_k: Learned ranker recall@K.
        deterministic_map_at_k: Deterministic ranker MAP@K on the same candidates.
        deterministic_recall_at_k: Deterministic ranker recall@K on the same candidates.
        baseline_map_at_k: Source-order baseline MAP@K on the same candidates.
        baseline_recall_at_k: Source-order baseline recall@K on the same candidates.
        delta_vs_deterministic_map_at_k: Learned MAP minus deterministic MAP.
        delta_vs_baseline_map_at_k: Learned MAP minus source-order baseline MAP.
        source_row_counts: Evaluation source row counts.
    """

    candidate_path: str
    candidate_rows: int
    unique_candidate_pairs: int
    evaluated_customers: int
    map_at_k: float
    recall_at_k: float
    deterministic_map_at_k: float
    deterministic_recall_at_k: float
    baseline_map_at_k: float
    baseline_recall_at_k: float
    delta_vs_deterministic_map_at_k: float
    delta_vs_baseline_map_at_k: float
    source_row_counts: dict[str, int]


@dataclass(frozen=True)
class LearnedLinearRankerReport:
    """Leakage-safe train/evaluate report for the learned linear ranker.

    Attributes:
        generated_at_utc: UTC timestamp for the report.
        train_cutoff: Training-label cutoff date.
        train_validation_end_exclusive: Exclusive end of the training label window.
        evaluation_cutoff: Evaluation-label cutoff date.
        evaluation_end_exclusive: Exclusive end of the evaluation label window.
        horizon_days: Label-window horizon in days.
        k: Recommendation depth for MAP/recall.
        candidate_k: Maximum candidates per source.
        config: Training configuration.
        model: Learned model weights.
        training: Training summary.
        evaluation: Evaluation metrics.
    """

    generated_at_utc: str
    train_cutoff: str
    train_validation_end_exclusive: str
    evaluation_cutoff: str
    evaluation_end_exclusive: str
    horizon_days: int
    k: int
    candidate_k: int
    config: LinearRankerConfig
    model: LinearRankerModel
    training: LinearRankerTrainingSummary
    evaluation: LinearRankerEvaluation


@dataclass(frozen=True)
class LinearRankerTrainingResult:
    """Learned model plus training metadata.

    Attributes:
        model: Learned linear ranker model.
        summary: Training metadata and final loss.
    """

    model: LinearRankerModel
    summary: LinearRankerTrainingSummary


def previous_window_split(evaluation_split: TemporalSplit) -> TemporalSplit:
    """Return the previous non-overlapping split used for ranker training.

    Args:
        evaluation_split: Evaluation split whose cutoff is the target validation
            window.

    Returns:
        Split with cutoff shifted back by ``horizon_days``.
    """

    return TemporalSplit(
        cutoff=evaluation_split.cutoff - timedelta(days=evaluation_split.horizon_days),
        horizon_days=evaluation_split.horizon_days,
    )


def feature_vector(features: CandidateFeatures) -> tuple[float, ...]:
    """Convert candidate features into a normalized linear-model vector.

    Args:
        features: Aggregated candidate-source features.

    Returns:
        Tuple matching ``LINEAR_FEATURE_NAMES``.
    """

    return (
        1.0,
        float(features.has_repeat),
        features.repeat_score,
        _rank_reciprocal(features.repeat_rank),
        float(features.has_recent_popularity),
        features.recent_popularity_score,
        _rank_reciprocal(features.recent_popularity_rank),
        float(features.has_all_time_popularity),
        features.all_time_popularity_score,
        _rank_reciprocal(features.all_time_popularity_rank),
        float(features.has_co_visitation),
        log1p(features.co_visitation_score),
        _rank_reciprocal(features.co_visitation_rank),
        float(features.has_age_segment_popularity),
        features.age_segment_popularity_score,
        _rank_reciprocal(features.age_segment_popularity_rank),
        float(features.has_garment_group_popularity),
        features.garment_group_popularity_score,
        _rank_reciprocal(features.garment_group_popularity_rank),
        float(features.has_content_similarity),
        features.content_similarity_score,
        _rank_reciprocal(features.content_similarity_rank),
        float(features.has_two_tower_retrieval),
        features.two_tower_retrieval_score,
        _rank_reciprocal(features.two_tower_retrieval_rank),
        features.source_count / 8.0,
        _rank_reciprocal(features.best_rank),
    )


def train_linear_ranker_from_csv(
    candidate_path: Path | str,
    validation_labels: Mapping[str, Iterable[str]],
    config: LinearRankerConfig = DEFAULT_LINEAR_RANKER_CONFIG,
) -> LinearRankerTrainingResult:
    """Train a weighted logistic linear ranker from candidate rows.

    Args:
        candidate_path: Ranker-ready candidate CSV for the training window.
        validation_labels: Training-window labels keyed by customer ID.
        config: Training hyperparameters.

    Returns:
        Learned model and training summary.
    """

    resolved_candidate_path = Path(candidate_path).expanduser().resolve()
    features_by_customer, source_counts, candidate_rows = _load_features_with_counts(
        resolved_candidate_path, validation_labels
    )
    examples = tuple(
        features
        for customer_features in features_by_customer.values()
        for features in customer_features.values()
    )
    positive_pairs = sum(features.label for features in examples)
    negative_pairs = len(examples) - positive_pairs
    positive_weight = _resolve_positive_weight(
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        config=config,
    )
    weights = [0.0] * len(LINEAR_FEATURE_NAMES)
    final_average_loss = 0.0
    for _ in range(config.epochs):
        total_loss = 0.0
        for features in sorted(examples, key=lambda item: (item.customer_id, item.article_id)):
            vector = feature_vector(features)
            label = float(features.label)
            prediction = _sigmoid(_dot(weights, vector))
            example_weight = positive_weight if features.label else 1.0
            gradient_scale = (prediction - label) * example_weight
            for index, value in enumerate(vector):
                regularization = config.l2 * weights[index] if index != 0 else 0.0
                weights[index] -= config.learning_rate * (gradient_scale * value + regularization)
            total_loss += _weighted_log_loss(
                label=label,
                prediction=prediction,
                weight=example_weight,
            )
        final_average_loss = total_loss / len(examples) if examples else 0.0
    model = LinearRankerModel(
        feature_names=LINEAR_FEATURE_NAMES,
        weights=tuple(weights),
    )
    summary = LinearRankerTrainingSummary(
        candidate_path=str(resolved_candidate_path),
        candidate_rows=candidate_rows,
        unique_candidate_pairs=len(examples),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        positive_weight=positive_weight,
        epochs=config.epochs,
        final_average_loss=final_average_loss,
        source_row_counts=dict(sorted(source_counts.items())),
    )
    return LinearRankerTrainingResult(model=model, summary=summary)


def score_with_linear_model(features: CandidateFeatures, model: LinearRankerModel) -> float:
    """Score one candidate feature row with a learned linear model.

    Args:
        features: Aggregated candidate features.
        model: Learned linear ranker model.

    Returns:
        Raw linear score; higher values rank earlier.
    """

    if model.feature_names != LINEAR_FEATURE_NAMES:
        raise ValueError("linear ranker model feature_names do not match expected schema")
    return _dot(model.weights, feature_vector(features))


def rank_with_linear_model(
    features_by_customer: Mapping[str, Mapping[str, CandidateFeatures]],
    model: LinearRankerModel,
    k: int,
) -> dict[str, tuple[str, ...]]:
    """Rank candidates per customer with a learned linear model.

    Args:
        features_by_customer: Aggregated candidate features by customer.
        model: Learned linear ranker model.
        k: Maximum recommendations per customer.

    Returns:
        Ranked article IDs per customer.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    predictions: dict[str, tuple[str, ...]] = {}
    for customer_id, article_features in features_by_customer.items():
        ranked_features = sorted(
            article_features.values(),
            key=lambda features: (
                -score_with_linear_model(features, model),
                -(features.source_count),
                features.best_rank if features.best_rank is not None else 10**9,
                features.article_id,
            ),
        )
        predictions[customer_id] = tuple(features.article_id for features in ranked_features[:k])
    return predictions


def evaluate_linear_ranker_from_csv(
    candidate_path: Path | str,
    validation_labels: Mapping[str, Iterable[str]],
    model: LinearRankerModel,
    k: int = 12,
) -> LinearRankerEvaluation:
    """Evaluate a learned linear ranker from a candidate CSV.

    Args:
        candidate_path: Ranker-ready candidate CSV for the evaluation window.
        validation_labels: Evaluation labels keyed by customer ID.
        model: Learned linear ranker model.
        k: MAP/recommendation depth.

    Returns:
        Evaluation metrics and same-scope deterministic/baseline comparisons.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    resolved_candidate_path = Path(candidate_path).expanduser().resolve()
    features_by_customer, source_counts, candidate_rows = _load_features_with_counts(
        resolved_candidate_path, validation_labels
    )
    labels_for_candidate_customers = {
        customer_id: tuple(validation_labels[customer_id])
        for customer_id in features_by_customer
        if customer_id in validation_labels
    }
    learned_predictions = rank_with_linear_model(features_by_customer, model=model, k=k)
    deterministic_predictions = rank_candidates_by_customer(features_by_customer, k=k)
    baseline_predictions = build_source_order_baseline_predictions(features_by_customer, k=k)
    learned_map = mean_average_precision_at_k(
        labels_for_candidate_customers, learned_predictions, k=k
    )
    deterministic_map = mean_average_precision_at_k(
        labels_for_candidate_customers, deterministic_predictions, k=k
    )
    baseline_map = mean_average_precision_at_k(
        labels_for_candidate_customers, baseline_predictions, k=k
    )
    learned_recall = _mean_recall_at_k(labels_for_candidate_customers, learned_predictions, k=k)
    deterministic_recall = _mean_recall_at_k(
        labels_for_candidate_customers, deterministic_predictions, k=k
    )
    baseline_recall = _mean_recall_at_k(labels_for_candidate_customers, baseline_predictions, k=k)
    unique_pairs = sum(
        len(customer_features) for customer_features in features_by_customer.values()
    )
    return LinearRankerEvaluation(
        candidate_path=str(resolved_candidate_path),
        candidate_rows=candidate_rows,
        unique_candidate_pairs=unique_pairs,
        evaluated_customers=len(labels_for_candidate_customers),
        map_at_k=learned_map,
        recall_at_k=learned_recall,
        deterministic_map_at_k=deterministic_map,
        deterministic_recall_at_k=deterministic_recall,
        baseline_map_at_k=baseline_map,
        baseline_recall_at_k=baseline_recall,
        delta_vs_deterministic_map_at_k=learned_map - deterministic_map,
        delta_vs_baseline_map_at_k=learned_map - baseline_map,
        source_row_counts=dict(sorted(source_counts.items())),
    )


def build_learned_linear_ranker_report(
    train_split: TemporalSplit,
    evaluation_split: TemporalSplit,
    k: int,
    candidate_k: int,
    config: LinearRankerConfig,
    training_result: LinearRankerTrainingResult,
    evaluation: LinearRankerEvaluation,
) -> LearnedLinearRankerReport:
    """Build a train/evaluate report for the learned linear ranker.

    Args:
        train_split: Previous temporal split used for training labels.
        evaluation_split: Target temporal split used for evaluation labels.
        k: Recommendation depth for MAP/recall.
        candidate_k: Maximum candidates per source.
        config: Training configuration.
        training_result: Learned model and training metadata.
        evaluation: Evaluation metrics.

    Returns:
        Complete learned ranker report.
    """

    return LearnedLinearRankerReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        train_cutoff=train_split.cutoff.isoformat(),
        train_validation_end_exclusive=train_split.validation_end.isoformat(),
        evaluation_cutoff=evaluation_split.cutoff.isoformat(),
        evaluation_end_exclusive=evaluation_split.validation_end.isoformat(),
        horizon_days=evaluation_split.horizon_days,
        k=k,
        candidate_k=candidate_k,
        config=config,
        model=training_result.model,
        training=training_result.summary,
        evaluation=evaluation,
    )


def learned_linear_ranker_report_to_dict(report: LearnedLinearRankerReport) -> dict[str, Any]:
    """Convert a learned linear ranker report to JSON-serializable primitives.

    Args:
        report: Report object to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """

    return asdict(report)


def write_learned_linear_ranker_report(
    report: LearnedLinearRankerReport,
    path: Path | str,
) -> Path:
    """Write a learned linear ranker report as deterministic JSON.

    Args:
        report: Report object to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(learned_linear_ranker_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _load_features_with_counts(
    candidate_path: Path,
    validation_labels: Mapping[str, Iterable[str]],
) -> tuple[dict[str, dict[str, CandidateFeatures]], Counter[str], int]:
    """Load candidate features while counting source rows.

    Args:
        candidate_path: Candidate CSV path.
        validation_labels: Labels keyed by customer ID.

    Returns:
        Tuple of aggregated features, source row counts, and total candidate rows.
    """

    source_counts: Counter[str] = Counter()
    row_count = 0

    def counting_records() -> Iterable[CandidateRecord]:
        """Yield candidate records while counting source rows."""

        nonlocal row_count
        for record in iter_candidate_records_from_csv(candidate_path):
            source_counts[record.source] += 1
            row_count += 1
            yield record

    features_by_customer = aggregate_candidate_features(counting_records(), validation_labels)
    return features_by_customer, source_counts, row_count


def _resolve_positive_weight(
    positive_pairs: int,
    negative_pairs: int,
    config: LinearRankerConfig,
) -> float:
    """Resolve explicit or automatic positive-class weight."""

    if config.positive_weight is not None:
        return config.positive_weight
    if positive_pairs <= 0:
        return 1.0
    return min(negative_pairs / positive_pairs, config.max_auto_positive_weight)


def _rank_reciprocal(rank: int | None) -> float:
    """Return reciprocal rank with ``0.0`` for missing ranks."""

    return 0.0 if rank is None else 1.0 / rank


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic sigmoid."""

    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _dot(weights: Sequence[float], vector: Sequence[float]) -> float:
    """Compute a dot product for equal-length numeric sequences."""

    return sum(weight * value for weight, value in zip(weights, vector, strict=True))


def _weighted_log_loss(label: float, prediction: float, weight: float) -> float:
    """Compute clipped weighted binary log loss for one example."""

    clipped = min(max(prediction, 1e-12), 1.0 - 1e-12)
    if label >= 0.5:
        return -weight * log1p(-1.0 + clipped)
    return -weight * log1p(-clipped)


def _mean_recall_at_k(
    actual_by_customer: Mapping[str, Iterable[str]],
    predicted_by_customer: Mapping[str, Iterable[str]],
    k: int,
) -> float:
    """Compute mean recall@K over labeled customers."""

    scores = [
        recall_at_k(actual, predicted_by_customer.get(customer_id, ()), k=k)
        for customer_id, actual in actual_by_customer.items()
    ]
    return sum(scores) / len(scores) if scores else 0.0
