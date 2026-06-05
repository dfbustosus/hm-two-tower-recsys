"""Transparent deterministic ranker baseline for candidate-source features."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import log1p
from pathlib import Path
from typing import Any

from hm_recsys.evaluation.metrics import mean_average_precision_at_k, recall_at_k
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.candidate_export import CANDIDATE_EXPORT_HEADER, CandidateRecord
from hm_recsys.retrieval.source_names import (
    AGE_SEGMENT_POPULARITY_SOURCE,
    ALL_TIME_POPULARITY_SOURCE,
    CO_VISITATION_SOURCE,
    GARMENT_GROUP_POPULARITY_SOURCE,
    IMAGE_SIMILARITY_SOURCE,
    MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE,
    MULTIMODAL_SIMILARITY_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
    TEXT_SIMILARITY_SOURCE,
    TWO_TOWER_MULTIMODAL_SOURCE,
    TWO_TOWER_RETRIEVAL_SOURCE,
)

BASELINE_SOURCE_ORDER = (
    REPEAT_SOURCE,
    RECENT_POPULARITY_SOURCE,
    ALL_TIME_POPULARITY_SOURCE,
)
CONTENT_SIMILARITY_SOURCES = frozenset(
    {
        IMAGE_SIMILARITY_SOURCE,
        TEXT_SIMILARITY_SOURCE,
        MULTIMODAL_SIMILARITY_SOURCE,
        MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE,
        TWO_TOWER_MULTIMODAL_SOURCE,
    }
)


@dataclass(frozen=True)
class DeterministicRankerWeights:
    """Explicit weights for the deterministic candidate-source ranker.

    Attributes:
        repeat_presence_weight: Additive weight for repeat-purchase candidates.
        repeat_score_weight: Weight applied to repeat reciprocal-rank score.
        recent_popularity_presence_weight: Additive weight for recent popularity.
        recent_popularity_score_weight: Weight for recent popularity reciprocal rank.
        all_time_popularity_presence_weight: Additive weight for all-time popularity.
        all_time_popularity_score_weight: Weight for all-time popularity reciprocal rank.
        co_visitation_presence_weight: Additive weight for co-visitation candidates.
        co_visitation_score_weight: Weight for ``log1p(co_visitation_score)``.
        age_segment_popularity_presence_weight: Additive weight for age-segment
            popularity candidates.
        age_segment_popularity_score_weight: Weight for segment-popularity score.
        garment_group_popularity_presence_weight: Additive weight for
            garment-group affinity popularity candidates.
        garment_group_popularity_score_weight: Weight for garment-group score.
        content_similarity_presence_weight: Additive weight for content candidates.
        content_similarity_score_weight: Weight for content cosine/source score.
        two_tower_retrieval_presence_weight: Additive weight for two-tower retrieval.
        two_tower_retrieval_score_weight: Weight for two-tower retrieval score.
        source_count_weight: Weight for the number of sources emitting the pair.
        best_rank_score_weight: Weight for reciprocal best source rank.
    """

    repeat_presence_weight: float = 3.0
    repeat_score_weight: float = 2.0
    recent_popularity_presence_weight: float = 1.0
    recent_popularity_score_weight: float = 1.0
    all_time_popularity_presence_weight: float = 0.15
    all_time_popularity_score_weight: float = 0.15
    co_visitation_presence_weight: float = 0.35
    co_visitation_score_weight: float = 0.10
    age_segment_popularity_presence_weight: float = 0.30
    age_segment_popularity_score_weight: float = 0.20
    garment_group_popularity_presence_weight: float = 0.40
    garment_group_popularity_score_weight: float = 0.25
    content_similarity_presence_weight: float = 0.10
    content_similarity_score_weight: float = 0.05
    two_tower_retrieval_presence_weight: float = 0.10
    two_tower_retrieval_score_weight: float = 0.05
    source_count_weight: float = 0.05
    best_rank_score_weight: float = 0.05


DEFAULT_DETERMINISTIC_RANKER_WEIGHTS = DeterministicRankerWeights()


@dataclass
class CandidateFeatures:
    """Aggregated features for one ``(customer_id, article_id)`` pair.

    Attributes:
        customer_id: H&M customer identifier.
        article_id: H&M article identifier.
        label: Binary validation target label.
        repeat_rank: Optional source rank from repeat candidates.
        repeat_score: Source score from repeat candidates.
        recent_popularity_rank: Optional source rank from recent popularity.
        recent_popularity_score: Source score from recent popularity.
        all_time_popularity_rank: Optional source rank from all-time popularity.
        all_time_popularity_score: Source score from all-time popularity.
        co_visitation_rank: Optional source rank from co-visitation.
        co_visitation_score: Source score from co-visitation.
        age_segment_popularity_rank: Optional source rank from age-segment popularity.
        age_segment_popularity_score: Source score from age-segment popularity.
        garment_group_popularity_rank: Optional source rank from garment-group popularity.
        garment_group_popularity_score: Source score from garment-group popularity.
        content_similarity_rank: Optional source rank from cached content similarity.
        content_similarity_score: Source score from cached content similarity.
        two_tower_retrieval_rank: Optional source rank from two-tower retrieval.
        two_tower_retrieval_score: Source score from two-tower retrieval.
        source_count: Number of candidate sources emitting this pair.
        best_rank: Best one-based source rank across sources.
        max_source_score: Maximum raw source score across sources.
    """

    customer_id: str
    article_id: str
    label: int = 0
    repeat_rank: int | None = None
    repeat_score: float = 0.0
    recent_popularity_rank: int | None = None
    recent_popularity_score: float = 0.0
    all_time_popularity_rank: int | None = None
    all_time_popularity_score: float = 0.0
    co_visitation_rank: int | None = None
    co_visitation_score: float = 0.0
    age_segment_popularity_rank: int | None = None
    age_segment_popularity_score: float = 0.0
    garment_group_popularity_rank: int | None = None
    garment_group_popularity_score: float = 0.0
    content_similarity_rank: int | None = None
    content_similarity_score: float = 0.0
    two_tower_retrieval_rank: int | None = None
    two_tower_retrieval_score: float = 0.0
    source_count: int = 0
    best_rank: int | None = None
    max_source_score: float = 0.0

    def update_from_record(self, record: CandidateRecord) -> None:
        """Update feature fields from one source-specific candidate record.

        Args:
            record: Source-specific candidate row for the same customer/article.
        """

        self.source_count += 1
        self.best_rank = (
            record.source_rank
            if self.best_rank is None
            else min(self.best_rank, record.source_rank)
        )
        self.max_source_score = max(self.max_source_score, record.source_score)
        if record.source == REPEAT_SOURCE:
            self.repeat_rank = _min_optional_rank(self.repeat_rank, record.source_rank)
            self.repeat_score = max(self.repeat_score, record.source_score)
        elif record.source == RECENT_POPULARITY_SOURCE:
            self.recent_popularity_rank = _min_optional_rank(
                self.recent_popularity_rank, record.source_rank
            )
            self.recent_popularity_score = max(self.recent_popularity_score, record.source_score)
        elif record.source == ALL_TIME_POPULARITY_SOURCE:
            self.all_time_popularity_rank = _min_optional_rank(
                self.all_time_popularity_rank, record.source_rank
            )
            self.all_time_popularity_score = max(
                self.all_time_popularity_score, record.source_score
            )
        elif record.source == CO_VISITATION_SOURCE:
            self.co_visitation_rank = _min_optional_rank(
                self.co_visitation_rank, record.source_rank
            )
            self.co_visitation_score = max(self.co_visitation_score, record.source_score)
        elif record.source == AGE_SEGMENT_POPULARITY_SOURCE:
            self.age_segment_popularity_rank = _min_optional_rank(
                self.age_segment_popularity_rank,
                record.source_rank,
            )
            self.age_segment_popularity_score = max(
                self.age_segment_popularity_score,
                record.source_score,
            )
        elif record.source == GARMENT_GROUP_POPULARITY_SOURCE:
            self.garment_group_popularity_rank = _min_optional_rank(
                self.garment_group_popularity_rank,
                record.source_rank,
            )
            self.garment_group_popularity_score = max(
                self.garment_group_popularity_score,
                record.source_score,
            )
        elif record.source == TWO_TOWER_RETRIEVAL_SOURCE:
            self.two_tower_retrieval_rank = _min_optional_rank(
                self.two_tower_retrieval_rank,
                record.source_rank,
            )
            self.two_tower_retrieval_score = max(
                self.two_tower_retrieval_score,
                record.source_score,
            )
        elif record.source in CONTENT_SIMILARITY_SOURCES:
            self.content_similarity_rank = _min_optional_rank(
                self.content_similarity_rank, record.source_rank
            )
            self.content_similarity_score = max(self.content_similarity_score, record.source_score)

    @property
    def has_repeat(self) -> bool:
        """Return whether repeat retrieval emitted this pair."""

        return self.repeat_rank is not None

    @property
    def has_recent_popularity(self) -> bool:
        """Return whether recent popularity emitted this pair."""

        return self.recent_popularity_rank is not None

    @property
    def has_all_time_popularity(self) -> bool:
        """Return whether all-time popularity emitted this pair."""

        return self.all_time_popularity_rank is not None

    @property
    def has_co_visitation(self) -> bool:
        """Return whether co-visitation emitted this pair."""

        return self.co_visitation_rank is not None

    @property
    def has_age_segment_popularity(self) -> bool:
        """Return whether age-segment popularity emitted this pair."""

        return self.age_segment_popularity_rank is not None

    @property
    def has_garment_group_popularity(self) -> bool:
        """Return whether garment-group affinity popularity emitted this pair."""

        return self.garment_group_popularity_rank is not None

    @property
    def has_content_similarity(self) -> bool:
        """Return whether a content-similarity source emitted this pair."""

        return self.content_similarity_rank is not None

    @property
    def has_two_tower_retrieval(self) -> bool:
        """Return whether two-tower retrieval emitted this pair."""

        return self.two_tower_retrieval_rank is not None


@dataclass(frozen=True)
class DeterministicRankerReport:
    """Evaluation report for the deterministic ranker baseline.

    Attributes:
        generated_at_utc: UTC timestamp for the evaluation.
        cutoff: Validation cutoff date.
        validation_end_exclusive: Exclusive validation-window end date.
        horizon_days: Validation horizon in days.
        k: MAP/recommendation depth.
        candidate_path: Candidate CSV path consumed.
        candidate_rows: Source-specific candidate rows read.
        unique_candidate_pairs: Unique ``(customer_id, article_id)`` pairs.
        evaluated_customers: Customers with validation labels in the candidate file.
        map_at_k: Deterministic ranker MAP@K.
        recall_at_k: Deterministic ranker recall@K.
        baseline_map_at_k: Same-scope repeat→popularity source-order MAP@K.
        baseline_recall_at_k: Same-scope repeat→popularity recall@K.
        delta_map_at_k: Ranker MAP@K minus same-scope baseline MAP@K.
        duplicate_prediction_rows: Ranked rows containing duplicate article IDs.
        average_candidates_per_customer: Mean unique candidate pairs per customer.
        source_row_counts: Source-specific row counts in the candidate CSV.
        weights: Explicit deterministic ranker weights.
    """

    generated_at_utc: str
    cutoff: str
    validation_end_exclusive: str
    horizon_days: int
    k: int
    candidate_path: str
    candidate_rows: int
    unique_candidate_pairs: int
    evaluated_customers: int
    map_at_k: float
    recall_at_k: float
    baseline_map_at_k: float
    baseline_recall_at_k: float
    delta_map_at_k: float
    duplicate_prediction_rows: int
    average_candidates_per_customer: float
    source_row_counts: dict[str, int]
    weights: DeterministicRankerWeights


def iter_candidate_records_from_csv(path: Path | str) -> Iterable[CandidateRecord]:
    """Stream candidate records from a ranker-ready candidate CSV.

    Args:
        path: Candidate CSV path with ``CANDIDATE_EXPORT_HEADER``.

    Yields:
        Parsed candidate records.

    Raises:
        ValueError: If the header or typed fields are invalid.
    """

    candidate_path = Path(path).expanduser().resolve()
    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANDIDATE_EXPORT_HEADER:
            raise ValueError(
                f"candidate CSV header must be exactly {','.join(CANDIDATE_EXPORT_HEADER)}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                yield CandidateRecord(
                    customer_id=row["customer_id"],
                    article_id=row["article_id"],
                    source=row["source"],
                    source_rank=int(row["source_rank"]),
                    source_score=float(row["source_score"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: invalid candidate row") from exc


def aggregate_candidate_features(
    records: Iterable[CandidateRecord],
    validation_labels: Mapping[str, Iterable[str]],
) -> dict[str, dict[str, CandidateFeatures]]:
    """Aggregate source rows into one feature record per customer/article pair.

    Args:
        records: Source-specific candidate records.
        validation_labels: Validation labels keyed by customer ID.

    Returns:
        Nested mapping ``customer_id -> article_id -> CandidateFeatures``.
    """

    label_sets = {
        customer_id: set(article_ids) for customer_id, article_ids in validation_labels.items()
    }
    features_by_customer: dict[str, dict[str, CandidateFeatures]] = defaultdict(dict)
    for record in records:
        customer_features = features_by_customer[record.customer_id]
        features = customer_features.get(record.article_id)
        if features is None:
            features = CandidateFeatures(
                customer_id=record.customer_id,
                article_id=record.article_id,
                label=int(record.article_id in label_sets.get(record.customer_id, set())),
            )
            customer_features[record.article_id] = features
        features.update_from_record(record)
    return {
        customer_id: dict(article_features)
        for customer_id, article_features in features_by_customer.items()
    }


def score_candidate(
    features: CandidateFeatures,
    weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
) -> float:
    """Score one candidate feature row with transparent deterministic weights.

    Args:
        features: Aggregated candidate-source features.
        weights: Explicit deterministic ranker weights.

    Returns:
        Numeric score; higher values rank earlier.
    """

    score = weights.source_count_weight * features.source_count
    if features.best_rank is not None:
        score += weights.best_rank_score_weight / features.best_rank
    if features.has_repeat:
        score += weights.repeat_presence_weight
        score += weights.repeat_score_weight * features.repeat_score
    if features.has_recent_popularity:
        score += weights.recent_popularity_presence_weight
        score += weights.recent_popularity_score_weight * features.recent_popularity_score
    if features.has_all_time_popularity:
        score += weights.all_time_popularity_presence_weight
        score += weights.all_time_popularity_score_weight * features.all_time_popularity_score
    if features.has_co_visitation:
        score += weights.co_visitation_presence_weight
        score += weights.co_visitation_score_weight * log1p(features.co_visitation_score)
    if features.has_age_segment_popularity:
        score += weights.age_segment_popularity_presence_weight
        score += weights.age_segment_popularity_score_weight * features.age_segment_popularity_score
    if features.has_garment_group_popularity:
        score += weights.garment_group_popularity_presence_weight
        score += (
            weights.garment_group_popularity_score_weight * features.garment_group_popularity_score
        )
    if features.has_content_similarity:
        score += weights.content_similarity_presence_weight
        score += weights.content_similarity_score_weight * features.content_similarity_score
    if features.has_two_tower_retrieval:
        score += weights.two_tower_retrieval_presence_weight
        score += weights.two_tower_retrieval_score_weight * features.two_tower_retrieval_score
    return score


def rank_candidates_by_customer(
    features_by_customer: Mapping[str, Mapping[str, CandidateFeatures]],
    k: int,
    weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
) -> dict[str, tuple[str, ...]]:
    """Rank candidates per customer with deterministic score and tie-breaks.

    Args:
        features_by_customer: Aggregated candidate features by customer.
        k: Maximum recommendations per customer.
        weights: Explicit deterministic ranker weights.

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
                -score_candidate(features, weights),
                -(features.source_count),
                features.best_rank if features.best_rank is not None else 10**9,
                features.article_id,
            ),
        )
        predictions[customer_id] = tuple(features.article_id for features in ranked_features[:k])
    return predictions


def build_source_order_baseline_predictions(
    features_by_customer: Mapping[str, Mapping[str, CandidateFeatures]],
    k: int,
    source_order: tuple[str, ...] = BASELINE_SOURCE_ORDER,
) -> dict[str, tuple[str, ...]]:
    """Reconstruct an ordered source-blend baseline from aggregated features.

    Args:
        features_by_customer: Aggregated candidate features by customer.
        k: Maximum recommendations per customer.
        source_order: Source priority order used for deterministic blending.

    Returns:
        Ranked article IDs per customer using source order and source ranks.
    """

    predictions: dict[str, tuple[str, ...]] = {}
    for customer_id, article_features in features_by_customer.items():
        selected: list[str] = []
        seen: set[str] = set()
        for source in source_order:
            ranked_for_source = sorted(
                (
                    features
                    for features in article_features.values()
                    if _source_rank(features, source) is not None
                ),
                key=lambda features: (_source_rank(features, source) or 10**9, features.article_id),
            )
            for features in ranked_for_source:
                if features.article_id in seen:
                    continue
                selected.append(features.article_id)
                seen.add(features.article_id)
                if len(selected) == k:
                    break
            if len(selected) == k:
                break
        predictions[customer_id] = tuple(selected)
    return predictions


def evaluate_deterministic_ranker_from_csv(
    candidate_path: Path | str,
    validation_labels: Mapping[str, Iterable[str]],
    split: TemporalSplit,
    k: int = 12,
    weights: DeterministicRankerWeights = DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
) -> DeterministicRankerReport:
    """Evaluate deterministic ranker predictions from a candidate CSV.

    Args:
        candidate_path: Ranker-ready candidate CSV path.
        validation_labels: Validation labels keyed by customer ID.
        split: Temporal split used to create candidate rows and labels.
        k: MAP/recommendation depth.
        weights: Explicit deterministic ranker weights.

    Returns:
        Deterministic ranker evaluation report.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    resolved_candidate_path = Path(candidate_path).expanduser().resolve()
    source_row_counts: Counter[str] = Counter()

    def counting_records() -> Iterable[CandidateRecord]:
        """Yield candidate records while counting source rows for reporting."""

        for record in iter_candidate_records_from_csv(resolved_candidate_path):
            source_row_counts[record.source] += 1
            yield record

    features_by_customer = aggregate_candidate_features(counting_records(), validation_labels)
    labels_for_candidate_customers = {
        customer_id: tuple(validation_labels[customer_id])
        for customer_id in features_by_customer
        if customer_id in validation_labels
    }
    ranker_predictions = rank_candidates_by_customer(features_by_customer, k=k, weights=weights)
    baseline_predictions = build_source_order_baseline_predictions(features_by_customer, k=k)
    ranker_map = mean_average_precision_at_k(
        labels_for_candidate_customers, ranker_predictions, k=k
    )
    baseline_map = mean_average_precision_at_k(
        labels_for_candidate_customers, baseline_predictions, k=k
    )
    ranker_recall = _mean_recall_at_k(labels_for_candidate_customers, ranker_predictions, k=k)
    baseline_recall = _mean_recall_at_k(labels_for_candidate_customers, baseline_predictions, k=k)
    candidate_counts = [len(article_features) for article_features in features_by_customer.values()]
    duplicate_prediction_rows = sum(
        1
        for predictions in ranker_predictions.values()
        if len(set(predictions)) != len(predictions)
    )
    return DeterministicRankerReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoff=split.cutoff.isoformat(),
        validation_end_exclusive=split.validation_end.isoformat(),
        horizon_days=split.horizon_days,
        k=k,
        candidate_path=str(resolved_candidate_path),
        candidate_rows=sum(source_row_counts.values()),
        unique_candidate_pairs=sum(candidate_counts),
        evaluated_customers=len(labels_for_candidate_customers),
        map_at_k=ranker_map,
        recall_at_k=ranker_recall,
        baseline_map_at_k=baseline_map,
        baseline_recall_at_k=baseline_recall,
        delta_map_at_k=ranker_map - baseline_map,
        duplicate_prediction_rows=duplicate_prediction_rows,
        average_candidates_per_customer=(
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
        ),
        source_row_counts=dict(sorted(source_row_counts.items())),
        weights=weights,
    )


def deterministic_ranker_report_to_dict(report: DeterministicRankerReport) -> dict[str, Any]:
    """Convert a deterministic ranker report to JSON-serializable primitives.

    Args:
        report: Report object to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """

    return asdict(report)


def write_deterministic_ranker_report(report: DeterministicRankerReport, path: Path | str) -> Path:
    """Write a deterministic ranker report as deterministic JSON.

    Args:
        report: Evaluation report to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(deterministic_ranker_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _source_rank(features: CandidateFeatures, source: str) -> int | None:
    """Return the rank field associated with a retrieval source name."""

    if source == REPEAT_SOURCE:
        return features.repeat_rank
    if source == RECENT_POPULARITY_SOURCE:
        return features.recent_popularity_rank
    if source == ALL_TIME_POPULARITY_SOURCE:
        return features.all_time_popularity_rank
    if source == CO_VISITATION_SOURCE:
        return features.co_visitation_rank
    if source == AGE_SEGMENT_POPULARITY_SOURCE:
        return features.age_segment_popularity_rank
    if source == GARMENT_GROUP_POPULARITY_SOURCE:
        return features.garment_group_popularity_rank
    return None


def _min_optional_rank(current: int | None, candidate: int) -> int:
    """Return the minimum rank while supporting an initially missing value."""

    return candidate if current is None else min(current, candidate)


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
