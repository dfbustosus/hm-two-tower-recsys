"""Final-data learned linear ranker submission generation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.submission import SubmissionValidationResult
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.deterministic import (
    DeterministicRankerWeights,
    aggregate_candidate_features,
    rank_candidates_by_customer,
)
from hm_recsys.ranking.deterministic_tuning import DeterministicRankerWeightSelection
from hm_recsys.ranking.linear import (
    LinearRankerConfig,
    LinearRankerModel,
    LinearRankerTrainingSummary,
    rank_with_linear_model,
)
from hm_recsys.retrieval.baselines import (
    build_repeat_popularity_candidate_sources,
    merge_ranked_sources,
)
from hm_recsys.retrieval.candidate_export import iter_candidate_records_for_customer
from hm_recsys.retrieval.co_visitation import (
    DEFAULT_MAX_HISTORY_ITEMS,
    DEFAULT_MAX_NEIGHBORS_PER_ITEM,
    build_co_visitation_index,
)
from hm_recsys.retrieval.metadata_affinity import (
    GARMENT_GROUP_COLUMN,
    build_article_attribute_popularity_index,
)
from hm_recsys.retrieval.segment_popularity import (
    DEFAULT_AGE_SEGMENT_BUCKET_SIZE,
    build_age_segment_popularity_index,
)
from hm_recsys.retrieval.source_names import TWO_TOWER_RETRIEVAL_SOURCE
from hm_recsys.training.two_tower_retrieval import TwoTowerSmokeModel


@dataclass(frozen=True)
class LinearRankerSubmissionDiagnostics:
    """Shape and source diagnostics for learned-ranker final predictions.

    Attributes:
        target_customers: Number of customers for whom predictions were generated.
        customers_with_full_length_predictions: Customers with exactly ``k`` predictions.
        prediction_coverage: Share of target customers with exactly ``k`` predictions.
        duplicate_prediction_rows: Rows containing duplicate article IDs.
        average_prediction_count: Mean prediction count per customer.
        predicted_article_coverage: Number of unique predicted articles.
        unique_candidate_pairs: Unique customer/article candidate pairs scored.
        source_row_counts: Candidate source rows emitted during final scoring.
    """

    target_customers: int
    customers_with_full_length_predictions: int
    prediction_coverage: float
    duplicate_prediction_rows: int
    average_prediction_count: float
    predicted_article_coverage: int
    unique_candidate_pairs: int
    source_row_counts: dict[str, int]


@dataclass(frozen=True)
class LinearRankerSubmissionPredictions:
    """Final-data predictions produced by a learned linear ranker.

    Attributes:
        predictions: Ranked article IDs keyed by customer ID.
        final_training_cutoff: Exclusive cutoff used for final-data candidate features.
        max_transaction_date: Latest observed transaction date, if known by caller.
        runtime_seconds: Wall-clock runtime for final prediction generation.
        diagnostics: Prediction shape and candidate-source diagnostics.
    """

    predictions: dict[str, tuple[str, ...]]
    final_training_cutoff: str
    max_transaction_date: str | None
    runtime_seconds: float
    diagnostics: LinearRankerSubmissionDiagnostics


@dataclass(frozen=True)
class LearnedLinearRankerSubmissionReport:
    """Experiment report for a generated learned-ranker submission.

    Attributes:
        generated_at_utc: UTC timestamp for the report.
        train_cutoff: Cutoff date for the supervised label window used to train.
        train_validation_end_exclusive: Exclusive end of the supervised training label window.
        final_training_cutoff: Exclusive cutoff used for final-data candidate features.
        max_transaction_date: Latest observed transaction date from the raw training file.
        horizon_days: Label horizon used for the supervised training window.
        k: Recommendation depth and submission row length.
        candidate_k: Maximum candidates per source.
        popularity_lookback_days: Recent popularity lookback length.
        include_co_visitation: Whether co-visitation source rows were used.
        co_visitation_max_history_items: Co-visitation customer-history length.
        co_visitation_max_neighbors_per_item: Co-visitation neighbor cap per item.
        config: Learned linear ranker training configuration.
        model: Learned linear ranker weights used for final scoring.
        training: Training summary for the latest supervised label window.
        submission: Final prediction diagnostics.
        submission_path: Generated CSV path under ``submissions/``.
        validation_report_path: Submission-validation JSON report path.
        validation: Submission-validation result.
    """

    generated_at_utc: str
    train_cutoff: str
    train_validation_end_exclusive: str
    final_training_cutoff: str
    max_transaction_date: str | None
    horizon_days: int
    k: int
    candidate_k: int
    popularity_lookback_days: int
    include_co_visitation: bool
    co_visitation_max_history_items: int
    co_visitation_max_neighbors_per_item: int
    config: LinearRankerConfig
    model: LinearRankerModel
    training: LinearRankerTrainingSummary
    submission: LinearRankerSubmissionDiagnostics
    submission_path: str
    validation_report_path: str
    validation: SubmissionValidationResult


@dataclass(frozen=True)
class DeterministicRankerSubmissionPredictions:
    """Final-data predictions produced by deterministic ranker weights.

    Attributes mirror ``LinearRankerSubmissionPredictions`` but store the explicit
    deterministic weights used for scoring.
    """

    predictions: dict[str, tuple[str, ...]]
    final_training_cutoff: str
    max_transaction_date: str | None
    runtime_seconds: float
    weights: DeterministicRankerWeights
    diagnostics: LinearRankerSubmissionDiagnostics


@dataclass(frozen=True)
class DeterministicRankerSubmissionReport:
    """Experiment report for a tuned deterministic-ranker submission.

    Attributes:
        generated_at_utc: UTC timestamp for the report.
        tuning_cutoff: Cutoff date for the label window used to select weights.
        tuning_validation_end_exclusive: Exclusive end of the tuning label window.
        final_training_cutoff: Exclusive cutoff used for final-data candidate features.
        max_transaction_date: Latest observed transaction date from the raw training file.
        horizon_days: Label horizon used for the tuning window.
        k: Recommendation depth and submission row length.
        candidate_k: Maximum candidates per source.
        popularity_lookback_days: Recent popularity lookback length.
        include_co_visitation: Whether co-visitation source rows were used.
        co_visitation_max_history_items: Co-visitation customer-history length.
        co_visitation_max_neighbors_per_item: Co-visitation neighbor cap per item.
        include_age_segment_popularity: Whether age-segment source rows were used.
        age_segment_bucket_size: Age bucket width when age source is enabled.
        age_segment_popularity_lookback_days: Age-segment source lookback.
        include_garment_group_popularity: Whether garment-group source rows were used.
        garment_group_popularity_lookback_days: Garment-group source lookback.
        garment_group_max_history_items: History length used for garment affinities.
        include_two_tower_retrieval: Whether two-tower retrieval rows were used.
        two_tower_source_name: Source label used for two-tower retrieval rows.
        two_tower_max_retrieval_articles: Exact-scoring article-pool cap for two-tower retrieval.
        weight_selection: Leakage-safe tuning-window weight selection diagnostics.
        weights: Selected deterministic weights used for final scoring.
        submission: Final prediction diagnostics.
        submission_path: Generated CSV path under ``submissions/``.
        validation_report_path: Submission-validation JSON report path.
        validation: Submission-validation result.
    """

    generated_at_utc: str
    tuning_cutoff: str
    tuning_validation_end_exclusive: str
    final_training_cutoff: str
    max_transaction_date: str | None
    horizon_days: int
    k: int
    candidate_k: int
    popularity_lookback_days: int
    include_co_visitation: bool
    co_visitation_max_history_items: int
    co_visitation_max_neighbors_per_item: int
    include_age_segment_popularity: bool
    age_segment_bucket_size: int | None
    age_segment_popularity_lookback_days: int | None
    include_garment_group_popularity: bool
    garment_group_popularity_lookback_days: int | None
    garment_group_max_history_items: int | None
    include_two_tower_retrieval: bool
    two_tower_source_name: str | None
    two_tower_max_retrieval_articles: int | None
    weight_selection: DeterministicRankerWeightSelection
    weights: DeterministicRankerWeights
    submission: LinearRankerSubmissionDiagnostics
    submission_path: str
    validation_report_path: str
    validation: SubmissionValidationResult


def build_linear_ranker_submission_predictions(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    split: TemporalSplit,
    target_customer_ids: Iterable[str],
    model: LinearRankerModel,
    k: int = 12,
    candidate_k: int = 12,
    popularity_lookback_days: int = 7,
    include_co_visitation: bool = True,
    co_visitation_max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    co_visitation_max_neighbors_per_item: int = DEFAULT_MAX_NEIGHBORS_PER_ITEM,
    max_transaction_date: date | None = None,
    transaction_progress_interval: int | None = None,
    transaction_progress_callback: Callable[[str, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    progress_interval: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> LinearRankerSubmissionPredictions:
    """Generate final-data predictions by scoring source candidates per customer.

    The supplied ``split.cutoff`` is the exclusive feature cutoff. For H&M final
    submissions this should be one day after the last transaction date, so all
    official training rows are usable and no hidden target rows are touched.

    Args:
        transaction_iter_factory: Callable returning a fresh transaction iterable
            for each full pass over the raw transactions.
        split: Final-data split whose cutoff is the exclusive feature boundary.
        target_customer_ids: Ordered ``sample_submission.csv`` customer IDs.
        model: Learned linear ranker model to score candidate features.
        k: Final recommendation depth.
        candidate_k: Maximum candidates emitted per source.
        popularity_lookback_days: Recent popularity lookback length.
        include_co_visitation: Whether to include co-visitation source rows.
        co_visitation_max_history_items: Co-visitation customer-history length.
        co_visitation_max_neighbors_per_item: Co-visitation neighbor cap per item.
        max_transaction_date: Optional latest observed transaction date for reports.
        transaction_progress_interval: Optional transaction-row interval for source
            builder progress callbacks.
        transaction_progress_callback: Optional callback receiving a phase name and
            scanned transaction rows.
        status_callback: Optional callback receiving coarse source-building status.
        progress_interval: Optional customer interval for progress callbacks.
        progress_callback: Optional callback receiving completed and total customers.

    Returns:
        Final predictions plus diagnostics.

    Raises:
        ValueError: If numeric limits are invalid.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if popularity_lookback_days <= 0:
        raise ValueError("popularity_lookback_days must be positive")
    if transaction_progress_interval is not None and transaction_progress_interval <= 0:
        raise ValueError("transaction_progress_interval must be positive when provided")
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError("progress_interval must be positive when provided")

    started_at = perf_counter()
    target_customer_tuple = tuple(target_customer_ids)
    if status_callback is not None:
        status_callback("building repeat/popularity candidate sources")
    baseline_sources = build_repeat_popularity_candidate_sources(
        transactions=transaction_iter_factory(),
        split=split,
        target_customer_ids=target_customer_tuple,
        k=candidate_k,
        popularity_lookback_days=popularity_lookback_days,
        progress_interval=transaction_progress_interval,
        progress_callback=(
            None
            if transaction_progress_callback is None
            else lambda rows: transaction_progress_callback("repeat_popularity", rows)
        ),
    )
    if status_callback is not None:
        status_callback("repeat/popularity candidate sources ready")
        if include_co_visitation:
            status_callback("building co-visitation index")
    co_visitation_index = (
        build_co_visitation_index(
            transactions=transaction_iter_factory(),
            split=split,
            target_customer_ids=target_customer_tuple,
            max_history_items=co_visitation_max_history_items,
            max_neighbors_per_item=co_visitation_max_neighbors_per_item,
            progress_interval=transaction_progress_interval,
            progress_callback=(
                None
                if transaction_progress_callback is None
                else lambda rows: transaction_progress_callback("co_visitation", rows)
            ),
            status_callback=status_callback,
        )
        if include_co_visitation
        else None
    )
    if status_callback is not None and include_co_visitation:
        status_callback("co-visitation index ready")
    popularity_backfill = merge_ranked_sources(
        (baseline_sources.recent_popularity, baseline_sources.all_time_popularity),
        k=k,
    )

    predictions: dict[str, tuple[str, ...]] = {}
    source_row_counts: Counter[str] = Counter()
    unique_candidate_pairs = 0
    total_customers = len(target_customer_tuple)
    for customer_index, customer_id in enumerate(target_customer_tuple, start=1):
        records = tuple(
            iter_candidate_records_for_customer(
                customer_id=customer_id,
                repeat_recommendations=baseline_sources.repeat_recommendations,
                recent_popularity=baseline_sources.recent_popularity,
                all_time_popularity=baseline_sources.all_time_popularity,
                co_visitation_index=co_visitation_index,
                k=candidate_k,
            )
        )
        source_row_counts.update(record.source for record in records)
        unique_candidate_pairs += len({record.article_id for record in records})
        features_by_customer = aggregate_candidate_features(records, validation_labels={})
        ranked = rank_with_linear_model(features_by_customer, model=model, k=k).get(
            customer_id,
            (),
        )
        predictions[customer_id] = merge_ranked_sources((ranked, popularity_backfill), k=k)
        if (
            progress_callback is not None
            and progress_interval is not None
            and (customer_index % progress_interval == 0 or customer_index == total_customers)
        ):
            progress_callback(customer_index, total_customers)

    diagnostics = build_linear_ranker_submission_diagnostics(
        predictions=predictions,
        k=k,
        unique_candidate_pairs=unique_candidate_pairs,
        source_row_counts=dict(sorted(source_row_counts.items())),
    )
    return LinearRankerSubmissionPredictions(
        predictions=predictions,
        final_training_cutoff=split.cutoff.isoformat(),
        max_transaction_date=(max_transaction_date.isoformat() if max_transaction_date else None),
        runtime_seconds=perf_counter() - started_at,
        diagnostics=diagnostics,
    )


def build_deterministic_ranker_submission_predictions(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    split: TemporalSplit,
    target_customer_ids: Iterable[str],
    weights: DeterministicRankerWeights,
    k: int = 12,
    candidate_k: int = 12,
    popularity_lookback_days: int = 7,
    include_co_visitation: bool = True,
    co_visitation_max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    co_visitation_max_neighbors_per_item: int = DEFAULT_MAX_NEIGHBORS_PER_ITEM,
    include_age_segment_popularity: bool = False,
    customer_segment_by_id: dict[str, str] | None = None,
    age_segment_bucket_size: int = DEFAULT_AGE_SEGMENT_BUCKET_SIZE,
    age_segment_popularity_lookback_days: int | None = None,
    include_garment_group_popularity: bool = False,
    article_garment_group_by_id: dict[str, str] | None = None,
    garment_group_popularity_lookback_days: int | None = None,
    garment_group_max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    two_tower_model: TwoTowerSmokeModel | None = None,
    two_tower_source_name: str = TWO_TOWER_RETRIEVAL_SOURCE,
    two_tower_max_retrieval_articles: int | None = 5000,
    max_transaction_date: date | None = None,
    transaction_progress_interval: int | None = None,
    transaction_progress_callback: Callable[[str, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    progress_interval: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DeterministicRankerSubmissionPredictions:
    """Generate final-data predictions with explicit deterministic ranker weights.

    Args mirror ``build_linear_ranker_submission_predictions`` with optional
    age-segment and garment-group candidate sources added for the deterministic
    champion path.

    Raises:
        ValueError: If numeric limits or required metadata mappings are invalid.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if popularity_lookback_days <= 0:
        raise ValueError("popularity_lookback_days must be positive")
    if transaction_progress_interval is not None and transaction_progress_interval <= 0:
        raise ValueError("transaction_progress_interval must be positive when provided")
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError("progress_interval must be positive when provided")
    if include_age_segment_popularity and customer_segment_by_id is None:
        raise ValueError("customer_segment_by_id is required for age-segment popularity")
    if age_segment_bucket_size <= 0:
        raise ValueError("age_segment_bucket_size must be positive")
    resolved_age_segment_lookback_days = (
        age_segment_popularity_lookback_days
        if age_segment_popularity_lookback_days is not None
        else popularity_lookback_days
    )
    if resolved_age_segment_lookback_days <= 0:
        raise ValueError("age_segment_popularity_lookback_days must be positive")
    if include_garment_group_popularity and article_garment_group_by_id is None:
        raise ValueError("article_garment_group_by_id is required for garment-group popularity")
    if garment_group_max_history_items <= 0:
        raise ValueError("garment_group_max_history_items must be positive")
    resolved_garment_group_lookback_days = (
        garment_group_popularity_lookback_days
        if garment_group_popularity_lookback_days is not None
        else popularity_lookback_days
    )
    if resolved_garment_group_lookback_days <= 0:
        raise ValueError("garment_group_popularity_lookback_days must be positive")
    if two_tower_model is not None and not two_tower_source_name:
        raise ValueError("two_tower_source_name must not be empty")
    if two_tower_max_retrieval_articles is not None and two_tower_max_retrieval_articles <= 0:
        raise ValueError("two_tower_max_retrieval_articles must be positive when provided")

    started_at = perf_counter()
    target_customer_tuple = tuple(target_customer_ids)
    if status_callback is not None:
        status_callback("building repeat/popularity candidate sources")
    baseline_sources = build_repeat_popularity_candidate_sources(
        transactions=transaction_iter_factory(),
        split=split,
        target_customer_ids=target_customer_tuple,
        k=candidate_k,
        popularity_lookback_days=popularity_lookback_days,
        progress_interval=transaction_progress_interval,
        progress_callback=(
            None
            if transaction_progress_callback is None
            else lambda rows: transaction_progress_callback("repeat_popularity", rows)
        ),
    )
    if status_callback is not None:
        status_callback("repeat/popularity candidate sources ready")
        if include_co_visitation:
            status_callback("building co-visitation index")
    co_visitation_index = (
        build_co_visitation_index(
            transactions=transaction_iter_factory(),
            split=split,
            target_customer_ids=target_customer_tuple,
            max_history_items=co_visitation_max_history_items,
            max_neighbors_per_item=co_visitation_max_neighbors_per_item,
            progress_interval=transaction_progress_interval,
            progress_callback=(
                None
                if transaction_progress_callback is None
                else lambda rows: transaction_progress_callback("co_visitation", rows)
            ),
            status_callback=status_callback,
        )
        if include_co_visitation
        else None
    )
    if status_callback is not None and include_co_visitation:
        status_callback("co-visitation index ready")
    age_segment_index = (
        build_age_segment_popularity_index(
            transactions=transaction_iter_factory(),
            split=split,
            customer_segment_by_id=customer_segment_by_id or {},
            lookback_days=resolved_age_segment_lookback_days,
            max_articles_per_segment=candidate_k,
        )
        if include_age_segment_popularity
        else None
    )
    garment_group_index = (
        build_article_attribute_popularity_index(
            transactions=transaction_iter_factory(),
            split=split,
            target_customer_ids=target_customer_tuple,
            article_attribute_by_id=article_garment_group_by_id or {},
            attribute_name=GARMENT_GROUP_COLUMN,
            lookback_days=resolved_garment_group_lookback_days,
            max_history_items=garment_group_max_history_items,
            max_articles_per_attribute=candidate_k,
        )
        if include_garment_group_popularity
        else None
    )
    popularity_backfill = merge_ranked_sources(
        (baseline_sources.recent_popularity, baseline_sources.all_time_popularity),
        k=k,
    )

    predictions: dict[str, tuple[str, ...]] = {}
    source_row_counts: Counter[str] = Counter()
    unique_candidate_pairs = 0
    total_customers = len(target_customer_tuple)
    for customer_index, customer_id in enumerate(target_customer_tuple, start=1):
        records = tuple(
            iter_candidate_records_for_customer(
                customer_id=customer_id,
                repeat_recommendations=baseline_sources.repeat_recommendations,
                recent_popularity=baseline_sources.recent_popularity,
                all_time_popularity=baseline_sources.all_time_popularity,
                co_visitation_index=co_visitation_index,
                age_segment_index=age_segment_index,
                garment_group_index=garment_group_index,
                two_tower_model=two_tower_model,
                two_tower_source_name=two_tower_source_name,
                two_tower_max_retrieval_articles=two_tower_max_retrieval_articles,
                k=candidate_k,
            )
        )
        source_row_counts.update(record.source for record in records)
        unique_candidate_pairs += len({record.article_id for record in records})
        features_by_customer = aggregate_candidate_features(records, validation_labels={})
        ranked = rank_candidates_by_customer(features_by_customer, k=k, weights=weights).get(
            customer_id,
            (),
        )
        predictions[customer_id] = merge_ranked_sources((ranked, popularity_backfill), k=k)
        if (
            progress_callback is not None
            and progress_interval is not None
            and (customer_index % progress_interval == 0 or customer_index == total_customers)
        ):
            progress_callback(customer_index, total_customers)

    diagnostics = build_linear_ranker_submission_diagnostics(
        predictions=predictions,
        k=k,
        unique_candidate_pairs=unique_candidate_pairs,
        source_row_counts=dict(sorted(source_row_counts.items())),
    )
    return DeterministicRankerSubmissionPredictions(
        predictions=predictions,
        final_training_cutoff=split.cutoff.isoformat(),
        max_transaction_date=(max_transaction_date.isoformat() if max_transaction_date else None),
        runtime_seconds=perf_counter() - started_at,
        weights=weights,
        diagnostics=diagnostics,
    )


def build_linear_ranker_submission_diagnostics(
    predictions: dict[str, tuple[str, ...]],
    k: int,
    unique_candidate_pairs: int,
    source_row_counts: dict[str, int],
) -> LinearRankerSubmissionDiagnostics:
    """Build prediction-shape diagnostics for learned-ranker submissions.

    Args:
        predictions: Ranked article IDs keyed by customer ID.
        k: Expected recommendation depth.
        unique_candidate_pairs: Unique customer/article pairs scored.
        source_row_counts: Candidate source rows emitted during scoring.

    Returns:
        Aggregate diagnostics for final prediction rows.
    """

    prediction_lengths = [len(article_ids) for article_ids in predictions.values()]
    target_customers = len(predictions)
    full_length_rows = sum(1 for length in prediction_lengths if length == k)
    duplicate_prediction_rows = sum(
        1 for article_ids in predictions.values() if len(set(article_ids)) != len(article_ids)
    )
    predicted_article_coverage = len(
        {article_id for article_ids in predictions.values() for article_id in article_ids}
    )
    return LinearRankerSubmissionDiagnostics(
        target_customers=target_customers,
        customers_with_full_length_predictions=full_length_rows,
        prediction_coverage=full_length_rows / target_customers if target_customers else 0.0,
        duplicate_prediction_rows=duplicate_prediction_rows,
        average_prediction_count=(
            sum(prediction_lengths) / target_customers if target_customers else 0.0
        ),
        predicted_article_coverage=predicted_article_coverage,
        unique_candidate_pairs=unique_candidate_pairs,
        source_row_counts=source_row_counts,
    )


def build_learned_linear_ranker_submission_report(
    train_split: TemporalSplit,
    final_split: TemporalSplit,
    k: int,
    candidate_k: int,
    popularity_lookback_days: int,
    include_co_visitation: bool,
    co_visitation_max_history_items: int,
    co_visitation_max_neighbors_per_item: int,
    config: LinearRankerConfig,
    model: LinearRankerModel,
    training: LinearRankerTrainingSummary,
    submission: LinearRankerSubmissionPredictions,
    submission_path: Path | str,
    validation_report_path: Path | str,
    validation: SubmissionValidationResult,
) -> LearnedLinearRankerSubmissionReport:
    """Build a reproducibility report for a learned-ranker submission.

    Args mirror ``LearnedLinearRankerSubmissionReport`` fields.

    Returns:
        Structured report suitable for JSON persistence.
    """

    return LearnedLinearRankerSubmissionReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        train_cutoff=train_split.cutoff.isoformat(),
        train_validation_end_exclusive=train_split.validation_end.isoformat(),
        final_training_cutoff=final_split.cutoff.isoformat(),
        max_transaction_date=submission.max_transaction_date,
        horizon_days=train_split.horizon_days,
        k=k,
        candidate_k=candidate_k,
        popularity_lookback_days=popularity_lookback_days,
        include_co_visitation=include_co_visitation,
        co_visitation_max_history_items=co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=co_visitation_max_neighbors_per_item,
        config=config,
        model=model,
        training=training,
        submission=submission.diagnostics,
        submission_path=str(Path(submission_path).expanduser().resolve()),
        validation_report_path=str(Path(validation_report_path).expanduser().resolve()),
        validation=validation,
    )


def build_deterministic_ranker_submission_report(
    tuning_split: TemporalSplit,
    final_split: TemporalSplit,
    k: int,
    candidate_k: int,
    popularity_lookback_days: int,
    include_co_visitation: bool,
    co_visitation_max_history_items: int,
    co_visitation_max_neighbors_per_item: int,
    include_age_segment_popularity: bool,
    age_segment_bucket_size: int | None,
    age_segment_popularity_lookback_days: int | None,
    include_garment_group_popularity: bool,
    garment_group_popularity_lookback_days: int | None,
    garment_group_max_history_items: int | None,
    include_two_tower_retrieval: bool,
    two_tower_source_name: str | None,
    two_tower_max_retrieval_articles: int | None,
    weight_selection: DeterministicRankerWeightSelection,
    submission: DeterministicRankerSubmissionPredictions,
    submission_path: Path | str,
    validation_report_path: Path | str,
    validation: SubmissionValidationResult,
) -> DeterministicRankerSubmissionReport:
    """Build a reproducibility report for deterministic-ranker submission."""

    return DeterministicRankerSubmissionReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        tuning_cutoff=tuning_split.cutoff.isoformat(),
        tuning_validation_end_exclusive=tuning_split.validation_end.isoformat(),
        final_training_cutoff=final_split.cutoff.isoformat(),
        max_transaction_date=submission.max_transaction_date,
        horizon_days=tuning_split.horizon_days,
        k=k,
        candidate_k=candidate_k,
        popularity_lookback_days=popularity_lookback_days,
        include_co_visitation=include_co_visitation,
        co_visitation_max_history_items=co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=co_visitation_max_neighbors_per_item,
        include_age_segment_popularity=include_age_segment_popularity,
        age_segment_bucket_size=age_segment_bucket_size,
        age_segment_popularity_lookback_days=age_segment_popularity_lookback_days,
        include_garment_group_popularity=include_garment_group_popularity,
        garment_group_popularity_lookback_days=garment_group_popularity_lookback_days,
        garment_group_max_history_items=garment_group_max_history_items,
        include_two_tower_retrieval=include_two_tower_retrieval,
        two_tower_source_name=two_tower_source_name if include_two_tower_retrieval else None,
        two_tower_max_retrieval_articles=(
            two_tower_max_retrieval_articles if include_two_tower_retrieval else None
        ),
        weight_selection=weight_selection,
        weights=submission.weights,
        submission=submission.diagnostics,
        submission_path=str(Path(submission_path).expanduser().resolve()),
        validation_report_path=str(Path(validation_report_path).expanduser().resolve()),
        validation=validation,
    )


def learned_linear_ranker_submission_report_to_dict(
    report: LearnedLinearRankerSubmissionReport,
) -> dict[str, Any]:
    """Convert a learned-ranker submission report to serializable primitives.

    Args:
        report: Report object to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """

    return asdict(report)


def deterministic_ranker_submission_report_to_dict(
    report: DeterministicRankerSubmissionReport,
) -> dict[str, Any]:
    """Convert a deterministic-ranker submission report to serializable primitives."""

    return asdict(report)


def write_learned_linear_ranker_submission_report(
    report: LearnedLinearRankerSubmissionReport,
    path: Path | str,
) -> Path:
    """Write a learned-ranker submission report as deterministic JSON.

    Args:
        report: Report object to serialize.
        path: Destination JSON path.

    Returns:
        Resolved report path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            learned_linear_ranker_submission_report_to_dict(report),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def write_deterministic_ranker_submission_report(
    report: DeterministicRankerSubmissionReport,
    path: Path | str,
) -> Path:
    """Write a deterministic-ranker submission report as JSON."""

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            deterministic_ranker_submission_report_to_dict(report),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path
