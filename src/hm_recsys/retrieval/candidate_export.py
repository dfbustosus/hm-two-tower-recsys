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
from hm_recsys.embeddings.cache_io import load_article_embedding_cache
from hm_recsys.embeddings.cache_manifest import read_article_embedding_cache_manifest
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
from hm_recsys.retrieval.content_similarity import (
    ContentSimilarityIndex,
    build_content_similarity_candidate_source_records,
    build_content_similarity_index,
)
from hm_recsys.retrieval.metadata_affinity import (
    GARMENT_GROUP_COLUMN,
    PRODUCT_CODE_COLUMN,
    ArticleAttributePopularityIndex,
    build_article_attribute_popularity_candidates,
    build_article_attribute_popularity_index,
)
from hm_recsys.retrieval.seasonality import (
    DEFAULT_SEASONAL_SHIFT_DAYS,
    DEFAULT_SEASONAL_WINDOW_DAYS,
    SeasonalPopularityIndex,
    build_seasonal_popularity_candidates,
    build_seasonal_popularity_index,
)
from hm_recsys.retrieval.segment_popularity import (
    DEFAULT_AGE_SEGMENT_BUCKET_SIZE,
    AgeSegmentPopularityIndex,
    build_age_segment_popularity_candidates,
    build_age_segment_popularity_index,
)
from hm_recsys.retrieval.source_names import (
    AGE_SEGMENT_POPULARITY_SOURCE,
    ALL_TIME_POPULARITY_SOURCE,
    CO_VISITATION_SOURCE,
    GARMENT_GROUP_POPULARITY_SOURCE,
    MULTIMODAL_SIMILARITY_SOURCE,
    PRODUCT_CODE_POPULARITY_SOURCE,
    RECENT_POPULARITY_1D_SOURCE,
    RECENT_POPULARITY_3D_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
    SEASONAL_POPULARITY_SOURCE,
    TWO_TOWER_RETRIEVAL_SOURCE,
)
from hm_recsys.training.two_tower_retrieval import TwoTowerSmokeModel, score_two_tower_candidates

_SHORT_TERM_POPULARITY_SOURCE_BY_LOOKBACK: dict[int, str] = {
    1: RECENT_POPULARITY_1D_SOURCE,
    3: RECENT_POPULARITY_3D_SOURCE,
}
"""Map known short-term lookbacks to their canonical source names.

Adding a new short-term window is a single-line change here; unknown lookbacks
are exported under a synthesized ``recent_popularity_<N>d`` name so the data
stays consumable even before the ranker schema catches up.
"""

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
        include_seasonal_popularity: Whether shifted-window seasonal rows were exported.
        seasonal_shift_days: Historical shift between cutoff and seasonal window start.
        seasonal_window_days: Length of the historical seasonal popularity window.
        include_age_segment_popularity: Whether age-segment popularity rows were exported.
        age_segment_bucket_size: Width of customer age buckets when segment rows are exported.
        age_segment_popularity_lookback_days: Pre-cutoff lookback used for segment counts.
        include_garment_group_popularity: Whether garment-group affinity rows were exported.
        garment_group_popularity_lookback_days: Pre-cutoff lookback used for group counts.
        garment_group_max_history_items: Customer history length used for group affinities.
        content_similarity_manifest_path: Optional cached embedding manifest path.
        content_similarity_source_name: Source name for content-similarity rows.
        content_similarity_max_history_items: Customer-history length for content queries.
        content_similarity_popularity_prior_weight: Popularity-prior blend weight.
        content_similarity_popularity_lookback_days: Optional popularity-prior lookback.
        content_similarity_candidate_pool_size: Optional content neighbor pool size.
        include_two_tower_retrieval: Whether two-tower retrieval rows were exported.
        two_tower_source_name: Source name for two-tower rows.
        two_tower_max_retrieval_articles: Article pool cap used by two-tower exact scoring.
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
    include_seasonal_popularity: bool
    seasonal_shift_days: int | None
    seasonal_window_days: int | None
    include_age_segment_popularity: bool
    age_segment_bucket_size: int | None
    age_segment_popularity_lookback_days: int | None
    include_garment_group_popularity: bool
    garment_group_popularity_lookback_days: int | None
    garment_group_max_history_items: int | None
    include_product_code_popularity: bool
    product_code_popularity_lookback_days: int | None
    product_code_max_history_items: int | None
    content_similarity_manifest_path: str | None
    content_similarity_source_name: str | None
    content_similarity_max_history_items: int | None
    content_similarity_popularity_prior_weight: float | None
    content_similarity_popularity_lookback_days: int | None
    content_similarity_candidate_pool_size: int | None
    include_two_tower_retrieval: bool
    two_tower_source_name: str | None
    two_tower_max_retrieval_articles: int | None
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
    include_seasonal_popularity: bool = False,
    seasonal_shift_days: int = DEFAULT_SEASONAL_SHIFT_DAYS,
    seasonal_window_days: int = DEFAULT_SEASONAL_WINDOW_DAYS,
    include_age_segment_popularity: bool = False,
    customer_segment_by_id: Mapping[str, str] | None = None,
    age_segment_bucket_size: int = DEFAULT_AGE_SEGMENT_BUCKET_SIZE,
    age_segment_popularity_lookback_days: int | None = None,
    include_garment_group_popularity: bool = False,
    article_garment_group_by_id: Mapping[str, str] | None = None,
    garment_group_popularity_lookback_days: int | None = None,
    garment_group_max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    include_product_code_popularity: bool = False,
    article_product_code_by_id: Mapping[str, str] | None = None,
    product_code_popularity_lookback_days: int | None = None,
    product_code_max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    content_similarity_manifest_path: Path | str | None = None,
    content_similarity_source_name: str = MULTIMODAL_SIMILARITY_SOURCE,
    content_similarity_max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    content_similarity_exclude_history: bool = True,
    content_similarity_popularity_prior_weight: float = 0.0,
    content_similarity_popularity_lookback_days: int | None = None,
    content_similarity_candidate_pool_size: int | None = None,
    two_tower_model: TwoTowerSmokeModel | None = None,
    two_tower_source_name: str = TWO_TOWER_RETRIEVAL_SOURCE,
    two_tower_max_retrieval_articles: int | None = 5000,
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
        include_seasonal_popularity: Whether to export global shifted-window
            seasonal popularity rows.
        seasonal_shift_days: Number of days between cutoff and historical window
            start when seasonal rows are enabled.
        seasonal_window_days: Historical seasonal window length when seasonal
            rows are enabled.
        include_age_segment_popularity: Whether to export customer age-segment
            popularity rows.
        customer_segment_by_id: Customer-to-age-segment mapping required when
            ``include_age_segment_popularity`` is enabled.
        age_segment_bucket_size: Width of age buckets used to build the mapping.
        age_segment_popularity_lookback_days: Optional pre-cutoff segment
            popularity lookback. Defaults to ``popularity_lookback_days``.
        include_garment_group_popularity: Whether to export recent popularity rows
            from garment groups seen in the customer's pre-cutoff history.
        article_garment_group_by_id: Article-to-garment-group mapping required
            when garment-group popularity is enabled.
        garment_group_popularity_lookback_days: Optional pre-cutoff lookback used
            for garment-group article counts. Defaults to ``popularity_lookback_days``.
        garment_group_max_history_items: Recent unique customer-history articles
            used to infer garment-group affinities.
        content_similarity_manifest_path: Optional embedding-cache manifest used to
            export content-similarity rows.
        content_similarity_source_name: Source label for content rows.
        content_similarity_max_history_items: Recent unique customer-history length
            used for content query vectors.
        content_similarity_exclude_history: Whether content retrieval filters
            articles already in pre-cutoff customer history.
        content_similarity_popularity_prior_weight: Popularity-prior blend weight
            for content candidate reranking.
        content_similarity_popularity_lookback_days: Optional pre-cutoff lookback
            used to compute content popularity priors.
        content_similarity_candidate_pool_size: Optional content neighbor pool size
            before popularity-prior reranking.
        two_tower_model: Optional trained two-tower smoke model used to append
            model-retrieval candidate rows for mapped validation customers.
        two_tower_source_name: Source label for two-tower rows.
        two_tower_max_retrieval_articles: Optional exact-scoring article pool cap
            for two-tower retrieval.
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
    if seasonal_shift_days <= 0:
        raise ValueError("seasonal_shift_days must be positive")
    if seasonal_window_days <= 0:
        raise ValueError("seasonal_window_days must be positive")
    if content_similarity_manifest_path is not None and not content_similarity_source_name:
        raise ValueError("content_similarity_source_name must not be empty")
    if content_similarity_manifest_path is not None and content_similarity_max_history_items <= 0:
        raise ValueError("content_similarity_max_history_items must be positive")
    if two_tower_model is not None and not two_tower_source_name:
        raise ValueError("two_tower_source_name must not be empty")
    if two_tower_max_retrieval_articles is not None and two_tower_max_retrieval_articles <= 0:
        raise ValueError("two_tower_max_retrieval_articles must be positive when provided")
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
    if include_product_code_popularity and article_product_code_by_id is None:
        raise ValueError("article_product_code_by_id is required for product-code popularity")
    if product_code_max_history_items <= 0:
        raise ValueError("product_code_max_history_items must be positive")
    resolved_product_code_lookback_days = (
        product_code_popularity_lookback_days
        if product_code_popularity_lookback_days is not None
        else popularity_lookback_days
    )
    if resolved_product_code_lookback_days <= 0:
        raise ValueError("product_code_popularity_lookback_days must be positive")

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
    seasonal_popularity_index = (
        build_seasonal_popularity_index(
            transactions=transaction_iter_factory(),
            split=split,
            shift_days=seasonal_shift_days,
            window_days=seasonal_window_days,
            max_articles=k,
        )
        if include_seasonal_popularity
        else None
    )
    age_segment_index = (
        build_age_segment_popularity_index(
            transactions=transaction_iter_factory(),
            split=split,
            customer_segment_by_id=customer_segment_by_id or {},
            lookback_days=resolved_age_segment_lookback_days,
            max_articles_per_segment=k,
        )
        if include_age_segment_popularity
        else None
    )
    garment_group_index = (
        build_article_attribute_popularity_index(
            transactions=transaction_iter_factory(),
            split=split,
            target_customer_ids=target_customer_ids,
            article_attribute_by_id=article_garment_group_by_id or {},
            attribute_name=GARMENT_GROUP_COLUMN,
            lookback_days=resolved_garment_group_lookback_days,
            max_history_items=garment_group_max_history_items,
            max_articles_per_attribute=k,
        )
        if include_garment_group_popularity
        else None
    )
    product_code_index = (
        build_article_attribute_popularity_index(
            transactions=transaction_iter_factory(),
            split=split,
            target_customer_ids=target_customer_ids,
            article_attribute_by_id=article_product_code_by_id or {},
            attribute_name=PRODUCT_CODE_COLUMN,
            lookback_days=resolved_product_code_lookback_days,
            max_history_items=product_code_max_history_items,
            max_articles_per_attribute=k,
        )
        if include_product_code_popularity
        else None
    )
    content_similarity_index = (
        _build_content_similarity_index_from_manifest(
            transaction_iter_factory=transaction_iter_factory,
            split=split,
            target_customer_ids=target_customer_ids,
            manifest_path=content_similarity_manifest_path,
            source_name=content_similarity_source_name,
            max_history_items=content_similarity_max_history_items,
            exclude_history=content_similarity_exclude_history,
            popularity_prior_weight=content_similarity_popularity_prior_weight,
            popularity_lookback_days=content_similarity_popularity_lookback_days,
            candidate_pool_size=content_similarity_candidate_pool_size,
        )
        if content_similarity_manifest_path is not None
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
            seasonal_popularity_index=seasonal_popularity_index,
            age_segment_index=age_segment_index,
            garment_group_index=garment_group_index,
            product_code_index=product_code_index,
            content_similarity_index=content_similarity_index,
            two_tower_model=two_tower_model,
            two_tower_source_name=two_tower_source_name,
            two_tower_max_retrieval_articles=two_tower_max_retrieval_articles,
            k=k,
            recent_popularity_by_lookback={
                lookback: articles[:k]
                for lookback, articles in baseline_sources.recent_popularity_by_lookback.items()
            },
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
        include_seasonal_popularity=include_seasonal_popularity,
        seasonal_shift_days=seasonal_shift_days if include_seasonal_popularity else None,
        seasonal_window_days=seasonal_window_days if include_seasonal_popularity else None,
        include_age_segment_popularity=include_age_segment_popularity,
        age_segment_bucket_size=(
            age_segment_bucket_size if include_age_segment_popularity else None
        ),
        age_segment_popularity_lookback_days=(
            resolved_age_segment_lookback_days if include_age_segment_popularity else None
        ),
        include_garment_group_popularity=include_garment_group_popularity,
        garment_group_popularity_lookback_days=(
            resolved_garment_group_lookback_days if include_garment_group_popularity else None
        ),
        garment_group_max_history_items=(
            garment_group_max_history_items if include_garment_group_popularity else None
        ),
        include_product_code_popularity=include_product_code_popularity,
        product_code_popularity_lookback_days=(
            resolved_product_code_lookback_days if include_product_code_popularity else None
        ),
        product_code_max_history_items=(
            product_code_max_history_items if include_product_code_popularity else None
        ),
        content_similarity_manifest_path=(
            str(Path(content_similarity_manifest_path).expanduser().resolve())
            if content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_source_name=(
            content_similarity_source_name if content_similarity_manifest_path is not None else None
        ),
        content_similarity_max_history_items=(
            content_similarity_max_history_items
            if content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_prior_weight=(
            content_similarity_popularity_prior_weight
            if content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_popularity_lookback_days=(
            content_similarity_popularity_lookback_days
            if content_similarity_manifest_path is not None
            else None
        ),
        content_similarity_candidate_pool_size=(
            content_similarity_candidate_pool_size
            if content_similarity_manifest_path is not None
            else None
        ),
        include_two_tower_retrieval=two_tower_model is not None,
        two_tower_source_name=two_tower_source_name if two_tower_model is not None else None,
        two_tower_max_retrieval_articles=(
            two_tower_max_retrieval_articles if two_tower_model is not None else None
        ),
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
    seasonal_popularity_index: SeasonalPopularityIndex | None = None,
    age_segment_index: AgeSegmentPopularityIndex | None = None,
    garment_group_index: ArticleAttributePopularityIndex | None = None,
    product_code_index: ArticleAttributePopularityIndex | None = None,
    content_similarity_index: ContentSimilarityIndex | None = None,
    two_tower_model: TwoTowerSmokeModel | None = None,
    two_tower_source_name: str = TWO_TOWER_RETRIEVAL_SOURCE,
    two_tower_max_retrieval_articles: int | None = 5000,
    recent_popularity_by_lookback: Mapping[int, Sequence[str]] | None = None,
) -> Iterable[CandidateRecord]:
    """Yield source-specific candidate records in deterministic source order."""

    for customer_id in target_customer_ids:
        yield from iter_candidate_records_for_customer(
            customer_id=customer_id,
            repeat_recommendations=repeat_recommendations,
            recent_popularity=recent_popularity,
            all_time_popularity=all_time_popularity,
            co_visitation_index=co_visitation_index,
            seasonal_popularity_index=seasonal_popularity_index,
            age_segment_index=age_segment_index,
            garment_group_index=garment_group_index,
            product_code_index=product_code_index,
            content_similarity_index=content_similarity_index,
            two_tower_model=two_tower_model,
            recent_popularity_by_lookback=recent_popularity_by_lookback,
            two_tower_source_name=two_tower_source_name,
            two_tower_max_retrieval_articles=two_tower_max_retrieval_articles,
            k=k,
        )


def iter_candidate_records_for_customer(
    customer_id: str,
    repeat_recommendations: Mapping[str, tuple[str, ...]],
    recent_popularity: Sequence[str],
    all_time_popularity: Sequence[str],
    co_visitation_index: CoVisitationIndex | None,
    k: int,
    seasonal_popularity_index: SeasonalPopularityIndex | None = None,
    age_segment_index: AgeSegmentPopularityIndex | None = None,
    garment_group_index: ArticleAttributePopularityIndex | None = None,
    product_code_index: ArticleAttributePopularityIndex | None = None,
    content_similarity_index: ContentSimilarityIndex | None = None,
    two_tower_model: TwoTowerSmokeModel | None = None,
    two_tower_source_name: str = TWO_TOWER_RETRIEVAL_SOURCE,
    two_tower_max_retrieval_articles: int | None = 5000,
    recent_popularity_by_lookback: Mapping[int, Sequence[str]] | None = None,
) -> Iterable[CandidateRecord]:
    """Yield ranker-ready source rows for one customer.

    Args:
        customer_id: Customer requiring candidate records.
        repeat_recommendations: Customer-specific repeat-purchase rankings.
        recent_popularity: Global recent-popularity ranking.
        all_time_popularity: Global all-time popularity ranking.
        co_visitation_index: Optional co-visitation index for item-item rows.
        seasonal_popularity_index: Optional shifted-window seasonal popularity index.
        age_segment_index: Optional age-segment popularity index.
        garment_group_index: Optional garment-group affinity popularity index.
        k: Maximum candidates emitted per source.
        content_similarity_index: Optional cached content embedding index.
        two_tower_model: Optional trained two-tower smoke model.
        two_tower_source_name: Source label for two-tower rows.
        two_tower_max_retrieval_articles: Optional article pool cap for two-tower scoring.
        recent_popularity_by_lookback: Optional short-term recent-popularity
            rankings keyed by lookback length in days. Each entry produces a
            ``recent_popularity_<N>d`` source row. Closes the Phase-0 emission
            gap so ranker features ``has_recent_popularity_1d/3d`` are no
            longer guaranteed-zero placeholders.

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
    if recent_popularity_by_lookback is not None:
        for lookback in sorted(recent_popularity_by_lookback.keys()):
            articles = recent_popularity_by_lookback[lookback]
            if not articles:
                continue
            source_name = _SHORT_TERM_POPULARITY_SOURCE_BY_LOOKBACK.get(
                lookback, f"recent_popularity_{lookback}d"
            )
            yield from _ranked_article_records(
                customer_id=customer_id,
                source=source_name,
                article_ids=tuple(articles)[:k],
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
    if seasonal_popularity_index is not None:
        yield from (
            CandidateRecord(
                customer_id=customer_id,
                article_id=candidate.article_id,
                source=SEASONAL_POPULARITY_SOURCE,
                source_rank=candidate.rank,
                source_score=candidate.score,
            )
            for candidate in build_seasonal_popularity_candidates(
                seasonal_popularity_index,
                k=k,
            )
        )
    if age_segment_index is not None:
        yield from (
            CandidateRecord(
                customer_id=customer_id,
                article_id=candidate.article_id,
                source=AGE_SEGMENT_POPULARITY_SOURCE,
                source_rank=candidate.rank,
                source_score=candidate.score,
            )
            for candidate in build_age_segment_popularity_candidates(
                age_segment_index,
                customer_id,
                k=k,
            )
        )
    if garment_group_index is not None:
        yield from (
            CandidateRecord(
                customer_id=customer_id,
                article_id=candidate.article_id,
                source=GARMENT_GROUP_POPULARITY_SOURCE,
                source_rank=candidate.rank,
                source_score=candidate.score,
            )
            for candidate in build_article_attribute_popularity_candidates(
                garment_group_index,
                customer_id,
                k=k,
            )
        )
    if product_code_index is not None:
        yield from (
            CandidateRecord(
                customer_id=customer_id,
                article_id=candidate.article_id,
                source=PRODUCT_CODE_POPULARITY_SOURCE,
                source_rank=candidate.rank,
                source_score=candidate.score,
            )
            for candidate in build_article_attribute_popularity_candidates(
                product_code_index,
                customer_id,
                k=k,
            )
        )
    if content_similarity_index is not None:
        yield from build_content_similarity_candidate_source_records(
            content_similarity_index,
            customer_id,
            k=k,
        )
    if two_tower_model is not None:
        yield from (
            CandidateRecord(
                customer_id=customer_id,
                article_id=article_id,
                source=two_tower_source_name,
                source_rank=rank,
                source_score=score,
            )
            for rank, (article_id, score) in enumerate(
                score_two_tower_candidates(
                    two_tower_model,
                    customer_id,
                    k=k,
                    max_retrieval_articles=two_tower_max_retrieval_articles,
                ),
                start=1,
            )
        )


def _build_content_similarity_index_from_manifest(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    split: TemporalSplit,
    target_customer_ids: Sequence[str],
    manifest_path: Path | str,
    source_name: str,
    max_history_items: int,
    exclude_history: bool,
    popularity_prior_weight: float,
    popularity_lookback_days: int | None,
    candidate_pool_size: int | None,
) -> ContentSimilarityIndex:
    manifest = read_article_embedding_cache_manifest(manifest_path)
    embedding_records = load_article_embedding_cache(manifest)
    return build_content_similarity_index(
        transactions=transaction_iter_factory(),
        split=split,
        target_customer_ids=target_customer_ids,
        embedding_records=embedding_records,
        source_name=source_name,
        max_history_items=max_history_items,
        exclude_history=exclude_history,
        popularity_prior_weight=popularity_prior_weight,
        popularity_lookback_days=popularity_lookback_days,
        candidate_pool_size=candidate_pool_size,
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
