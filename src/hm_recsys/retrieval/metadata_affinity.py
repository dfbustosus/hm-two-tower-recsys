"""Article-metadata affinity popularity retrieval sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from math import log1p
from pathlib import Path

from hm_recsys.core.ids import is_article_id
from hm_recsys.data.io import CsvValueError, TransactionEvent, iter_csv_rows
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.co_visitation import DEFAULT_MAX_HISTORY_ITEMS

GARMENT_GROUP_COLUMN = "garment_group_name"
PRODUCT_CODE_COLUMN = "product_code"
UNKNOWN_ARTICLE_ATTRIBUTE = "unknown"


@dataclass(frozen=True)
class ArticleAttributePopularityCandidate:
    """One metadata-affinity popularity candidate for a customer."""

    article_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class ArticleAttributePopularityIndex:
    """Pre-cutoff article popularity by metadata attribute and customer affinity.

    Attributes:
        article_attribute_by_id: Article ID to metadata value mapping.
        customer_attributes: Recent unique metadata values from each target
            customer's pre-cutoff history, newest first.
        rankings_by_attribute: Ranked article IDs and normalized recent-popularity
            scores by metadata value.
        attribute_name: Article metadata column used for the source.
        lookback_days: Recent pre-cutoff window used for article counts.
        max_history_items: Recent unique customer-history article count scanned.
        train_rows_used: Number of pre-cutoff transactions scanned.
    """

    article_attribute_by_id: Mapping[str, str]
    customer_attributes: dict[str, tuple[str, ...]]
    rankings_by_attribute: dict[str, tuple[tuple[str, float], ...]]
    attribute_name: str
    lookback_days: int
    max_history_items: int
    train_rows_used: int


def load_article_attribute_values(
    raw_data_dir: Path | str,
    *,
    attribute_column: str = GARMENT_GROUP_COLUMN,
) -> dict[str, str]:
    """Load article metadata values from ``articles.csv`` while preserving IDs."""

    if not attribute_column:
        raise ValueError("attribute_column must not be empty")
    path = Path(raw_data_dir).expanduser().resolve() / "articles.csv"
    required_columns = ("article_id", attribute_column)
    values: dict[str, str] = {}
    for line_number, row in enumerate(iter_csv_rows(path, required_columns), start=2):
        article_id = row["article_id"]
        if not is_article_id(article_id):
            raise CsvValueError(f"line {line_number}: invalid article_id {article_id!r}")
        values[article_id] = _normalize_attribute_value(row[attribute_column])
    return values


def build_article_attribute_popularity_index(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    target_customer_ids: Iterable[str],
    article_attribute_by_id: Mapping[str, str],
    *,
    attribute_name: str = GARMENT_GROUP_COLUMN,
    lookback_days: int = 7,
    max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    max_articles_per_attribute: int | None = None,
) -> ArticleAttributePopularityIndex:
    """Build cutoff-safe recent article-popularity rankings by metadata value."""

    if not attribute_name:
        raise ValueError("attribute_name must not be empty")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if max_history_items <= 0:
        raise ValueError("max_history_items must be positive")
    if max_articles_per_attribute is not None and max_articles_per_attribute <= 0:
        raise ValueError("max_articles_per_attribute must be positive when provided")

    target_customer_set = set(target_customer_ids)
    window_start = split.cutoff - timedelta(days=lookback_days)
    counts_by_attribute: dict[str, Counter[str]] = {}
    mutable_customer_attributes: dict[str, list[str]] = {}
    train_rows_used = 0

    for transaction in transactions:
        if transaction.t_dat >= split.cutoff:
            continue
        train_rows_used += 1
        attribute_value = article_attribute_by_id.get(transaction.article_id)
        if attribute_value is None:
            continue
        if transaction.t_dat >= window_start:
            counts_by_attribute.setdefault(attribute_value, Counter())[transaction.article_id] += 1
        if transaction.customer_id in target_customer_set:
            _update_recent_unique_attribute(
                mutable_customer_attributes.setdefault(transaction.customer_id, []),
                attribute_value,
                max_history_items=max_history_items,
            )

    customer_attributes = {
        customer_id: tuple(reversed(attributes))
        for customer_id, attributes in mutable_customer_attributes.items()
    }
    rankings_by_attribute = {
        attribute_value: _rank_attribute_counts(counts, limit=max_articles_per_attribute)
        for attribute_value, counts in counts_by_attribute.items()
    }
    return ArticleAttributePopularityIndex(
        article_attribute_by_id=article_attribute_by_id,
        customer_attributes=customer_attributes,
        rankings_by_attribute=rankings_by_attribute,
        attribute_name=attribute_name,
        lookback_days=lookback_days,
        max_history_items=max_history_items,
        train_rows_used=train_rows_used,
    )


def build_article_attribute_popularity_candidates(
    index: ArticleAttributePopularityIndex,
    customer_id: str,
    k: int,
) -> tuple[ArticleAttributePopularityCandidate, ...]:
    """Return metadata-affinity popularity candidates for one customer."""

    if k <= 0:
        raise ValueError("k must be positive")
    attributes = index.customer_attributes.get(customer_id, ())
    if not attributes:
        return ()
    selected: dict[str, float] = {}
    for attribute_rank, attribute_value in enumerate(attributes, start=1):
        attribute_weight = 1.0 / attribute_rank
        for article_id, article_score in index.rankings_by_attribute.get(attribute_value, ()):
            score = article_score * attribute_weight
            current_score = selected.get(article_id)
            if current_score is None or score > current_score:
                selected[article_id] = score
    ranked = sorted(selected.items(), key=lambda item: (-item[1], item[0]))[:k]
    return tuple(
        ArticleAttributePopularityCandidate(article_id=article_id, score=score, rank=rank)
        for rank, (article_id, score) in enumerate(ranked, start=1)
    )


def _normalize_attribute_value(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized if normalized else UNKNOWN_ARTICLE_ATTRIBUTE


def _update_recent_unique_attribute(
    attributes: list[str],
    attribute_value: str,
    *,
    max_history_items: int,
) -> None:
    if attribute_value in attributes:
        attributes.remove(attribute_value)
    attributes.append(attribute_value)
    if len(attributes) > max_history_items:
        del attributes[0 : len(attributes) - max_history_items]


def _rank_attribute_counts(
    counts: Counter[str],
    *,
    limit: int | None,
) -> tuple[tuple[str, float], ...]:
    if not counts:
        return ()
    # Counts already reflect a bounded recent window. Use log-count and article ID
    # for deterministic ranking; no target-window data is present in ``counts``.
    log_counts = {article_id: log1p(count) for article_id, count in counts.items()}
    max_log_count = max(log_counts.values())
    ranked = sorted(log_counts.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ranked = ranked[:limit]
    return tuple(
        (article_id, log_count / max_log_count if max_log_count > 0.0 else 0.0)
        for article_id, log_count in ranked
    )


__all__ = [
    "GARMENT_GROUP_COLUMN",
    "PRODUCT_CODE_COLUMN",
    "UNKNOWN_ARTICLE_ATTRIBUTE",
    "ArticleAttributePopularityCandidate",
    "ArticleAttributePopularityIndex",
    "build_article_attribute_popularity_candidates",
    "build_article_attribute_popularity_index",
    "load_article_attribute_values",
]
