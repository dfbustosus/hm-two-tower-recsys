"""Cutoff-safe behavioral ranker features for candidate reranking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from math import log1p
from typing import TypeVar

from hm_recsys.data.io import TransactionEvent
from hm_recsys.ranking.deterministic import CandidateFeatures
from hm_recsys.ranking.linear import LINEAR_FEATURE_NAMES, feature_vector

BEHAVIORAL_FEATURE_NAMES = (
    "customer_transaction_count_log",
    "customer_unique_article_count_log",
    "customer_days_since_last_purchase",
    "article_all_time_purchase_count_log",
    "article_1d_purchase_count_log",
    "article_3d_purchase_count_log",
    "article_7d_purchase_count_log",
    "article_14d_purchase_count_log",
    "article_30d_purchase_count_log",
    "article_days_since_last_purchase",
    "user_article_purchase_count_log",
    "user_article_days_since_last_purchase",
)

FeatureKey = TypeVar("FeatureKey", str, tuple[str, str])

SOURCE_AND_BEHAVIORAL_FEATURE_NAMES = (*LINEAR_FEATURE_NAMES, *BEHAVIORAL_FEATURE_NAMES)


@dataclass(frozen=True)
class CutoffBehavioralFeatures:
    """Precomputed behavioral feature store for one cutoff date.

    The store is intentionally narrow: it only uses transactions with
    ``t_dat < cutoff`` and can be scoped to target customers/candidate articles
    for memory-safe ranker experiments. Duplicate transaction rows are counted
    as repeated purchases, matching the H&M transaction semantics.
    """

    cutoff: date
    missing_days_since: float
    customer_transaction_counts: dict[str, int]
    customer_unique_article_counts: dict[str, int]
    customer_last_purchase_dates: dict[str, date]
    article_purchase_counts: dict[str, int]
    article_purchase_counts_1d: dict[str, int]
    article_purchase_counts_3d: dict[str, int]
    article_purchase_counts_7d: dict[str, int]
    article_purchase_counts_14d: dict[str, int]
    article_purchase_counts_30d: dict[str, int]
    article_last_purchase_dates: dict[str, date]
    user_article_purchase_counts: dict[tuple[str, str], int]
    user_article_last_purchase_dates: dict[tuple[str, str], date]

    def vector_for(self, customer_id: str, article_id: str) -> tuple[float, ...]:
        """Return behavioral features for one candidate pair.

        Args:
            customer_id: H&M customer identifier preserved as a string.
            article_id: H&M article identifier preserved as a string.

        Returns:
            Tuple matching :data:`BEHAVIORAL_FEATURE_NAMES`.
        """

        customer_last_date = self.customer_last_purchase_dates.get(customer_id)
        article_last_date = self.article_last_purchase_dates.get(article_id)
        user_article_last_date = self.user_article_last_purchase_dates.get(
            (customer_id, article_id)
        )
        return (
            log1p(self.customer_transaction_counts.get(customer_id, 0)),
            log1p(self.customer_unique_article_counts.get(customer_id, 0)),
            self._days_since(customer_last_date),
            log1p(self.article_purchase_counts.get(article_id, 0)),
            log1p(self.article_purchase_counts_1d.get(article_id, 0)),
            log1p(self.article_purchase_counts_3d.get(article_id, 0)),
            log1p(self.article_purchase_counts_7d.get(article_id, 0)),
            log1p(self.article_purchase_counts_14d.get(article_id, 0)),
            log1p(self.article_purchase_counts_30d.get(article_id, 0)),
            self._days_since(article_last_date),
            log1p(self.user_article_purchase_counts.get((customer_id, article_id), 0)),
            self._days_since(user_article_last_date),
        )

    def _days_since(self, last_date: date | None) -> float:
        if last_date is None:
            return self.missing_days_since
        return float((self.cutoff - last_date).days)


def build_cutoff_behavioral_features(
    transactions: Iterable[TransactionEvent],
    cutoff: date,
    *,
    target_customer_ids: Iterable[str] | None = None,
    candidate_article_ids: Iterable[str] | None = None,
    missing_days_since: float = 999.0,
) -> CutoffBehavioralFeatures:
    """Build cutoff-safe behavioral ranker features.

    Args:
        transactions: Transaction events to scan. Only events with
            ``event.t_dat < cutoff`` contribute features.
        cutoff: Exclusive feature cutoff date.
        target_customer_ids: Optional customer scope. When provided, customer
            and user-article features are computed only for these customers.
        candidate_article_ids: Optional article scope. When provided, article
            and user-article features are computed only for these articles.
        missing_days_since: Sentinel used when no prior purchase date exists.

    Returns:
        Cutoff-scoped behavioral feature store.

    Raises:
        ValueError: If ``missing_days_since`` is negative.
    """

    if missing_days_since < 0:
        raise ValueError("missing_days_since must be non-negative")

    customer_scope = set(target_customer_ids) if target_customer_ids is not None else None
    article_scope = set(candidate_article_ids) if candidate_article_ids is not None else None

    customer_transaction_counts: dict[str, int] = {}
    customer_unique_articles: dict[str, set[str]] = {}
    customer_last_purchase_dates: dict[str, date] = {}
    article_purchase_counts: dict[str, int] = {}
    article_purchase_counts_1d: dict[str, int] = {}
    article_purchase_counts_3d: dict[str, int] = {}
    article_purchase_counts_7d: dict[str, int] = {}
    article_purchase_counts_14d: dict[str, int] = {}
    article_purchase_counts_30d: dict[str, int] = {}
    article_last_purchase_dates: dict[str, date] = {}
    user_article_purchase_counts: dict[tuple[str, str], int] = {}
    user_article_last_purchase_dates: dict[tuple[str, str], date] = {}

    for transaction in transactions:
        if transaction.t_dat >= cutoff:
            continue
        customer_in_scope = customer_scope is None or transaction.customer_id in customer_scope
        article_in_scope = article_scope is None or transaction.article_id in article_scope
        days_before_cutoff = (cutoff - transaction.t_dat).days

        if customer_in_scope:
            _increment(customer_transaction_counts, transaction.customer_id)
            customer_unique_articles.setdefault(transaction.customer_id, set()).add(
                transaction.article_id
            )
            _set_latest(
                customer_last_purchase_dates,
                transaction.customer_id,
                transaction.t_dat,
            )

        if article_in_scope:
            _increment(article_purchase_counts, transaction.article_id)
            if days_before_cutoff <= 1:
                _increment(article_purchase_counts_1d, transaction.article_id)
            if days_before_cutoff <= 3:
                _increment(article_purchase_counts_3d, transaction.article_id)
            if days_before_cutoff <= 7:
                _increment(article_purchase_counts_7d, transaction.article_id)
            if days_before_cutoff <= 14:
                _increment(article_purchase_counts_14d, transaction.article_id)
            if days_before_cutoff <= 30:
                _increment(article_purchase_counts_30d, transaction.article_id)
            _set_latest(article_last_purchase_dates, transaction.article_id, transaction.t_dat)

        if customer_in_scope and article_in_scope:
            key = (transaction.customer_id, transaction.article_id)
            _increment(user_article_purchase_counts, key)
            _set_latest(user_article_last_purchase_dates, key, transaction.t_dat)

    return CutoffBehavioralFeatures(
        cutoff=cutoff,
        missing_days_since=missing_days_since,
        customer_transaction_counts=customer_transaction_counts,
        customer_unique_article_counts={
            customer_id: len(article_ids)
            for customer_id, article_ids in customer_unique_articles.items()
        },
        customer_last_purchase_dates=customer_last_purchase_dates,
        article_purchase_counts=article_purchase_counts,
        article_purchase_counts_1d=article_purchase_counts_1d,
        article_purchase_counts_3d=article_purchase_counts_3d,
        article_purchase_counts_7d=article_purchase_counts_7d,
        article_purchase_counts_14d=article_purchase_counts_14d,
        article_purchase_counts_30d=article_purchase_counts_30d,
        article_last_purchase_dates=article_last_purchase_dates,
        user_article_purchase_counts=user_article_purchase_counts,
        user_article_last_purchase_dates=user_article_last_purchase_dates,
    )


def source_and_behavioral_feature_vector(
    candidate_features: CandidateFeatures,
    behavioral_features: CutoffBehavioralFeatures,
) -> tuple[float, ...]:
    """Combine candidate-source and cutoff-safe behavioral features.

    Args:
        candidate_features: Aggregated source features for one candidate pair.
        behavioral_features: Feature store built with the same feature cutoff as
            the candidate export.

    Returns:
        Tuple matching :data:`SOURCE_AND_BEHAVIORAL_FEATURE_NAMES`.
    """

    return (
        *feature_vector(candidate_features),
        *behavioral_features.vector_for(
            candidate_features.customer_id,
            candidate_features.article_id,
        ),
    )


def _increment(mapping: dict[FeatureKey, int], key: FeatureKey) -> None:
    mapping[key] = mapping.get(key, 0) + 1


def _set_latest(
    mapping: dict[FeatureKey, date],
    key: FeatureKey,
    value: date,
) -> None:
    current = mapping.get(key)
    if current is None or value > current:
        mapping[key] = value
