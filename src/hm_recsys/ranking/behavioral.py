"""Cutoff-safe behavioral ranker features for candidate reranking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from math import log1p
from pathlib import Path
from typing import Protocol, TypeVar

from hm_recsys.core.ids import is_article_id
from hm_recsys.data.io import CsvValueError, iter_csv_rows
from hm_recsys.ranking.deterministic import CandidateFeatures
from hm_recsys.ranking.linear import LINEAR_FEATURE_NAMES, feature_vector

ARTICLE_ATTRIBUTE_FEATURES = (
    ("product_type_no", "product_type"),
    ("product_group_name", "product_group"),
    ("department_no", "department"),
    ("section_no", "section"),
    ("garment_group_no", "garment_group"),
    ("colour_group_code", "colour_group"),
    ("index_group_no", "index_group"),
)
ARTICLE_ATTRIBUTE_WINDOW_FEATURES = (
    ("product_type_no", "product_type"),
    ("section_no", "section"),
    ("garment_group_no", "garment_group"),
)
ARTICLE_ATTRIBUTE_COLUMNS = tuple(column for column, _ in ARTICLE_ATTRIBUTE_FEATURES)

BEHAVIORAL_FEATURE_NAMES = (
    "customer_transaction_count_log",
    "customer_unique_article_count_log",
    "customer_1d_transaction_count_log",
    "customer_3d_transaction_count_log",
    "customer_7d_transaction_count_log",
    "customer_30d_transaction_count_log",
    "customer_1d_to_7d_transaction_ratio",
    "customer_3d_to_7d_transaction_ratio",
    "customer_7d_to_30d_transaction_ratio",
    "customer_30d_to_all_time_transaction_ratio",
    "customer_days_since_last_purchase",
    "customer_tenure_days_log",
    "article_all_time_purchase_count_log",
    "article_1d_purchase_count_log",
    "article_3d_purchase_count_log",
    "article_7d_purchase_count_log",
    "article_14d_purchase_count_log",
    "article_30d_purchase_count_log",
    "article_1d_to_7d_purchase_ratio",
    "article_3d_to_14d_purchase_ratio",
    "article_7d_to_30d_purchase_ratio",
    "article_7d_to_all_time_purchase_ratio",
    "article_1d_purchase_count_delta_vs_7d_rate",
    "article_3d_purchase_count_delta_vs_14d_rate",
    "article_7d_purchase_count_delta_vs_30d_rate",
    "article_days_since_last_purchase",
    "user_article_purchase_count_log",
    "user_article_days_since_last_purchase",
    "customer_mean_price",
    "article_mean_price",
    "article_mean_price_minus_customer_mean",
    "article_mean_price_ratio_customer_mean",
    "customer_sales_channel_2_share",
    "article_sales_channel_2_share",
    "article_customer_sales_channel_2_share_gap_abs",
    *(f"user_{feature_slug}_purchase_count_log" for _, feature_slug in ARTICLE_ATTRIBUTE_FEATURES),
    *(
        f"user_{feature_slug}_30d_purchase_count_log"
        for _, feature_slug in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
    ),
    *(
        f"user_{feature_slug}_30d_to_all_time_purchase_ratio"
        for _, feature_slug in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
    ),
    *(
        f"user_{feature_slug}_days_since_last_purchase"
        for _, feature_slug in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
    ),
    *(
        f"article_{feature_slug}_7d_purchase_count_log"
        for _, feature_slug in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
    ),
    *(
        f"article_{feature_slug}_30d_purchase_count_log"
        for _, feature_slug in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
    ),
    *(
        f"article_{feature_slug}_7d_to_30d_purchase_ratio"
        for _, feature_slug in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
    ),
)

FeatureKey = TypeVar("FeatureKey", str, tuple[str, str], tuple[str, str, str])
ArticleAttributeMap = Mapping[str, Mapping[str, str]]


class BehavioralTransaction(Protocol):
    """Transaction fields required for cutoff-safe behavioral features."""

    @property
    def t_dat(self) -> date: ...

    @property
    def customer_id(self) -> str: ...

    @property
    def article_id(self) -> str: ...


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
    customer_transaction_counts_1d: dict[str, int]
    customer_transaction_counts_3d: dict[str, int]
    customer_transaction_counts_7d: dict[str, int]
    customer_transaction_counts_30d: dict[str, int]
    customer_last_purchase_dates: dict[str, date]
    customer_first_purchase_dates: dict[str, date]
    article_purchase_counts: dict[str, int]
    article_purchase_counts_1d: dict[str, int]
    article_purchase_counts_3d: dict[str, int]
    article_purchase_counts_7d: dict[str, int]
    article_purchase_counts_14d: dict[str, int]
    article_purchase_counts_30d: dict[str, int]
    article_last_purchase_dates: dict[str, date]
    user_article_purchase_counts: dict[tuple[str, str], int]
    user_article_last_purchase_dates: dict[tuple[str, str], date]
    customer_price_sums: dict[str, float]
    customer_price_counts: dict[str, int]
    article_price_sums: dict[str, float]
    article_price_counts: dict[str, int]
    customer_sales_channel_2_counts: dict[str, int]
    article_sales_channel_2_counts: dict[str, int]
    article_attributes_by_id: ArticleAttributeMap
    customer_attribute_purchase_counts: dict[tuple[str, str, str], int]
    customer_attribute_purchase_counts_30d: dict[tuple[str, str, str], int]
    customer_attribute_last_purchase_dates: dict[tuple[str, str, str], date]
    article_attribute_purchase_counts_7d: dict[tuple[str, str], int]
    article_attribute_purchase_counts_30d: dict[tuple[str, str], int]

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
        customer_transaction_count = self.customer_transaction_counts.get(customer_id, 0)
        customer_transaction_count_1d = self.customer_transaction_counts_1d.get(customer_id, 0)
        customer_transaction_count_3d = self.customer_transaction_counts_3d.get(customer_id, 0)
        customer_transaction_count_7d = self.customer_transaction_counts_7d.get(customer_id, 0)
        customer_transaction_count_30d = self.customer_transaction_counts_30d.get(customer_id, 0)
        article_purchase_count = self.article_purchase_counts.get(article_id, 0)
        article_purchase_count_1d = self.article_purchase_counts_1d.get(article_id, 0)
        article_purchase_count_3d = self.article_purchase_counts_3d.get(article_id, 0)
        article_purchase_count_7d = self.article_purchase_counts_7d.get(article_id, 0)
        article_purchase_count_14d = self.article_purchase_counts_14d.get(article_id, 0)
        article_purchase_count_30d = self.article_purchase_counts_30d.get(article_id, 0)
        customer_mean_price = _mean_for(
            self.customer_price_sums,
            self.customer_price_counts,
            customer_id,
        )
        article_mean_price = _mean_for(
            self.article_price_sums,
            self.article_price_counts,
            article_id,
        )
        customer_channel_2_share = _share_for(
            self.customer_sales_channel_2_counts,
            self.customer_transaction_counts,
            customer_id,
        )
        article_channel_2_share = _share_for(
            self.article_sales_channel_2_counts,
            self.article_purchase_counts,
            article_id,
        )
        attribute_count_features = self._attribute_count_features(customer_id, article_id)
        attribute_30d_features = self._attribute_30d_features(customer_id, article_id)
        attribute_30d_ratio_features = self._attribute_30d_ratio_features(customer_id, article_id)
        attribute_recency_features = self._attribute_recency_features(customer_id, article_id)
        article_attribute_trend_features = self._article_attribute_trend_features(article_id)
        customer_first_date = self.customer_first_purchase_dates.get(customer_id)
        customer_tenure_days = (
            (self.cutoff - customer_first_date).days
            if customer_first_date is not None
            else self.missing_days_since
        )
        return (
            log1p(customer_transaction_count),
            log1p(self.customer_unique_article_counts.get(customer_id, 0)),
            log1p(customer_transaction_count_1d),
            log1p(customer_transaction_count_3d),
            log1p(customer_transaction_count_7d),
            log1p(customer_transaction_count_30d),
            _safe_ratio(customer_transaction_count_1d, customer_transaction_count_7d),
            _safe_ratio(customer_transaction_count_3d, customer_transaction_count_7d),
            _safe_ratio(customer_transaction_count_7d, customer_transaction_count_30d),
            _safe_ratio(customer_transaction_count_30d, customer_transaction_count),
            self._days_since(customer_last_date),
            log1p(max(0, customer_tenure_days)),
            log1p(article_purchase_count),
            log1p(article_purchase_count_1d),
            log1p(article_purchase_count_3d),
            log1p(article_purchase_count_7d),
            log1p(article_purchase_count_14d),
            log1p(article_purchase_count_30d),
            _safe_ratio(article_purchase_count_1d, article_purchase_count_7d),
            _safe_ratio(article_purchase_count_3d, article_purchase_count_14d),
            _safe_ratio(article_purchase_count_7d, article_purchase_count_30d),
            _safe_ratio(article_purchase_count_7d, article_purchase_count),
            _rate_delta(article_purchase_count_1d, article_purchase_count_7d, 1, 7),
            _rate_delta(article_purchase_count_3d, article_purchase_count_14d, 3, 14),
            _rate_delta(article_purchase_count_7d, article_purchase_count_30d, 7, 30),
            self._days_since(article_last_date),
            log1p(self.user_article_purchase_counts.get((customer_id, article_id), 0)),
            self._days_since(user_article_last_date),
            customer_mean_price,
            article_mean_price,
            article_mean_price - customer_mean_price,
            _safe_ratio(article_mean_price, customer_mean_price),
            customer_channel_2_share,
            article_channel_2_share,
            abs(article_channel_2_share - customer_channel_2_share),
            *attribute_count_features,
            *attribute_30d_features,
            *attribute_30d_ratio_features,
            *attribute_recency_features,
            *article_attribute_trend_features,
        )

    def _days_since(self, last_date: date | None) -> float:
        if last_date is None:
            return self.missing_days_since
        return float((self.cutoff - last_date).days)

    def _attribute_count_features(self, customer_id: str, article_id: str) -> tuple[float, ...]:
        attributes = self.article_attributes_by_id.get(article_id, {})
        return tuple(
            (
                log1p(
                    self.customer_attribute_purchase_counts.get(
                        (customer_id, column, attributes.get(column, "")),
                        0,
                    )
                )
                if attributes.get(column, "")
                else 0.0
            )
            for column, _ in ARTICLE_ATTRIBUTE_FEATURES
        )

    def _attribute_30d_features(self, customer_id: str, article_id: str) -> tuple[float, ...]:
        attributes = self.article_attributes_by_id.get(article_id, {})
        return tuple(
            (
                log1p(
                    self.customer_attribute_purchase_counts_30d.get(
                        (customer_id, column, attributes.get(column, "")),
                        0,
                    )
                )
                if attributes.get(column, "")
                else 0.0
            )
            for column, _ in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
        )

    def _attribute_30d_ratio_features(
        self,
        customer_id: str,
        article_id: str,
    ) -> tuple[float, ...]:
        attributes = self.article_attributes_by_id.get(article_id, {})
        return tuple(
            (
                _safe_ratio(
                    self.customer_attribute_purchase_counts_30d.get(
                        (customer_id, column, attributes.get(column, "")),
                        0,
                    ),
                    self.customer_attribute_purchase_counts.get(
                        (customer_id, column, attributes.get(column, "")),
                        0,
                    ),
                )
                if attributes.get(column, "")
                else 0.0
            )
            for column, _ in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
        )

    def _attribute_recency_features(self, customer_id: str, article_id: str) -> tuple[float, ...]:
        attributes = self.article_attributes_by_id.get(article_id, {})
        return tuple(
            (
                self._days_since(
                    self.customer_attribute_last_purchase_dates.get(
                        (customer_id, column, attributes.get(column, ""))
                    )
                )
                if attributes.get(column, "")
                else self.missing_days_since
            )
            for column, _ in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
        )

    def _article_attribute_trend_features(self, article_id: str) -> tuple[float, ...]:
        attributes = self.article_attributes_by_id.get(article_id, {})
        counts_7d = tuple(
            (
                self.article_attribute_purchase_counts_7d.get(
                    (column, attributes.get(column, "")),
                    0,
                )
                if attributes.get(column, "")
                else 0
            )
            for column, _ in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
        )
        counts_30d = tuple(
            (
                self.article_attribute_purchase_counts_30d.get(
                    (column, attributes.get(column, "")),
                    0,
                )
                if attributes.get(column, "")
                else 0
            )
            for column, _ in ARTICLE_ATTRIBUTE_WINDOW_FEATURES
        )
        ratio_features = tuple(
            _safe_ratio(count_7d, count_30d)
            for count_7d, count_30d in zip(counts_7d, counts_30d, strict=True)
        )
        return (
            *(log1p(count) for count in counts_7d),
            *(log1p(count) for count in counts_30d),
            *ratio_features,
        )


def build_cutoff_behavioral_features(
    transactions: Iterable[BehavioralTransaction],
    cutoff: date,
    *,
    target_customer_ids: Iterable[str] | None = None,
    candidate_article_ids: Iterable[str] | None = None,
    article_attributes_by_id: ArticleAttributeMap | None = None,
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
        article_attributes_by_id: Optional article metadata keyed by article ID.
            When provided, cutoff-safe customer/category affinity features are
            computed against the candidate article's static attributes.
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
    resolved_article_attributes = article_attributes_by_id or {}

    customer_transaction_counts: dict[str, int] = {}
    customer_unique_articles: dict[str, set[str]] = {}
    customer_transaction_counts_1d: dict[str, int] = {}
    customer_transaction_counts_3d: dict[str, int] = {}
    customer_transaction_counts_7d: dict[str, int] = {}
    customer_transaction_counts_30d: dict[str, int] = {}
    customer_last_purchase_dates: dict[str, date] = {}
    customer_first_purchase_dates: dict[str, date] = {}
    article_purchase_counts: dict[str, int] = {}
    article_purchase_counts_1d: dict[str, int] = {}
    article_purchase_counts_3d: dict[str, int] = {}
    article_purchase_counts_7d: dict[str, int] = {}
    article_purchase_counts_14d: dict[str, int] = {}
    article_purchase_counts_30d: dict[str, int] = {}
    article_last_purchase_dates: dict[str, date] = {}
    user_article_purchase_counts: dict[tuple[str, str], int] = {}
    user_article_last_purchase_dates: dict[tuple[str, str], date] = {}
    customer_price_sums: dict[str, float] = {}
    customer_price_counts: dict[str, int] = {}
    article_price_sums: dict[str, float] = {}
    article_price_counts: dict[str, int] = {}
    customer_sales_channel_2_counts: dict[str, int] = {}
    article_sales_channel_2_counts: dict[str, int] = {}
    customer_attribute_purchase_counts: dict[tuple[str, str, str], int] = {}
    customer_attribute_purchase_counts_30d: dict[tuple[str, str, str], int] = {}
    customer_attribute_last_purchase_dates: dict[tuple[str, str, str], date] = {}
    article_attribute_purchase_counts_7d: dict[tuple[str, str], int] = {}
    article_attribute_purchase_counts_30d: dict[tuple[str, str], int] = {}

    for transaction in transactions:
        if transaction.t_dat >= cutoff:
            continue
        customer_in_scope = customer_scope is None or transaction.customer_id in customer_scope
        article_in_scope = article_scope is None or transaction.article_id in article_scope
        days_before_cutoff = (cutoff - transaction.t_dat).days
        price = _optional_float(getattr(transaction, "price", None))
        sales_channel_id = getattr(transaction, "sales_channel_id", None)

        if customer_in_scope:
            _increment(customer_transaction_counts, transaction.customer_id)
            customer_unique_articles.setdefault(transaction.customer_id, set()).add(
                transaction.article_id
            )
            if days_before_cutoff <= 1:
                _increment(customer_transaction_counts_1d, transaction.customer_id)
            if days_before_cutoff <= 3:
                _increment(customer_transaction_counts_3d, transaction.customer_id)
            if days_before_cutoff <= 7:
                _increment(customer_transaction_counts_7d, transaction.customer_id)
            if days_before_cutoff <= 30:
                _increment(customer_transaction_counts_30d, transaction.customer_id)
            _set_latest(
                customer_last_purchase_dates,
                transaction.customer_id,
                transaction.t_dat,
            )
            _set_earliest(
                customer_first_purchase_dates,
                transaction.customer_id,
                transaction.t_dat,
            )
            if price is not None:
                _add_to_sum(customer_price_sums, transaction.customer_id, price)
                _increment(customer_price_counts, transaction.customer_id)
            if sales_channel_id == 2:
                _increment(customer_sales_channel_2_counts, transaction.customer_id)
            _update_customer_attribute_features(
                customer_attribute_purchase_counts,
                customer_attribute_purchase_counts_30d,
                customer_attribute_last_purchase_dates,
                resolved_article_attributes.get(transaction.article_id, {}),
                transaction.customer_id,
                transaction.t_dat,
                days_before_cutoff,
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
            if price is not None:
                _add_to_sum(article_price_sums, transaction.article_id, price)
                _increment(article_price_counts, transaction.article_id)
            if sales_channel_id == 2:
                _increment(article_sales_channel_2_counts, transaction.article_id)

        if customer_in_scope and article_in_scope:
            key = (transaction.customer_id, transaction.article_id)
            _increment(user_article_purchase_counts, key)
            _set_latest(user_article_last_purchase_dates, key, transaction.t_dat)

        _update_article_attribute_trend_features(
            article_attribute_purchase_counts_7d,
            article_attribute_purchase_counts_30d,
            resolved_article_attributes.get(transaction.article_id, {}),
            days_before_cutoff,
        )

    return CutoffBehavioralFeatures(
        cutoff=cutoff,
        missing_days_since=missing_days_since,
        customer_transaction_counts=customer_transaction_counts,
        customer_unique_article_counts={
            customer_id: len(article_ids)
            for customer_id, article_ids in customer_unique_articles.items()
        },
        customer_transaction_counts_1d=customer_transaction_counts_1d,
        customer_transaction_counts_3d=customer_transaction_counts_3d,
        customer_transaction_counts_7d=customer_transaction_counts_7d,
        customer_transaction_counts_30d=customer_transaction_counts_30d,
        customer_last_purchase_dates=customer_last_purchase_dates,
        customer_first_purchase_dates=customer_first_purchase_dates,
        article_purchase_counts=article_purchase_counts,
        article_purchase_counts_1d=article_purchase_counts_1d,
        article_purchase_counts_3d=article_purchase_counts_3d,
        article_purchase_counts_7d=article_purchase_counts_7d,
        article_purchase_counts_14d=article_purchase_counts_14d,
        article_purchase_counts_30d=article_purchase_counts_30d,
        article_last_purchase_dates=article_last_purchase_dates,
        user_article_purchase_counts=user_article_purchase_counts,
        user_article_last_purchase_dates=user_article_last_purchase_dates,
        customer_price_sums=customer_price_sums,
        customer_price_counts=customer_price_counts,
        article_price_sums=article_price_sums,
        article_price_counts=article_price_counts,
        customer_sales_channel_2_counts=customer_sales_channel_2_counts,
        article_sales_channel_2_counts=article_sales_channel_2_counts,
        article_attributes_by_id=resolved_article_attributes,
        customer_attribute_purchase_counts=customer_attribute_purchase_counts,
        customer_attribute_purchase_counts_30d=customer_attribute_purchase_counts_30d,
        customer_attribute_last_purchase_dates=customer_attribute_last_purchase_dates,
        article_attribute_purchase_counts_7d=article_attribute_purchase_counts_7d,
        article_attribute_purchase_counts_30d=article_attribute_purchase_counts_30d,
    )


def load_article_attribute_maps(
    raw_data_dir: Path | str,
    columns: Iterable[str] = ARTICLE_ATTRIBUTE_COLUMNS,
) -> dict[str, dict[str, str]]:
    """Load selected static article metadata for ranker affinity features.

    IDs and categorical values are read as strings. The resulting mapping is safe
    to use with temporal ranker features because H&M article metadata is static
    and does not include validation-window purchases.
    """

    requested_columns = tuple(dict.fromkeys(columns))
    path = Path(raw_data_dir).expanduser().resolve() / "articles.csv"
    required_columns = ("article_id", *requested_columns)
    article_attributes: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(iter_csv_rows(path, required_columns), start=2):
        article_id = row["article_id"]
        if not is_article_id(article_id):
            raise CsvValueError(f"line {line_number}: invalid article_id {article_id!r}")
        if article_id in article_attributes:
            raise CsvValueError(f"line {line_number}: duplicate article_id {article_id!r}")
        article_attributes[article_id] = {column: row[column] for column in requested_columns}
    return article_attributes


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


def _add_to_sum(mapping: dict[FeatureKey, float], key: FeatureKey, value: float) -> None:
    mapping[key] = mapping.get(key, 0.0) + value


def _set_latest(
    mapping: dict[FeatureKey, date],
    key: FeatureKey,
    value: date,
) -> None:
    current = mapping.get(key)
    if current is None or value > current:
        mapping[key] = value


def _set_earliest(
    mapping: dict[FeatureKey, date],
    key: FeatureKey,
    value: date,
) -> None:
    current = mapping.get(key)
    if current is None or value < current:
        mapping[key] = value


def _update_customer_attribute_features(
    customer_attribute_purchase_counts: dict[tuple[str, str, str], int],
    customer_attribute_purchase_counts_30d: dict[tuple[str, str, str], int],
    customer_attribute_last_purchase_dates: dict[tuple[str, str, str], date],
    article_attributes: Mapping[str, str],
    customer_id: str,
    transaction_date: date,
    days_before_cutoff: int,
) -> None:
    for column, _ in ARTICLE_ATTRIBUTE_FEATURES:
        value = article_attributes.get(column, "")
        if not value:
            continue
        key = (customer_id, column, value)
        _increment(customer_attribute_purchase_counts, key)
        if column in _WINDOW_ATTRIBUTE_COLUMNS and days_before_cutoff <= 30:
            _increment(customer_attribute_purchase_counts_30d, key)
        if column in _WINDOW_ATTRIBUTE_COLUMNS:
            _set_latest(customer_attribute_last_purchase_dates, key, transaction_date)


def _update_article_attribute_trend_features(
    article_attribute_purchase_counts_7d: dict[tuple[str, str], int],
    article_attribute_purchase_counts_30d: dict[tuple[str, str], int],
    article_attributes: Mapping[str, str],
    days_before_cutoff: int,
) -> None:
    if days_before_cutoff > 30:
        return
    for column, _ in ARTICLE_ATTRIBUTE_WINDOW_FEATURES:
        value = article_attributes.get(column, "")
        if not value:
            continue
        key = (column, value)
        if days_before_cutoff <= 7:
            _increment(article_attribute_purchase_counts_7d, key)
        _increment(article_attribute_purchase_counts_30d, key)


_WINDOW_ATTRIBUTE_COLUMNS = frozenset(column for column, _ in ARTICLE_ATTRIBUTE_WINDOW_FEATURES)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _mean_for(
    sums: Mapping[FeatureKey, float],
    counts: Mapping[FeatureKey, int],
    key: FeatureKey,
) -> float:
    count = counts.get(key, 0)
    if count <= 0:
        return 0.0
    return sums.get(key, 0.0) / count


def _share_for(
    numerators: Mapping[FeatureKey, int],
    denominators: Mapping[FeatureKey, int],
    key: FeatureKey,
) -> float:
    denominator = denominators.get(key, 0)
    if denominator <= 0:
        return 0.0
    return numerators.get(key, 0) / denominator


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _rate_delta(
    recent_count: int,
    context_count: int,
    recent_days: int,
    context_days: int,
) -> float:
    if context_count <= 0 or context_days <= 0:
        return 0.0
    return recent_count - (context_count * (recent_days / context_days))
