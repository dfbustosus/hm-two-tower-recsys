"""Leakage-safe descriptive EDA for the H&M dataset.

The report is computed in two streaming passes over ``transactions_train.csv``
plus one pass over each of the smaller H&M CSV files. The implementation is
deliberately stdlib-only so it can run on any project virtual environment.

Design rules:
    * Memory bounded: per-customer state stays integer-typed; the
      repeat-purchase rolling-window state purges entries older than
      ``REPEAT_PROXY_PURGE_DAYS`` so the dict size is proportional to a
      short window of transactions rather than the full history.
    * Leakage safe: every metric that is conditioned on a cutoff date uses
      strictly ``t < cutoff``. Descriptive (non-cutoff) metrics use the
      full dataset.
    * Deterministic: outputs are sorted; floats are rounded with
      ``round`` to the requested precision so JSON diffs are stable across
      runs on the same data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from hm_recsys.data.io import (
    TransactionEvent,
    iter_csv_rows,
    iter_transaction_events,
    iter_transactions,
    load_submission_customer_ids,
)

REPEAT_PROXY_PURGE_DAYS = 21
"""Sliding window kept in memory while detecting same-pair rapid repurchase."""

REPEAT_PROXY_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("within_2d", 0, 2),
    ("within_3_to_7d", 3, 7),
    ("within_8_to_14d", 8, 14),
)
"""Inter-purchase day-gap bins used for the returns-proxy histogram."""

HIERARCHY_COLUMNS: tuple[str, ...] = (
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
)
"""Article hierarchy columns reported in the fanout table."""

CUSTOMER_METADATA_COLUMNS: tuple[str, ...] = (
    "FN",
    "Active",
    "club_member_status",
    "fashion_news_frequency",
)
"""Customer categorical columns reported in the metadata distribution."""

DEFAULT_AGE_BUCKETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("missing", None, None),
    ("under_20", None, 20),
    ("20_to_24", 20, 25),
    ("25_to_29", 25, 30),
    ("30_to_39", 30, 40),
    ("40_to_49", 40, 50),
    ("50_to_59", 50, 60),
    ("60_to_69", 60, 70),
    ("70_plus", 70, None),
)
"""Inclusive-lower, exclusive-upper age buckets for customer reporting."""

DEFAULT_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90, 95, 99)
"""Percentiles reported for distributions (history depth, repeat counts, etc.)."""

DEFAULT_TOP_HIERARCHY_VALUES = 20
"""Maximum number of values reported per hierarchy column."""

DEFAULT_TOP_BUSY_DAYS = 30
"""Number of busiest daily transaction counts surfaced in the report."""

DEFAULT_ROLLING_CUTOFFS: tuple[str, ...] = (
    "2020-09-02",
    "2020-09-09",
    "2020-09-16",
)
"""Default rolling validation cutoffs used by the gap-closure plan."""

WEEKDAY_NAMES: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


@dataclass(frozen=True)
class EdaSegmentThresholds:
    """Cold/sparse/dense thresholds derived from history-depth distribution.

    Attributes:
        cold_max_transactions: Maximum transaction count for ``cold`` segment.
        sparse_max_transactions: Maximum transaction count for ``sparse``
            segment (inclusive). Dense customers exceed this threshold.
    """

    cold_max_transactions: int = 0
    sparse_max_transactions: int = 4


@dataclass(frozen=True)
class EdaReportConfig:
    """Configuration controlling EDA scope and bucketing.

    Attributes:
        rolling_cutoffs: ISO ``YYYY-MM-DD`` strings used for cold-user share.
        percentiles: Percentile breakpoints for distribution summaries.
        top_hierarchy_values: Maximum values surfaced per hierarchy column.
        top_busy_days: Number of busiest days surfaced.
        segment_thresholds: Cold/sparse/dense thresholds.
    """

    rolling_cutoffs: tuple[str, ...] = DEFAULT_ROLLING_CUTOFFS
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES
    top_hierarchy_values: int = DEFAULT_TOP_HIERARCHY_VALUES
    top_busy_days: int = DEFAULT_TOP_BUSY_DAYS
    segment_thresholds: EdaSegmentThresholds = field(default_factory=EdaSegmentThresholds)

    def __post_init__(self) -> None:
        """Validate configuration values."""

        if not self.rolling_cutoffs:
            raise ValueError("rolling_cutoffs must not be empty")
        for cutoff in self.rolling_cutoffs:
            date.fromisoformat(cutoff)
        for percentile in self.percentiles:
            if not 0 < percentile < 100:
                raise ValueError(f"percentile must be between 1 and 99, got {percentile}")
        if self.top_hierarchy_values <= 0:
            raise ValueError("top_hierarchy_values must be positive")
        if self.top_busy_days <= 0:
            raise ValueError("top_busy_days must be positive")
        if self.segment_thresholds.cold_max_transactions < 0:
            raise ValueError("cold_max_transactions must be non-negative")
        if (
            self.segment_thresholds.sparse_max_transactions
            < self.segment_thresholds.cold_max_transactions
        ):
            raise ValueError("sparse_max_transactions must be >= cold_max_transactions")


@dataclass(frozen=True)
class TransactionVolumeStats:
    """Aggregate transaction-volume statistics."""

    total_rows: int
    distinct_customers: int
    distinct_articles: int
    distinct_customer_article_pairs: int
    date_min: str
    date_max: str
    monthly_counts: dict[str, int]
    weekday_counts: dict[str, int]
    top_busy_days: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ChannelStats:
    """Sales-channel distribution and per-customer channel-mix summary."""

    rows_by_channel: dict[str, int]
    rows_by_channel_share: dict[str, float]
    customers_with_observed_channel_count: int
    customers_online_dominant: int
    customers_store_dominant: int
    customers_mixed: int
    customer_online_share_percentiles: dict[str, float]


@dataclass(frozen=True)
class HistoryDepthStats:
    """Per-customer history-depth (transaction count) distribution."""

    customers_with_any_history: int
    submission_customers: int
    submission_customers_with_any_history: int
    transaction_count_mean: float
    transaction_count_percentiles: dict[str, int]
    segment_thresholds: EdaSegmentThresholds
    submission_segment_counts: dict[str, int]
    cold_user_share_by_cutoff: dict[str, float]
    cold_customer_counts_by_cutoff: dict[str, int]


@dataclass(frozen=True)
class ArticleStats:
    """Article-level descriptive statistics."""

    total_articles_in_articles_csv: int
    articles_with_at_least_one_transaction: int
    article_age_days_percentiles: dict[str, int]
    hierarchy_distinct_value_counts: dict[str, int]
    hierarchy_top_values: dict[str, tuple[tuple[str, int], ...]]


@dataclass(frozen=True)
class RepeatPurchaseStats:
    """Repeat-purchase structure and rapid same-pair returns proxy."""

    distinct_customer_article_pairs: int
    pairs_with_repeat: int
    pairs_with_repeat_share: float
    repeat_count_percentiles: dict[str, int]
    same_pair_returns_proxy: dict[str, int]
    returns_proxy_purge_window_days: int


@dataclass(frozen=True)
class CustomerMetadataStats:
    """Customer-metadata categorical distributions."""

    total_customers_in_customers_csv: int
    age_buckets: dict[str, int]
    categorical_distributions: dict[str, dict[str, int]]


@dataclass(frozen=True)
class EdaReport:
    """Top-level EDA report combining all per-section results."""

    generated_at_utc: str
    raw_data_dir: str
    config: EdaReportConfig
    transactions: TransactionVolumeStats
    channels: ChannelStats
    customers: HistoryDepthStats
    articles: ArticleStats
    repeat_purchase: RepeatPurchaseStats
    customer_metadata: CustomerMetadataStats


# ---------------------------------------------------------------------------
# Streaming aggregators


class _TransactionStreamingAggregator:
    """Single-pass aggregator over ``transactions_train.csv``.

    The aggregator keeps memory bounded by:
        * storing only integer counters per customer/article,
        * purging the same-pair last-date dictionary on each new day so the
          dict size is proportional to ``REPEAT_PROXY_PURGE_DAYS`` of
          activity rather than the full history.
    """

    def __init__(self, rolling_cutoffs: Sequence[str]) -> None:
        self._cutoffs: tuple[tuple[str, date], ...] = tuple(
            (raw, date.fromisoformat(raw)) for raw in rolling_cutoffs
        )
        self.total_rows: int = 0
        self.distinct_pairs: set[tuple[str, str]] = set()
        self.date_min: date | None = None
        self.date_max: date | None = None
        self.daily_counts: dict[date, int] = {}
        self.monthly_counts: dict[str, int] = {}
        self.weekday_counts: dict[int, int] = dict.fromkeys(range(7), 0)
        self.rows_by_channel: dict[str, int] = {}
        self.customer_transaction_counts: dict[str, int] = {}
        self.customer_article_counts: dict[str, int] = {}
        self.customer_channel_counts: dict[str, list[int]] = {}
        self.article_transaction_counts: dict[str, int] = {}
        self.article_first_date: dict[str, date] = {}
        self.article_last_date: dict[str, date] = {}
        self.cold_customers_seen: dict[str, set[str]] = {raw: set() for raw, _ in self._cutoffs}
        self.pair_repeat_counts: dict[tuple[str, str], int] = {}
        self.same_pair_returns_proxy: dict[str, int] = {
            label: 0 for label, _, _ in REPEAT_PROXY_BUCKETS
        }
        self._recent_pair_last_date: dict[tuple[str, str], date] = {}
        self._recent_purge_cursor: date | None = None

    def consume(
        self,
        event_date: date,
        customer_id: str,
        article_id: str,
        channel_id: int | None,
    ) -> None:
        """Update aggregators with a single transaction row.

        Args:
            event_date: Transaction date parsed from ``t_dat``.
            customer_id: H&M customer identifier preserved as a string.
            article_id: H&M article identifier preserved as a string.
            channel_id: Integer sales channel (1 or 2) or ``None`` if unknown.
        """

        self.total_rows += 1
        if self.date_min is None or event_date < self.date_min:
            self.date_min = event_date
        if self.date_max is None or event_date > self.date_max:
            self.date_max = event_date

        self.daily_counts[event_date] = self.daily_counts.get(event_date, 0) + 1
        month_key = f"{event_date.year:04d}-{event_date.month:02d}"
        self.monthly_counts[month_key] = self.monthly_counts.get(month_key, 0) + 1
        self.weekday_counts[event_date.weekday()] += 1

        if channel_id is not None:
            key = str(channel_id)
            self.rows_by_channel[key] = self.rows_by_channel.get(key, 0) + 1
            channel_index = 0 if channel_id == 1 else 1
            channel_history = self.customer_channel_counts.get(customer_id)
            if channel_history is None:
                channel_history = [0, 0]
                self.customer_channel_counts[customer_id] = channel_history
            channel_history[channel_index] += 1

        self.customer_transaction_counts[customer_id] = (
            self.customer_transaction_counts.get(customer_id, 0) + 1
        )
        self.customer_article_counts[customer_id] = (
            self.customer_article_counts.get(customer_id, 0) + 1
        )

        self.article_transaction_counts[article_id] = (
            self.article_transaction_counts.get(article_id, 0) + 1
        )
        if article_id not in self.article_first_date:
            self.article_first_date[article_id] = event_date
            self.article_last_date[article_id] = event_date
        else:
            if event_date < self.article_first_date[article_id]:
                self.article_first_date[article_id] = event_date
            if event_date > self.article_last_date[article_id]:
                self.article_last_date[article_id] = event_date

        for raw_cutoff, cutoff_date in self._cutoffs:
            if event_date < cutoff_date:
                self.cold_customers_seen[raw_cutoff].add(customer_id)

        pair = (customer_id, article_id)
        self.distinct_pairs.add(pair)
        self.pair_repeat_counts[pair] = self.pair_repeat_counts.get(pair, 0) + 1

        last_seen = self._recent_pair_last_date.get(pair)
        if last_seen is not None:
            gap_days = (event_date - last_seen).days
            for label, lower, upper in REPEAT_PROXY_BUCKETS:
                if lower <= gap_days <= upper:
                    self.same_pair_returns_proxy[label] += 1
                    break
        self._recent_pair_last_date[pair] = event_date
        self._purge_recent_pairs(event_date)

    def _purge_recent_pairs(self, event_date: date) -> None:
        """Drop recent-pair entries older than the purge window.

        Args:
            event_date: Current row's transaction date used to bound the
                rolling window.
        """

        if self._recent_purge_cursor == event_date:
            return
        self._recent_purge_cursor = event_date
        threshold_days = REPEAT_PROXY_PURGE_DAYS
        stale = [
            pair
            for pair, last_seen in self._recent_pair_last_date.items()
            if (event_date - last_seen).days > threshold_days
        ]
        for pair in stale:
            self._recent_pair_last_date.pop(pair, None)


# ---------------------------------------------------------------------------
# Report assembly


def build_eda_report(
    raw_data_dir: Path | str,
    config: EdaReportConfig | None = None,
) -> EdaReport:
    """Build an EDA report from the raw H&M data directory.

    Args:
        raw_data_dir: Directory containing the unmodified Kaggle CSV files.
        config: Optional configuration overriding the defaults.

    Returns:
        Fully populated :class:`EdaReport`.
    """

    config = config or EdaReportConfig()
    raw_dir = Path(raw_data_dir).expanduser().resolve()
    aggregator = _aggregate_transactions(raw_dir, config)
    submission_customer_ids = load_submission_customer_ids(raw_dir)
    transactions = _build_transaction_stats(aggregator, top_busy_days=config.top_busy_days)
    channels = _build_channel_stats(aggregator, percentiles=config.percentiles)
    customers = _build_history_depth_stats(
        aggregator,
        submission_customer_ids=submission_customer_ids,
        percentiles=config.percentiles,
        thresholds=config.segment_thresholds,
    )
    articles = _build_article_stats(
        raw_dir=raw_dir,
        aggregator=aggregator,
        percentiles=config.percentiles,
        top_values=config.top_hierarchy_values,
    )
    repeat_purchase = _build_repeat_purchase_stats(aggregator, percentiles=config.percentiles)
    customer_metadata = _build_customer_metadata_stats(raw_dir)
    return EdaReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_data_dir=str(raw_dir),
        config=config,
        transactions=transactions,
        channels=channels,
        customers=customers,
        articles=articles,
        repeat_purchase=repeat_purchase,
        customer_metadata=customer_metadata,
    )


def _aggregate_transactions(
    raw_dir: Path,
    config: EdaReportConfig,
) -> _TransactionStreamingAggregator:
    """Run the single streaming pass over ``transactions_train.csv``."""

    aggregator = _TransactionStreamingAggregator(rolling_cutoffs=config.rolling_cutoffs)
    for record in iter_transactions(raw_dir):
        aggregator.consume(
            event_date=record.t_dat,
            customer_id=record.customer_id,
            article_id=record.article_id,
            channel_id=record.sales_channel_id,
        )
    return aggregator


def build_eda_report_from_events(
    events: Iterable[TransactionEvent],
    submission_customer_ids: Iterable[str],
    customer_metadata_rows: Iterable[Mapping[str, str]] | None = None,
    article_metadata_rows: Iterable[Mapping[str, str]] | None = None,
    total_articles_in_articles_csv: int | None = None,
    total_customers_in_customers_csv: int | None = None,
    config: EdaReportConfig | None = None,
    raw_data_dir: str = "synthetic://",
    channel_id_provider: Mapping[tuple[str, str, date], int] | None = None,
) -> EdaReport:
    """Build an EDA report from in-memory transaction events.

    This helper is provided for tests and for callers that already have
    transactions in memory. ``channel_id_provider`` lets the caller attach
    sales-channel identifiers because :class:`TransactionEvent` does not
    carry them.

    Args:
        events: Iterable of transaction events.
        submission_customer_ids: Submission-set customer universe.
        customer_metadata_rows: Optional iterable of ``customers.csv`` rows.
        article_metadata_rows: Optional iterable of ``articles.csv`` rows.
        total_articles_in_articles_csv: Optional total articles count
            overriding the deduced size when ``article_metadata_rows`` is
            ``None``.
        total_customers_in_customers_csv: Optional total customers count
            overriding the deduced size when ``customer_metadata_rows`` is
            ``None``.
        config: Optional configuration overriding the defaults.
        raw_data_dir: Identifier surfaced in the report for traceability.
        channel_id_provider: Mapping from ``(customer_id, article_id, date)``
            to channel id. Missing keys are treated as ``None``.

    Returns:
        Fully populated :class:`EdaReport`.
    """

    config = config or EdaReportConfig()
    aggregator = _TransactionStreamingAggregator(rolling_cutoffs=config.rolling_cutoffs)
    for event in events:
        channel_id: int | None = None
        if channel_id_provider is not None:
            channel_id = channel_id_provider.get((event.customer_id, event.article_id, event.t_dat))
        aggregator.consume(
            event_date=event.t_dat,
            customer_id=event.customer_id,
            article_id=event.article_id,
            channel_id=channel_id,
        )

    submission_set = set(submission_customer_ids)
    transactions = _build_transaction_stats(aggregator, top_busy_days=config.top_busy_days)
    channels = _build_channel_stats(aggregator, percentiles=config.percentiles)
    customers = _build_history_depth_stats(
        aggregator,
        submission_customer_ids=submission_set,
        percentiles=config.percentiles,
        thresholds=config.segment_thresholds,
    )
    articles = _build_article_stats_from_rows(
        rows=article_metadata_rows or (),
        aggregator=aggregator,
        percentiles=config.percentiles,
        top_values=config.top_hierarchy_values,
        total_articles_override=total_articles_in_articles_csv,
    )
    repeat_purchase = _build_repeat_purchase_stats(aggregator, percentiles=config.percentiles)
    customer_metadata = _build_customer_metadata_stats_from_rows(
        rows=customer_metadata_rows or (),
        total_customers_override=total_customers_in_customers_csv,
    )
    return EdaReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_data_dir=raw_data_dir,
        config=config,
        transactions=transactions,
        channels=channels,
        customers=customers,
        articles=articles,
        repeat_purchase=repeat_purchase,
        customer_metadata=customer_metadata,
    )


def _build_transaction_stats(
    aggregator: _TransactionStreamingAggregator,
    top_busy_days: int,
) -> TransactionVolumeStats:
    """Convert raw aggregator state into the transaction-volume section."""

    sorted_monthly = dict(sorted(aggregator.monthly_counts.items()))
    weekday_named = {WEEKDAY_NAMES[index]: aggregator.weekday_counts[index] for index in range(7)}
    busiest = sorted(aggregator.daily_counts.items(), key=lambda item: (-item[1], item[0]))[
        :top_busy_days
    ]
    busiest_serialized = tuple((event_date.isoformat(), count) for event_date, count in busiest)
    return TransactionVolumeStats(
        total_rows=aggregator.total_rows,
        distinct_customers=len(aggregator.customer_transaction_counts),
        distinct_articles=len(aggregator.article_transaction_counts),
        distinct_customer_article_pairs=len(aggregator.distinct_pairs),
        date_min=aggregator.date_min.isoformat() if aggregator.date_min else "",
        date_max=aggregator.date_max.isoformat() if aggregator.date_max else "",
        monthly_counts=sorted_monthly,
        weekday_counts=weekday_named,
        top_busy_days=busiest_serialized,
    )


def _build_channel_stats(
    aggregator: _TransactionStreamingAggregator,
    percentiles: tuple[int, ...],
) -> ChannelStats:
    """Convert raw aggregator state into the channel-mix section."""

    total = sum(aggregator.rows_by_channel.values())
    shares: dict[str, float] = {}
    if total > 0:
        shares = {
            channel: round(count / total, 6)
            for channel, count in aggregator.rows_by_channel.items()
        }
    online_dominant = 0
    store_dominant = 0
    mixed = 0
    online_shares: list[float] = []
    for _customer_id, counts in aggregator.customer_channel_counts.items():
        observed = counts[0] + counts[1]
        if observed == 0:
            continue
        online_share = counts[1] / observed
        online_shares.append(online_share)
        if online_share >= 0.8:
            online_dominant += 1
        elif online_share <= 0.2:
            store_dominant += 1
        else:
            mixed += 1
    return ChannelStats(
        rows_by_channel=dict(sorted(aggregator.rows_by_channel.items())),
        rows_by_channel_share=dict(sorted(shares.items())),
        customers_with_observed_channel_count=len(online_shares),
        customers_online_dominant=online_dominant,
        customers_store_dominant=store_dominant,
        customers_mixed=mixed,
        customer_online_share_percentiles=_float_percentiles(online_shares, percentiles),
    )


def _build_history_depth_stats(
    aggregator: _TransactionStreamingAggregator,
    submission_customer_ids: set[str],
    percentiles: tuple[int, ...],
    thresholds: EdaSegmentThresholds,
) -> HistoryDepthStats:
    """Convert raw aggregator state into the customer-history section."""

    counts = list(aggregator.customer_transaction_counts.values())
    submission_total = len(submission_customer_ids)
    transaction_counts = aggregator.customer_transaction_counts
    submission_with_history = sum(
        1 for customer_id in submission_customer_ids if customer_id in transaction_counts
    )
    cold_count = sum(
        1
        for customer_id in submission_customer_ids
        if aggregator.customer_transaction_counts.get(customer_id, 0)
        <= thresholds.cold_max_transactions
    )
    sparse_count = sum(
        1
        for customer_id in submission_customer_ids
        if thresholds.cold_max_transactions
        < aggregator.customer_transaction_counts.get(customer_id, 0)
        <= thresholds.sparse_max_transactions
    )
    dense_count = submission_total - cold_count - sparse_count
    cold_share_by_cutoff: dict[str, float] = {}
    cold_count_by_cutoff: dict[str, int] = {}
    for cutoff_raw, seen_customers in aggregator.cold_customers_seen.items():
        cold_for_cutoff = sum(
            1 for customer_id in submission_customer_ids if customer_id not in seen_customers
        )
        cold_count_by_cutoff[cutoff_raw] = cold_for_cutoff
        cold_share_by_cutoff[cutoff_raw] = (
            round(cold_for_cutoff / submission_total, 6) if submission_total else 0.0
        )
    mean_count = round(sum(counts) / len(counts), 4) if counts else 0.0
    return HistoryDepthStats(
        customers_with_any_history=len(counts),
        submission_customers=submission_total,
        submission_customers_with_any_history=submission_with_history,
        transaction_count_mean=mean_count,
        transaction_count_percentiles=_integer_percentiles(counts, percentiles),
        segment_thresholds=thresholds,
        submission_segment_counts={
            "cold": cold_count,
            "sparse": sparse_count,
            "dense": dense_count,
        },
        cold_user_share_by_cutoff=dict(sorted(cold_share_by_cutoff.items())),
        cold_customer_counts_by_cutoff=dict(sorted(cold_count_by_cutoff.items())),
    )


def _build_article_stats(
    raw_dir: Path,
    aggregator: _TransactionStreamingAggregator,
    percentiles: tuple[int, ...],
    top_values: int,
) -> ArticleStats:
    """Build the article-statistics section by streaming ``articles.csv``."""

    rows = iter_csv_rows(
        raw_dir / "articles.csv",
        required_columns=("article_id", *HIERARCHY_COLUMNS),
    )
    return _build_article_stats_from_rows(
        rows=rows,
        aggregator=aggregator,
        percentiles=percentiles,
        top_values=top_values,
        total_articles_override=None,
    )


def _build_article_stats_from_rows(
    rows: Iterable[Mapping[str, str]],
    aggregator: _TransactionStreamingAggregator,
    percentiles: tuple[int, ...],
    top_values: int,
    total_articles_override: int | None,
) -> ArticleStats:
    """Build article statistics from an iterable of ``articles.csv`` rows."""

    distinct_values: dict[str, set[str]] = {column: set() for column in HIERARCHY_COLUMNS}
    counts: dict[str, dict[str, int]] = {column: {} for column in HIERARCHY_COLUMNS}
    total_articles = 0
    for row in rows:
        total_articles += 1
        for column in HIERARCHY_COLUMNS:
            value = row.get(column, "")
            if value == "":
                continue
            distinct_values[column].add(value)
            counts[column][value] = counts[column].get(value, 0) + 1
    if total_articles_override is not None:
        total_articles = total_articles_override
    age_days = [
        (last_date - first_date).days
        for article_id, first_date in aggregator.article_first_date.items()
        for last_date in (aggregator.article_last_date[article_id],)
    ]
    top_per_column = {
        column: _top_values(value_counts, top_values) for column, value_counts in counts.items()
    }
    distinct_counts = {column: len(values) for column, values in distinct_values.items()}
    return ArticleStats(
        total_articles_in_articles_csv=total_articles,
        articles_with_at_least_one_transaction=len(aggregator.article_transaction_counts),
        article_age_days_percentiles=_integer_percentiles(age_days, percentiles),
        hierarchy_distinct_value_counts=dict(sorted(distinct_counts.items())),
        hierarchy_top_values=dict(sorted(top_per_column.items())),
    )


def _build_repeat_purchase_stats(
    aggregator: _TransactionStreamingAggregator,
    percentiles: tuple[int, ...],
) -> RepeatPurchaseStats:
    """Convert raw aggregator state into the repeat-purchase section."""

    pair_counts = list(aggregator.pair_repeat_counts.values())
    pairs_total = len(pair_counts)
    pairs_with_repeat = sum(1 for count in pair_counts if count >= 2)
    share = round(pairs_with_repeat / pairs_total, 6) if pairs_total else 0.0
    return RepeatPurchaseStats(
        distinct_customer_article_pairs=pairs_total,
        pairs_with_repeat=pairs_with_repeat,
        pairs_with_repeat_share=share,
        repeat_count_percentiles=_integer_percentiles(pair_counts, percentiles),
        same_pair_returns_proxy=dict(sorted(aggregator.same_pair_returns_proxy.items())),
        returns_proxy_purge_window_days=REPEAT_PROXY_PURGE_DAYS,
    )


def _build_customer_metadata_stats(raw_dir: Path) -> CustomerMetadataStats:
    """Build the customer-metadata section by streaming ``customers.csv``."""

    rows = iter_csv_rows(
        raw_dir / "customers.csv",
        required_columns=("customer_id", "age", *CUSTOMER_METADATA_COLUMNS),
    )
    return _build_customer_metadata_stats_from_rows(rows=rows, total_customers_override=None)


def _build_customer_metadata_stats_from_rows(
    rows: Iterable[Mapping[str, str]],
    total_customers_override: int | None,
) -> CustomerMetadataStats:
    """Build customer-metadata statistics from ``customers.csv`` rows."""

    age_bucket_counts: dict[str, int] = {label: 0 for label, _, _ in DEFAULT_AGE_BUCKETS}
    categorical_counts: dict[str, dict[str, int]] = {
        column: {} for column in CUSTOMER_METADATA_COLUMNS
    }
    total = 0
    for index, row in enumerate(rows, start=1):
        total = index
        age_bucket = _assign_age_bucket(row.get("age", ""))
        age_bucket_counts[age_bucket] += 1
        for column in CUSTOMER_METADATA_COLUMNS:
            raw_value = row.get(column, "")
            value = raw_value if raw_value != "" else "missing"
            categorical_counts[column][value] = categorical_counts[column].get(value, 0) + 1
    if total_customers_override is not None:
        total = total_customers_override
    sorted_categoricals = {
        column: dict(sorted(values.items()))
        for column, values in sorted(categorical_counts.items())
    }
    return CustomerMetadataStats(
        total_customers_in_customers_csv=total,
        age_buckets=age_bucket_counts,
        categorical_distributions=sorted_categoricals,
    )


# ---------------------------------------------------------------------------
# Utility functions


def _assign_age_bucket(raw_age: str) -> str:
    """Return the bucket label for a raw age string."""

    if raw_age == "":
        return "missing"
    try:
        age = int(raw_age)
    except ValueError:
        try:
            age = int(float(raw_age))
        except ValueError:
            return "missing"
    for label, lower, upper in DEFAULT_AGE_BUCKETS:
        if label == "missing":
            continue
        if lower is None and upper is not None and age < upper:
            return label
        if lower is not None and upper is not None and lower <= age < upper:
            return label
        if lower is not None and upper is None and age >= lower:
            return label
    return "missing"


def _integer_percentiles(values: Sequence[int], percentiles: tuple[int, ...]) -> dict[str, int]:
    """Compute integer percentile breakpoints, plus min and max."""

    if not values:
        labels = ["min", *(f"p{p}" for p in percentiles), "max"]
        return dict.fromkeys(labels, 0)
    sorted_values = sorted(values)
    n = len(sorted_values)
    result: dict[str, int] = {"min": sorted_values[0]}
    for percentile in percentiles:
        rank = max(0, min(n - 1, round(percentile / 100 * (n - 1))))
        result[f"p{percentile}"] = sorted_values[rank]
    result["max"] = sorted_values[-1]
    return result


def _float_percentiles(values: Sequence[float], percentiles: tuple[int, ...]) -> dict[str, float]:
    """Compute float percentile breakpoints with deterministic rounding."""

    if not values:
        labels = ["min", *(f"p{p}" for p in percentiles), "max"]
        return dict.fromkeys(labels, 0.0)
    sorted_values = sorted(values)
    n = len(sorted_values)
    result: dict[str, float] = {"min": round(sorted_values[0], 6)}
    for percentile in percentiles:
        rank = max(0, min(n - 1, round(percentile / 100 * (n - 1))))
        result[f"p{percentile}"] = round(sorted_values[rank], 6)
    result["max"] = round(sorted_values[-1], 6)
    return result


def _top_values(value_counts: Mapping[str, int], limit: int) -> tuple[tuple[str, int], ...]:
    """Return the top ``limit`` value/count pairs sorted by descending count."""

    sorted_items = sorted(value_counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(sorted_items[:limit])


# ---------------------------------------------------------------------------
# Serialization


def eda_report_to_dict(report: EdaReport) -> dict[str, Any]:
    """Convert an :class:`EdaReport` into JSON-serializable primitives.

    Args:
        report: Report object to serialize.

    Returns:
        Dictionary suitable for ``json.dumps``.
    """

    data = asdict(report)
    transactions = data["transactions"]
    transactions["top_busy_days"] = [
        [event_date, count] for event_date, count in transactions["top_busy_days"]
    ]
    articles = data["articles"]
    articles["hierarchy_top_values"] = {
        column: [[value, count] for value, count in entries]
        for column, entries in articles["hierarchy_top_values"].items()
    }
    return data


def write_eda_report(report: EdaReport, path: Path | str) -> Path:
    """Write an EDA report as deterministic JSON.

    Args:
        report: Report object to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(eda_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def render_eda_report_markdown(report: EdaReport) -> str:
    """Render an EDA report as a human-readable Markdown summary.

    Args:
        report: Report object to render.

    Returns:
        Markdown string.
    """

    lines: list[str] = []
    lines.append("# H&M Exploratory Data Analysis (EDA)")
    lines.append("")
    lines.append(f"- Generated: `{report.generated_at_utc}`")
    lines.append(f"- Raw data: `{report.raw_data_dir}`")
    lines.append(
        f"- Rolling cutoffs: {', '.join(f'`{cutoff}`' for cutoff in report.config.rolling_cutoffs)}"
    )
    lines.append("")

    transactions = report.transactions
    lines.append("## Transaction volume")
    lines.append("")
    lines.append(f"- Total rows: {transactions.total_rows:,}")
    lines.append(f"- Distinct customers: {transactions.distinct_customers:,}")
    lines.append(f"- Distinct articles: {transactions.distinct_articles:,}")
    lines.append(
        f"- Distinct (customer, article) pairs: {transactions.distinct_customer_article_pairs:,}"
    )
    lines.append(f"- Date range: `{transactions.date_min}` to `{transactions.date_max}`")
    lines.append("")
    lines.append("### Monthly volume")
    lines.append("")
    lines.append("| Month | Rows |")
    lines.append("| --- | ---: |")
    for month, count in transactions.monthly_counts.items():
        lines.append(f"| {month} | {count:,} |")
    lines.append("")
    lines.append("### Weekday volume")
    lines.append("")
    lines.append("| Weekday | Rows |")
    lines.append("| --- | ---: |")
    for weekday, count in transactions.weekday_counts.items():
        lines.append(f"| {weekday} | {count:,} |")
    lines.append("")
    lines.append(f"### Busiest days (top {len(transactions.top_busy_days)})")
    lines.append("")
    lines.append("| Date | Rows |")
    lines.append("| --- | ---: |")
    for day, count in transactions.top_busy_days:
        lines.append(f"| {day} | {count:,} |")
    lines.append("")

    channels = report.channels
    lines.append("## Sales channels")
    lines.append("")
    lines.append("| Channel | Rows | Share |")
    lines.append("| --- | ---: | ---: |")
    for channel, count in channels.rows_by_channel.items():
        share = channels.rows_by_channel_share.get(channel, 0.0)
        lines.append(f"| {channel} | {count:,} | {share:.3f} |")
    lines.append("")
    lines.append(
        f"- Customers with at least one observed channel: "
        f"{channels.customers_with_observed_channel_count:,}"
    )
    lines.append(f"- Online-dominant (>=80% channel 2): {channels.customers_online_dominant:,}")
    lines.append(f"- Store-dominant (>=80% channel 1): {channels.customers_store_dominant:,}")
    lines.append(f"- Mixed channels (20-80% online): {channels.customers_mixed:,}")
    lines.append("")
    lines.append("### Customer online-channel-share percentiles")
    lines.append("")
    for label, value in channels.customer_online_share_percentiles.items():
        lines.append(f"- {label}: {value:.3f}")
    lines.append("")

    customers = report.customers
    lines.append("## Customer history depth")
    lines.append("")
    lines.append(
        "- Customers with any pre-cutoff history: " f"{customers.customers_with_any_history:,}"
    )
    lines.append(f"- Submission customers: {customers.submission_customers:,}")
    lines.append(
        f"- Submission customers with any history: "
        f"{customers.submission_customers_with_any_history:,}"
    )
    lines.append(f"- Mean transactions per customer: {customers.transaction_count_mean:.2f}")
    lines.append("")
    lines.append("### Transaction-count percentiles (all customers with history)")
    lines.append("")
    for label, value in customers.transaction_count_percentiles.items():
        lines.append(f"- {label}: {value}")
    lines.append("")
    lines.append("### Submission-set segmentation")
    lines.append("")
    cold_max = customers.segment_thresholds.cold_max_transactions
    sparse_max = customers.segment_thresholds.sparse_max_transactions
    cold_count_total = customers.submission_segment_counts.get("cold", 0)
    sparse_count_total = customers.submission_segment_counts.get("sparse", 0)
    dense_count_total = customers.submission_segment_counts.get("dense", 0)
    lines.append(f"- Cold (<= {cold_max} transactions): {cold_count_total:,}")
    lines.append(f"- Sparse ({cold_max + 1}-{sparse_max} transactions): {sparse_count_total:,}")
    lines.append(f"- Dense (> {sparse_max} transactions): {dense_count_total:,}")
    lines.append("")
    lines.append("### Cold-user share by rolling cutoff")
    lines.append("")
    lines.append("| Cutoff | Cold customers | Cold share |")
    lines.append("| --- | ---: | ---: |")
    for cutoff, share in customers.cold_user_share_by_cutoff.items():
        count = customers.cold_customer_counts_by_cutoff.get(cutoff, 0)
        lines.append(f"| {cutoff} | {count:,} | {share:.4f} |")
    lines.append("")

    articles = report.articles
    lines.append("## Article catalog")
    lines.append("")
    lines.append(f"- Total articles in `articles.csv`: {articles.total_articles_in_articles_csv:,}")
    lines.append(
        f"- Articles with >= 1 transaction: {articles.articles_with_at_least_one_transaction:,}"
    )
    lines.append("")
    lines.append("### Article active span (days) percentiles")
    lines.append("")
    for label, value in articles.article_age_days_percentiles.items():
        lines.append(f"- {label}: {value}")
    lines.append("")
    lines.append("### Hierarchy fanout (distinct values per column)")
    lines.append("")
    lines.append("| Column | Distinct values |")
    lines.append("| --- | ---: |")
    for column, count in articles.hierarchy_distinct_value_counts.items():
        lines.append(f"| {column} | {count:,} |")
    lines.append("")
    lines.append(f"### Top hierarchy values per column (top {report.config.top_hierarchy_values})")
    lines.append("")
    for column, entries in articles.hierarchy_top_values.items():
        lines.append(f"**{column}**")
        lines.append("")
        lines.append("| Value | Articles |")
        lines.append("| --- | ---: |")
        for hierarchy_value, hierarchy_count in entries:
            lines.append(f"| {hierarchy_value} | {hierarchy_count:,} |")
        lines.append("")

    repeat = report.repeat_purchase
    lines.append("## Repeat purchases")
    lines.append("")
    lines.append(
        f"- Distinct (customer, article) pairs: " f"{repeat.distinct_customer_article_pairs:,}"
    )
    lines.append(
        f"- Pairs purchased >= 2 times: {repeat.pairs_with_repeat:,} "
        f"({repeat.pairs_with_repeat_share:.4f} share)"
    )
    lines.append("")
    lines.append("### Per-pair purchase-count percentiles")
    lines.append("")
    for label, value in repeat.repeat_count_percentiles.items():
        lines.append(f"- {label}: {value}")
    lines.append("")
    lines.append(
        "### Same-pair rapid re-purchase proxy (within rolling window of "
        f"{repeat.returns_proxy_purge_window_days} days)"
    )
    lines.append("")
    for label, count in repeat.same_pair_returns_proxy.items():
        lines.append(f"- {label}: {count:,}")
    lines.append("")

    metadata = report.customer_metadata
    lines.append("## Customer metadata")
    lines.append("")
    lines.append(
        f"- Total customers in `customers.csv`: {metadata.total_customers_in_customers_csv:,}"
    )
    lines.append("")
    lines.append("### Age buckets")
    lines.append("")
    lines.append("| Bucket | Customers |")
    lines.append("| --- | ---: |")
    for bucket, count in metadata.age_buckets.items():
        lines.append(f"| {bucket} | {count:,} |")
    lines.append("")
    for column, distribution in metadata.categorical_distributions.items():
        lines.append(f"### {column}")
        lines.append("")
        lines.append("| Value | Customers |")
        lines.append("| --- | ---: |")
        for metadata_value, metadata_count in distribution.items():
            lines.append(f"| {metadata_value} | {metadata_count:,} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_eda_report_markdown(report: EdaReport, path: Path | str) -> Path:
    """Write the human-readable Markdown summary to disk.

    Args:
        report: Report object to render.
        path: Destination Markdown path.

    Returns:
        Resolved path written to disk.
    """

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_eda_report_markdown(report), encoding="utf-8")
    return output_path


def iter_transaction_events_for_eda(raw_data_dir: Path | str) -> Iterable[TransactionEvent]:
    """Expose the project's existing transaction iterator for EDA callers."""

    return iter_transaction_events(raw_data_dir)
