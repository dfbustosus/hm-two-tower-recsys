"""Temporal validation splitting and label collection utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from hm_recsys.data.io import TransactionEvent

SplitBucket = Literal["train", "validation", "future"]


@dataclass(frozen=True)
class TemporalSplit:
    """Cutoff-based next-window split for H&M-style validation.

    Attributes:
        cutoff: First date included in the validation target window.
        horizon_days: Number of days in the validation target window.
    """

    cutoff: date
    horizon_days: int = 7

    def __post_init__(self) -> None:
        """Validate split parameters after dataclass initialization.

        Raises:
            ValueError: If ``horizon_days`` is not positive.
        """

        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")

    @classmethod
    def from_isoformat(cls, cutoff: str, horizon_days: int = 7) -> TemporalSplit:
        """Create a split from an ISO ``YYYY-MM-DD`` cutoff string.

        Args:
            cutoff: Validation start date as an ISO date string.
            horizon_days: Number of days in the validation window.

        Returns:
            Parsed ``TemporalSplit`` instance.
        """

        return cls(cutoff=date.fromisoformat(cutoff), horizon_days=horizon_days)

    @property
    def validation_end(self) -> date:
        """Return the exclusive validation-window end date.

        Returns:
            ``cutoff + horizon_days`` as a ``date``.
        """

        return self.cutoff + timedelta(days=self.horizon_days)


@dataclass(frozen=True)
class TemporalSplitSummary:
    """Aggregate row, customer, article, and label counts for a split.

    Attributes mirror the serialized JSON report emitted by the temporal split
    CLI command.
    """

    cutoff: str
    validation_start: str
    validation_end_exclusive: str
    horizon_days: int
    train_rows: int
    validation_rows: int
    future_rows: int
    train_customers: int
    validation_customers: int
    future_customers: int
    train_articles: int
    validation_articles: int
    future_articles: int
    validation_label_customers: int
    validation_unique_customer_article_pairs: int


@dataclass(frozen=True)
class TemporalValidationData:
    """Temporal split summary plus validation labels.

    Attributes:
        summary: Aggregate counts for train, validation, and future buckets.
        validation_labels: Mapping of customer ID to unique validation articles in
            first-seen order.
    """

    summary: TemporalSplitSummary
    validation_labels: dict[str, tuple[str, ...]]


def assign_split_bucket(transaction_date: date, split: TemporalSplit) -> SplitBucket:
    """Assign a transaction date to train, validation, or future.

    Args:
        transaction_date: Purchase date to classify.
        split: Cutoff and horizon definition.

    Returns:
        ``"train"`` for dates before cutoff, ``"validation"`` for dates in the
        target window, and ``"future"`` for dates after the target window.
    """

    if transaction_date < split.cutoff:
        return "train"
    if transaction_date < split.validation_end:
        return "validation"
    return "future"


def summarize_temporal_split(
    transactions: Iterable[TransactionEvent], split: TemporalSplit
) -> TemporalSplitSummary:
    """Summarize a temporal split without returning labels.

    Args:
        transactions: Transaction events to bucket by date.
        split: Cutoff and horizon definition.

    Returns:
        Aggregate split summary.
    """

    return summarize_temporal_split_with_labels(transactions, split).summary


def summarize_temporal_split_with_labels(
    transactions: Iterable[TransactionEvent], split: TemporalSplit
) -> TemporalValidationData:
    """Summarize a split and collect unique validation target labels.

    Args:
        transactions: Transaction events to bucket by date.
        split: Cutoff and horizon definition.

    Returns:
        Split summary plus customer-to-article validation labels.
    """

    row_counts = {"train": 0, "validation": 0, "future": 0}
    customers: dict[SplitBucket, set[str]] = {
        "train": set(),
        "validation": set(),
        "future": set(),
    }
    articles: dict[SplitBucket, set[str]] = {
        "train": set(),
        "validation": set(),
        "future": set(),
    }
    validation_pairs: set[tuple[str, str]] = set()
    labels: dict[str, list[str]] = defaultdict(list)
    seen_labels: dict[str, set[str]] = defaultdict(set)

    for transaction in transactions:
        bucket = assign_split_bucket(transaction.t_dat, split)
        row_counts[bucket] += 1
        customers[bucket].add(transaction.customer_id)
        articles[bucket].add(transaction.article_id)
        if bucket == "validation":
            validation_pairs.add((transaction.customer_id, transaction.article_id))
            customer_seen = seen_labels[transaction.customer_id]
            if transaction.article_id not in customer_seen:
                labels[transaction.customer_id].append(transaction.article_id)
                customer_seen.add(transaction.article_id)

    summary = TemporalSplitSummary(
        cutoff=split.cutoff.isoformat(),
        validation_start=split.cutoff.isoformat(),
        validation_end_exclusive=split.validation_end.isoformat(),
        horizon_days=split.horizon_days,
        train_rows=row_counts["train"],
        validation_rows=row_counts["validation"],
        future_rows=row_counts["future"],
        train_customers=len(customers["train"]),
        validation_customers=len(customers["validation"]),
        future_customers=len(customers["future"]),
        train_articles=len(articles["train"]),
        validation_articles=len(articles["validation"]),
        future_articles=len(articles["future"]),
        validation_label_customers=len(customers["validation"]),
        validation_unique_customer_article_pairs=len(validation_pairs),
    )
    validation_labels = {
        customer_id: tuple(article_ids) for customer_id, article_ids in labels.items()
    }
    return TemporalValidationData(summary=summary, validation_labels=validation_labels)


def collect_validation_labels(
    transactions: Iterable[TransactionEvent], split: TemporalSplit
) -> dict[str, tuple[str, ...]]:
    """Collect unique validation target articles per customer.

    Args:
        transactions: Transaction events to bucket by date.
        split: Cutoff and horizon definition.

    Returns:
        Mapping of customer ID to unique validation article IDs in first-seen
        order.
    """

    return summarize_temporal_split_with_labels(transactions, split).validation_labels


def temporal_split_summary_to_dict(summary: TemporalSplitSummary) -> dict[str, int | str]:
    """Convert a temporal split summary to JSON-serializable primitives.

    Args:
        summary: Summary object to convert.

    Returns:
        Dictionary suitable for ``json.dumps``.
    """

    return asdict(summary)


def write_temporal_split_summary(summary: TemporalSplitSummary, path: Path | str) -> Path:
    """Write a temporal split summary as deterministic JSON.

    Args:
        summary: Summary object to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(temporal_split_summary_to_dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def assert_no_validation_leakage(
    train_features: Mapping[str, object], split: TemporalSplit
) -> None:
    """Guard feature metadata against validation leakage.

    Args:
        train_features: Feature metadata expected to contain a ``cutoff`` key.
        split: Temporal split whose cutoff must match the metadata.

    Raises:
        ValueError: If metadata lacks a cutoff or records the wrong cutoff.
    """

    cutoff = train_features.get("cutoff")
    if cutoff is None:
        raise ValueError("train_features metadata must include a cutoff")
    if str(cutoff) != split.cutoff.isoformat():
        raise ValueError(
            f"feature cutoff {cutoff!r} does not match split cutoff {split.cutoff.isoformat()!r}"
        )
