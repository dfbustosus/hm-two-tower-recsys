"""Leakage-safe customer-segment popularity candidate retrieval."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from math import log1p
from pathlib import Path

from hm_recsys.core.ids import is_customer_id
from hm_recsys.data.io import CsvValueError, TransactionEvent, iter_csv_rows
from hm_recsys.evaluation.temporal import TemporalSplit

UNKNOWN_AGE_SEGMENT = "age_unknown"
DEFAULT_AGE_SEGMENT_BUCKET_SIZE = 10


@dataclass(frozen=True)
class AgeSegmentPopularityCandidate:
    """One segment-popularity candidate for a customer."""

    article_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class AgeSegmentPopularityIndex:
    """Pre-cutoff popularity rankings keyed by customer age segment.

    Attributes:
        customer_segment_by_id: Customer-to-segment mapping loaded from
            ``customers.csv``.
        rankings_by_segment: Ranked article IDs and normalized scores for each
            age segment.
        lookback_days: Recent pre-cutoff window used for the popularity counts.
        train_rows_used: Number of pre-cutoff transactions scanned.
    """

    customer_segment_by_id: Mapping[str, str]
    rankings_by_segment: dict[str, tuple[tuple[str, float], ...]]
    lookback_days: int
    train_rows_used: int


def load_customer_age_segments(
    raw_data_dir: Path | str,
    *,
    bucket_size: int = DEFAULT_AGE_SEGMENT_BUCKET_SIZE,
) -> dict[str, str]:
    """Load customer age-bucket segments from ``customers.csv``.

    Args:
        raw_data_dir: Directory containing H&M ``customers.csv``.
        bucket_size: Width of age buckets, for example ``10`` gives
            ``age_30_39``.

    Returns:
        Customer IDs mapped to deterministic age-segment labels. Missing ages are
        mapped to ``age_unknown``.

    Raises:
        ValueError: If ``bucket_size`` is invalid.
        CsvSchemaError: If required columns are missing.
        CsvValueError: If a customer ID or age value is invalid.
    """

    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    path = Path(raw_data_dir).expanduser().resolve() / "customers.csv"
    required_columns = ("customer_id", "age")
    segments: dict[str, str] = {}
    for line_number, row in enumerate(iter_csv_rows(path, required_columns), start=2):
        customer_id = row["customer_id"]
        if not is_customer_id(customer_id):
            raise CsvValueError(f"line {line_number}: invalid customer_id {customer_id!r}")
        segments[customer_id] = age_to_segment(
            row["age"],
            bucket_size=bucket_size,
            line_number=line_number,
        )
    return segments


def age_to_segment(age_value: str, *, bucket_size: int, line_number: int = 0) -> str:
    """Convert a raw age string to a deterministic age-bucket label."""

    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    normalized = age_value.strip()
    if not normalized:
        return UNKNOWN_AGE_SEGMENT
    try:
        age = int(normalized)
    except ValueError as exc:
        location = f"line {line_number}: " if line_number else ""
        raise CsvValueError(f"{location}invalid age {age_value!r}") from exc
    if age <= 0:
        location = f"line {line_number}: " if line_number else ""
        raise CsvValueError(f"{location}invalid age {age_value!r}")
    lower = (age // bucket_size) * bucket_size
    upper = lower + bucket_size - 1
    return f"age_{lower}_{upper}"


def build_age_segment_popularity_index(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    customer_segment_by_id: Mapping[str, str],
    *,
    lookback_days: int = 7,
    max_articles_per_segment: int | None = None,
) -> AgeSegmentPopularityIndex:
    """Build cutoff-safe recent-popularity rankings by customer age segment."""

    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if max_articles_per_segment is not None and max_articles_per_segment <= 0:
        raise ValueError("max_articles_per_segment must be positive when provided")

    window_start = split.cutoff - timedelta(days=lookback_days)
    counts_by_segment: dict[str, Counter[str]] = {}
    train_rows_used = 0
    for transaction in transactions:
        if transaction.t_dat >= split.cutoff:
            continue
        train_rows_used += 1
        if transaction.t_dat < window_start:
            continue
        segment = customer_segment_by_id.get(transaction.customer_id)
        if segment is None:
            continue
        counts_by_segment.setdefault(segment, Counter())[transaction.article_id] += 1

    rankings_by_segment = {
        segment: _rank_segment_counts(counts, limit=max_articles_per_segment)
        for segment, counts in counts_by_segment.items()
    }
    return AgeSegmentPopularityIndex(
        customer_segment_by_id=customer_segment_by_id,
        rankings_by_segment=rankings_by_segment,
        lookback_days=lookback_days,
        train_rows_used=train_rows_used,
    )


def build_age_segment_popularity_candidates(
    index: AgeSegmentPopularityIndex,
    customer_id: str,
    k: int,
) -> tuple[AgeSegmentPopularityCandidate, ...]:
    """Return ranked age-segment popularity candidates for one customer."""

    if k <= 0:
        raise ValueError("k must be positive")
    segment = index.customer_segment_by_id.get(customer_id)
    if segment is None:
        return ()
    ranking = index.rankings_by_segment.get(segment, ())[:k]
    return tuple(
        AgeSegmentPopularityCandidate(article_id=article_id, score=score, rank=rank)
        for rank, (article_id, score) in enumerate(ranking, start=1)
    )


def _rank_segment_counts(
    counts: Counter[str],
    *,
    limit: int | None,
) -> tuple[tuple[str, float], ...]:
    if not counts:
        return ()
    log_counts = {article_id: log1p(count) for article_id, count in counts.items()}
    max_log_count = max(log_counts.values())
    ranked = sorted(
        log_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if limit is not None:
        ranked = ranked[:limit]
    return tuple(
        (article_id, log_count / max_log_count if max_log_count > 0.0 else 0.0)
        for article_id, log_count in ranked
    )


__all__ = [
    "DEFAULT_AGE_SEGMENT_BUCKET_SIZE",
    "UNKNOWN_AGE_SEGMENT",
    "AgeSegmentPopularityCandidate",
    "AgeSegmentPopularityIndex",
    "age_to_segment",
    "build_age_segment_popularity_candidates",
    "build_age_segment_popularity_index",
    "load_customer_age_segments",
]
