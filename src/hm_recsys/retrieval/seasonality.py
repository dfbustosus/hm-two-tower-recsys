"""Leakage-safe shifted-window seasonal popularity retrieval."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from math import log1p

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit

DEFAULT_SEASONAL_SHIFT_DAYS = 366
DEFAULT_SEASONAL_WINDOW_DAYS = 7


@dataclass(frozen=True)
class SeasonalPopularityCandidate:
    """One shifted-window seasonal-popularity candidate."""

    article_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class SeasonalPopularityIndex:
    """Pre-cutoff article popularity in a historical shifted target window.

    Attributes:
        ranking: Ranked article IDs and normalized scores from the shifted
            historical window.
        shift_days: Number of days between the validation cutoff and the start
            of the historical seasonal window.
        window_days: Length of the historical seasonal window.
        window_start: Inclusive historical window start date.
        window_end_exclusive: Exclusive historical window end date.
        train_rows_used: Number of pre-cutoff rows scanned.
        seasonal_rows_used: Number of rows inside the shifted seasonal window.
    """

    ranking: tuple[tuple[str, float], ...]
    shift_days: int
    window_days: int
    window_start: date
    window_end_exclusive: date
    train_rows_used: int
    seasonal_rows_used: int


def build_seasonal_popularity_index(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    *,
    shift_days: int = DEFAULT_SEASONAL_SHIFT_DAYS,
    window_days: int = DEFAULT_SEASONAL_WINDOW_DAYS,
    max_articles: int | None = None,
) -> SeasonalPopularityIndex:
    """Build a global shifted-window popularity ranking without target leakage.

    The historical window is ``[split.cutoff - shift_days,
    split.cutoff - shift_days + window_days)``. The function rejects windows
    that overlap the validation/training cutoff, so validation labels cannot
    contribute to candidate availability or scores.
    """

    if shift_days <= 0:
        raise ValueError("shift_days must be positive")
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if max_articles is not None and max_articles <= 0:
        raise ValueError("max_articles must be positive when provided")

    window_start = split.cutoff - timedelta(days=shift_days)
    window_end_exclusive = window_start + timedelta(days=window_days)
    if window_end_exclusive > split.cutoff:
        raise ValueError("seasonal popularity window must end no later than the cutoff")

    counts: Counter[str] = Counter()
    train_rows_used = 0
    seasonal_rows_used = 0
    for transaction in transactions:
        if transaction.t_dat >= split.cutoff:
            continue
        train_rows_used += 1
        if window_start <= transaction.t_dat < window_end_exclusive:
            counts[transaction.article_id] += 1
            seasonal_rows_used += 1

    return SeasonalPopularityIndex(
        ranking=_rank_counts(counts, limit=max_articles),
        shift_days=shift_days,
        window_days=window_days,
        window_start=window_start,
        window_end_exclusive=window_end_exclusive,
        train_rows_used=train_rows_used,
        seasonal_rows_used=seasonal_rows_used,
    )


def build_seasonal_popularity_candidates(
    index: SeasonalPopularityIndex,
    k: int,
) -> tuple[SeasonalPopularityCandidate, ...]:
    """Return ranked seasonal-popularity candidates."""

    if k <= 0:
        raise ValueError("k must be positive")
    return tuple(
        SeasonalPopularityCandidate(article_id=article_id, score=score, rank=rank)
        for rank, (article_id, score) in enumerate(index.ranking[:k], start=1)
    )


def _rank_counts(
    counts: Counter[str],
    *,
    limit: int | None,
) -> tuple[tuple[str, float], ...]:
    if not counts:
        return ()
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
    "DEFAULT_SEASONAL_SHIFT_DAYS",
    "DEFAULT_SEASONAL_WINDOW_DAYS",
    "SeasonalPopularityCandidate",
    "SeasonalPopularityIndex",
    "build_seasonal_popularity_candidates",
    "build_seasonal_popularity_index",
]
