"""Per-customer ensemble blenders for combining multiple ranker outputs.

Two strategies are implemented:

* :func:`reciprocal_rank_fusion` — Reciprocal Rank Fusion (Cormack et al.,
  2009). Scale-free and resilient to wildly different score
  distributions; the recommended default when blending heterogeneous
  rankers (e.g. LightGBM scores + two-tower cosine).
* :func:`zscore_blend` — Per-customer z-score normalization then weighted
  sum. Useful when the rankers are well-calibrated and you want to
  preserve magnitude differences between top and tail candidates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import fmean

DEFAULT_RRF_K: int = 60


@dataclass(frozen=True)
class RankedSlate:
    """One ranker's ordered predictions for a single customer.

    Attributes:
        ranker_name: Stable name identifying the ranker (e.g. ``"lightgbm"``).
        ranked_article_ids: Article IDs ordered from best to worst.
        scores: Optional aligned per-article scores. Required for
            :func:`zscore_blend`; ignored by :func:`reciprocal_rank_fusion`.
    """

    ranker_name: str
    ranked_article_ids: tuple[str, ...]
    scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.ranker_name:
            raise ValueError("ranker_name must not be empty")
        if self.scores is not None and len(self.scores) != len(self.ranked_article_ids):
            raise ValueError("scores length must match ranked_article_ids length")


def reciprocal_rank_fusion(
    slates: Iterable[RankedSlate],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> tuple[tuple[str, float], ...]:
    """Combine ranker slates with weighted Reciprocal Rank Fusion.

    The fused score for an article is ``sum_i w_i / (k + rank_i)`` summed
    across the rankers that placed the article.

    Args:
        slates: Per-ranker ordered article lists for one customer.
        k: RRF stabilisation constant. Defaults to ``60`` (Cormack et al.).
        weights: Optional per-ranker weights. Missing ranker names default
            to ``1.0``.

    Returns:
        Articles sorted by descending fused score with their fused scores.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    weights = weights or {}
    aggregated: dict[str, float] = {}
    for slate in slates:
        weight = weights.get(slate.ranker_name, 1.0)
        if weight < 0:
            raise ValueError(f"weight for {slate.ranker_name!r} must be non-negative")
        for rank, article_id in enumerate(slate.ranked_article_ids, start=1):
            aggregated[article_id] = aggregated.get(article_id, 0.0) + weight / (k + rank)
    return tuple(sorted(aggregated.items(), key=lambda item: -item[1]))


def zscore_blend(
    slates: Iterable[RankedSlate],
    *,
    weights: Mapping[str, float] | None = None,
    eps: float = 1e-9,
) -> tuple[tuple[str, float], ...]:
    """Combine ranker slates by per-customer z-score then weighted sum.

    Each slate's scores are converted to z-scores (subtract mean, divide
    by std-dev). Missing items are imputed at the slate's minimum z-score
    minus one so that they fall below any item the slate ranked.

    Args:
        slates: Per-ranker slates with populated scores.
        weights: Optional per-ranker weights. Missing ranker names default
            to ``1.0``.
        eps: Numerical floor for the standard deviation.

    Returns:
        Articles sorted by descending blended z-score.

    Raises:
        ValueError: If any slate is missing scores or contains <= 1 item.
    """

    slate_list = list(slates)
    if not slate_list:
        return ()
    weights = weights or {}
    aggregated: dict[str, float] = {}
    universe: set[str] = set()
    per_slate_min: dict[str, float] = {}
    normalized: list[tuple[str, dict[str, float]]] = []
    for slate in slate_list:
        if slate.scores is None:
            raise ValueError(f"slate {slate.ranker_name!r} requires scores for zscore_blend")
        if len(slate.scores) < 2:
            raise ValueError(
                f"slate {slate.ranker_name!r} needs >= 2 scored items for zscore_blend"
            )
        mean = fmean(slate.scores)
        variance = fmean([(score - mean) ** 2 for score in slate.scores])
        std = sqrt(variance) if variance > 0 else eps
        per_article = {
            article: (score - mean) / std
            for article, score in zip(slate.ranked_article_ids, slate.scores, strict=True)
        }
        per_slate_min[slate.ranker_name] = min(per_article.values())
        universe.update(per_article)
        normalized.append((slate.ranker_name, per_article))
    for ranker_name, per_article in normalized:
        weight = weights.get(ranker_name, 1.0)
        if weight < 0:
            raise ValueError(f"weight for {ranker_name!r} must be non-negative")
        fallback = per_slate_min[ranker_name] - 1.0
        for article in universe:
            aggregated[article] = aggregated.get(article, 0.0) + weight * per_article.get(
                article, fallback
            )
    return tuple(sorted(aggregated.items(), key=lambda item: -item[1]))


def top_k_articles(fused: Sequence[tuple[str, float]], *, k: int) -> tuple[str, ...]:
    """Return the top ``k`` articles from a fused ranking, preserving order."""

    if k < 0:
        raise ValueError("k must be non-negative")
    return tuple(article for article, _ in fused[:k])


__all__ = (
    "DEFAULT_RRF_K",
    "RankedSlate",
    "reciprocal_rank_fusion",
    "top_k_articles",
    "zscore_blend",
)
