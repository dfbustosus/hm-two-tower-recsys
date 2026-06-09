"""Maximal Marginal Relevance (MMR) diversity reranker.

MMR is the cheapest principled way to inject diversity into a slate that
was produced by a relevance-only ranker. It is well-suited to fashion
recommendations where the relevance model often clusters near-duplicates
(same garment in 5 colours) and where MAP@K rewards covering more
distinct ground-truth items.

The implementation here is dependency-light: it accepts any callable that
returns a similarity score between two articles, so it works with any
embedding backend (Marqo content vectors, two-tower outputs, attribute
overlap, etc.). For very long slates the greedy selection is O(k*N)
in similarity calls; this is appropriate for the typical N <= 200 inputs
after retrieval/ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

ArticleSimilarity = Callable[[str, str], float]


@dataclass(frozen=True)
class MMRConfig:
    """Configuration for :func:`mmr_rerank`."""

    lambda_relevance: float = 0.7
    similarity_floor: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_relevance <= 1.0:
            raise ValueError("lambda_relevance must be in [0, 1]")
        if not 0.0 <= self.similarity_floor < 1.0:
            raise ValueError("similarity_floor must be in [0, 1)")


def mmr_rerank(
    ranked_articles: Sequence[tuple[str, float]],
    similarity: ArticleSimilarity,
    *,
    k: int,
    config: MMRConfig | None = None,
) -> tuple[str, ...]:
    """Greedy MMR over a relevance-scored slate.

    Args:
        ranked_articles: Sequence of ``(article_id, relevance_score)`` pairs
            ordered by descending relevance.
        similarity: Callable returning a similarity in ``[-1, 1]`` for two
            article IDs.
        k: Number of articles to return.
        config: Optional MMR configuration. Defaults to ``MMRConfig()``.

    Returns:
        Selected article IDs ordered by greedy selection.

    Raises:
        ValueError: If ``k`` is negative.
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0 or not ranked_articles:
        return ()
    cfg = config or MMRConfig()
    candidates = list(ranked_articles)
    selected: list[str] = []
    while len(selected) < k and candidates:
        best_article: str | None = None
        best_score = float("-inf")
        for article, relevance in candidates:
            if not selected:
                penalty = 0.0
            else:
                penalty = max(
                    (max(similarity(article, chosen), cfg.similarity_floor) for chosen in selected),
                    default=cfg.similarity_floor,
                )
            mmr_score = cfg.lambda_relevance * relevance - (1 - cfg.lambda_relevance) * penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_article = article
        assert best_article is not None
        selected.append(best_article)
        candidates = [
            (article, score) for article, score in candidates if article != best_article
        ]
    return tuple(selected)


def build_attribute_overlap_similarity(
    attributes_by_article: dict[str, dict[str, str]],
) -> ArticleSimilarity:
    """Return a similarity callable based on shared attribute count.

    The similarity is the fraction of attributes that have identical
    values in both articles, in ``[0, 1]``. This makes a strong baseline
    when content embeddings are unavailable.
    """

    def _similarity(article_a: str, article_b: str) -> float:
        attrs_a = attributes_by_article.get(article_a, {})
        attrs_b = attributes_by_article.get(article_b, {})
        if not attrs_a or not attrs_b:
            return 0.0
        shared_keys = set(attrs_a) & set(attrs_b)
        if not shared_keys:
            return 0.0
        matches = sum(1 for key in shared_keys if attrs_a[key] == attrs_b[key])
        return matches / len(shared_keys)

    return _similarity


__all__ = (
    "ArticleSimilarity",
    "MMRConfig",
    "build_attribute_overlap_similarity",
    "mmr_rerank",
)
