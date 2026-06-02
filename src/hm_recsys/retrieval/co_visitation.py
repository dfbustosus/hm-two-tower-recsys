"""Leakage-safe co-visitation item-item retrieval for H&M candidates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit

DEFAULT_MAX_HISTORY_ITEMS = 8
DEFAULT_MAX_NEIGHBORS_PER_ITEM = 100


@dataclass(frozen=True)
class ArticleNeighbor:
    """Ranked co-visitation neighbor for one source article.

    Attributes:
        article_id: Neighbor article identifier.
        score: Co-visitation count score.
        rank: One-based rank within the source article's neighbor list.
    """

    article_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class CoVisitationIndex:
    """Bounded item-item co-visitation index and customer histories.

    Attributes:
        neighbors_by_article: Ranked neighbors keyed by source article ID.
        customer_histories: Recent unique customer histories ordered newest first.
        train_rows_used: Number of pre-cutoff transaction rows consumed.
        max_history_items: Maximum recent unique articles retained per customer.
        max_neighbors_per_item: Maximum neighbors retained per article.
    """

    neighbors_by_article: dict[str, tuple[ArticleNeighbor, ...]]
    customer_histories: dict[str, tuple[str, ...]]
    train_rows_used: int
    max_history_items: int
    max_neighbors_per_item: int


@dataclass(frozen=True)
class CoVisitationCandidate:
    """Customer-level co-visitation candidate with aggregate retrieval score.

    Attributes:
        article_id: Candidate article identifier.
        score: Aggregate co-visitation score after history recency weighting.
        rank: One-based rank within the customer's co-visitation candidate list.
    """

    article_id: str
    score: float
    rank: int


def build_co_visitation_index(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    target_customer_ids: Iterable[str],
    max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    max_neighbors_per_item: int = DEFAULT_MAX_NEIGHBORS_PER_ITEM,
) -> CoVisitationIndex:
    """Build a co-visitation index from pre-cutoff customer histories.

    The index is leakage-safe because only transactions with ``t_dat < cutoff``
    contribute to customer histories and item-item counts.

    Args:
        transactions: Transaction events in any iterable order.
        split: Temporal split defining the exclusive training cutoff.
        target_customer_ids: Customers for whom histories should be retained.
        max_history_items: Maximum recent unique articles kept per customer.
        max_neighbors_per_item: Maximum ranked neighbors kept per source article.

    Returns:
        Co-visitation index with deterministic ranked neighbors.

    Raises:
        ValueError: If ``max_history_items`` or ``max_neighbors_per_item`` is not
        positive.
    """

    if max_history_items <= 0:
        raise ValueError("max_history_items must be positive")
    if max_neighbors_per_item <= 0:
        raise ValueError("max_neighbors_per_item must be positive")

    target_customer_set = set(target_customer_ids)
    mutable_histories: dict[str, list[str]] = {}
    train_rows_used = 0

    for transaction in transactions:
        if transaction.t_dat >= split.cutoff:
            continue
        train_rows_used += 1
        if transaction.customer_id not in target_customer_set:
            continue
        _update_recent_unique_history(
            mutable_histories.setdefault(transaction.customer_id, []),
            transaction.article_id,
            max_history_items=max_history_items,
        )

    pair_counts = _build_pair_counts(mutable_histories.values())
    neighbors_by_article = _rank_pair_counts(
        pair_counts,
        max_neighbors_per_item=max_neighbors_per_item,
    )
    customer_histories = {
        customer_id: tuple(reversed(history)) for customer_id, history in mutable_histories.items()
    }
    return CoVisitationIndex(
        neighbors_by_article=neighbors_by_article,
        customer_histories=customer_histories,
        train_rows_used=train_rows_used,
        max_history_items=max_history_items,
        max_neighbors_per_item=max_neighbors_per_item,
    )


def build_co_visitation_candidates(
    index: CoVisitationIndex,
    customer_id: str,
    k: int,
) -> tuple[str, ...]:
    """Build ranked co-visitation candidates for one customer.

    Args:
        index: Co-visitation index produced from pre-cutoff transactions.
        customer_id: Customer identifier requiring candidates.
        k: Maximum number of candidates to return.

    Returns:
        Ranked article IDs with duplicates removed.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    return tuple(
        candidate.article_id
        for candidate in build_co_visitation_candidate_records(index, customer_id, k=k)
    )


def build_co_visitation_candidate_records(
    index: CoVisitationIndex,
    customer_id: str,
    k: int,
) -> tuple[CoVisitationCandidate, ...]:
    """Build scored co-visitation candidate records for one customer.

    Args:
        index: Co-visitation index produced from pre-cutoff transactions.
        customer_id: Customer identifier requiring candidates.
        k: Maximum number of candidates to return.

    Returns:
        Ranked co-visitation candidates with aggregate scores and one-based ranks.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    best_neighbor_rank: dict[str, int] = {}
    for history_rank, source_article_id in enumerate(
        index.customer_histories.get(customer_id, ()),
        start=1,
    ):
        history_weight = 1.0 / history_rank
        for neighbor in index.neighbors_by_article.get(source_article_id, ()):
            scores[neighbor.article_id] = (
                scores.get(neighbor.article_id, 0.0) + neighbor.score * history_weight
            )
            best_neighbor_rank[neighbor.article_id] = min(
                best_neighbor_rank.get(neighbor.article_id, neighbor.rank),
                neighbor.rank,
            )

    ranked = sorted(
        scores,
        key=lambda article_id: (
            -scores[article_id],
            best_neighbor_rank[article_id],
            article_id,
        ),
    )
    return tuple(
        CoVisitationCandidate(article_id=article_id, score=scores[article_id], rank=rank)
        for rank, article_id in enumerate(ranked[:k], start=1)
    )


def co_visitation_candidate_count(index: CoVisitationIndex, customer_id: str, k: int) -> int:
    """Count unique co-visitation candidates for one customer up to ``k``.

    Args:
        index: Co-visitation index produced from pre-cutoff transactions.
        customer_id: Customer identifier requiring candidates.
        k: Maximum count to compute before early stopping.

    Returns:
        Number of unique candidates, capped at ``k``.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    seen: set[str] = set()
    for source_article_id in index.customer_histories.get(customer_id, ()):
        for neighbor in index.neighbors_by_article.get(source_article_id, ()):
            seen.add(neighbor.article_id)
            if len(seen) == k:
                return k
    return len(seen)


def co_visitation_article_coverage(
    index: CoVisitationIndex,
    customer_ids: Iterable[str],
) -> int:
    """Count unique articles reachable from target customer histories.

    Args:
        index: Co-visitation index produced from pre-cutoff transactions.
        customer_ids: Customer universe whose histories define emitted candidates.

    Returns:
        Number of unique neighbor article IDs reachable for the customer universe.
    """

    history_articles = {
        article_id
        for customer_id in customer_ids
        for article_id in index.customer_histories.get(customer_id, ())
    }
    return len(
        {
            neighbor.article_id
            for source_article_id in history_articles
            for neighbor in index.neighbors_by_article.get(source_article_id, ())
        }
    )


def _update_recent_unique_history(
    history: list[str], article_id: str, max_history_items: int
) -> None:
    """Update a bounded oldest-to-newest unique article history in place."""

    if article_id in history:
        history.remove(article_id)
    history.append(article_id)
    if len(history) > max_history_items:
        del history[0]


def _build_pair_counts(
    histories: Iterable[Sequence[str]],
) -> dict[str, dict[str, int]]:
    """Build directed co-visitation counts from bounded customer histories."""

    pair_counts: dict[str, dict[str, int]] = {}
    for history in histories:
        history_tuple = tuple(history)
        history_length = len(history_tuple)
        for source_index in range(history_length):
            source_article_id = history_tuple[source_index]
            source_counts = pair_counts.setdefault(source_article_id, {})
            for neighbor_index in range(history_length):
                if source_index == neighbor_index:
                    continue
                neighbor_article_id = history_tuple[neighbor_index]
                source_counts[neighbor_article_id] = source_counts.get(neighbor_article_id, 0) + 1
    return pair_counts


def _rank_pair_counts(
    pair_counts: dict[str, dict[str, int]],
    max_neighbors_per_item: int,
) -> dict[str, tuple[ArticleNeighbor, ...]]:
    """Rank co-visitation counts deterministically for every source article."""

    ranked_neighbors: dict[str, tuple[ArticleNeighbor, ...]] = {}
    for source_article_id, counts in pair_counts.items():
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :max_neighbors_per_item
        ]
        ranked_neighbors[source_article_id] = tuple(
            ArticleNeighbor(article_id=article_id, score=float(score), rank=rank)
            for rank, (article_id, score) in enumerate(ranked, start=1)
        )
    return ranked_neighbors
