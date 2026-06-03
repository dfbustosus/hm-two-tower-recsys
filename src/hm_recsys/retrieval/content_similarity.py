"""Leakage-safe content-embedding similarity retrieval for H&M candidates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hm_recsys.data.io import TransactionEvent
from hm_recsys.embeddings.contracts import ArticleEmbeddingRecord
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.indexing.contracts import VectorIndexConfig
from hm_recsys.indexing.exact import ExactVectorIndex, l2_normalize, mean_vector
from hm_recsys.retrieval.candidate_export import CandidateRecord
from hm_recsys.retrieval.co_visitation import DEFAULT_MAX_HISTORY_ITEMS
from hm_recsys.retrieval.source_names import MULTIMODAL_SIMILARITY_SOURCE


@dataclass(frozen=True)
class ContentSimilarityIndex:
    """Exact embedding index plus cutoff-safe customer history vectors.

    Attributes:
        article_index: Exact vector index over cached article embeddings.
        customer_histories: Recent unique pre-cutoff article histories, newest
            first, retained only for target customers.
        source_name: Candidate source label written to ranker-ready records.
        train_rows_used: Number of pre-cutoff transaction rows scanned.
        max_history_items: Maximum recent unique articles retained per customer.
        exclude_history: Whether historical articles are filtered from retrieved
            content-similarity candidates.
    """

    article_index: ExactVectorIndex
    customer_histories: dict[str, tuple[str, ...]]
    source_name: str
    train_rows_used: int
    max_history_items: int
    exclude_history: bool


@dataclass(frozen=True)
class ContentSimilarityCandidate:
    """Customer-level content-similarity candidate."""

    article_id: str
    score: float
    rank: int


def build_content_similarity_index(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    target_customer_ids: Iterable[str],
    embedding_records: Iterable[ArticleEmbeddingRecord],
    *,
    source_name: str = MULTIMODAL_SIMILARITY_SOURCE,
    max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    exclude_history: bool = True,
) -> ContentSimilarityIndex:
    """Build a leakage-safe content-similarity retrieval index.

    Customer query histories use only transactions with ``t_dat < split.cutoff``.
    Article embeddings are assumed to be article-content-only features and must
    not be trained on validation target interactions.

    Args:
        transactions: Transaction events used to collect customer histories.
        split: Temporal split whose cutoff is the exclusive training boundary.
        target_customer_ids: Customers for whom histories should be retained.
        embedding_records: Cached article embeddings to index.
        source_name: Source name for emitted candidate rows.
        max_history_items: Recent unique article history length.
        exclude_history: Whether to suppress direct historical articles.

    Returns:
        Content-similarity index with exact article search and customer histories.

    Raises:
        ValueError: If configuration or embeddings are invalid.
    """

    if not source_name:
        raise ValueError("source_name must not be empty")
    if max_history_items <= 0:
        raise ValueError("max_history_items must be positive")

    records = tuple(embedding_records)
    if not records:
        raise ValueError("embedding_records must not be empty")
    dimension = records[0].dimension
    provider_name = records[0].provider_name
    if any(record.dimension != dimension for record in records):
        raise ValueError("all embedding records must have the same dimension")
    if any(record.provider_name != provider_name for record in records):
        raise ValueError("all embedding records must use the same provider_name")

    article_index = ExactVectorIndex(
        VectorIndexConfig(
            name=f"{source_name}_{provider_name}",
            metric="cosine",
            dimension=dimension,
        )
    )
    article_index.build(records)

    target_customer_set = set(target_customer_ids)
    mutable_histories: dict[str, list[str]] = {}
    train_rows_used = 0
    for transaction in transactions:
        if transaction.t_dat >= split.cutoff:
            continue
        train_rows_used += 1
        if transaction.customer_id not in target_customer_set:
            continue
        if article_index.vector_for_article(transaction.article_id) is None:
            continue
        _update_recent_unique_history(
            mutable_histories.setdefault(transaction.customer_id, []),
            transaction.article_id,
            max_history_items=max_history_items,
        )

    customer_histories = {
        customer_id: tuple(reversed(history)) for customer_id, history in mutable_histories.items()
    }
    return ContentSimilarityIndex(
        article_index=article_index,
        customer_histories=customer_histories,
        source_name=source_name,
        train_rows_used=train_rows_used,
        max_history_items=max_history_items,
        exclude_history=exclude_history,
    )


def build_content_similarity_candidate_records(
    index: ContentSimilarityIndex,
    customer_id: str,
    k: int,
) -> tuple[ContentSimilarityCandidate, ...]:
    """Build scored content-similarity candidates for one customer.

    Args:
        index: Content-similarity index built from pre-cutoff histories.
        customer_id: Customer requiring candidates.
        k: Maximum candidate count.

    Returns:
        Ranked content candidates with one-based ranks after history filtering.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    history = index.customer_histories.get(customer_id, ())
    if not history:
        return ()
    history_vectors = tuple(
        vector
        for article_id in history
        if (vector := index.article_index.vector_for_article(article_id)) is not None
    )
    if not history_vectors:
        return ()
    query_vector = l2_normalize(mean_vector(history_vectors))
    query_k = k + len(history) if index.exclude_history else k
    history_set = set(history)
    candidates: list[ContentSimilarityCandidate] = []
    for result in index.article_index.query(query_vector, top_k=query_k):
        if index.exclude_history and result.article_id in history_set:
            continue
        candidates.append(
            ContentSimilarityCandidate(
                article_id=result.article_id,
                score=result.score,
                rank=len(candidates) + 1,
            )
        )
        if len(candidates) == k:
            break
    return tuple(candidates)


def build_content_similarity_candidate_source_records(
    index: ContentSimilarityIndex,
    customer_id: str,
    k: int,
) -> tuple[CandidateRecord, ...]:
    """Build ranker-ready candidate records for one customer's content source."""

    return tuple(
        CandidateRecord(
            customer_id=customer_id,
            article_id=candidate.article_id,
            source=index.source_name,
            source_rank=candidate.rank,
            source_score=candidate.score,
        )
        for candidate in build_content_similarity_candidate_records(index, customer_id, k=k)
    )


def content_similarity_article_coverage(
    index: ContentSimilarityIndex,
    customer_ids: Iterable[str],
    k: int,
) -> int:
    """Count unique articles retrieved by content similarity for customers."""

    if k <= 0:
        raise ValueError("k must be positive")
    article_ids: set[str] = set()
    for customer_id in customer_ids:
        for candidate in build_content_similarity_candidate_records(index, customer_id, k=k):
            article_ids.add(candidate.article_id)
    return len(article_ids)


def _update_recent_unique_history(
    history: list[str],
    article_id: str,
    *,
    max_history_items: int,
) -> None:
    if article_id in history:
        history.remove(article_id)
    history.append(article_id)
    if len(history) > max_history_items:
        del history[0 : len(history) - max_history_items]
