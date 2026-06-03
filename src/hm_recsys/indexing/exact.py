"""Pure-Python exact vector index for correctness-first retrieval smoke tests."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

from hm_recsys.embeddings.contracts import ArticleEmbeddingRecord, EmbeddingVector
from hm_recsys.indexing.contracts import IndexSearchResult, VectorIndexConfig


class ExactVectorIndex:
    """In-memory exact nearest-neighbor index.

    This index is intentionally simple and deterministic.  It is suitable for
    unit tests, small smoke runs, and validating article-ID mappings before
    introducing approximate indexes such as FAISS or HNSW.
    """

    def __init__(self, config: VectorIndexConfig) -> None:
        """Create an empty exact vector index."""

        self._config = config
        self._records: tuple[ArticleEmbeddingRecord, ...] = ()
        self._vectors_by_article_id: dict[str, EmbeddingVector] = {}

    @property
    def config(self) -> VectorIndexConfig:
        """Return the immutable vector-index configuration."""

        return self._config

    def build(self, records: Iterable[ArticleEmbeddingRecord]) -> None:
        """Build the index from article embedding records.

        Args:
            records: Article embeddings to index.

        Raises:
            ValueError: If a vector dimension or duplicate article ID is invalid.
        """

        indexed_records: list[ArticleEmbeddingRecord] = []
        vectors_by_article_id: dict[str, EmbeddingVector] = {}
        for record in records:
            if record.dimension != self.config.dimension:
                raise ValueError(
                    f"article {record.article_id} dimension {record.dimension} does not match "
                    f"index dimension {self.config.dimension}"
                )
            if record.article_id in vectors_by_article_id:
                raise ValueError(f"duplicate article_id in index: {record.article_id!r}")
            indexed_records.append(record)
            vectors_by_article_id[record.article_id] = record.vector
        self._records = tuple(indexed_records)
        self._vectors_by_article_id = vectors_by_article_id

    def query(self, vector: EmbeddingVector, top_k: int) -> tuple[IndexSearchResult, ...]:
        """Retrieve exact nearest neighbors for a query vector.

        Args:
            vector: Query vector with ``config.dimension`` values.
            top_k: Maximum result count.

        Returns:
            Ranked search results with deterministic article-ID tie-breaking.

        Raises:
            ValueError: If query shape or ``top_k`` is invalid.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(vector) != self.config.dimension:
            raise ValueError(
                f"query dimension {len(vector)} does not match index dimension "
                f"{self.config.dimension}"
            )
        scored = [
            IndexSearchResult(
                article_id=record.article_id,
                score=_score(vector, record.vector, self.config.metric),
            )
            for record in self._records
        ]
        ranked = sorted(scored, key=lambda result: (-result.score, result.article_id))
        return tuple(ranked[:top_k])

    def vector_for_article(self, article_id: str) -> EmbeddingVector | None:
        """Return the stored vector for an article, if present."""

        return self._vectors_by_article_id.get(article_id)

    @property
    def article_count(self) -> int:
        """Return the number of indexed articles."""

        return len(self._records)


def mean_vector(vectors: Iterable[EmbeddingVector]) -> EmbeddingVector:
    """Return the element-wise mean of non-empty equal-length vectors.

    Raises:
        ValueError: If no vectors are provided or dimensions differ.
    """

    vector_list = list(vectors)
    if not vector_list:
        raise ValueError("vectors must not be empty")
    dimension = len(vector_list[0])
    if dimension == 0:
        raise ValueError("vectors must not be empty")
    totals = [0.0] * dimension
    for vector in vector_list:
        if len(vector) != dimension:
            raise ValueError("all vectors must have the same dimension")
        for index, value in enumerate(vector):
            totals[index] += value
    return tuple(total / len(vector_list) for total in totals)


def l2_normalize(vector: EmbeddingVector) -> EmbeddingVector:
    """Return an L2-normalized copy of ``vector``.

    Zero vectors are returned unchanged so callers can decide whether an empty
    query should produce candidates.
    """

    norm = sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return tuple(value / norm for value in vector)


def _score(query: EmbeddingVector, candidate: EmbeddingVector, metric: str) -> float:
    if metric == "dot":
        return _dot(query, candidate)
    if metric == "cosine":
        return _cosine(query, candidate)
    if metric == "l2":
        return -sum((left - right) ** 2 for left, right in zip(query, candidate, strict=True))
    raise ValueError(f"unsupported distance metric: {metric!r}")


def _dot(left: EmbeddingVector, right: EmbeddingVector) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def _cosine(left: EmbeddingVector, right: EmbeddingVector) -> float:
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)
