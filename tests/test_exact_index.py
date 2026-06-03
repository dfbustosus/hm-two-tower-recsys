import pytest

from hm_recsys.embeddings.contracts import ArticleEmbeddingRecord
from hm_recsys.indexing.contracts import IndexSearchResult, VectorIndexConfig
from hm_recsys.indexing.exact import ExactVectorIndex, l2_normalize, mean_vector


def test_exact_vector_index_ranks_cosine_with_deterministic_ties() -> None:
    index = ExactVectorIndex(VectorIndexConfig(name="test", metric="cosine", dimension=2))
    index.build(
        (
            ArticleEmbeddingRecord("0300000000", (1.0, 0.0), "test-provider"),
            ArticleEmbeddingRecord("0200000000", (1.0, 0.0), "test-provider"),
            ArticleEmbeddingRecord("0100000000", (0.0, 1.0), "test-provider"),
        )
    )

    assert index.query((1.0, 0.0), top_k=2) == (
        IndexSearchResult("0200000000", 1.0),
        IndexSearchResult("0300000000", 1.0),
    )
    assert index.vector_for_article("0100000000") == (0.0, 1.0)
    assert index.article_count == 3


def test_exact_vector_index_supports_dot_and_l2_metrics() -> None:
    dot_index = ExactVectorIndex(VectorIndexConfig(name="dot", metric="dot", dimension=2))
    dot_index.build(
        (
            ArticleEmbeddingRecord("0100000000", (2.0, 0.0), "test-provider"),
            ArticleEmbeddingRecord("0200000000", (0.0, 1.0), "test-provider"),
        )
    )
    assert dot_index.query((1.0, 0.0), top_k=1) == (IndexSearchResult("0100000000", 2.0),)

    l2_index = ExactVectorIndex(VectorIndexConfig(name="l2", metric="l2", dimension=2))
    l2_index.build(
        (
            ArticleEmbeddingRecord("0100000000", (0.0, 0.0), "test-provider"),
            ArticleEmbeddingRecord("0200000000", (3.0, 0.0), "test-provider"),
        )
    )
    assert l2_index.query((1.0, 0.0), top_k=1) == (IndexSearchResult("0100000000", -1.0),)


def test_exact_vector_index_rejects_invalid_inputs() -> None:
    index = ExactVectorIndex(VectorIndexConfig(name="test", metric="cosine", dimension=2))
    with pytest.raises(ValueError, match="dimension"):
        index.build((ArticleEmbeddingRecord("0100000000", (1.0,), "test-provider"),))
    with pytest.raises(ValueError, match="duplicate"):
        index.build(
            (
                ArticleEmbeddingRecord("0100000000", (1.0, 0.0), "test-provider"),
                ArticleEmbeddingRecord("0100000000", (0.0, 1.0), "test-provider"),
            )
        )
    index.build((ArticleEmbeddingRecord("0100000000", (1.0, 0.0), "test-provider"),))
    with pytest.raises(ValueError, match="top_k"):
        index.query((1.0, 0.0), top_k=0)
    with pytest.raises(ValueError, match="query dimension"):
        index.query((1.0,), top_k=1)


def test_vector_helpers_validate_and_normalize() -> None:
    assert mean_vector(((1.0, 0.0), (0.0, 2.0))) == (0.5, 1.0)
    assert l2_normalize((3.0, 4.0)) == (0.6, 0.8)
    assert l2_normalize((0.0, 0.0)) == (0.0, 0.0)
    with pytest.raises(ValueError, match="must not be empty"):
        mean_vector(())
    with pytest.raises(ValueError, match="same dimension"):
        mean_vector(((1.0,), (1.0, 2.0)))
