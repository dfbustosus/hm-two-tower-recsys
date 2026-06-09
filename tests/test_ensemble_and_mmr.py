"""Tests for the ensemble blender and MMR diversity reranker."""

from __future__ import annotations

import pytest

from hm_recsys.ranking.ensemble import (
    DEFAULT_RRF_K,
    RankedSlate,
    reciprocal_rank_fusion,
    top_k_articles,
    zscore_blend,
)
from hm_recsys.ranking.mmr_rerank import MMRConfig, build_attribute_overlap_similarity, mmr_rerank


def test_reciprocal_rank_fusion_promotes_articles_in_multiple_slates() -> None:
    slates = [
        RankedSlate("ranker_a", ("A", "B", "C")),
        RankedSlate("ranker_b", ("B", "A", "D")),
    ]

    fused = reciprocal_rank_fusion(slates)

    fused_ids = [article for article, _ in fused]
    assert fused_ids[0] in {"A", "B"}
    assert fused[0][1] > fused[-1][1]
    assert set(fused_ids) == {"A", "B", "C", "D"}


def test_reciprocal_rank_fusion_respects_weights() -> None:
    slates = [
        RankedSlate("strong", ("A", "B")),
        RankedSlate("weak", ("B", "A")),
    ]

    fused = reciprocal_rank_fusion(slates, weights={"strong": 2.0, "weak": 1.0})

    assert fused[0][0] == "A"


def test_reciprocal_rank_fusion_default_k_is_60() -> None:
    assert DEFAULT_RRF_K == 60


def test_reciprocal_rank_fusion_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([], k=0)


def test_ranked_slate_rejects_mismatched_score_length() -> None:
    with pytest.raises(ValueError, match="length"):
        RankedSlate("ranker", ("A", "B"), scores=(1.0,))


def test_zscore_blend_requires_scores() -> None:
    slates = [RankedSlate("ranker", ("A", "B"))]

    with pytest.raises(ValueError, match="requires scores"):
        zscore_blend(slates)


def test_zscore_blend_combines_slates() -> None:
    slates = [
        RankedSlate("a", ("X", "Y", "Z"), scores=(3.0, 1.0, 0.5)),
        RankedSlate("b", ("Y", "X", "W"), scores=(2.0, 1.5, 0.0)),
    ]

    fused = zscore_blend(slates)
    fused_ids = [article for article, _ in fused]

    assert fused_ids[0] in {"X", "Y"}
    assert set(fused_ids) == {"X", "Y", "Z", "W"}


def test_top_k_articles_truncates() -> None:
    fused = (("A", 1.0), ("B", 0.5), ("C", 0.25))

    assert top_k_articles(fused, k=2) == ("A", "B")


def test_mmr_rerank_pure_relevance_when_lambda_one() -> None:
    ranked = (("A", 1.0), ("B", 0.9), ("C", 0.8))
    sim = build_attribute_overlap_similarity({})

    result = mmr_rerank(ranked, sim, k=3, config=MMRConfig(lambda_relevance=1.0))

    assert result == ("A", "B", "C")


def test_mmr_rerank_penalizes_near_duplicates() -> None:
    attributes = {
        "A": {"product_type": "shirt", "color": "blue"},
        "B": {"product_type": "shirt", "color": "blue"},  # near-duplicate of A
        "C": {"product_type": "pants", "color": "red"},
    }
    sim = build_attribute_overlap_similarity(attributes)
    ranked = (("A", 1.0), ("B", 0.95), ("C", 0.9))

    selected = mmr_rerank(ranked, sim, k=2, config=MMRConfig(lambda_relevance=0.4))

    assert selected[0] == "A"
    assert selected[1] == "C", "MMR should prefer a diverse item over a near-duplicate"


def test_mmr_rerank_empty_input_returns_empty_tuple() -> None:
    sim = build_attribute_overlap_similarity({})
    assert mmr_rerank((), sim, k=5) == ()


def test_mmr_rerank_rejects_negative_k() -> None:
    sim = build_attribute_overlap_similarity({})
    with pytest.raises(ValueError, match="k must be non-negative"):
        mmr_rerank((("A", 1.0),), sim, k=-1)


def test_mmr_config_rejects_out_of_range_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_relevance"):
        MMRConfig(lambda_relevance=1.1)
