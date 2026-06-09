"""Tests for the Ranker protocol and its concrete adapters."""

from __future__ import annotations

from hm_recsys.ranking.deterministic import (
    DEFAULT_DETERMINISTIC_RANKER_WEIGHTS,
    CandidateFeatures,
    DeterministicRankerAdapter,
)
from hm_recsys.ranking.linear import LINEAR_FEATURE_NAMES, LinearRankerAdapter, LinearRankerModel
from hm_recsys.ranking.protocol import Ranker


def _candidate_features(
    customer_id: str, article_id: str, *, repeat: bool = False
) -> CandidateFeatures:
    return CandidateFeatures(
        customer_id=customer_id,
        article_id=article_id,
        label=0,
        repeat_rank=1 if repeat else None,
        repeat_score=1.0 if repeat else 0.0,
        recent_popularity_rank=1,
        recent_popularity_score=1.0,
        all_time_popularity_rank=1,
        all_time_popularity_score=1.0,
        source_count=2 + int(repeat),
        best_rank=1,
    )


def test_default_linear_adapter_satisfies_ranker_protocol() -> None:
    adapter = LinearRankerAdapter(
        model=LinearRankerModel(
            feature_names=LINEAR_FEATURE_NAMES,
            weights=tuple(0.0 for _ in LINEAR_FEATURE_NAMES),
        )
    )
    assert isinstance(adapter, Ranker)
    assert adapter.name == "linear"


def test_default_deterministic_adapter_satisfies_ranker_protocol() -> None:
    adapter = DeterministicRankerAdapter()
    assert isinstance(adapter, Ranker)
    assert adapter.weights == DEFAULT_DETERMINISTIC_RANKER_WEIGHTS
    assert adapter.name == "deterministic"


def test_deterministic_adapter_returns_ranked_articles_for_each_customer() -> None:
    adapter = DeterministicRankerAdapter()
    features = {
        "alice": {
            "art-a": _candidate_features("alice", "art-a", repeat=True),
            "art-b": _candidate_features("alice", "art-b"),
        },
        "bob": {
            "art-z": _candidate_features("bob", "art-z"),
        },
    }
    ranked = adapter.rank_customer_batch(features, k=12)
    assert set(ranked.keys()) == {"alice", "bob"}
    assert len(ranked["alice"]) == 2
    assert ranked["bob"] == ("art-z",)


def test_linear_adapter_invokes_underlying_scorer() -> None:
    adapter = LinearRankerAdapter(
        model=LinearRankerModel(
            feature_names=LINEAR_FEATURE_NAMES,
            weights=tuple(1.0 if name == "has_repeat" else 0.0 for name in LINEAR_FEATURE_NAMES),
        )
    )
    features = {
        "alice": {
            "art-a": _candidate_features("alice", "art-a", repeat=True),
            "art-b": _candidate_features("alice", "art-b"),
        },
    }
    ranked = adapter.rank_customer_batch(features, k=12)
    assert ranked["alice"][0] == "art-a"
