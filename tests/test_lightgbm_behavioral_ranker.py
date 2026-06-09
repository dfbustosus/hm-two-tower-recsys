import csv
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.behavioral import BEHAVIORAL_FEATURE_NAMES, build_cutoff_behavioral_features
from hm_recsys.ranking.deterministic import CandidateFeatures
from hm_recsys.ranking.lightgbm_behavioral import (
    LIGHTGBM_BEHAVIORAL_FEATURE_NAMES,
    LightGBMBehavioralRankerAdapter,
    LightGBMBehavioralRankerConfig,
    iter_grouped_candidate_features_from_csv,
    lightgbm_behavioral_feature_vector,
    train_lightgbm_behavioral_ranker_from_csv,
)
from hm_recsys.ranking.linear import LINEAR_FEATURE_NAMES
from hm_recsys.retrieval.candidate_export import CANDIDATE_EXPORT_HEADER
from hm_recsys.retrieval.source_names import RECENT_POPULARITY_SOURCE, REPEAT_SOURCE

CUSTOMER_1 = "a" * 64
CUSTOMER_2 = "b" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"


def test_lightgbm_behavioral_feature_vector_matches_schema() -> None:
    cutoff = date(2020, 9, 16)
    behavioral_features = build_cutoff_behavioral_features(
        (TransactionEvent(date(2020, 9, 15), CUSTOMER_1, ARTICLE_1),),
        cutoff,
    )
    candidate_features = CandidateFeatures(
        customer_id=CUSTOMER_1,
        article_id=ARTICLE_1,
        repeat_rank=1,
        repeat_score=1.0,
        source_count=1,
        best_rank=1,
    )

    vector = lightgbm_behavioral_feature_vector(candidate_features, behavioral_features)

    assert (
        *LINEAR_FEATURE_NAMES,
        "deterministic_score",
        *BEHAVIORAL_FEATURE_NAMES,
    ) == LIGHTGBM_BEHAVIORAL_FEATURE_NAMES
    assert len(set(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES)) == len(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES)
    assert len(vector) == len(LIGHTGBM_BEHAVIORAL_FEATURE_NAMES)
    assert vector[LINEAR_FEATURE_NAMES.index("has_repeat")] == 1.0
    assert vector[LIGHTGBM_BEHAVIORAL_FEATURE_NAMES.index("deterministic_score")] > 0.0


def test_grouped_candidate_iterator_rejects_repeated_customer_block(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    _write_candidate_csv(
        path,
        [
            (CUSTOMER_1, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
            (CUSTOMER_2, ARTICLE_2, REPEAT_SOURCE, 1, 1.0),
            (CUSTOMER_1, ARTICLE_2, RECENT_POPULARITY_SOURCE, 1, 1.0),
        ],
    )

    with pytest.raises(ValueError, match="repeated customer block"):
        tuple(
            iter_grouped_candidate_features_from_csv(
                path,
                {CUSTOMER_1: (ARTICLE_1,), CUSTOMER_2: (ARTICLE_2,)},
            )
        )


def test_grouped_candidate_iterator_aggregates_one_customer(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    _write_candidate_csv(
        path,
        [
            (CUSTOMER_1, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
            (CUSTOMER_1, ARTICLE_1, RECENT_POPULARITY_SOURCE, 2, 0.5),
            (CUSTOMER_2, ARTICLE_2, REPEAT_SOURCE, 1, 1.0),
        ],
    )

    groups = tuple(
        iter_grouped_candidate_features_from_csv(
            path,
            {CUSTOMER_1: (ARTICLE_1,), CUSTOMER_2: (ARTICLE_2,)},
        )
    )

    assert [customer_id for customer_id, _ in groups] == [CUSTOMER_1, CUSTOMER_2]
    assert groups[0][1][ARTICLE_1].source_count == 2
    assert groups[0][1][ARTICLE_1].label == 1


def test_lightgbm_behavioral_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="blend_lambda"):
        LightGBMBehavioralRankerConfig(blend_lambda=-0.1)


def test_train_then_rank_smoke_with_lightgbm(tmp_path: Path) -> None:
    """End-to-end smoke: train a tiny LightGBM ranker and use the adapter to rank.

    Skips when the optional ``lightgbm`` dependency is not installed so the
    suite stays green on minimal environments. Also skips when PyTorch has
    already been imported into the same process: LightGBM and PyTorch ship
    incompatible OpenMP runtimes on macOS and co-loading them crashes the
    process (``Fatal Python error: Aborted``). In that case run this test
    in its own subprocess via ``pytest tests/test_lightgbm_behavioral_ranker.py``.
    """

    import sys

    pytest.importorskip("lightgbm")
    pytest.importorskip("numpy")
    if "torch" in sys.modules:
        pytest.skip(
            "skipping: torch already loaded into this process; "
            "co-loading LightGBM crashes on macOS (libomp duplicate). "
            "Run this test in isolation: "
            "`pytest tests/test_lightgbm_behavioral_ranker.py`."
        )

    train_split = TemporalSplit.from_isoformat("2020-09-16")
    candidate_path = tmp_path / "train_candidates.csv"
    _write_candidate_csv(
        candidate_path,
        [
            (CUSTOMER_1, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
            (CUSTOMER_1, ARTICLE_2, RECENT_POPULARITY_SOURCE, 1, 1.0),
            (CUSTOMER_2, ARTICLE_1, RECENT_POPULARITY_SOURCE, 1, 1.0),
            (CUSTOMER_2, ARTICLE_2, REPEAT_SOURCE, 1, 1.0),
        ],
    )
    events = (
        TransactionEvent(date(2020, 9, 8), CUSTOMER_1, ARTICLE_1),
        TransactionEvent(date(2020, 9, 9), CUSTOMER_1, ARTICLE_2),
        TransactionEvent(date(2020, 9, 10), CUSTOMER_2, ARTICLE_2),
    )
    config = LightGBMBehavioralRankerConfig(num_leaves=4, num_boost_round=5, k=2)

    np, model, summary = train_lightgbm_behavioral_ranker_from_csv(
        transaction_iter_factory=lambda: iter(events),
        train_split=train_split,
        train_candidate_path=candidate_path,
        train_validation_labels={CUSTOMER_1: (ARTICLE_2,), CUSTOMER_2: (ARTICLE_2,)},
        config=config,
    )

    behavioral_features = build_cutoff_behavioral_features(events, train_split.cutoff)
    adapter = LightGBMBehavioralRankerAdapter(
        np=np,
        model=model,
        behavioral_features=behavioral_features,
        config=config,
    )
    features_by_customer = {
        CUSTOMER_1: {
            ARTICLE_1: CandidateFeatures(
                customer_id=CUSTOMER_1,
                article_id=ARTICLE_1,
                repeat_rank=1,
                repeat_score=1.0,
                source_count=1,
                best_rank=1,
            ),
            ARTICLE_2: CandidateFeatures(
                customer_id=CUSTOMER_1,
                article_id=ARTICLE_2,
                recent_popularity_rank=1,
                recent_popularity_score=1.0,
                source_count=1,
                best_rank=1,
            ),
        }
    }
    ranked = adapter.rank_customer_batch(features_by_customer, k=2)
    assert summary.train_unique_candidate_pairs == 4
    assert summary.train_label_customers == 2
    assert set(ranked.keys()) == {CUSTOMER_1}
    assert set(ranked[CUSTOMER_1]) == {ARTICLE_1, ARTICLE_2}


def _write_candidate_csv(path: Path, rows: list[tuple[str, str, str, int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        writer.writerows(rows)
