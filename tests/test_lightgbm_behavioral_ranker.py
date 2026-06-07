import csv
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.ranking.behavioral import BEHAVIORAL_FEATURE_NAMES, build_cutoff_behavioral_features
from hm_recsys.ranking.deterministic import CandidateFeatures
from hm_recsys.ranking.lightgbm_behavioral import (
    LIGHTGBM_BEHAVIORAL_FEATURE_NAMES,
    LightGBMBehavioralRankerConfig,
    iter_grouped_candidate_features_from_csv,
    lightgbm_behavioral_feature_vector,
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


def _write_candidate_csv(path: Path, rows: list[tuple[str, str, str, int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        writer.writerows(rows)
