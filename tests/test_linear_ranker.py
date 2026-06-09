import csv
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.deterministic import CandidateFeatures
from hm_recsys.ranking.linear import (
    LINEAR_FEATURE_NAMES,
    LinearRankerConfig,
    LinearRankerModel,
    build_learned_linear_ranker_report,
    evaluate_linear_ranker_from_csv,
    feature_vector,
    previous_window_split,
    rank_with_linear_model,
    score_with_linear_model,
    train_linear_ranker_from_csv,
    write_learned_linear_ranker_report,
)
from hm_recsys.retrieval.candidate_export import CANDIDATE_EXPORT_HEADER
from hm_recsys.retrieval.source_names import (
    MULTIMODAL_SIMILARITY_SOURCE,
    RECENT_POPULARITY_1D_SOURCE,
    RECENT_POPULARITY_3D_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
    SEASONAL_POPULARITY_SOURCE,
    TWO_TOWER_RETRIEVAL_LATEST_CUSTOMER_SOURCE,
    TWO_TOWER_RETRIEVAL_SOURCE,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"


def test_linear_ranker_num_sources_matches_feature_vector_emission() -> None:
    from hm_recsys.ranking.linear import LINEAR_RANKER_NUM_SOURCES

    has_count = sum(1 for name in LINEAR_FEATURE_NAMES if name.startswith("has_"))
    assert has_count == LINEAR_RANKER_NUM_SOURCES
    assert LINEAR_RANKER_NUM_SOURCES > 9, (
        "LINEAR_RANKER_NUM_SOURCES must reflect the post-Phase-0.6 source count; "
        "if you add or remove sources, the denominator should follow automatically."
    )


def test_previous_window_split_is_non_overlapping() -> None:
    evaluation_split = TemporalSplit.from_isoformat("2020-01-15", horizon_days=7)

    training_split = previous_window_split(evaluation_split)

    assert training_split.cutoff == date(2020, 1, 8)
    assert training_split.validation_end == evaluation_split.cutoff


def test_feature_vector_matches_schema() -> None:
    features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_1,
        repeat_rank=2,
        repeat_score=0.5,
        recent_popularity_1d_rank=1,
        recent_popularity_1d_score=1.0,
        recent_popularity_3d_rank=2,
        recent_popularity_3d_score=0.5,
        seasonal_popularity_rank=2,
        seasonal_popularity_score=0.4,
        age_segment_popularity_rank=3,
        age_segment_popularity_score=0.7,
        garment_group_popularity_rank=4,
        garment_group_popularity_score=0.6,
        product_code_popularity_rank=3,
        product_code_popularity_score=0.55,
        content_similarity_rank=1,
        content_similarity_score=0.8,
        two_tower_retrieval_rank=5,
        two_tower_retrieval_score=1.2,
        two_tower_retrieval_latest_customer_rank=6,
        two_tower_retrieval_latest_customer_score=0.7,
        source_count=1,
        best_rank=2,
    )

    vector = feature_vector(features)

    assert len(vector) == len(LINEAR_FEATURE_NAMES)
    assert vector[0] == 1.0
    assert vector[1] == 1.0
    assert vector[3] == pytest.approx(0.5)
    assert vector[LINEAR_FEATURE_NAMES.index("has_recent_popularity_1d")] == 1.0
    assert vector[LINEAR_FEATURE_NAMES.index("recent_popularity_1d_score")] == pytest.approx(1.0)
    assert vector[LINEAR_FEATURE_NAMES.index("has_recent_popularity_3d")] == 1.0
    assert vector[LINEAR_FEATURE_NAMES.index("recent_popularity_3d_score")] == pytest.approx(0.5)
    assert vector[LINEAR_FEATURE_NAMES.index("has_seasonal_popularity")] == 1.0
    assert vector[LINEAR_FEATURE_NAMES.index("seasonal_popularity_score")] == pytest.approx(0.4)
    assert vector[LINEAR_FEATURE_NAMES.index("has_age_segment_popularity")] == 1.0
    assert vector[LINEAR_FEATURE_NAMES.index("age_segment_popularity_score")] == pytest.approx(0.7)
    assert vector[LINEAR_FEATURE_NAMES.index("has_garment_group_popularity")] == 1.0
    assert vector[LINEAR_FEATURE_NAMES.index("garment_group_popularity_score")] == pytest.approx(
        0.6
    )
    assert vector[LINEAR_FEATURE_NAMES.index("has_product_code_popularity")] == 1.0
    assert vector[LINEAR_FEATURE_NAMES.index("product_code_popularity_score")] == pytest.approx(
        0.55
    )
    assert vector[
        LINEAR_FEATURE_NAMES.index("product_code_popularity_rank_reciprocal")
    ] == pytest.approx(1.0 / 3)
    assert vector[LINEAR_FEATURE_NAMES.index("has_content_similarity")] == 1.0
    assert vector[LINEAR_FEATURE_NAMES.index("content_similarity_score")] == pytest.approx(0.8)
    assert vector[LINEAR_FEATURE_NAMES.index("has_two_tower_retrieval")] == 1.0
    assert vector[LINEAR_FEATURE_NAMES.index("two_tower_retrieval_score")] == pytest.approx(1.2)
    assert vector[LINEAR_FEATURE_NAMES.index("has_two_tower_retrieval_latest_customer")] == 1.0
    assert vector[
        LINEAR_FEATURE_NAMES.index("two_tower_retrieval_latest_customer_score")
    ] == pytest.approx(0.7)


def test_train_and_evaluate_linear_ranker_from_csv(tmp_path: Path) -> None:
    train_path = tmp_path / "train_candidates.csv"
    eval_path = tmp_path / "eval_candidates.csv"
    rows = [
        (CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
        (CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_SOURCE, 1, 1.0),
        (CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_1D_SOURCE, 1, 1.0),
        (CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_3D_SOURCE, 1, 1.0),
        (CUSTOMER_ID, ARTICLE_2, SEASONAL_POPULARITY_SOURCE, 1, 0.5),
        (CUSTOMER_ID, ARTICLE_2, MULTIMODAL_SIMILARITY_SOURCE, 1, 0.9),
        (CUSTOMER_ID, ARTICLE_2, TWO_TOWER_RETRIEVAL_SOURCE, 1, 1.1),
        (CUSTOMER_ID, ARTICLE_2, TWO_TOWER_RETRIEVAL_LATEST_CUSTOMER_SOURCE, 1, 0.8),
        (SECOND_CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
        (SECOND_CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_SOURCE, 1, 1.0),
        (SECOND_CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_1D_SOURCE, 1, 1.0),
        (SECOND_CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_3D_SOURCE, 1, 1.0),
        (SECOND_CUSTOMER_ID, ARTICLE_2, MULTIMODAL_SIMILARITY_SOURCE, 1, 0.9),
        (SECOND_CUSTOMER_ID, ARTICLE_2, TWO_TOWER_RETRIEVAL_SOURCE, 1, 1.1),
        (SECOND_CUSTOMER_ID, ARTICLE_2, TWO_TOWER_RETRIEVAL_LATEST_CUSTOMER_SOURCE, 1, 0.8),
    ]
    write_candidate_csv(train_path, rows)
    write_candidate_csv(eval_path, rows)

    training = train_linear_ranker_from_csv(
        candidate_path=train_path,
        validation_labels={CUSTOMER_ID: (ARTICLE_2,), SECOND_CUSTOMER_ID: (ARTICLE_2,)},
        config=LinearRankerConfig(
            epochs=30,
            learning_rate=0.25,
            l2=0.0,
            positive_weight=1.0,
        ),
    )
    evaluation = evaluate_linear_ranker_from_csv(
        candidate_path=eval_path,
        validation_labels={CUSTOMER_ID: (ARTICLE_2,), SECOND_CUSTOMER_ID: (ARTICLE_2,)},
        model=training.model,
        k=2,
    )

    assert training.summary.positive_pairs == 2
    assert training.summary.negative_pairs == 2
    assert evaluation.map_at_k == pytest.approx(1.0)
    assert evaluation.baseline_map_at_k == pytest.approx(0.5)
    assert evaluation.delta_vs_baseline_map_at_k == pytest.approx(0.5)


def test_linear_model_schema_is_validated() -> None:
    features = CandidateFeatures(customer_id=CUSTOMER_ID, article_id=ARTICLE_1)
    model = LinearRankerModel(feature_names=("bad",), weights=(0.0,))

    with pytest.raises(ValueError, match="feature_names"):
        score_with_linear_model(features, model)


def test_rank_with_linear_model_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        rank_with_linear_model(
            {}, LinearRankerModel(LINEAR_FEATURE_NAMES, (0.0,) * len(LINEAR_FEATURE_NAMES)), k=0
        )


def test_linear_ranker_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="epochs"):
        LinearRankerConfig(epochs=0)


def test_write_learned_linear_ranker_report(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    write_candidate_csv(train_path, [(CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0)])
    write_candidate_csv(eval_path, [(CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0)])
    config = LinearRankerConfig(epochs=1, learning_rate=0.1)
    training = train_linear_ranker_from_csv(
        candidate_path=train_path,
        validation_labels={CUSTOMER_ID: (ARTICLE_1,)},
        config=config,
    )
    evaluation = evaluate_linear_ranker_from_csv(
        candidate_path=eval_path,
        validation_labels={CUSTOMER_ID: (ARTICLE_1,)},
        model=training.model,
    )
    report = build_learned_linear_ranker_report(
        train_split=TemporalSplit.from_isoformat("2020-01-01"),
        evaluation_split=TemporalSplit.from_isoformat("2020-01-08"),
        k=12,
        candidate_k=12,
        config=config,
        training_result=training,
        evaluation=evaluation,
    )

    path = write_learned_linear_ranker_report(report, tmp_path / "report.json")

    assert path.exists()
    assert '"evaluation"' in path.read_text(encoding="utf-8")


def write_candidate_csv(path: Path, rows: list[tuple[str, str, str, int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        writer.writerows(rows)
