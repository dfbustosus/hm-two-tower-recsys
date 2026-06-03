import csv
from pathlib import Path

import pytest

from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.deterministic import (
    CandidateFeatures,
    DeterministicRankerWeights,
    aggregate_candidate_features,
    build_source_order_baseline_predictions,
    evaluate_deterministic_ranker_from_csv,
    rank_candidates_by_customer,
    score_candidate,
    write_deterministic_ranker_report,
)
from hm_recsys.retrieval.candidate_export import CANDIDATE_EXPORT_HEADER, CandidateRecord
from hm_recsys.retrieval.source_names import (
    CO_VISITATION_SOURCE,
    MULTIMODAL_SIMILARITY_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
)

CUSTOMER_ID = "a" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"


def test_aggregate_candidate_features_combines_sources_and_labels() -> None:
    features_by_customer = aggregate_candidate_features(
        records=(
            CandidateRecord(CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
            CandidateRecord(CUSTOMER_ID, ARTICLE_1, RECENT_POPULARITY_SOURCE, 2, 0.5),
            CandidateRecord(CUSTOMER_ID, ARTICLE_2, CO_VISITATION_SOURCE, 1, 3.0),
            CandidateRecord(CUSTOMER_ID, ARTICLE_2, MULTIMODAL_SIMILARITY_SOURCE, 3, 0.8),
        ),
        validation_labels={CUSTOMER_ID: (ARTICLE_2,)},
    )

    article_1 = features_by_customer[CUSTOMER_ID][ARTICLE_1]
    article_2 = features_by_customer[CUSTOMER_ID][ARTICLE_2]

    assert article_1.label == 0
    assert article_1.has_repeat
    assert article_1.has_recent_popularity
    assert article_1.source_count == 2
    assert article_1.best_rank == 1
    assert article_2.label == 1
    assert article_2.has_co_visitation
    assert article_2.co_visitation_score == 3.0
    assert article_2.has_content_similarity
    assert article_2.content_similarity_score == 0.8


def test_deterministic_ranker_scoring_and_source_order_baseline() -> None:
    features_by_customer = {
        CUSTOMER_ID: {
            ARTICLE_1: CandidateFeatures(
                customer_id=CUSTOMER_ID,
                article_id=ARTICLE_1,
                repeat_rank=1,
                repeat_score=1.0,
                source_count=1,
                best_rank=1,
                max_source_score=1.0,
            ),
            ARTICLE_2: CandidateFeatures(
                customer_id=CUSTOMER_ID,
                article_id=ARTICLE_2,
                recent_popularity_rank=1,
                recent_popularity_score=1.0,
                source_count=1,
                best_rank=1,
                max_source_score=1.0,
            ),
        }
    }
    weights = DeterministicRankerWeights(
        repeat_presence_weight=0.0,
        repeat_score_weight=0.0,
        recent_popularity_presence_weight=10.0,
        recent_popularity_score_weight=0.0,
        content_similarity_presence_weight=0.0,
        content_similarity_score_weight=0.0,
    )

    assert score_candidate(features_by_customer[CUSTOMER_ID][ARTICLE_2], weights) > score_candidate(
        features_by_customer[CUSTOMER_ID][ARTICLE_1], weights
    )
    assert rank_candidates_by_customer(features_by_customer, k=2, weights=weights)[CUSTOMER_ID] == (
        ARTICLE_2,
        ARTICLE_1,
    )
    assert build_source_order_baseline_predictions(features_by_customer, k=2)[CUSTOMER_ID] == (
        ARTICLE_1,
        ARTICLE_2,
    )


def test_evaluate_deterministic_ranker_from_csv_reports_same_scope_delta(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.csv"
    write_candidate_csv(
        candidate_path,
        [
            (CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
            (CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_SOURCE, 1, 1.0),
        ],
    )
    split = TemporalSplit.from_isoformat("2020-01-08")
    weights = DeterministicRankerWeights(
        repeat_presence_weight=0.0,
        repeat_score_weight=0.0,
        recent_popularity_presence_weight=10.0,
        recent_popularity_score_weight=0.0,
    )

    report = evaluate_deterministic_ranker_from_csv(
        candidate_path=candidate_path,
        validation_labels={CUSTOMER_ID: (ARTICLE_2,)},
        split=split,
        k=2,
        weights=weights,
    )

    assert report.candidate_rows == 2
    assert report.unique_candidate_pairs == 2
    assert report.evaluated_customers == 1
    assert report.map_at_k == pytest.approx(1.0)
    assert report.baseline_map_at_k == pytest.approx(0.5)
    assert report.delta_map_at_k == pytest.approx(0.5)
    assert report.duplicate_prediction_rows == 0


def test_evaluate_deterministic_ranker_rejects_invalid_header(tmp_path: Path) -> None:
    candidate_path = tmp_path / "bad.csv"
    candidate_path.write_text("customer_id,article_id\n", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate CSV header"):
        evaluate_deterministic_ranker_from_csv(
            candidate_path=candidate_path,
            validation_labels={},
            split=TemporalSplit.from_isoformat("2020-01-08"),
        )


def test_write_deterministic_ranker_report(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.csv"
    write_candidate_csv(candidate_path, [(CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0)])
    report = evaluate_deterministic_ranker_from_csv(
        candidate_path=candidate_path,
        validation_labels={CUSTOMER_ID: (ARTICLE_1,)},
        split=TemporalSplit.from_isoformat("2020-01-08"),
    )

    report_path = write_deterministic_ranker_report(report, tmp_path / "report.json")

    assert report_path.exists()
    assert '"map_at_k"' in report_path.read_text(encoding="utf-8")


def write_candidate_csv(path: Path, rows: list[tuple[str, str, str, int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        writer.writerows(rows)
