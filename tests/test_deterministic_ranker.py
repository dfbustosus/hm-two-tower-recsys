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
    AGE_SEGMENT_POPULARITY_SOURCE,
    CO_VISITATION_SOURCE,
    GARMENT_GROUP_POPULARITY_SOURCE,
    ITEM2VEC_SIMILARITY_POPULARITY_PRIOR_SOURCE,
    MULTIMODAL_SIMILARITY_SOURCE,
    PRODUCT_CODE_POPULARITY_SOURCE,
    RECENT_POPULARITY_1D_SOURCE,
    RECENT_POPULARITY_3D_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
    SEASONAL_POPULARITY_SOURCE,
    TWO_TOWER_RETRIEVAL_LATEST_CUSTOMER_SOURCE,
    TWO_TOWER_RETRIEVAL_SOURCE,
)

CUSTOMER_ID = "a" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"


def test_aggregate_candidate_features_combines_sources_and_labels() -> None:
    features_by_customer = aggregate_candidate_features(
        records=(
            CandidateRecord(CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0),
            CandidateRecord(CUSTOMER_ID, ARTICLE_1, RECENT_POPULARITY_SOURCE, 2, 0.5),
            CandidateRecord(CUSTOMER_ID, ARTICLE_1, RECENT_POPULARITY_1D_SOURCE, 1, 1.0),
            CandidateRecord(CUSTOMER_ID, ARTICLE_1, RECENT_POPULARITY_3D_SOURCE, 1, 1.0),
            CandidateRecord(CUSTOMER_ID, ARTICLE_1, SEASONAL_POPULARITY_SOURCE, 3, 0.4),
            CandidateRecord(CUSTOMER_ID, ARTICLE_2, CO_VISITATION_SOURCE, 1, 3.0),
            CandidateRecord(CUSTOMER_ID, ARTICLE_2, MULTIMODAL_SIMILARITY_SOURCE, 3, 0.8),
            CandidateRecord(
                CUSTOMER_ID,
                ARTICLE_2,
                ITEM2VEC_SIMILARITY_POPULARITY_PRIOR_SOURCE,
                2,
                0.95,
            ),
            CandidateRecord(CUSTOMER_ID, ARTICLE_2, TWO_TOWER_RETRIEVAL_SOURCE, 4, 1.2),
            CandidateRecord(
                CUSTOMER_ID,
                ARTICLE_2,
                TWO_TOWER_RETRIEVAL_LATEST_CUSTOMER_SOURCE,
                5,
                0.7,
            ),
            CandidateRecord(CUSTOMER_ID, ARTICLE_2, AGE_SEGMENT_POPULARITY_SOURCE, 2, 0.6),
            CandidateRecord(CUSTOMER_ID, ARTICLE_2, GARMENT_GROUP_POPULARITY_SOURCE, 1, 0.9),
            CandidateRecord(CUSTOMER_ID, ARTICLE_2, PRODUCT_CODE_POPULARITY_SOURCE, 2, 0.75),
        ),
        validation_labels={CUSTOMER_ID: (ARTICLE_2,)},
    )

    article_1 = features_by_customer[CUSTOMER_ID][ARTICLE_1]
    article_2 = features_by_customer[CUSTOMER_ID][ARTICLE_2]

    assert article_1.label == 0
    assert article_1.has_repeat
    assert article_1.has_recent_popularity
    assert article_1.has_recent_popularity_1d
    assert article_1.has_recent_popularity_3d
    assert article_1.has_seasonal_popularity
    assert article_1.seasonal_popularity_score == 0.4
    assert article_1.source_count == 5
    assert article_1.best_rank == 1
    assert article_2.label == 1
    assert article_2.has_co_visitation
    assert article_2.co_visitation_score == 3.0
    assert article_2.has_content_similarity
    assert article_2.content_similarity_rank == 2
    assert article_2.content_similarity_score == 0.95
    assert article_2.has_item2vec_similarity
    assert article_2.item2vec_similarity_rank == 2
    assert article_2.item2vec_similarity_score == 0.95
    assert article_2.has_two_tower_retrieval
    assert article_2.two_tower_retrieval_score == 1.2
    assert article_2.two_tower_retrieval_rank == 4
    assert article_2.has_two_tower_retrieval_latest_customer
    assert article_2.two_tower_retrieval_latest_customer_score == 0.7
    assert article_2.two_tower_retrieval_latest_customer_rank == 5
    assert article_2.has_age_segment_popularity
    assert article_2.age_segment_popularity_score == 0.6
    assert article_2.has_garment_group_popularity
    assert article_2.garment_group_popularity_score == 0.9
    assert article_2.has_product_code_popularity
    assert article_2.product_code_popularity_score == 0.75
    assert article_2.product_code_popularity_rank == 2


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


def test_deterministic_ranker_scores_two_tower_features_separately() -> None:
    two_tower_features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_1,
        two_tower_retrieval_rank=1,
        two_tower_retrieval_score=2.0,
    )
    content_features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_2,
        content_similarity_rank=1,
        content_similarity_score=2.0,
    )
    weights = DeterministicRankerWeights(
        content_similarity_presence_weight=0.0,
        content_similarity_score_weight=0.0,
        two_tower_retrieval_presence_weight=1.0,
        two_tower_retrieval_score_weight=0.5,
    )

    assert score_candidate(two_tower_features, weights) > score_candidate(content_features, weights)


def test_deterministic_ranker_scores_two_tower_variants_separately() -> None:
    latest_features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_1,
        two_tower_retrieval_rank=1,
        two_tower_retrieval_score=2.0,
    )
    latest_customer_features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_2,
        two_tower_retrieval_latest_customer_rank=1,
        two_tower_retrieval_latest_customer_score=2.0,
    )
    weights = DeterministicRankerWeights(
        two_tower_retrieval_presence_weight=0.0,
        two_tower_retrieval_score_weight=0.0,
        two_tower_retrieval_latest_customer_presence_weight=1.0,
        two_tower_retrieval_latest_customer_score_weight=0.5,
    )

    assert score_candidate(latest_customer_features, weights) > score_candidate(
        latest_features, weights
    )


def test_deterministic_ranker_scores_two_tower_rank_reciprocal() -> None:
    top_ranked_features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_1,
        two_tower_retrieval_rank=1,
        two_tower_retrieval_latest_customer_rank=1,
    )
    lower_ranked_features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_2,
        two_tower_retrieval_rank=10,
        two_tower_retrieval_latest_customer_rank=10,
    )
    weights = DeterministicRankerWeights(
        two_tower_retrieval_presence_weight=0.0,
        two_tower_retrieval_score_weight=0.0,
        two_tower_retrieval_rank_weight=1.0,
        two_tower_retrieval_latest_customer_presence_weight=0.0,
        two_tower_retrieval_latest_customer_score_weight=0.0,
        two_tower_retrieval_latest_customer_rank_weight=1.0,
    )

    assert score_candidate(top_ranked_features, weights) > score_candidate(
        lower_ranked_features, weights
    )


def test_deterministic_ranker_scores_product_code_popularity() -> None:
    product_code_features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_1,
        product_code_popularity_rank=1,
        product_code_popularity_score=1.0,
    )
    other_features = CandidateFeatures(
        customer_id=CUSTOMER_ID,
        article_id=ARTICLE_2,
        garment_group_popularity_rank=1,
        garment_group_popularity_score=1.0,
    )
    weights = DeterministicRankerWeights(
        product_code_popularity_presence_weight=1.0,
        product_code_popularity_score_weight=0.5,
        garment_group_popularity_presence_weight=0.0,
        garment_group_popularity_score_weight=0.0,
    )

    assert score_candidate(product_code_features, weights) > score_candidate(
        other_features,
        weights,
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


def test_iter_candidate_records_accepts_both_optional_columns(tmp_path: Path) -> None:
    """Reader must accept canonical + ``two_tower_score`` + ``content_user_cosine``.

    Guards the schema seam between
    ``score-two-tower-candidates`` + ``score-content-similarity-candidates``
    (which can both write to the same CSV) and downstream rankers.
    """

    from hm_recsys.ranking.deterministic import iter_candidate_records_from_csv
    from hm_recsys.retrieval.candidate_export import (
        CONTENT_USER_COSINE_COLUMN,
        TWO_TOWER_SCORE_COLUMN,
    )

    candidate_path = tmp_path / "both.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                *CANDIDATE_EXPORT_HEADER,
                TWO_TOWER_SCORE_COLUMN,
                CONTENT_USER_COSINE_COLUMN,
            )
        )
        writer.writerow((CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0, 0.42, 0.91))

    [record] = list(iter_candidate_records_from_csv(candidate_path))
    assert record.two_tower_score == pytest.approx(0.42)
    assert record.content_user_cosine == pytest.approx(0.91)


def test_iter_candidate_records_accepts_only_content_user_cosine(tmp_path: Path) -> None:
    """``content_user_cosine`` alone (without two_tower_score) is valid."""

    from hm_recsys.ranking.deterministic import iter_candidate_records_from_csv
    from hm_recsys.retrieval.candidate_export import CONTENT_USER_COSINE_COLUMN

    candidate_path = tmp_path / "content_only.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((*CANDIDATE_EXPORT_HEADER, CONTENT_USER_COSINE_COLUMN))
        writer.writerow((CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0, 0.5))

    [record] = list(iter_candidate_records_from_csv(candidate_path))
    assert record.two_tower_score == 0.0
    assert record.content_user_cosine == pytest.approx(0.5)


def test_iter_candidate_records_rejects_optional_columns_out_of_order(
    tmp_path: Path,
) -> None:
    """Optional columns MUST be in the canonical order to prevent ambiguity."""

    from hm_recsys.ranking.deterministic import iter_candidate_records_from_csv
    from hm_recsys.retrieval.candidate_export import (
        CONTENT_USER_COSINE_COLUMN,
        TWO_TOWER_SCORE_COLUMN,
    )

    candidate_path = tmp_path / "wrong_order.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # Wrong order: content_user_cosine BEFORE two_tower_score.
        writer.writerow(
            (*CANDIDATE_EXPORT_HEADER, CONTENT_USER_COSINE_COLUMN, TWO_TOWER_SCORE_COLUMN)
        )
        writer.writerow((CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0, 0.5, 0.6))

    with pytest.raises(ValueError, match="unknown or out-of-order column"):
        list(iter_candidate_records_from_csv(candidate_path))


def test_iter_candidate_records_accepts_augmented_two_tower_score_column(
    tmp_path: Path,
) -> None:
    """Reader must accept the canonical header + ``two_tower_score`` sidecar.

    This is the seam between ``score-two-tower-candidates`` and any
    downstream ranker (deterministic, LightGBM, ensemble). If the
    augmented header is rejected, the entire two-tower-as-feature
    pipeline breaks silently the moment a real scored CSV reaches a
    ranker.
    """

    from hm_recsys.ranking.deterministic import iter_candidate_records_from_csv
    from hm_recsys.retrieval.candidate_export import TWO_TOWER_SCORE_COLUMN

    candidate_path = tmp_path / "augmented.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((*CANDIDATE_EXPORT_HEADER, TWO_TOWER_SCORE_COLUMN))
        writer.writerow((CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0, 0.42))
        writer.writerow((CUSTOMER_ID, ARTICLE_2, RECENT_POPULARITY_SOURCE, 1, 1.0, -0.10))

    records = list(iter_candidate_records_from_csv(candidate_path))
    assert len(records) == 2
    assert records[0].two_tower_score == pytest.approx(0.42)
    assert records[1].two_tower_score == pytest.approx(-0.10)


def test_iter_candidate_records_defaults_two_tower_score_when_canonical_header(
    tmp_path: Path,
) -> None:
    """Canonical header must produce records with zero ``two_tower_score``.

    Guards backwards compatibility: all existing candidate CSVs on disk
    have the canonical 5-column header, and they must keep yielding
    valid records without surprise non-zero defaults.
    """

    from hm_recsys.ranking.deterministic import iter_candidate_records_from_csv

    candidate_path = tmp_path / "canonical.csv"
    write_candidate_csv(
        candidate_path,
        [(CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0)],
    )

    records = list(iter_candidate_records_from_csv(candidate_path))
    assert len(records) == 1
    assert records[0].two_tower_score == 0.0


def test_iter_candidate_records_rejects_unknown_header_columns(tmp_path: Path) -> None:
    """Unknown trailing columns are rejected loudly to prevent schema drift.

    Only ``two_tower_score`` is an accepted optional column today; any
    other extra column likely indicates a writer/reader version mismatch
    and would silently corrupt feature vectors if accepted.
    """

    from hm_recsys.ranking.deterministic import iter_candidate_records_from_csv

    candidate_path = tmp_path / "stray.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((*CANDIDATE_EXPORT_HEADER, "stray_extra_column"))
        writer.writerow((CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0, "x"))

    with pytest.raises(ValueError, match="candidate CSV header"):
        list(iter_candidate_records_from_csv(candidate_path))


def test_update_from_record_propagates_two_tower_score_to_features() -> None:
    """Aggregation must surface the pair-level ``two_tower_score`` on features.

    This is the contract that downstream LightGBM feature-vector builders
    rely on: after aggregating all source rows for a (customer, article)
    pair, ``CandidateFeatures.two_tower_score`` must equal the maximum
    seen across source rows (which, by design, is the single shared
    pair-level value).
    """

    feature = CandidateFeatures(customer_id=CUSTOMER_ID, article_id=ARTICLE_1)
    feature.update_from_record(
        CandidateRecord(
            CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, 1, 1.0, two_tower_score=0.7
        )
    )
    feature.update_from_record(
        CandidateRecord(
            CUSTOMER_ID, ARTICLE_1, RECENT_POPULARITY_SOURCE, 1, 0.5,
            two_tower_score=0.7,  # pair-level, identical
        )
    )
    assert feature.two_tower_score == pytest.approx(0.7)
