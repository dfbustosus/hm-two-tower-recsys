import csv
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.submission import SubmissionValidationResult
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.deterministic import DeterministicRankerWeights
from hm_recsys.ranking.deterministic_tuning import (
    DeterministicRankerTuningGrid,
    select_deterministic_ranker_weights_from_csv,
)
from hm_recsys.ranking.submission import (
    build_deterministic_ranker_submission_predictions,
    build_deterministic_ranker_submission_report,
    write_deterministic_ranker_submission_report,
)
from hm_recsys.retrieval.candidate_export import CANDIDATE_EXPORT_HEADER
from hm_recsys.retrieval.source_names import (
    ALL_TIME_POPULARITY_SOURCE,
    GARMENT_GROUP_POPULARITY_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
COLD_CUSTOMER_ID = "c" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"


def test_deterministic_ranker_submission_predictions_with_garment_source() -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_2),
    ]

    submission = build_deterministic_ranker_submission_predictions(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-03"),
        target_customer_ids=(CUSTOMER_ID, SECOND_CUSTOMER_ID, COLD_CUSTOMER_ID),
        weights=DeterministicRankerWeights(garment_group_popularity_presence_weight=2.0),
        k=2,
        candidate_k=2,
        include_co_visitation=False,
        include_garment_group_popularity=True,
        article_garment_group_by_id={ARTICLE_1: "Tops", ARTICLE_2: "Tops"},
        max_transaction_date=date(2020, 1, 2),
    )

    assert tuple(submission.predictions) == (CUSTOMER_ID, SECOND_CUSTOMER_ID, COLD_CUSTOMER_ID)
    assert all(len(predictions) == 2 for predictions in submission.predictions.values())
    assert submission.final_training_cutoff == "2020-01-03"
    assert submission.max_transaction_date == "2020-01-02"
    assert submission.diagnostics.duplicate_prediction_rows == 0
    assert submission.diagnostics.source_row_counts == {
        ALL_TIME_POPULARITY_SOURCE: 6,
        GARMENT_GROUP_POPULARITY_SOURCE: 4,
        RECENT_POPULARITY_SOURCE: 6,
        REPEAT_SOURCE: 2,
    }


def test_deterministic_ranker_submission_rejects_missing_metadata() -> None:
    with pytest.raises(ValueError, match="article_garment_group_by_id"):
        build_deterministic_ranker_submission_predictions(
            transaction_iter_factory=lambda: iter(()),
            split=TemporalSplit.from_isoformat("2020-01-03"),
            target_customer_ids=(),
            weights=DeterministicRankerWeights(),
            include_garment_group_popularity=True,
        )


def test_write_deterministic_ranker_submission_report(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.csv"
    _write_candidates(candidate_path)
    weight_selection = select_deterministic_ranker_weights_from_csv(
        candidate_path=candidate_path,
        validation_labels={CUSTOMER_ID: (ARTICLE_2,)},
        k=1,
        grid=DeterministicRankerTuningGrid(
            garment_group_popularity_presence_weights=(3.0,),
            garment_group_popularity_score_weights=(0.0,),
            age_segment_popularity_presence_weights=(0.0,),
            age_segment_popularity_score_weights=(0.0,),
            source_count_weights=(0.0,),
            best_rank_score_weights=(0.0,),
        ),
    )
    submission = build_deterministic_ranker_submission_predictions(
        transaction_iter_factory=lambda: iter(
            [TransactionEvent(date(2020, 1, 2), CUSTOMER_ID, ARTICLE_1)]
        ),
        split=TemporalSplit.from_isoformat("2020-01-03"),
        target_customer_ids=(CUSTOMER_ID,),
        weights=weight_selection.selected_weights,
        k=1,
        candidate_k=1,
        include_co_visitation=False,
        max_transaction_date=date(2020, 1, 2),
    )
    validation = SubmissionValidationResult(
        path=str(tmp_path / "submission.csv"),
        valid=True,
        row_count=1,
        expected_customer_count=1,
        missing_customer_count=0,
        extra_customer_count=0,
        duplicate_customer_rows=0,
        rows_with_too_many_predictions=0,
        rows_with_too_few_predictions=0,
        rows_with_duplicate_predictions=0,
        rows_with_invalid_customer_id_format=0,
        rows_with_invalid_article_id_format=0,
        rows_with_unknown_article_ids=0,
        failures=(),
        examples=(),
    )
    report = build_deterministic_ranker_submission_report(
        tuning_split=TemporalSplit.from_isoformat("2019-12-27"),
        final_split=TemporalSplit.from_isoformat("2020-01-03"),
        k=1,
        candidate_k=1,
        popularity_lookback_days=7,
        include_co_visitation=False,
        co_visitation_max_history_items=8,
        co_visitation_max_neighbors_per_item=100,
        include_age_segment_popularity=False,
        age_segment_bucket_size=None,
        age_segment_popularity_lookback_days=None,
        include_garment_group_popularity=False,
        garment_group_popularity_lookback_days=None,
        garment_group_max_history_items=None,
        weight_selection=weight_selection,
        submission=submission,
        submission_path=tmp_path / "submission.csv",
        validation_report_path=tmp_path / "validation.json",
        validation=validation,
    )

    path = write_deterministic_ranker_submission_report(report, tmp_path / "report.json")

    assert path.exists()
    assert "selected_weights" in path.read_text(encoding="utf-8")


def _write_candidates(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_EXPORT_HEADER)
        writer.writerow((CUSTOMER_ID, ARTICLE_1, RECENT_POPULARITY_SOURCE, 1, 1.0))
        writer.writerow((CUSTOMER_ID, ARTICLE_2, GARMENT_GROUP_POPULARITY_SOURCE, 1, 1.0))
