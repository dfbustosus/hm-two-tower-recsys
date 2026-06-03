from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.submission import SubmissionValidationResult
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.ranking.linear import (
    LINEAR_FEATURE_NAMES,
    LinearRankerConfig,
    LinearRankerModel,
    LinearRankerTrainingSummary,
)
from hm_recsys.ranking.submission import (
    build_learned_linear_ranker_submission_report,
    build_linear_ranker_submission_predictions,
    write_learned_linear_ranker_submission_report,
)
from hm_recsys.retrieval.source_names import (
    ALL_TIME_POPULARITY_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
COLD_CUSTOMER_ID = "c" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"


def test_linear_ranker_submission_predictions_backfill_and_preserve_scope() -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 3), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 4), CUSTOMER_ID, ARTICLE_1),
    ]
    weights = [0.0] * len(LINEAR_FEATURE_NAMES)
    weights[LINEAR_FEATURE_NAMES.index("has_recent_popularity")] = 2.0
    model = LinearRankerModel(feature_names=LINEAR_FEATURE_NAMES, weights=tuple(weights))

    submission = build_linear_ranker_submission_predictions(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-05"),
        target_customer_ids=(CUSTOMER_ID, SECOND_CUSTOMER_ID, COLD_CUSTOMER_ID),
        model=model,
        k=2,
        candidate_k=2,
        include_co_visitation=False,
        max_transaction_date=date(2020, 1, 4),
    )

    assert tuple(submission.predictions) == (CUSTOMER_ID, SECOND_CUSTOMER_ID, COLD_CUSTOMER_ID)
    assert all(len(predictions) == 2 for predictions in submission.predictions.values())
    assert all(
        len(set(predictions)) == len(predictions) for predictions in submission.predictions.values()
    )
    assert submission.final_training_cutoff == "2020-01-05"
    assert submission.max_transaction_date == "2020-01-04"
    assert submission.diagnostics.target_customers == 3
    assert submission.diagnostics.customers_with_full_length_predictions == 3
    assert submission.diagnostics.duplicate_prediction_rows == 0
    assert submission.diagnostics.source_row_counts == {
        ALL_TIME_POPULARITY_SOURCE: 6,
        RECENT_POPULARITY_SOURCE: 6,
        REPEAT_SOURCE: 2,
    }


def test_linear_ranker_submission_rejects_invalid_limits() -> None:
    model = LinearRankerModel(feature_names=LINEAR_FEATURE_NAMES, weights=(0.0,) * 15)

    with pytest.raises(ValueError, match="k must be positive"):
        build_linear_ranker_submission_predictions(
            transaction_iter_factory=lambda: iter(()),
            split=TemporalSplit.from_isoformat("2020-01-05"),
            target_customer_ids=(),
            model=model,
            k=0,
        )


def test_write_learned_linear_ranker_submission_report(tmp_path: Path) -> None:
    config = LinearRankerConfig(epochs=1, learning_rate=0.1)
    model = LinearRankerModel(feature_names=LINEAR_FEATURE_NAMES, weights=(0.0,) * 15)
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
    submission = build_linear_ranker_submission_predictions(
        transaction_iter_factory=lambda: iter(
            [TransactionEvent(date(2020, 1, 4), CUSTOMER_ID, ARTICLE_1)]
        ),
        split=TemporalSplit.from_isoformat("2020-01-05"),
        target_customer_ids=(CUSTOMER_ID,),
        model=model,
        k=1,
        candidate_k=1,
        include_co_visitation=False,
        max_transaction_date=date(2020, 1, 4),
    )
    report = build_learned_linear_ranker_submission_report(
        train_split=TemporalSplit.from_isoformat("2019-12-29"),
        final_split=TemporalSplit.from_isoformat("2020-01-05"),
        k=1,
        candidate_k=1,
        popularity_lookback_days=7,
        include_co_visitation=False,
        co_visitation_max_history_items=8,
        co_visitation_max_neighbors_per_item=100,
        config=config,
        model=model,
        training=_training_summary(tmp_path / "train.csv"),
        submission=submission,
        submission_path=tmp_path / "submission.csv",
        validation_report_path=tmp_path / "validation.json",
        validation=validation,
    )

    path = write_learned_linear_ranker_submission_report(report, tmp_path / "report.json")

    assert path.exists()
    assert '"submission"' in path.read_text(encoding="utf-8")


def _training_summary(path: Path) -> LinearRankerTrainingSummary:
    return LinearRankerTrainingSummary(
        candidate_path=str(path),
        candidate_rows=1,
        unique_candidate_pairs=1,
        positive_pairs=1,
        negative_pairs=0,
        positive_weight=1.0,
        epochs=1,
        final_average_loss=0.0,
        source_row_counts={REPEAT_SOURCE: 1},
    )
