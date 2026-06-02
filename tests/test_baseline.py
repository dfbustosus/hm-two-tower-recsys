from datetime import date

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.baselines import (
    ArticleStats,
    build_repeat_popularity_baseline,
    build_repeat_popularity_submission_baseline,
    evaluate_repeat_popularity_baseline,
    find_max_transaction_date,
    rank_article_stats,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
COLD_CUSTOMER_ID = "c" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
ARTICLE_3 = "0000000003"
ARTICLE_4 = "0000000004"


def test_rank_article_stats_orders_by_count_recency_then_article_id() -> None:
    stats = {
        ARTICLE_2: make_stats(count=2, last_seen=date(2020, 1, 1)),
        ARTICLE_1: make_stats(count=2, last_seen=date(2020, 1, 2)),
        ARTICLE_3: make_stats(count=1, last_seen=date(2020, 1, 9)),
    }

    assert rank_article_stats(stats, limit=3) == (ARTICLE_1, ARTICLE_2, ARTICLE_3)


def test_repeat_popularity_baseline_repeats_then_popularity_backfill() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 3), SECOND_CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 6), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 7), CUSTOMER_ID, ARTICLE_1),
    ]

    baseline = build_repeat_popularity_baseline(
        transactions=events,
        split=split,
        target_customer_ids=[CUSTOMER_ID, COLD_CUSTOMER_ID],
        k=3,
        popularity_lookback_days=7,
    )

    assert baseline.repeat_recommendations[CUSTOMER_ID] == (ARTICLE_1, ARTICLE_2)
    assert baseline.predictions[CUSTOMER_ID] == (ARTICLE_1, ARTICLE_2, ARTICLE_3)
    assert baseline.predictions[COLD_CUSTOMER_ID] == (ARTICLE_1, ARTICLE_3, ARTICLE_2)


def test_baseline_evaluation_uses_only_pre_cutoff_rows_for_candidates() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    validation_only_article = ARTICLE_4
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 7), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, validation_only_article),
        TransactionEvent(date(2020, 1, 9), SECOND_CUSTOMER_ID, ARTICLE_2),
    ]

    report = evaluate_repeat_popularity_baseline(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        k=2,
        popularity_lookback_days=7,
    )
    baseline = build_repeat_popularity_baseline(
        transactions=events,
        split=split,
        target_customer_ids=[CUSTOMER_ID, SECOND_CUSTOMER_ID],
        k=2,
        popularity_lookback_days=7,
    )

    assert validation_only_article not in baseline.recent_popularity
    assert validation_only_article not in baseline.predictions[CUSTOMER_ID]
    assert report.split_summary.validation_rows == 2
    assert report.split_summary.train_rows == 2
    assert report.map_at_k == pytest.approx(0.5)


def test_baseline_backfills_to_k_when_repeat_items_overlap_popularity() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 3), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 4), SECOND_CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 5), SECOND_CUSTOMER_ID, ARTICLE_4),
    ]

    baseline = build_repeat_popularity_baseline(
        transactions=events,
        split=split,
        target_customer_ids=[CUSTOMER_ID],
        k=4,
        popularity_lookback_days=7,
    )

    assert len(baseline.predictions[CUSTOMER_ID]) == 4
    assert len(set(baseline.predictions[CUSTOMER_ID])) == 4


def test_baseline_evaluation_reports_perfect_score_for_repeat_hit() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_1),
    ]

    report = evaluate_repeat_popularity_baseline(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        k=1,
        popularity_lookback_days=7,
    )

    assert report.map_at_k == pytest.approx(1.0)
    assert report.recall_at_k == pytest.approx(1.0)
    assert report.diagnostics.customers_with_full_length_predictions == 1


def test_baseline_evaluation_predicts_target_universe_but_scores_label_customers() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_1),
    ]

    report = evaluate_repeat_popularity_baseline(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        target_customer_ids=[CUSTOMER_ID, COLD_CUSTOMER_ID],
        k=1,
        popularity_lookback_days=7,
    )

    assert report.diagnostics.target_customers == 2
    assert report.diagnostics.evaluated_customers == 1
    assert report.diagnostics.customers_with_full_length_predictions == 2
    assert report.diagnostics.prediction_coverage == pytest.approx(1.0)
    assert report.map_at_k == pytest.approx(1.0)


def test_submission_baseline_uses_all_training_rows_and_sample_universe() -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_2),
    ]

    submission = build_repeat_popularity_submission_baseline(
        transaction_iter_factory=lambda: iter(events),
        target_customer_ids=[CUSTOMER_ID, COLD_CUSTOMER_ID],
        k=2,
        popularity_lookback_days=7,
    )

    assert submission.max_transaction_date == date(2020, 1, 8)
    assert submission.training_cutoff == date(2020, 1, 9)
    assert submission.predictions.train_rows_used == 2
    assert submission.diagnostics.target_customers == 2
    assert submission.predictions.predictions[CUSTOMER_ID] == (ARTICLE_2, ARTICLE_1)
    assert submission.predictions.predictions[COLD_CUSTOMER_ID] == (ARTICLE_2, ARTICLE_1)


def test_find_max_transaction_date_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="without transactions"):
        find_max_transaction_date(())


def make_stats(count: int, last_seen: date) -> ArticleStats:
    stats = ArticleStats()
    stats.count = count
    stats.last_seen = last_seen
    return stats
