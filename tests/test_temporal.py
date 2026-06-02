from datetime import date

from hm_recsys.safe_io import TransactionEvent
from hm_recsys.temporal import (
    TemporalSplit,
    assign_split_bucket,
    collect_validation_labels,
    summarize_temporal_split,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
ARTICLE_ID = "0000000001"
SECOND_ARTICLE_ID = "0000000002"
THIRD_ARTICLE_ID = "0000000003"


def test_assign_split_bucket_uses_inclusive_cutoff_and_exclusive_end() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")

    assert assign_split_bucket(date(2020, 9, 15), split) == "train"
    assert assign_split_bucket(date(2020, 9, 16), split) == "validation"
    assert assign_split_bucket(date(2020, 9, 22), split) == "validation"
    assert assign_split_bucket(date(2020, 9, 23), split) == "future"


def test_summarize_temporal_split_counts_rows_customers_articles_and_pairs() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    events = [
        TransactionEvent(date(2020, 9, 15), CUSTOMER_ID, ARTICLE_ID),
        TransactionEvent(date(2020, 9, 16), CUSTOMER_ID, ARTICLE_ID),
        TransactionEvent(date(2020, 9, 16), CUSTOMER_ID, ARTICLE_ID),
        TransactionEvent(date(2020, 9, 22), SECOND_CUSTOMER_ID, SECOND_ARTICLE_ID),
        TransactionEvent(date(2020, 9, 23), SECOND_CUSTOMER_ID, THIRD_ARTICLE_ID),
    ]

    summary = summarize_temporal_split(events, split)

    assert summary.train_rows == 1
    assert summary.validation_rows == 3
    assert summary.future_rows == 1
    assert summary.validation_customers == 2
    assert summary.validation_articles == 2
    assert summary.validation_unique_customer_article_pairs == 2


def test_collect_validation_labels_deduplicates_repeated_actual_purchases() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    events = [
        TransactionEvent(date(2020, 9, 16), CUSTOMER_ID, ARTICLE_ID),
        TransactionEvent(date(2020, 9, 16), CUSTOMER_ID, ARTICLE_ID),
        TransactionEvent(date(2020, 9, 17), CUSTOMER_ID, SECOND_ARTICLE_ID),
        TransactionEvent(date(2020, 9, 23), CUSTOMER_ID, THIRD_ARTICLE_ID),
    ]

    assert collect_validation_labels(events, split) == {
        CUSTOMER_ID: (ARTICLE_ID, SECOND_ARTICLE_ID)
    }
