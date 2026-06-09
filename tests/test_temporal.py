from datetime import date

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import (
    TemporalSplit,
    assign_split_bucket,
    collect_validation_labels,
    collect_validation_labels_for_splits,
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


def test_collect_validation_labels_for_splits_matches_per_split_helper() -> None:
    splits = (
        TemporalSplit.from_isoformat("2020-09-02"),
        TemporalSplit.from_isoformat("2020-09-09"),
        TemporalSplit.from_isoformat("2020-09-16"),
    )
    events = [
        TransactionEvent(date(2020, 8, 30), CUSTOMER_ID, ARTICLE_ID),
        TransactionEvent(date(2020, 9, 2), CUSTOMER_ID, ARTICLE_ID),
        TransactionEvent(date(2020, 9, 2), CUSTOMER_ID, ARTICLE_ID),
        TransactionEvent(date(2020, 9, 4), CUSTOMER_ID, SECOND_ARTICLE_ID),
        TransactionEvent(date(2020, 9, 10), SECOND_CUSTOMER_ID, SECOND_ARTICLE_ID),
        TransactionEvent(date(2020, 9, 12), SECOND_CUSTOMER_ID, THIRD_ARTICLE_ID),
        TransactionEvent(date(2020, 9, 19), CUSTOMER_ID, THIRD_ARTICLE_ID),
        TransactionEvent(date(2020, 9, 30), CUSTOMER_ID, THIRD_ARTICLE_ID),
    ]

    actual = collect_validation_labels_for_splits(events, splits)
    expected = {split.cutoff: collect_validation_labels(events, split) for split in splits}
    assert actual == expected


def test_collect_validation_labels_for_splits_rejects_duplicates() -> None:
    split = TemporalSplit.from_isoformat("2020-09-09")
    with pytest.raises(ValueError, match="unique cutoff dates"):
        collect_validation_labels_for_splits([], (split, split))


def test_collect_validation_labels_for_splits_rejects_empty_splits() -> None:
    with pytest.raises(ValueError, match="at least one temporal split"):
        collect_validation_labels_for_splits([], ())
