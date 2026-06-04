import csv
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import CsvValueError, TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.segment_popularity import (
    UNKNOWN_AGE_SEGMENT,
    age_to_segment,
    build_age_segment_popularity_candidates,
    build_age_segment_popularity_index,
    load_customer_age_segments,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64


def test_load_customer_age_segments_preserves_ids_and_buckets(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with (raw_dir / "customers.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("customer_id", "age"))
        writer.writerow((CUSTOMER_ID, "34"))
        writer.writerow((SECOND_CUSTOMER_ID, ""))

    segments = load_customer_age_segments(raw_dir, bucket_size=10)

    assert segments[CUSTOMER_ID] == "age_30_39"
    assert segments[SECOND_CUSTOMER_ID] == UNKNOWN_AGE_SEGMENT


def test_age_to_segment_rejects_invalid_values() -> None:
    assert age_to_segment("25", bucket_size=5) == "age_25_29"
    with pytest.raises(ValueError, match="bucket_size"):
        age_to_segment("25", bucket_size=0)
    with pytest.raises(CsvValueError, match="invalid age"):
        age_to_segment("not-an-age", bucket_size=10)
    with pytest.raises(CsvValueError, match="invalid age"):
        age_to_segment("0", bucket_size=10)


def test_age_segment_popularity_uses_only_recent_pre_cutoff_rows() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    article_1 = "0100000000"
    article_2 = "0200000000"
    article_3 = "0300000000"
    transactions = (
        TransactionEvent(date(2020, 9, 10), CUSTOMER_ID, article_1),
        TransactionEvent(date(2020, 9, 11), SECOND_CUSTOMER_ID, article_2),
        TransactionEvent(date(2020, 9, 1), SECOND_CUSTOMER_ID, article_3),
        TransactionEvent(date(2020, 9, 16), SECOND_CUSTOMER_ID, article_3),
    )
    segments = {CUSTOMER_ID: "age_30_39", SECOND_CUSTOMER_ID: "age_30_39"}

    index = build_age_segment_popularity_index(
        transactions,
        split,
        segments,
        lookback_days=7,
    )
    candidates = build_age_segment_popularity_candidates(index, CUSTOMER_ID, k=3)

    assert index.train_rows_used == 3
    assert tuple(candidate.article_id for candidate in candidates) == (article_1, article_2)
    assert article_3 not in {candidate.article_id for candidate in candidates}


def test_age_segment_popularity_handles_missing_segment_and_invalid_k() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    index = build_age_segment_popularity_index((), split, {}, lookback_days=7)

    assert build_age_segment_popularity_candidates(index, CUSTOMER_ID, k=1) == ()
    with pytest.raises(ValueError, match="k"):
        build_age_segment_popularity_candidates(index, CUSTOMER_ID, k=0)
    with pytest.raises(ValueError, match="lookback_days"):
        build_age_segment_popularity_index((), split, {}, lookback_days=0)
