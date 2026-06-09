from datetime import date

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.seasonality import (
    build_seasonal_popularity_candidates,
    build_seasonal_popularity_index,
)

CUSTOMER_ID = "a" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
VALIDATION_ARTICLE = "0000000003"


def test_seasonal_popularity_uses_shifted_pre_cutoff_window_only() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    events = [
        TransactionEvent(date(2019, 9, 16), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2019, 9, 17), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2019, 9, 18), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2019, 9, 23), CUSTOMER_ID, "0000000004"),
        TransactionEvent(date(2020, 9, 16), CUSTOMER_ID, VALIDATION_ARTICLE),
    ]

    index = build_seasonal_popularity_index(
        events,
        split,
        shift_days=366,
        window_days=7,
        max_articles=2,
    )
    candidates = build_seasonal_popularity_candidates(index, k=2)

    assert index.window_start == date(2019, 9, 16)
    assert index.window_end_exclusive == date(2019, 9, 23)
    assert index.seasonal_rows_used == 3
    assert [candidate.article_id for candidate in candidates] == [ARTICLE_1, ARTICLE_2]
    assert VALIDATION_ARTICLE not in {candidate.article_id for candidate in candidates}


def test_seasonal_popularity_rejects_cutoff_overlapping_window() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")

    with pytest.raises(ValueError, match="must end no later than the cutoff"):
        build_seasonal_popularity_index(
            (),
            split,
            shift_days=3,
            window_days=7,
        )
