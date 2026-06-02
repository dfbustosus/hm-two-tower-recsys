from datetime import date

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.co_visitation import (
    build_co_visitation_candidates,
    build_co_visitation_index,
    co_visitation_article_coverage,
    co_visitation_candidate_count,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
ARTICLE_3 = "0000000003"
ARTICLE_4 = "0000000004"


def test_co_visitation_index_uses_only_pre_cutoff_transactions() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    validation_only_article = ARTICLE_4
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, validation_only_article),
    ]

    index = build_co_visitation_index(
        transactions=events,
        split=split,
        target_customer_ids=(CUSTOMER_ID,),
        max_history_items=3,
        max_neighbors_per_item=10,
    )

    assert validation_only_article not in index.customer_histories[CUSTOMER_ID]
    assert validation_only_article not in index.neighbors_by_article
    assert build_co_visitation_candidates(index, CUSTOMER_ID, k=3) == (ARTICLE_1, ARTICLE_2)


def test_co_visitation_candidates_are_scored_by_neighbor_and_history_recency() -> None:
    split = TemporalSplit.from_isoformat("2020-01-10")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 3), CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 1), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_3),
    ]

    index = build_co_visitation_index(
        transactions=events,
        split=split,
        target_customer_ids=(CUSTOMER_ID, SECOND_CUSTOMER_ID),
        max_history_items=3,
        max_neighbors_per_item=10,
    )

    assert index.customer_histories[CUSTOMER_ID] == (ARTICLE_3, ARTICLE_2, ARTICLE_1)
    assert build_co_visitation_candidates(index, CUSTOMER_ID, k=3) == (
        ARTICLE_2,
        ARTICLE_1,
        ARTICLE_3,
    )
    assert co_visitation_candidate_count(index, CUSTOMER_ID, k=2) == 2
    assert co_visitation_article_coverage(index, (CUSTOMER_ID,)) == 3


def test_co_visitation_rejects_invalid_configuration() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")

    with pytest.raises(ValueError, match="max_history_items"):
        build_co_visitation_index(
            transactions=(),
            split=split,
            target_customer_ids=(),
            max_history_items=0,
        )

    index = build_co_visitation_index(
        transactions=(),
        split=split,
        target_customer_ids=(),
    )
    with pytest.raises(ValueError, match="k must be positive"):
        build_co_visitation_candidates(index, CUSTOMER_ID, k=0)
