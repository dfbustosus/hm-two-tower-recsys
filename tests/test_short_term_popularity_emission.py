"""Tests for the Phase-0.6 recent_popularity_1d / _3d emission fix.

Prior to the fix the retrieval stack only emitted the generic
``recent_popularity`` source. Consumers (linear ranker, deterministic ranker,
ranker-ready candidate CSV) reserved feature slots for ``recent_popularity_1d``
and ``recent_popularity_3d`` but their weights were silently dead — a real
ranker bug that depressed MAP@12 because the features were guaranteed zero.

These tests pin the corrected behaviour so the regression cannot reappear.
"""

from __future__ import annotations

from datetime import date

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.baselines import (
    DEFAULT_SHORT_TERM_POPULARITY_LOOKBACK_DAYS,
    build_repeat_popularity_candidate_sources,
)
from hm_recsys.retrieval.candidate_export import iter_candidate_records_for_customer

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
ARTICLE_ONE = "0000000001"
ARTICLE_TWO = "0000000002"
ARTICLE_THREE = "0000000003"


def _event(t_dat: date, customer_id: str, article_id: str) -> TransactionEvent:
    return TransactionEvent(t_dat=t_dat, customer_id=customer_id, article_id=article_id)


def test_default_lookbacks_include_one_and_three_days() -> None:
    assert DEFAULT_SHORT_TERM_POPULARITY_LOOKBACK_DAYS == (1, 3)


def test_build_sources_emits_separate_short_term_rankings() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    events = [
        _event(date(2020, 9, 9), CUSTOMER_ID, ARTICLE_ONE),
        _event(date(2020, 9, 14), CUSTOMER_ID, ARTICLE_TWO),
        _event(date(2020, 9, 15), CUSTOMER_ID, ARTICLE_THREE),
        _event(date(2020, 9, 15), SECOND_CUSTOMER_ID, ARTICLE_THREE),
    ]
    sources = build_repeat_popularity_candidate_sources(
        transactions=events,
        split=split,
        target_customer_ids=(CUSTOMER_ID,),
    )
    one_day = sources.recent_popularity_by_lookback[1]
    three_day = sources.recent_popularity_by_lookback[3]
    seven_day = sources.recent_popularity
    assert one_day[0] == ARTICLE_THREE
    assert ARTICLE_THREE in three_day
    assert ARTICLE_TWO in three_day
    assert ARTICLE_ONE not in one_day
    assert ARTICLE_ONE not in three_day
    assert ARTICLE_ONE in seven_day


def test_build_sources_with_no_short_term_lookbacks_omits_extra_rankings() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    events = [_event(date(2020, 9, 15), CUSTOMER_ID, ARTICLE_ONE)]
    sources = build_repeat_popularity_candidate_sources(
        transactions=events,
        split=split,
        target_customer_ids=(CUSTOMER_ID,),
        short_term_popularity_lookback_days=(),
    )
    assert sources.recent_popularity_by_lookback == {}


def test_build_sources_rejects_non_positive_short_term_lookbacks() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    with pytest.raises(ValueError, match="short_term_popularity_lookback_days"):
        build_repeat_popularity_candidate_sources(
            transactions=(),
            split=split,
            target_customer_ids=(CUSTOMER_ID,),
            short_term_popularity_lookback_days=(0, 1),
        )


def test_iter_candidate_records_emits_short_term_source_rows() -> None:
    records = list(
        iter_candidate_records_for_customer(
            customer_id=CUSTOMER_ID,
            repeat_recommendations={CUSTOMER_ID: (ARTICLE_ONE,)},
            recent_popularity=(ARTICLE_ONE, ARTICLE_TWO),
            all_time_popularity=(ARTICLE_ONE, ARTICLE_TWO, ARTICLE_THREE),
            co_visitation_index=None,
            k=4,
            recent_popularity_by_lookback={
                1: (ARTICLE_THREE,),
                3: (ARTICLE_THREE, ARTICLE_TWO),
            },
        )
    )
    sources_emitted = {record.source for record in records}
    assert {"recent_popularity_1d", "recent_popularity_3d"}.issubset(sources_emitted)
    one_day_record = next(record for record in records if record.source == "recent_popularity_1d")
    assert one_day_record.article_id == ARTICLE_THREE
    assert one_day_record.source_rank == 1


def test_iter_candidate_records_handles_unknown_short_term_lookback_label() -> None:
    records = list(
        iter_candidate_records_for_customer(
            customer_id=CUSTOMER_ID,
            repeat_recommendations={CUSTOMER_ID: (ARTICLE_ONE,)},
            recent_popularity=(ARTICLE_ONE,),
            all_time_popularity=(ARTICLE_ONE,),
            co_visitation_index=None,
            k=4,
            recent_popularity_by_lookback={5: (ARTICLE_TWO, ARTICLE_THREE)},
        )
    )
    five_day_records = [record for record in records if record.source == "recent_popularity_5d"]
    assert len(five_day_records) == 2
    assert five_day_records[0].article_id == ARTICLE_TWO
    assert five_day_records[0].source_rank == 1
    assert five_day_records[1].source_rank == 2


def test_iter_candidate_records_drops_empty_short_term_lookbacks() -> None:
    records = list(
        iter_candidate_records_for_customer(
            customer_id=CUSTOMER_ID,
            repeat_recommendations={CUSTOMER_ID: (ARTICLE_ONE,)},
            recent_popularity=(ARTICLE_ONE,),
            all_time_popularity=(ARTICLE_ONE,),
            co_visitation_index=None,
            k=4,
            recent_popularity_by_lookback={1: (), 3: (ARTICLE_TWO,)},
        )
    )
    sources_emitted = {record.source for record in records}
    assert "recent_popularity_1d" not in sources_emitted
    assert "recent_popularity_3d" in sources_emitted
