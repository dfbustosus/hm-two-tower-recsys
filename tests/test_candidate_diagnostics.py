from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.candidate_diagnostics import (
    ALL_TIME_POPULARITY_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_POPULARITY_BLEND_SOURCE,
    REPEAT_SOURCE,
    evaluate_baseline_candidate_diagnostics,
    write_candidate_diagnostics_report,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
COLD_CUSTOMER_ID = "c" * 64
NO_HISTORY_CUSTOMER_ID = "d" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"


def test_candidate_diagnostics_compare_sources_without_validation_leakage() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 3), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 8), COLD_CUSTOMER_ID, ARTICLE_2),
    ]

    report = evaluate_baseline_candidate_diagnostics(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        target_customer_ids=(
            CUSTOMER_ID,
            SECOND_CUSTOMER_ID,
            COLD_CUSTOMER_ID,
            NO_HISTORY_CUSTOMER_ID,
        ),
        popularity_lookback_days=7,
        evaluation_ks=(1, 2),
    )
    source_metrics = {source.source: source for source in report.sources}

    assert report.target_customers == 4
    assert report.evaluated_customers == 2
    assert source_metrics[REPEAT_SOURCE].candidate_coverage == pytest.approx(0.5)
    assert source_metrics[REPEAT_SOURCE].recall_at_k == {"1": 0.5, "2": 0.5}
    assert source_metrics[RECENT_POPULARITY_SOURCE].recall_at_k == {"1": 0.5, "2": 1.0}
    assert source_metrics[ALL_TIME_POPULARITY_SOURCE].map_at_12 == pytest.approx(0.75)
    assert source_metrics[REPEAT_POPULARITY_BLEND_SOURCE].map_at_12 == pytest.approx(1.0)
    assert source_metrics[REPEAT_POPULARITY_BLEND_SOURCE].recall_at_k == {
        "1": 1.0,
        "2": 1.0,
    }


def test_candidate_diagnostics_reject_invalid_cutoffs() -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")

    with pytest.raises(ValueError, match="positive"):
        evaluate_baseline_candidate_diagnostics(
            transaction_iter_factory=lambda: iter(()),
            split=split,
            target_customer_ids=(),
            evaluation_ks=(0,),
        )


def test_write_candidate_diagnostics_report(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    report = evaluate_baseline_candidate_diagnostics(
        transaction_iter_factory=lambda: iter(
            [
                TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
                TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_1),
            ]
        ),
        split=split,
        target_customer_ids=(CUSTOMER_ID,),
        evaluation_ks=(1,),
    )

    written_path = write_candidate_diagnostics_report(report, tmp_path / "report.json")

    assert written_path.exists()
    assert '"repeat_popularity_blend"' in written_path.read_text(encoding="utf-8")
