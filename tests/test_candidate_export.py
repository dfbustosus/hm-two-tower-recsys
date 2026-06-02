import csv
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.candidate_export import (
    CANDIDATE_EXPORT_HEADER,
    CandidateRecord,
    candidate_record_to_row,
    write_candidate_export_summary,
    write_validation_candidate_export,
)
from hm_recsys.retrieval.source_names import (
    ALL_TIME_POPULARITY_SOURCE,
    CO_VISITATION_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
THIRD_CUSTOMER_ID = "c" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
VALIDATION_ONLY_ARTICLE = "0000000003"


def test_candidate_record_to_row_uses_schema_and_stable_score_format() -> None:
    row = candidate_record_to_row(
        CandidateRecord(
            customer_id=CUSTOMER_ID,
            article_id=ARTICLE_1,
            source=REPEAT_SOURCE,
            source_rank=2,
            source_score=0.5,
        )
    )

    assert row == (CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, "2", "0.5")


def test_validation_candidate_export_is_leakage_safe_and_ranker_ready(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 3), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, VALIDATION_ONLY_ARTICLE),
    ]
    output_path = tmp_path / "candidates.csv"

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        submission_customer_ids=(CUSTOMER_ID, SECOND_CUSTOMER_ID),
        output_path=output_path,
        k=2,
        popularity_lookback_days=7,
    )

    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))

    assert tuple(rows[0]) == CANDIDATE_EXPORT_HEADER
    assert summary.target_scope == "validation_label_customers"
    assert summary.target_customers == 1
    assert summary.rows_written == 8
    assert summary.source_row_counts == {
        ALL_TIME_POPULARITY_SOURCE: 2,
        CO_VISITATION_SOURCE: 2,
        RECENT_POPULARITY_SOURCE: 2,
        REPEAT_SOURCE: 2,
    }
    assert {row["customer_id"] for row in rows} == {CUSTOMER_ID}
    assert VALIDATION_ONLY_ARTICLE not in {row["article_id"] for row in rows}
    assert rows[0] == {
        "customer_id": CUSTOMER_ID,
        "article_id": ARTICLE_2,
        "source": REPEAT_SOURCE,
        "source_rank": "1",
        "source_score": "1",
    }


def test_candidate_export_supports_deterministic_smoke_cap(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 1), THIRD_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 8), THIRD_CUSTOMER_ID, ARTICLE_2),
    ]

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        submission_customer_ids=(CUSTOMER_ID, THIRD_CUSTOMER_ID),
        output_path=tmp_path / "capped.csv",
        k=1,
        max_target_customers=1,
        include_co_visitation=False,
    )

    assert summary.target_customers == 1
    assert summary.max_target_customers == 1
    assert summary.source_row_counts == {
        ALL_TIME_POPULARITY_SOURCE: 1,
        RECENT_POPULARITY_SOURCE: 1,
        REPEAT_SOURCE: 1,
    }


def test_candidate_export_rejects_invalid_limits(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")

    with pytest.raises(ValueError, match="k must be positive"):
        write_validation_candidate_export(
            transaction_iter_factory=lambda: iter(()),
            split=split,
            submission_customer_ids=(),
            output_path=tmp_path / "invalid.csv",
            k=0,
        )


def test_write_candidate_export_summary(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter(
            [
                TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
                TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_1),
            ]
        ),
        split=split,
        submission_customer_ids=(CUSTOMER_ID,),
        output_path=tmp_path / "candidates.csv",
        k=1,
    )

    report_path = write_candidate_export_summary(summary, tmp_path / "summary.json")

    assert report_path.exists()
    assert '"rows_written"' in report_path.read_text(encoding="utf-8")
