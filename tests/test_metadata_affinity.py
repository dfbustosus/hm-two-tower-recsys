import csv
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.metadata_affinity import (
    GARMENT_GROUP_COLUMN,
    UNKNOWN_ARTICLE_ATTRIBUTE,
    build_article_attribute_popularity_candidates,
    build_article_attribute_popularity_index,
    load_article_attribute_values,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
ARTICLE_3 = "0000000003"
FUTURE_ARTICLE = "0000000004"


def test_load_article_attribute_values_preserves_ids_and_normalizes_values(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with (raw_dir / "articles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("article_id", GARMENT_GROUP_COLUMN))
        writer.writerow((ARTICLE_1, "  Dresses  "))
        writer.writerow((ARTICLE_2, ""))

    values = load_article_attribute_values(raw_dir)

    assert values[ARTICLE_1] == "Dresses"
    assert values[ARTICLE_2] == UNKNOWN_ARTICLE_ATTRIBUTE


def test_article_attribute_popularity_is_cutoff_safe_and_affinity_ranked() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    transactions = (
        TransactionEvent(date(2020, 9, 10), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 9, 11), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 9, 12), CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 9, 16), SECOND_CUSTOMER_ID, FUTURE_ARTICLE),
    )
    article_attribute_by_id = {
        ARTICLE_1: "Trousers",
        ARTICLE_2: "Trousers",
        ARTICLE_3: "Shirts",
        FUTURE_ARTICLE: "Trousers",
    }

    index = build_article_attribute_popularity_index(
        transactions=transactions,
        split=split,
        target_customer_ids=(CUSTOMER_ID,),
        article_attribute_by_id=article_attribute_by_id,
        lookback_days=7,
        max_history_items=2,
    )
    candidates = build_article_attribute_popularity_candidates(index, CUSTOMER_ID, k=3)

    assert index.train_rows_used == 3
    assert index.customer_attributes[CUSTOMER_ID] == ("Shirts", "Trousers")
    assert tuple(candidate.article_id for candidate in candidates) == (
        ARTICLE_3,
        ARTICLE_1,
        ARTICLE_2,
    )
    assert FUTURE_ARTICLE not in {candidate.article_id for candidate in candidates}


def test_article_attribute_popularity_handles_missing_history_and_invalid_limits() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16")
    index = build_article_attribute_popularity_index(
        transactions=(),
        split=split,
        target_customer_ids=(CUSTOMER_ID,),
        article_attribute_by_id={},
    )

    assert build_article_attribute_popularity_candidates(index, CUSTOMER_ID, k=1) == ()
    with pytest.raises(ValueError, match="k"):
        build_article_attribute_popularity_candidates(index, CUSTOMER_ID, k=0)
    with pytest.raises(ValueError, match="lookback_days"):
        build_article_attribute_popularity_index((), split, (), {}, lookback_days=0)
