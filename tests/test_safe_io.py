import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from hm_recsys.data.io import (
    CsvSchemaError,
    CsvValueError,
    iter_transaction_events,
    iter_transactions,
    load_article_ids,
    load_submission_customer_ids_in_order,
)

CUSTOMER_ID = "a" * 64
ARTICLE_ID = "0000000001"


def test_iter_transactions_preserves_string_ids_and_parses_types(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_csv(
        raw_dir / "transactions_train.csv",
        ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        [
            {
                "t_dat": "2020-09-01",
                "customer_id": CUSTOMER_ID,
                "article_id": ARTICLE_ID,
                "price": "0.012345",
                "sales_channel_id": "2",
            }
        ],
    )

    transaction = next(iter_transactions(raw_dir))

    assert transaction.t_dat == date(2020, 9, 1)
    assert transaction.customer_id == CUSTOMER_ID
    assert transaction.article_id == ARTICLE_ID
    assert transaction.price == Decimal("0.012345")
    assert transaction.sales_channel_id == 2


def test_iter_transaction_events_raises_on_missing_required_column(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_csv(
        raw_dir / "transactions_train.csv",
        ["t_dat", "customer_id"],
        [{"t_dat": "2020-09-01", "customer_id": CUSTOMER_ID}],
    )

    with pytest.raises(CsvSchemaError):
        list(iter_transaction_events(raw_dir))


def test_iter_transaction_events_raises_on_invalid_id(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_csv(
        raw_dir / "transactions_train.csv",
        ["t_dat", "customer_id", "article_id"],
        [{"t_dat": "2020-09-01", "customer_id": CUSTOMER_ID, "article_id": "1"}],
    )

    with pytest.raises(CsvValueError):
        list(iter_transaction_events(raw_dir))


def test_load_article_ids_preserves_leading_zeroes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_csv(raw_dir / "articles.csv", ["article_id"], [{"article_id": ARTICLE_ID}])

    assert load_article_ids(raw_dir) == {ARTICLE_ID}


def test_load_submission_customer_ids_in_order_rejects_duplicates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_csv(
        raw_dir / "sample_submission.csv",
        ["customer_id", "prediction"],
        [
            {"customer_id": CUSTOMER_ID, "prediction": ARTICLE_ID},
            {"customer_id": CUSTOMER_ID, "prediction": ARTICLE_ID},
        ],
    )

    with pytest.raises(CsvValueError, match="duplicate customer_id"):
        load_submission_customer_ids_in_order(raw_dir)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
