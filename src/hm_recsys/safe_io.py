from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from hm_recsys.ids import is_article_id, is_customer_id


class CsvSchemaError(ValueError):
    """Raised when a CSV file does not match the expected lightweight schema."""


class CsvValueError(ValueError):
    """Raised when a CSV row contains an invalid typed value."""


@dataclass(frozen=True)
class TransactionEvent:
    """Lightweight transaction event for split and label logic."""

    t_dat: date
    customer_id: str
    article_id: str


@dataclass(frozen=True)
class TransactionRecord:
    """Typed transaction record preserving H&M string IDs."""

    t_dat: date
    customer_id: str
    article_id: str
    price: Decimal
    sales_channel_id: int


def iter_csv_rows(
    path: Path | str, required_columns: tuple[str, ...] = ()
) -> Iterator[dict[str, str]]:
    """Stream CSV rows as strings without numeric coercion."""
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing_columns = tuple(column for column in required_columns if column not in columns)
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise CsvSchemaError(f"{csv_path} is missing required column(s): {missing}")

        for row in reader:
            yield {column: row.get(column, "") or "" for column in columns}


def iter_transaction_events(raw_data_dir: Path | str) -> Iterator[TransactionEvent]:
    path = Path(raw_data_dir).expanduser().resolve() / "transactions_train.csv"
    required_columns = ("t_dat", "customer_id", "article_id")
    for line_number, row in enumerate(iter_csv_rows(path, required_columns), start=2):
        yield parse_transaction_event(row, line_number=line_number)


def iter_transactions(raw_data_dir: Path | str) -> Iterator[TransactionRecord]:
    path = Path(raw_data_dir).expanduser().resolve() / "transactions_train.csv"
    required_columns = ("t_dat", "customer_id", "article_id", "price", "sales_channel_id")
    for line_number, row in enumerate(iter_csv_rows(path, required_columns), start=2):
        event = parse_transaction_event(row, line_number=line_number)
        yield TransactionRecord(
            t_dat=event.t_dat,
            customer_id=event.customer_id,
            article_id=event.article_id,
            price=_parse_decimal(row["price"], column="price", line_number=line_number),
            sales_channel_id=_parse_int(
                row["sales_channel_id"], column="sales_channel_id", line_number=line_number
            ),
        )


def parse_transaction_event(row: dict[str, str], line_number: int) -> TransactionEvent:
    parsed_date = _parse_date(row["t_dat"], column="t_dat", line_number=line_number)
    customer_id = row["customer_id"]
    article_id = row["article_id"]
    if not is_customer_id(customer_id):
        raise CsvValueError(f"line {line_number}: invalid customer_id {customer_id!r}")
    if not is_article_id(article_id):
        raise CsvValueError(f"line {line_number}: invalid article_id {article_id!r}")
    return TransactionEvent(t_dat=parsed_date, customer_id=customer_id, article_id=article_id)


def load_article_ids(raw_data_dir: Path | str) -> set[str]:
    path = Path(raw_data_dir).expanduser().resolve() / "articles.csv"
    return load_id_set(path, column="article_id")


def load_submission_customer_ids(raw_data_dir: Path | str) -> set[str]:
    path = Path(raw_data_dir).expanduser().resolve() / "sample_submission.csv"
    return load_id_set(path, column="customer_id")


def load_id_set(path: Path | str, column: str) -> set[str]:
    values: set[str] = set()
    for line_number, row in enumerate(iter_csv_rows(path, (column,)), start=2):
        value = row[column]
        if column == "customer_id" and not is_customer_id(value):
            raise CsvValueError(f"line {line_number}: invalid customer_id {value!r}")
        if column == "article_id" and not is_article_id(value):
            raise CsvValueError(f"line {line_number}: invalid article_id {value!r}")
        values.add(value)
    return values


def _parse_date(value: str, column: str, line_number: int) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CsvValueError(f"line {line_number}: invalid {column} {value!r}") from exc


def _parse_decimal(value: str, column: str, line_number: int) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise CsvValueError(f"line {line_number}: invalid {column} {value!r}") from exc


def _parse_int(value: str, column: str, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise CsvValueError(f"line {line_number}: invalid {column} {value!r}") from exc
