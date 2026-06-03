"""Safe CSV readers that preserve H&M string identifiers exactly."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from hm_recsys.core.ids import is_article_id, is_customer_id


class CsvSchemaError(ValueError):
    """Raised when a CSV file does not match the expected lightweight schema."""


class CsvValueError(ValueError):
    """Raised when a CSV row contains an invalid typed value."""


@dataclass(frozen=True)
class TransactionEvent:
    """Lightweight transaction event for split and label logic.

    Attributes:
        t_dat: Purchase date parsed from ``transactions_train.csv``.
        customer_id: H&M customer identifier preserved as a string.
        article_id: H&M article identifier preserved as a string.
    """

    t_dat: date
    customer_id: str
    article_id: str


@dataclass(frozen=True)
class TransactionRecord:
    """Typed transaction record preserving H&M string IDs.

    Attributes:
        t_dat: Purchase date parsed from ``transactions_train.csv``.
        customer_id: H&M customer identifier preserved as a string.
        article_id: H&M article identifier preserved as a string.
        price: Transaction price parsed as ``Decimal`` to avoid float drift.
        sales_channel_id: Sales channel as an integer category.
    """

    t_dat: date
    customer_id: str
    article_id: str
    price: Decimal
    sales_channel_id: int


def iter_csv_rows(
    path: Path | str, required_columns: tuple[str, ...] = ()
) -> Iterator[dict[str, str]]:
    """Stream CSV rows as strings without numeric coercion.

    Args:
        path: CSV file path to stream.
        required_columns: Column names that must be present in the header.

    Yields:
        Row dictionaries whose values are always strings.

    Raises:
        CsvSchemaError: If any required column is missing.
    """
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
    """Yield lightweight transaction events from the raw H&M directory.

    Args:
        raw_data_dir: Directory containing ``transactions_train.csv``.

    Yields:
        Parsed ``TransactionEvent`` instances in file order.

    Raises:
        CsvSchemaError: If required transaction columns are missing.
        CsvValueError: If an event row contains an invalid date or ID.
    """

    path = Path(raw_data_dir).expanduser().resolve() / "transactions_train.csv"
    required_columns = ("t_dat", "customer_id", "article_id")
    for line_number, row in enumerate(iter_csv_rows(path, required_columns), start=2):
        yield parse_transaction_event(row, line_number=line_number)


def iter_transactions(raw_data_dir: Path | str) -> Iterator[TransactionRecord]:
    """Yield fully typed transaction records from the raw H&M directory.

    Args:
        raw_data_dir: Directory containing ``transactions_train.csv``.

    Yields:
        Parsed ``TransactionRecord`` instances in file order.

    Raises:
        CsvSchemaError: If required transaction columns are missing.
        CsvValueError: If a row contains an invalid typed value.
    """

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
    """Parse a raw CSV row into a lightweight transaction event.

    Args:
        row: Raw transaction row with string values.
        line_number: One-based CSV line number used in error messages.

    Returns:
        A validated transaction event.

    Raises:
        CsvValueError: If the date, customer ID, or article ID is invalid.
    """

    parsed_date = _parse_date(row["t_dat"], column="t_dat", line_number=line_number)
    customer_id = row["customer_id"]
    article_id = row["article_id"]
    if not is_customer_id(customer_id):
        raise CsvValueError(f"line {line_number}: invalid customer_id {customer_id!r}")
    if not is_article_id(article_id):
        raise CsvValueError(f"line {line_number}: invalid article_id {article_id!r}")
    return TransactionEvent(t_dat=parsed_date, customer_id=customer_id, article_id=article_id)


def load_article_ids(raw_data_dir: Path | str) -> set[str]:
    """Load the valid article ID universe from ``articles.csv``.

    Args:
        raw_data_dir: Directory containing H&M raw CSV files.

    Returns:
        Set of article IDs preserved as strings.
    """

    path = Path(raw_data_dir).expanduser().resolve() / "articles.csv"
    return load_id_set(path, column="article_id")


def load_submission_customer_ids(raw_data_dir: Path | str) -> set[str]:
    """Load the authoritative submission customer universe.

    Args:
        raw_data_dir: Directory containing ``sample_submission.csv``.

    Returns:
        Set of customer IDs preserved as strings.
    """

    path = Path(raw_data_dir).expanduser().resolve() / "sample_submission.csv"
    return load_id_set(path, column="customer_id")


def load_submission_customer_ids_in_order(raw_data_dir: Path | str) -> tuple[str, ...]:
    """Load submission customer IDs in the order provided by Kaggle.

    Args:
        raw_data_dir: Directory containing ``sample_submission.csv``.

    Returns:
        Ordered tuple of unique customer IDs preserved as strings.

    Raises:
        CsvSchemaError: If ``sample_submission.csv`` lacks ``customer_id``.
        CsvValueError: If a customer ID is malformed or duplicated.
    """

    path = Path(raw_data_dir).expanduser().resolve() / "sample_submission.csv"
    return load_id_sequence(path, column="customer_id", require_unique=True)


def load_id_set(path: Path | str, column: str) -> set[str]:
    """Load and validate a string ID column from a CSV file.

    Args:
        path: CSV file path.
        column: ID column to load; special validation applies to
            ``customer_id`` and ``article_id``.

    Returns:
        Set of unique string IDs from the requested column.

    Raises:
        CsvSchemaError: If ``column`` is absent from the CSV header.
        CsvValueError: If a customer or article ID has an invalid format.
    """

    values: set[str] = set()
    for line_number, row in enumerate(iter_csv_rows(path, (column,)), start=2):
        value = row[column]
        _validate_id_value(column=column, value=value, line_number=line_number)
        values.add(value)
    return values


def load_id_sequence(
    path: Path | str, column: str, require_unique: bool = False
) -> tuple[str, ...]:
    """Load and validate a string ID column while preserving file order.

    Args:
        path: CSV file path.
        column: ID column to load; special validation applies to
            ``customer_id`` and ``article_id``.
        require_unique: Whether duplicate IDs should raise ``CsvValueError``.

    Returns:
        Tuple of string IDs in file order.

    Raises:
        CsvSchemaError: If ``column`` is absent from the CSV header.
        CsvValueError: If an ID is malformed, or duplicated when uniqueness is
        required.
    """

    values: list[str] = []
    seen: set[str] = set()
    for line_number, row in enumerate(iter_csv_rows(path, (column,)), start=2):
        value = row[column]
        _validate_id_value(column=column, value=value, line_number=line_number)
        if require_unique and value in seen:
            raise CsvValueError(f"line {line_number}: duplicate {column} {value!r}")
        seen.add(value)
        values.append(value)
    return tuple(values)


def _validate_id_value(column: str, value: str, line_number: int) -> None:
    """Validate a known ID column value and raise with line context."""

    if column == "customer_id" and not is_customer_id(value):
        raise CsvValueError(f"line {line_number}: invalid customer_id {value!r}")
    if column == "article_id" and not is_article_id(value):
        raise CsvValueError(f"line {line_number}: invalid article_id {value!r}")


def _parse_date(value: str, column: str, line_number: int) -> date:
    """Parse an ISO date value and attach row context to errors."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CsvValueError(f"line {line_number}: invalid {column} {value!r}") from exc


def _parse_decimal(value: str, column: str, line_number: int) -> Decimal:
    """Parse a decimal value and attach row context to errors."""

    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise CsvValueError(f"line {line_number}: invalid {column} {value!r}") from exc


def _parse_int(value: str, column: str, line_number: int) -> int:
    """Parse an integer value and attach row context to errors."""

    try:
        return int(value)
    except ValueError as exc:
        raise CsvValueError(f"line {line_number}: invalid {column} {value!r}") from exc
