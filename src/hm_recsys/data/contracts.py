"""Validation contracts for the raw H&M Kaggle CSV files."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TypedDict

from hm_recsys.core.ids import is_article_id, is_customer_id


@dataclass(frozen=True)
class FileContract:
    """Expected schema and lightweight validation rules for one CSV file.

    Attributes:
        file_name: File name relative to the raw H&M data directory.
        required_columns: Header columns that must exist.
        id_columns: Columns validated as H&M customer or article IDs.
        date_columns: Columns parsed with ``date.fromisoformat``.
        decimal_columns: Columns parsed as ``Decimal``.
        enum_columns: Columns constrained to explicit string values.
        unique_columns: Columns expected to have no duplicate non-empty values.
        non_nullable_columns: Required columns that must not contain blanks.
    """

    file_name: str
    required_columns: tuple[str, ...]
    id_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    decimal_columns: tuple[str, ...] = ()
    enum_columns: dict[str, frozenset[str]] = field(default_factory=dict)
    unique_columns: tuple[str, ...] = ()
    non_nullable_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileValidationResult:
    """Validation outcome and summary statistics for one raw CSV file.

    Attributes:
        file_name: Validated file name.
        path: Absolute path examined by the validator.
        exists: Whether the path exists.
        row_count: Number of data rows scanned.
        columns: Header columns discovered in the file.
        missing_columns: Required columns absent from the header.
        null_counts: Blank-value counts for required columns.
        invalid_counts: Format validation failure counts by column.
        unique_counts: Unique non-empty counts for tracked ID/unique columns.
        duplicate_counts: Duplicate counts for columns declared unique.
        date_min: Minimum parsed date by date column.
        date_max: Maximum parsed date by date column.
        invalid_examples: Bounded examples for invalid values and duplicates.
        failures: Human-readable contract failures.
    """

    file_name: str
    path: str
    exists: bool
    row_count: int
    columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    null_counts: dict[str, int]
    invalid_counts: dict[str, int]
    unique_counts: dict[str, int]
    duplicate_counts: dict[str, int]
    date_min: dict[str, str]
    date_max: dict[str, str]
    invalid_examples: dict[str, tuple[str, ...]]
    failures: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Return whether the file passed all contract checks.

        Returns:
            ``True`` when no failures were recorded; otherwise ``False``.
        """

        return not self.failures


@dataclass(frozen=True)
class DataContractReport:
    """Top-level data-contract report for the raw H&M data directory.

    Attributes:
        generated_at_utc: UTC timestamp for the validation run.
        raw_data_dir: Absolute raw-data directory path.
        valid: Whether all required files and optional images status are valid.
        files: Per-file validation results.
        optional_images: Status for the optional article image directory.
    """

    generated_at_utc: str
    raw_data_dir: str
    valid: bool
    files: tuple[FileValidationResult, ...]
    optional_images: OptionalImagesStatus


class OptionalImagesStatus(TypedDict):
    """Typed status record for the optional H&M article image directory."""

    path: str
    exists: bool
    is_directory: bool


TRANSACTIONS_CONTRACT = FileContract(
    file_name="transactions_train.csv",
    required_columns=("t_dat", "customer_id", "article_id", "price", "sales_channel_id"),
    id_columns=("customer_id", "article_id"),
    date_columns=("t_dat",),
    decimal_columns=("price",),
    enum_columns={"sales_channel_id": frozenset({"1", "2"})},
    non_nullable_columns=("t_dat", "customer_id", "article_id", "price", "sales_channel_id"),
)

ARTICLES_CONTRACT = FileContract(
    file_name="articles.csv",
    required_columns=(
        "article_id",
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_no",
        "graphical_appearance_name",
        "colour_group_code",
        "colour_group_name",
        "perceived_colour_value_id",
        "perceived_colour_value_name",
        "perceived_colour_master_id",
        "perceived_colour_master_name",
        "department_no",
        "department_name",
        "index_code",
        "index_name",
        "index_group_no",
        "index_group_name",
        "section_no",
        "section_name",
        "garment_group_no",
        "garment_group_name",
        "detail_desc",
    ),
    id_columns=("article_id",),
    unique_columns=("article_id",),
    non_nullable_columns=("article_id",),
)

CUSTOMERS_CONTRACT = FileContract(
    file_name="customers.csv",
    required_columns=(
        "customer_id",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "postal_code",
    ),
    id_columns=("customer_id",),
    unique_columns=("customer_id",),
    non_nullable_columns=("customer_id",),
)

SAMPLE_SUBMISSION_CONTRACT = FileContract(
    file_name="sample_submission.csv",
    required_columns=("customer_id", "prediction"),
    id_columns=("customer_id",),
    unique_columns=("customer_id",),
    non_nullable_columns=("customer_id",),
)

HM_FILE_CONTRACTS = (
    TRANSACTIONS_CONTRACT,
    ARTICLES_CONTRACT,
    CUSTOMERS_CONTRACT,
    SAMPLE_SUBMISSION_CONTRACT,
)


def validate_hm_data_contract(raw_data_dir: Path | str) -> DataContractReport:
    """Validate all required H&M raw files and optional image path status.

    Args:
        raw_data_dir: Directory expected to contain H&M Kaggle raw files.

    Returns:
        Complete data-contract report with per-file validation details.
    """

    raw_dir = Path(raw_data_dir).expanduser().resolve()
    file_results = tuple(
        validate_file_contract(raw_dir, contract) for contract in HM_FILE_CONTRACTS
    )
    optional_images = validate_optional_images(raw_dir)
    images_valid = not optional_images["exists"] or optional_images["is_directory"]
    return DataContractReport(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_data_dir=str(raw_dir),
        valid=all(result.valid for result in file_results) and images_valid,
        files=file_results,
        optional_images=optional_images,
    )


def validate_optional_images(raw_data_dir: Path) -> OptionalImagesStatus:
    """Validate the optional ``images/`` path without scanning image files.

    Args:
        raw_data_dir: Resolved H&M raw-data directory.

    Returns:
        Typed status showing the image path, existence, and directory flag.
    """

    images_path = raw_data_dir / "images"
    return {
        "path": str(images_path),
        "exists": images_path.exists(),
        "is_directory": images_path.is_dir(),
    }


def validate_file_contract(raw_data_dir: Path, contract: FileContract) -> FileValidationResult:
    """Validate one CSV file against a ``FileContract``.

    Args:
        raw_data_dir: Resolved directory containing the target CSV file.
        contract: Schema and validation rules for the target file.

    Returns:
        Per-file validation result including counts, examples, and failures.
    """

    path = raw_data_dir / contract.file_name
    if not path.exists():
        return _missing_file_result(path, contract)
    if not path.is_file():
        return _not_a_file_result(path, contract)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing_columns = tuple(
            column for column in contract.required_columns if column not in columns
        )

        row_count = 0
        null_counts = dict.fromkeys(contract.required_columns, 0)
        invalid_counts = _empty_int_dict(
            (
                *contract.id_columns,
                *contract.date_columns,
                *contract.decimal_columns,
                *contract.enum_columns,
            )
        )
        duplicate_counts = _empty_int_dict(contract.unique_columns)
        invalid_examples: dict[str, list[str]] = {}
        date_min: dict[str, date] = {}
        date_max: dict[str, date] = {}
        unique_values = _empty_set_dict((*contract.id_columns, *contract.unique_columns))
        unique_column_seen = _empty_set_dict(contract.unique_columns)

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            _update_null_counts(row, contract.required_columns, null_counts)
            _validate_ids(row, line_number, contract.id_columns, invalid_counts, invalid_examples)
            _validate_dates(
                row,
                line_number,
                contract.date_columns,
                invalid_counts,
                invalid_examples,
                date_min,
                date_max,
            )
            _validate_decimals(
                row, line_number, contract.decimal_columns, invalid_counts, invalid_examples
            )
            _validate_enums(
                row, line_number, contract.enum_columns, invalid_counts, invalid_examples
            )
            _update_unique_counts(row, unique_values)
            _update_duplicate_counts(
                row, line_number, unique_column_seen, duplicate_counts, invalid_examples
            )

    unique_counts = {column: len(values) for column, values in unique_values.items()}
    failures = _build_failures(
        contract=contract,
        row_count=row_count,
        missing_columns=missing_columns,
        null_counts=null_counts,
        invalid_counts=invalid_counts,
        duplicate_counts=duplicate_counts,
    )
    return FileValidationResult(
        file_name=contract.file_name,
        path=str(path),
        exists=True,
        row_count=row_count,
        columns=columns,
        missing_columns=missing_columns,
        null_counts=null_counts,
        invalid_counts=invalid_counts,
        unique_counts=unique_counts,
        duplicate_counts=duplicate_counts,
        date_min={column: value.isoformat() for column, value in date_min.items()},
        date_max={column: value.isoformat() for column, value in date_max.items()},
        invalid_examples={
            column: tuple(examples) for column, examples in sorted(invalid_examples.items())
        },
        failures=failures,
    )


def write_data_contract_report(report: DataContractReport, path: Path | str) -> Path:
    """Write a data-contract report as deterministic JSON.

    Args:
        report: Report object to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(data_contract_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def data_contract_report_to_dict(report: DataContractReport) -> dict[str, Any]:
    """Convert a data-contract report into JSON-serializable primitives.

    Args:
        report: Report object to convert.

    Returns:
        Dictionary representation suitable for ``json.dumps``.
    """

    return asdict(report)


def _missing_file_result(path: Path, contract: FileContract) -> FileValidationResult:
    """Build a failed validation result for a missing required file."""

    return FileValidationResult(
        file_name=contract.file_name,
        path=str(path),
        exists=False,
        row_count=0,
        columns=(),
        missing_columns=contract.required_columns,
        null_counts=dict.fromkeys(contract.required_columns, 0),
        invalid_counts={},
        unique_counts={},
        duplicate_counts={},
        date_min={},
        date_max={},
        invalid_examples={},
        failures=(f"Missing required file: {path}",),
    )


def _not_a_file_result(path: Path, contract: FileContract) -> FileValidationResult:
    """Build a failed validation result for a non-file path."""

    return FileValidationResult(
        file_name=contract.file_name,
        path=str(path),
        exists=True,
        row_count=0,
        columns=(),
        missing_columns=contract.required_columns,
        null_counts=dict.fromkeys(contract.required_columns, 0),
        invalid_counts={},
        unique_counts={},
        duplicate_counts={},
        date_min={},
        date_max={},
        invalid_examples={},
        failures=(f"Expected file but found non-file path: {path}",),
    )


def _empty_int_dict(columns: tuple[str, ...]) -> dict[str, int]:
    """Return a de-duplicated zero-count dictionary for ``columns``."""

    return dict.fromkeys(dict.fromkeys(columns), 0)


def _empty_set_dict(columns: tuple[str, ...]) -> dict[str, set[str]]:
    """Return a de-duplicated set dictionary keyed by column name."""

    return {column: set() for column in dict.fromkeys(columns)}


def _update_null_counts(
    row: dict[str, str], columns: tuple[str, ...], counts: dict[str, int]
) -> None:
    """Increment blank-value counts for required columns in one row."""

    for column in columns:
        if row.get(column, "") == "":
            counts[column] += 1


def _validate_ids(
    row: dict[str, str],
    line_number: int,
    columns: tuple[str, ...],
    invalid_counts: dict[str, int],
    invalid_examples: dict[str, list[str]],
) -> None:
    """Validate configured ID columns and record invalid examples."""

    for column in columns:
        value = row.get(column, "")
        valid = is_customer_id(value) if column == "customer_id" else is_article_id(value)
        if not valid:
            _record_invalid(column, line_number, value, invalid_counts, invalid_examples)


def _validate_dates(
    row: dict[str, str],
    line_number: int,
    columns: tuple[str, ...],
    invalid_counts: dict[str, int],
    invalid_examples: dict[str, list[str]],
    date_min: dict[str, date],
    date_max: dict[str, date],
) -> None:
    """Validate date columns and update observed min/max dates."""

    for column in columns:
        value = row.get(column, "")
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            _record_invalid(column, line_number, value, invalid_counts, invalid_examples)
            continue
        date_min[column] = min(date_min.get(column, parsed), parsed)
        date_max[column] = max(date_max.get(column, parsed), parsed)


def _validate_decimals(
    row: dict[str, str],
    line_number: int,
    columns: tuple[str, ...],
    invalid_counts: dict[str, int],
    invalid_examples: dict[str, list[str]],
) -> None:
    """Validate decimal columns and record invalid examples."""

    for column in columns:
        value = row.get(column, "")
        try:
            Decimal(value)
        except InvalidOperation:
            _record_invalid(column, line_number, value, invalid_counts, invalid_examples)


def _validate_enums(
    row: dict[str, str],
    line_number: int,
    enum_columns: dict[str, frozenset[str]],
    invalid_counts: dict[str, int],
    invalid_examples: dict[str, list[str]],
) -> None:
    """Validate enum-constrained columns against allowed string values."""

    for column, allowed_values in enum_columns.items():
        value = row.get(column, "")
        if value not in allowed_values:
            _record_invalid(column, line_number, value, invalid_counts, invalid_examples)


def _update_unique_counts(row: dict[str, str], unique_values: dict[str, set[str]]) -> None:
    """Track unique non-empty values for selected columns."""

    for column, values in unique_values.items():
        value = row.get(column, "")
        if value:
            values.add(value)


def _update_duplicate_counts(
    row: dict[str, str],
    line_number: int,
    seen_values: dict[str, set[str]],
    duplicate_counts: dict[str, int],
    invalid_examples: dict[str, list[str]],
) -> None:
    """Track duplicates for columns declared unique by the contract."""

    for column, values in seen_values.items():
        value = row.get(column, "")
        if not value:
            continue
        if value in values:
            duplicate_counts[column] += 1
            _append_example(invalid_examples, f"duplicate_{column}", line_number, value)
        else:
            values.add(value)


def _record_invalid(
    column: str,
    line_number: int,
    value: str,
    invalid_counts: dict[str, int],
    invalid_examples: dict[str, list[str]],
) -> None:
    """Increment invalid-value counts and store a bounded example."""

    invalid_counts[column] += 1
    _append_example(invalid_examples, column, line_number, value)


def _append_example(
    invalid_examples: dict[str, list[str]], column: str, line_number: int, value: str
) -> None:
    """Append at most five invalid-value examples for a column."""

    examples = invalid_examples.setdefault(column, [])
    if len(examples) < 5:
        examples.append(f"line {line_number}: {value!r}")


def _build_failures(
    contract: FileContract,
    row_count: int,
    missing_columns: tuple[str, ...],
    null_counts: dict[str, int],
    invalid_counts: dict[str, int],
    duplicate_counts: dict[str, int],
) -> tuple[str, ...]:
    """Build human-readable failures from validation counters."""

    failures: list[str] = []
    if row_count == 0:
        failures.append(f"{contract.file_name} has no data rows")
    if missing_columns:
        failures.append(f"{contract.file_name} is missing columns: {', '.join(missing_columns)}")
    for column in contract.non_nullable_columns:
        if null_counts.get(column, 0) > 0:
            failures.append(
                f"{contract.file_name}.{column} has {null_counts[column]} empty required value(s)"
            )
    for column, count in invalid_counts.items():
        if count > 0:
            failures.append(f"{contract.file_name}.{column} has {count} invalid value(s)")
    for column, count in duplicate_counts.items():
        if count > 0:
            failures.append(f"{contract.file_name}.{column} has {count} duplicate value(s)")
    return tuple(failures)
