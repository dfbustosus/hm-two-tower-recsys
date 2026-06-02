from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TypedDict

from hm_recsys.ids import is_article_id, is_customer_id


@dataclass(frozen=True)
class FileContract:
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
        return not self.failures


@dataclass(frozen=True)
class DataContractReport:
    generated_at_utc: str
    raw_data_dir: str
    valid: bool
    files: tuple[FileValidationResult, ...]
    optional_images: OptionalImagesStatus


class OptionalImagesStatus(TypedDict):
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
    images_path = raw_data_dir / "images"
    return {
        "path": str(images_path),
        "exists": images_path.exists(),
        "is_directory": images_path.is_dir(),
    }


def validate_file_contract(raw_data_dir: Path, contract: FileContract) -> FileValidationResult:
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
    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(data_contract_report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def data_contract_report_to_dict(report: DataContractReport) -> dict[str, Any]:
    return asdict(report)


def _missing_file_result(path: Path, contract: FileContract) -> FileValidationResult:
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
    return dict.fromkeys(dict.fromkeys(columns), 0)


def _empty_set_dict(columns: tuple[str, ...]) -> dict[str, set[str]]:
    return {column: set() for column in dict.fromkeys(columns)}


def _update_null_counts(
    row: dict[str, str], columns: tuple[str, ...], counts: dict[str, int]
) -> None:
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
    for column, allowed_values in enum_columns.items():
        value = row.get(column, "")
        if value not in allowed_values:
            _record_invalid(column, line_number, value, invalid_counts, invalid_examples)


def _update_unique_counts(row: dict[str, str], unique_values: dict[str, set[str]]) -> None:
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
    invalid_counts[column] += 1
    _append_example(invalid_examples, column, line_number, value)


def _append_example(
    invalid_examples: dict[str, list[str]], column: str, line_number: int, value: str
) -> None:
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
