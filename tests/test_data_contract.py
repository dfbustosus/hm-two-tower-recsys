import csv
import json
from pathlib import Path
from typing import Any

from hm_recsys.data_contract import (
    HM_FILE_CONTRACTS,
    data_contract_report_to_dict,
    validate_hm_data_contract,
    write_data_contract_report,
)

CUSTOMER_ID = "0" * 64
SECOND_CUSTOMER_ID = "1" * 64
ARTICLE_ID = "0000000001"
SECOND_ARTICLE_ID = "0000000002"

VALID_ROWS: dict[str, dict[str, str]] = {
    "transactions_train.csv": {
        "t_dat": "2020-09-01",
        "customer_id": CUSTOMER_ID,
        "article_id": ARTICLE_ID,
        "price": "0.025423729",
        "sales_channel_id": "2",
    },
    "articles.csv": {
        "article_id": ARTICLE_ID,
        "product_code": "1",
        "prod_name": "Example product",
        "product_type_no": "252",
        "product_type_name": "Sweater",
        "product_group_name": "Garment Upper body",
        "graphical_appearance_no": "1010016",
        "graphical_appearance_name": "Solid",
        "colour_group_code": "9",
        "colour_group_name": "Black",
        "perceived_colour_value_id": "4",
        "perceived_colour_value_name": "Dark",
        "perceived_colour_master_id": "5",
        "perceived_colour_master_name": "Black",
        "department_no": "1676",
        "department_name": "Jersey",
        "index_code": "A",
        "index_name": "Ladieswear",
        "index_group_no": "1",
        "index_group_name": "Ladieswear",
        "section_no": "16",
        "section_name": "Womens Everyday Basics",
        "garment_group_no": "1002",
        "garment_group_name": "Jersey Basic",
        "detail_desc": "Synthetic product description.",
    },
    "customers.csv": {
        "customer_id": CUSTOMER_ID,
        "FN": "1.0",
        "Active": "1.0",
        "club_member_status": "ACTIVE",
        "fashion_news_frequency": "Regularly",
        "age": "25",
        "postal_code": "abc123",
    },
    "sample_submission.csv": {
        "customer_id": CUSTOMER_ID,
        "prediction": f"{ARTICLE_ID} {SECOND_ARTICLE_ID}",
    },
}


def test_valid_synthetic_contract_passes_and_writes_report(tmp_path: Path) -> None:
    raw_dir = write_valid_raw_data(tmp_path)

    report = validate_hm_data_contract(raw_dir)
    report_path = write_data_contract_report(report, tmp_path / "report.json")

    assert report.valid
    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["valid"] is True

    report_dict = data_contract_report_to_dict(report)
    transaction_result = file_result(report_dict, "transactions_train.csv")
    assert transaction_result["row_count"] == 1
    assert transaction_result["unique_counts"]["customer_id"] == 1
    assert transaction_result["date_min"]["t_dat"] == "2020-09-01"
    assert transaction_result["date_max"]["t_dat"] == "2020-09-01"


def test_missing_required_column_fails(tmp_path: Path) -> None:
    raw_dir = write_valid_raw_data(tmp_path, omitted_columns={"transactions_train.csv": {"price"}})

    report = validate_hm_data_contract(raw_dir)
    transaction_result = next(
        result for result in report.files if result.file_name == "transactions_train.csv"
    )

    assert not report.valid
    assert transaction_result.missing_columns == ("price",)


def test_invalid_article_id_fails_string_format_contract(tmp_path: Path) -> None:
    raw_dir = write_valid_raw_data(
        tmp_path,
        overrides={"articles.csv": {"article_id": "123"}},
    )

    report = validate_hm_data_contract(raw_dir)
    articles_result = next(result for result in report.files if result.file_name == "articles.csv")

    assert not report.valid
    assert articles_result.invalid_counts["article_id"] == 1


def test_duplicate_customer_metadata_ids_fail(tmp_path: Path) -> None:
    raw_dir = write_valid_raw_data(
        tmp_path,
        extra_rows={"customers.csv": [VALID_ROWS["customers.csv"]]},
    )

    report = validate_hm_data_contract(raw_dir)
    customers_result = next(
        result for result in report.files if result.file_name == "customers.csv"
    )

    assert not report.valid
    assert customers_result.duplicate_counts["customer_id"] == 1


def test_duplicate_transaction_rows_remain_valid_purchase_frequency(tmp_path: Path) -> None:
    raw_dir = write_valid_raw_data(
        tmp_path,
        extra_rows={"transactions_train.csv": [VALID_ROWS["transactions_train.csv"]]},
    )

    report = validate_hm_data_contract(raw_dir)
    transactions_result = next(
        result for result in report.files if result.file_name == "transactions_train.csv"
    )

    assert report.valid
    assert transactions_result.row_count == 2


def write_valid_raw_data(
    tmp_path: Path,
    overrides: dict[str, dict[str, str]] | None = None,
    omitted_columns: dict[str, set[str]] | None = None,
    extra_rows: dict[str, list[dict[str, str]]] | None = None,
) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "images").mkdir()
    overrides = overrides or {}
    omitted_columns = omitted_columns or {}
    extra_rows = extra_rows or {}

    for contract in HM_FILE_CONTRACTS:
        row = {**VALID_ROWS[contract.file_name], **overrides.get(contract.file_name, {})}
        rows = [row, *extra_rows.get(contract.file_name, [])]
        fieldnames = [
            column
            for column in contract.required_columns
            if column not in omitted_columns.get(contract.file_name, set())
        ]
        write_csv(raw_dir / contract.file_name, fieldnames, rows)
    return raw_dir


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_result(report: dict[str, Any], file_name: str) -> dict[str, Any]:
    files = report["files"]
    assert isinstance(files, (list, tuple))
    for result in files:
        assert isinstance(result, dict)
        if result["file_name"] == file_name:
            return result
    raise AssertionError(f"Missing result for {file_name}")
