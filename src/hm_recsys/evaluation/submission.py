from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hm_recsys.core.ids import is_article_id, is_customer_id

EXPECTED_SUBMISSION_HEADER = ("customer_id", "prediction")


@dataclass(frozen=True)
class SubmissionValidationResult:
    path: str
    valid: bool
    row_count: int
    expected_customer_count: int
    missing_customer_count: int
    extra_customer_count: int
    duplicate_customer_rows: int
    rows_with_too_many_predictions: int
    rows_with_too_few_predictions: int
    rows_with_duplicate_predictions: int
    rows_with_invalid_customer_id_format: int
    rows_with_invalid_article_id_format: int
    rows_with_unknown_article_ids: int
    failures: tuple[str, ...]
    examples: tuple[str, ...]


def validate_submission_file(
    submission_path: Path | str,
    expected_customer_ids: set[str],
    valid_article_ids: set[str],
    max_predictions: int = 12,
    require_full_length: bool = True,
) -> SubmissionValidationResult:
    if max_predictions <= 0:
        raise ValueError("max_predictions must be positive")

    path = Path(submission_path).expanduser().resolve()
    failures: list[str] = []
    examples: list[str] = []
    seen_customers: set[str] = set()
    duplicate_customer_rows = 0
    rows_with_too_many_predictions = 0
    rows_with_too_few_predictions = 0
    rows_with_duplicate_predictions = 0
    rows_with_invalid_customer_id_format = 0
    rows_with_invalid_article_id_format = 0
    rows_with_unknown_article_ids = 0
    row_count = 0

    if not path.exists():
        return _failed_submission_result(
            path=path,
            expected_customer_count=len(expected_customer_ids),
            failure=f"Missing submission file: {path}",
        )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, ()))
        if header != EXPECTED_SUBMISSION_HEADER:
            failures.append(
                "Submission header must be exactly "
                f"{','.join(EXPECTED_SUBMISSION_HEADER)!r}; got {','.join(header)!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != 2:
                failures.append(f"line {line_number}: expected 2 columns, got {len(row)}")
                _append_example(examples, f"line {line_number}: malformed row")
                continue
            customer_id, prediction = row
            predictions = prediction.split() if prediction else []

            if not is_customer_id(customer_id):
                rows_with_invalid_customer_id_format += 1
                _append_example(
                    examples, f"line {line_number}: invalid customer_id {customer_id!r}"
                )
            if customer_id in seen_customers:
                duplicate_customer_rows += 1
                _append_example(
                    examples, f"line {line_number}: duplicate customer_id {customer_id!r}"
                )
            seen_customers.add(customer_id)

            if len(predictions) > max_predictions:
                rows_with_too_many_predictions += 1
                _append_example(examples, f"line {line_number}: too many predictions")
            if require_full_length and len(predictions) < max_predictions:
                rows_with_too_few_predictions += 1
                _append_example(
                    examples, f"line {line_number}: fewer than {max_predictions} predictions"
                )
            if len(set(predictions)) != len(predictions):
                rows_with_duplicate_predictions += 1
                _append_example(examples, f"line {line_number}: duplicate predicted article_id")

            invalid_format_articles = [
                article for article in predictions if not is_article_id(article)
            ]
            if invalid_format_articles:
                rows_with_invalid_article_id_format += 1
                _append_example(
                    examples,
                    f"line {line_number}: invalid article_id format {invalid_format_articles[0]!r}",
                )
            unknown_articles = [
                article for article in predictions if article not in valid_article_ids
            ]
            if unknown_articles:
                rows_with_unknown_article_ids += 1
                _append_example(
                    examples,
                    f"line {line_number}: unknown article_id {unknown_articles[0]!r}",
                )

    missing_customer_ids = expected_customer_ids - seen_customers
    extra_customer_ids = seen_customers - expected_customer_ids

    _extend_count_failures(
        failures,
        {
            "missing customers": len(missing_customer_ids),
            "extra customers": len(extra_customer_ids),
            "duplicate customer rows": duplicate_customer_rows,
            "rows with too many predictions": rows_with_too_many_predictions,
            "rows with too few predictions": rows_with_too_few_predictions,
            "rows with duplicate predictions": rows_with_duplicate_predictions,
            "rows with invalid customer_id format": rows_with_invalid_customer_id_format,
            "rows with invalid article_id format": rows_with_invalid_article_id_format,
            "rows with unknown article IDs": rows_with_unknown_article_ids,
        },
    )

    return SubmissionValidationResult(
        path=str(path),
        valid=not failures,
        row_count=row_count,
        expected_customer_count=len(expected_customer_ids),
        missing_customer_count=len(missing_customer_ids),
        extra_customer_count=len(extra_customer_ids),
        duplicate_customer_rows=duplicate_customer_rows,
        rows_with_too_many_predictions=rows_with_too_many_predictions,
        rows_with_too_few_predictions=rows_with_too_few_predictions,
        rows_with_duplicate_predictions=rows_with_duplicate_predictions,
        rows_with_invalid_customer_id_format=rows_with_invalid_customer_id_format,
        rows_with_invalid_article_id_format=rows_with_invalid_article_id_format,
        rows_with_unknown_article_ids=rows_with_unknown_article_ids,
        failures=tuple(failures),
        examples=tuple(examples),
    )


def submission_validation_result_to_dict(
    result: SubmissionValidationResult,
) -> dict[str, Any]:
    return asdict(result)


def write_submission_validation_report(
    result: SubmissionValidationResult, path: Path | str
) -> Path:
    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(submission_validation_result_to_dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _failed_submission_result(
    path: Path, expected_customer_count: int, failure: str
) -> SubmissionValidationResult:
    return SubmissionValidationResult(
        path=str(path),
        valid=False,
        row_count=0,
        expected_customer_count=expected_customer_count,
        missing_customer_count=expected_customer_count,
        extra_customer_count=0,
        duplicate_customer_rows=0,
        rows_with_too_many_predictions=0,
        rows_with_too_few_predictions=0,
        rows_with_duplicate_predictions=0,
        rows_with_invalid_customer_id_format=0,
        rows_with_invalid_article_id_format=0,
        rows_with_unknown_article_ids=0,
        failures=(failure,),
        examples=(),
    )


def _append_example(examples: list[str], example: str) -> None:
    if len(examples) < 10:
        examples.append(example)


def _extend_count_failures(failures: list[str], counts: dict[str, int]) -> None:
    for label, count in counts.items():
        if count > 0:
            failures.append(f"{label}: {count}")
