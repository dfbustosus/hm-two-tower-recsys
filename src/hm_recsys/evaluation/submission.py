"""Kaggle submission validation for H&M recommendation CSV files."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hm_recsys.core.ids import is_article_id, is_customer_id

EXPECTED_SUBMISSION_HEADER = ("customer_id", "prediction")


def write_submission_file(
    predictions_by_customer: Mapping[str, Iterable[str]],
    customer_ids: Iterable[str],
    path: Path | str,
    max_predictions: int = 12,
) -> Path:
    """Write a Kaggle-style H&M submission CSV.

    Args:
        predictions_by_customer: Ranked article IDs keyed by customer ID.
        customer_ids: Authoritative customer order, usually from
            ``sample_submission.csv``.
        path: Destination CSV path.
        max_predictions: Maximum number of predictions allowed per row.

    Returns:
        Resolved path written to disk.

    Raises:
        ValueError: If identifiers are malformed, predictions are missing,
        duplicate article IDs appear in a row, or a row exceeds
        ``max_predictions``.
    """

    if max_predictions <= 0:
        raise ValueError("max_predictions must be positive")

    submission_path = Path(path).expanduser().resolve()
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_SUBMISSION_HEADER)
        for customer_id in customer_ids:
            if not is_customer_id(customer_id):
                raise ValueError(f"invalid customer_id: {customer_id!r}")
            if customer_id not in predictions_by_customer:
                raise ValueError(f"missing predictions for customer_id: {customer_id!r}")

            predictions = tuple(predictions_by_customer[customer_id])
            if len(predictions) > max_predictions:
                raise ValueError(
                    f"customer_id {customer_id!r} has more than {max_predictions} predictions"
                )
            if len(set(predictions)) != len(predictions):
                raise ValueError(f"customer_id {customer_id!r} has duplicate predictions")
            invalid_articles = [
                article_id for article_id in predictions if not is_article_id(article_id)
            ]
            if invalid_articles:
                raise ValueError(
                    f"customer_id {customer_id!r} has invalid article_id {invalid_articles[0]!r}"
                )
            writer.writerow((customer_id, " ".join(predictions)))
    return submission_path


@dataclass(frozen=True)
class SubmissionValidationResult:
    """Structured validation result for a Kaggle submission file.

    Attributes:
        path: Resolved submission path examined.
        valid: Whether all submission checks passed.
        row_count: Number of customer prediction rows read.
        expected_customer_count: Number of customers in ``sample_submission.csv``.
        missing_customer_count: Expected customers absent from the submission.
        extra_customer_count: Customers not present in ``sample_submission.csv``.
        duplicate_customer_rows: Duplicate customer rows in the submission.
        rows_with_too_many_predictions: Rows with more than the allowed top-k.
        rows_with_too_few_predictions: Rows with fewer predictions than required.
        rows_with_duplicate_predictions: Rows containing repeated article IDs.
        rows_with_invalid_customer_id_format: Rows with invalid customer IDs.
        rows_with_invalid_article_id_format: Rows with malformed article IDs.
        rows_with_unknown_article_ids: Rows containing articles outside articles.csv.
        failures: Human-readable failure summaries.
        examples: Bounded row-level examples for debugging.
    """

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
    """Validate a Kaggle-style H&M submission CSV.

    Args:
        submission_path: CSV path to validate.
        expected_customer_ids: Authoritative customer IDs from
            ``sample_submission.csv``.
        valid_article_ids: Valid article IDs from ``articles.csv``.
        max_predictions: Maximum number of article IDs per prediction row.
        require_full_length: Whether every row must contain ``max_predictions``
            article IDs.

    Returns:
        Structured submission validation result.

    Raises:
        ValueError: If ``max_predictions`` is not positive.
    """

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
    """Convert a submission validation result to serializable primitives.

    Args:
        result: Result object to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """

    return asdict(result)


def write_submission_validation_report(
    result: SubmissionValidationResult, path: Path | str
) -> Path:
    """Write a submission validation result as deterministic JSON.

    Args:
        result: Validation result to serialize.
        path: Destination JSON path.

    Returns:
        Resolved path written to disk.
    """

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
    """Build a validation result for an unreadable or missing submission file."""

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
    """Append at most ten row-level examples to a diagnostics list."""

    if len(examples) < 10:
        examples.append(example)


def _extend_count_failures(failures: list[str], counts: dict[str, int]) -> None:
    """Append failure summaries for non-zero validation counters."""

    for label, count in counts.items():
        if count > 0:
            failures.append(f"{label}: {count}")
