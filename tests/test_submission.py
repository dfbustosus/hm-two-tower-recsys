import csv
from pathlib import Path

from hm_recsys.evaluation.submission import validate_submission_file

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
EXTRA_CUSTOMER_ID = "c" * 64
ARTICLE_IDS = tuple(f"{value:010d}" for value in range(1, 14))


def test_valid_submission_passes_exact_customer_and_prediction_contract(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    write_submission(
        submission_path,
        {
            CUSTOMER_ID: ARTICLE_IDS[:12],
            SECOND_CUSTOMER_ID: ARTICLE_IDS[1:13],
        },
    )

    result = validate_submission_file(
        submission_path=submission_path,
        expected_customer_ids={CUSTOMER_ID, SECOND_CUSTOMER_ID},
        valid_article_ids=set(ARTICLE_IDS),
    )

    assert result.valid
    assert result.row_count == 2


def test_submission_validator_detects_common_invalid_output_modes(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    write_submission(
        submission_path,
        {
            CUSTOMER_ID: (*ARTICLE_IDS[:11], ARTICLE_IDS[0]),
            EXTRA_CUSTOMER_ID: ("not-an-article",),
        },
    )

    result = validate_submission_file(
        submission_path=submission_path,
        expected_customer_ids={CUSTOMER_ID, SECOND_CUSTOMER_ID},
        valid_article_ids=set(ARTICLE_IDS),
    )

    assert not result.valid
    assert result.missing_customer_count == 1
    assert result.extra_customer_count == 1
    assert result.rows_with_duplicate_predictions == 1
    assert result.rows_with_too_few_predictions == 1
    assert result.rows_with_invalid_article_id_format == 1
    assert result.rows_with_unknown_article_ids == 1


def test_submission_validator_allows_short_predictions_when_configured(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    write_submission(submission_path, {CUSTOMER_ID: ARTICLE_IDS[:3]})

    result = validate_submission_file(
        submission_path=submission_path,
        expected_customer_ids={CUSTOMER_ID},
        valid_article_ids=set(ARTICLE_IDS),
        require_full_length=False,
    )

    assert result.valid


def test_submission_validator_requires_exact_header(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer_id", "predictions"])
        writer.writerow([CUSTOMER_ID, " ".join(ARTICLE_IDS[:12])])

    result = validate_submission_file(
        submission_path=submission_path,
        expected_customer_ids={CUSTOMER_ID},
        valid_article_ids=set(ARTICLE_IDS),
    )

    assert not result.valid
    assert "header" in result.failures[0]


def write_submission(path: Path, rows: dict[str, tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer_id", "prediction"])
        for customer_id, predictions in rows.items():
            writer.writerow([customer_id, " ".join(predictions)])
