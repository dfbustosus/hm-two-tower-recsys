import csv
import json
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import CsvSchemaError, CsvValueError, TransactionEvent
from hm_recsys.embeddings.article_content import (
    DEFAULT_ARTICLE_TEXT_FIELD_NAMES,
    ArticleContentRecord,
    article_content_record_to_row,
    build_article_popularity_priority,
    combine_article_text_fields,
    iter_article_content_records,
    iter_article_content_records_from_csv,
    write_article_content_export,
)


def test_combine_article_text_fields_normalizes_and_labels_text() -> None:
    combined = combine_article_text_fields(
        {
            "prod_name": "  Wide   trousers ",
            "colour_group_name": "Black",
            "detail_desc": "",
        }
    )

    assert combined == "prod name: Wide trousers | colour group name: Black"


def test_article_content_records_include_text_and_image_paths(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_articles(raw_dir, ("0108775015", "Wide  trousers", "Black", "Loose fit"))
    _write_image(raw_dir, "010", "0108775015.jpg")

    records = tuple(
        iter_article_content_records(
            raw_dir,
            text_field_names=("prod_name", "colour_group_name", "detail_desc"),
        )
    )

    assert len(records) == 1
    assert records[0].article_id == "0108775015"
    assert records[0].text_fields == {
        "prod_name": "Wide trousers",
        "colour_group_name": "Black",
        "detail_desc": "Loose fit",
    }
    assert records[0].combined_text == (
        "prod name: Wide trousers | colour group name: Black | detail desc: Loose fit"
    )
    assert records[0].image_relative_path == "images/010/0108775015.jpg"
    assert records[0].image_exists is True
    embedding_input = records[0].to_embedding_input(raw_dir)
    assert embedding_input.image_path == (raw_dir / "images" / "010" / "0108775015.jpg").resolve()


def test_article_content_records_allow_missing_images_and_empty_text(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_articles(raw_dir, ("0108775015", "", "", ""))

    records = tuple(
        iter_article_content_records(
            raw_dir,
            text_field_names=("prod_name", "colour_group_name", "detail_desc"),
        )
    )

    assert records[0].combined_text == ""
    assert records[0].image_relative_path == ""
    assert records[0].image_exists is False
    assert records[0].to_embedding_input(raw_dir).image_path is None


def test_article_content_records_can_be_priority_ordered_and_limited(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_articles(
        raw_dir,
        ("0108775015", "shirt", "Black", ""),
        ("0201234567", "trousers", "Blue", ""),
        ("0300000000", "dress", "Red", ""),
    )

    records = tuple(
        iter_article_content_records(
            raw_dir,
            text_field_names=("prod_name", "colour_group_name", "detail_desc"),
            article_id_order=("0300000000", "0108775015"),
            max_articles=2,
        )
    )

    assert tuple(record.article_id for record in records) == ("0300000000", "0108775015")


def test_build_article_popularity_priority_is_cutoff_and_lookback_safe() -> None:
    events = (
        TransactionEvent(date(2020, 1, 1), "customer", "0108775015"),
        TransactionEvent(date(2020, 1, 2), "customer", "0108775015"),
        TransactionEvent(date(2020, 1, 3), "customer", "0201234567"),
        TransactionEvent(date(2020, 1, 4), "customer", "0300000000"),
    )

    assert build_article_popularity_priority(events, cutoff=date(2020, 1, 4)) == (
        "0108775015",
        "0201234567",
    )
    assert build_article_popularity_priority(
        events,
        cutoff=date(2020, 1, 4),
        lookback_days=1,
    ) == ("0201234567",)

    with pytest.raises(ValueError, match="cutoff"):
        build_article_popularity_priority(events, lookback_days=7)


def test_article_content_records_validate_article_schema_and_ids(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "articles.csv").write_text("article_id,prod_name\n123,bad\n", encoding="utf-8")

    with pytest.raises(CsvValueError, match="invalid article_id"):
        tuple(iter_article_content_records(raw_dir, text_field_names=("prod_name",)))

    raw_dir_missing_column = tmp_path / "raw_missing_column"
    raw_dir_missing_column.mkdir(parents=True)
    (raw_dir_missing_column / "articles.csv").write_text(
        "article_id,prod_name\n0108775015,good\n", encoding="utf-8"
    )
    with pytest.raises(CsvSchemaError, match="missing required column"):
        tuple(
            iter_article_content_records(
                raw_dir_missing_column,
                text_field_names=("detail_desc",),
            )
        )


def test_article_content_record_to_row_uses_requested_field_order() -> None:
    record = ArticleContentRecord(
        article_id="0108775015",
        text_fields={"detail_desc": "Loose fit", "prod_name": "Wide trousers"},
        combined_text="prod name: Wide trousers | detail desc: Loose fit",
        image_relative_path="images/010/0108775015.jpg",
        image_exists=True,
    )

    assert article_content_record_to_row(record, ("prod_name", "detail_desc")) == (
        "0108775015",
        "prod name: Wide trousers | detail desc: Loose fit",
        "images/010/0108775015.jpg",
        "true",
        "Wide trousers",
        "Loose fit",
    )


def test_write_article_content_export_writes_csv_and_summary(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_path = tmp_path / "artifacts" / "article_content.csv"
    report_path = tmp_path / "artifacts" / "article_content.json"
    _write_articles(
        raw_dir,
        ("0108775015", "Wide trousers", "Black", "Loose fit"),
        ("0201234567", "", "", ""),
    )
    _write_image(raw_dir, "010", "0108775015.jpg")

    summary = write_article_content_export(
        raw_dir,
        output_path=output_path,
        report_path=report_path,
        text_field_names=("prod_name", "colour_group_name", "detail_desc"),
    )

    assert summary.records_written == 2
    assert summary.image_available_count == 1
    assert summary.image_missing_count == 1
    assert summary.empty_combined_text_count == 1
    assert summary.missing_image_examples == ("0201234567",)
    assert summary.empty_text_examples == ("0201234567",)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "article_id",
        "combined_text",
        "image_relative_path",
        "image_exists",
        "prod_name",
        "colour_group_name",
        "detail_desc",
    ]
    assert rows[1][0] == "0108775015"
    assert rows[1][3] == "true"
    assert rows[2][0] == "0201234567"
    assert rows[2][3] == "false"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["text_field_names"] == ["prod_name", "colour_group_name", "detail_desc"]
    assert report["records_written"] == 2

    read_records = tuple(iter_article_content_records_from_csv(output_path))
    assert read_records[0].article_id == "0108775015"
    assert read_records[0].combined_text.startswith("prod name: Wide trousers")
    assert read_records[0].image_exists is True
    assert read_records[1].article_id == "0201234567"
    assert read_records[1].image_exists is False


def test_article_content_export_rejects_invalid_field_config(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_articles(raw_dir, ("0108775015", "Wide trousers", "Black", "Loose fit"))

    with pytest.raises(ValueError, match="must not be empty"):
        tuple(iter_article_content_records(raw_dir, text_field_names=()))
    with pytest.raises(ValueError, match="duplicates"):
        tuple(iter_article_content_records(raw_dir, text_field_names=("prod_name", "prod_name")))
    with pytest.raises(ValueError, match="max_examples"):
        write_article_content_export(
            raw_dir,
            output_path=tmp_path / "out.csv",
            report_path=tmp_path / "out.json",
            max_examples=-1,
        )
    with pytest.raises(ValueError, match="max_articles"):
        tuple(iter_article_content_records(raw_dir, max_articles=0))


def test_iter_article_content_records_from_csv_rejects_invalid_header_and_boolean(
    tmp_path: Path,
) -> None:
    invalid_header_path = tmp_path / "invalid_header.csv"
    invalid_header_path.write_text("article_id,image_exists\n0108775015,true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="article-content columns"):
        tuple(iter_article_content_records_from_csv(invalid_header_path))

    invalid_bool_path = tmp_path / "invalid_bool.csv"
    invalid_bool_path.write_text(
        "article_id,combined_text,image_relative_path,image_exists,prod_name\n"
        "0108775015,text,images/010/0108775015.jpg,yes,shirt\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="image_exists"):
        tuple(iter_article_content_records_from_csv(invalid_bool_path))


def test_default_article_text_fields_match_export_header_contract() -> None:
    assert "prod_name" in DEFAULT_ARTICLE_TEXT_FIELD_NAMES
    assert "detail_desc" in DEFAULT_ARTICLE_TEXT_FIELD_NAMES


def _write_articles(raw_dir: Path, *rows: tuple[str, str, str, str]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "articles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("article_id", "prod_name", "colour_group_name", "detail_desc"))
        for row in rows:
            writer.writerow(row)


def _write_image(raw_dir: Path, prefix: str, filename: str) -> None:
    image_dir = raw_dir / "images" / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / filename).write_bytes(b"not-real-image-pixels")
