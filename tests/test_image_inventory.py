import csv
import json
from pathlib import Path

import pytest

from hm_recsys.embeddings.image_inventory import (
    ARTICLE_IMAGE_INVENTORY_HEADER,
    build_article_image_inventory,
    expected_article_image_relative_path,
    write_article_image_inventory,
)


def test_expected_article_image_relative_path_uses_hm_layout() -> None:
    assert expected_article_image_relative_path("0108775015") == "images/010/0108775015.jpg"


def test_expected_article_image_relative_path_rejects_bad_article_id() -> None:
    with pytest.raises(ValueError, match="invalid article_id"):
        expected_article_image_relative_path("108775015")


def test_build_article_image_inventory_maps_present_and_missing_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_articles(raw_dir, "0108775015", "0201234567")
    _write_image(raw_dir, "010", "0108775015.jpg")

    inventory = build_article_image_inventory(raw_dir)

    assert inventory.summary.valid is True
    assert inventory.summary.article_count == 2
    assert inventory.summary.scanned_image_file_count == 1
    assert inventory.summary.canonical_image_file_count == 1
    assert inventory.summary.matched_article_count == 1
    assert inventory.summary.missing_article_count == 1
    assert inventory.summary.image_coverage == 0.5
    assert inventory.summary.missing_article_examples == ("0201234567",)
    assert inventory.records[0].article_id == "0108775015"
    assert inventory.records[0].image_relative_path == "images/010/0108775015.jpg"
    assert inventory.records[0].image_exists is True
    assert inventory.records[1].expected_image_relative_path == "images/020/0201234567.jpg"
    assert inventory.records[1].image_relative_path == ""
    assert inventory.records[1].image_exists is False


def test_build_article_image_inventory_reports_extra_and_malformed_files(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_articles(raw_dir, "0108775015")
    _write_image(raw_dir, "010", "0108775015.jpg")
    _write_image(raw_dir, "999", "9999999999.jpg")
    _write_image(raw_dir, "010", "0108775015.png")
    _write_image(raw_dir, "011", "0108775015.jpg")

    inventory = build_article_image_inventory(raw_dir, max_examples=1)

    assert inventory.summary.valid is True
    assert inventory.summary.scanned_image_file_count == 4
    assert inventory.summary.canonical_image_file_count == 2
    assert inventory.summary.matched_article_count == 1
    assert inventory.summary.extra_canonical_image_count == 1
    assert inventory.summary.extra_image_examples == ("images/999/9999999999.jpg",)
    assert inventory.summary.malformed_image_file_count == 2
    assert len(inventory.summary.malformed_image_examples) == 1


def test_build_article_image_inventory_allows_missing_images_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_articles(raw_dir, "0108775015")

    inventory = build_article_image_inventory(raw_dir)

    assert inventory.summary.valid is True
    assert inventory.summary.images_dir_exists is False
    assert inventory.summary.images_dir_is_directory is False
    assert inventory.summary.matched_article_count == 0
    assert inventory.summary.missing_article_count == 1
    assert inventory.summary.failures == ()


def test_build_article_image_inventory_marks_file_images_path_invalid(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_articles(raw_dir, "0108775015")
    (raw_dir / "images").write_text("not a directory", encoding="utf-8")

    inventory = build_article_image_inventory(raw_dir)

    assert inventory.summary.valid is False
    assert inventory.summary.images_dir_exists is True
    assert inventory.summary.images_dir_is_directory is False
    assert inventory.summary.failures
    assert inventory.summary.missing_article_count == 1


def test_write_article_image_inventory_writes_csv_and_json(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "artifacts" / "article_image_inventory.csv"
    report_path = tmp_path / "artifacts" / "article_image_inventory.json"
    _write_articles(raw_dir, "0108775015", "0201234567")
    _write_image(raw_dir, "010", "0108775015.jpg")

    summary = write_article_image_inventory(
        raw_dir,
        manifest_path=manifest_path,
        report_path=report_path,
    )

    assert summary.manifest_path == str(manifest_path.resolve())
    assert summary.report_path == str(report_path.resolve())

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert tuple(rows[0]) == ARTICLE_IMAGE_INVENTORY_HEADER
    assert rows[1] == [
        "0108775015",
        "images/010/0108775015.jpg",
        "images/010/0108775015.jpg",
        "true",
    ]
    assert rows[2] == ["0201234567", "images/020/0201234567.jpg", "", "false"]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["manifest_path"] == str(manifest_path.resolve())
    assert report["matched_article_count"] == 1


def _write_articles(raw_dir: Path, *article_ids: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "articles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("article_id",))
        for article_id in article_ids:
            writer.writerow((article_id,))


def _write_image(raw_dir: Path, prefix: str, filename: str) -> None:
    image_dir = raw_dir / "images" / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / filename).write_bytes(b"not-real-image-pixels")
