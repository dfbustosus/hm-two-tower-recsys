"""Article image inventory helpers for H&M multimodal retrieval.

The H&M image directory uses a deterministic convention:
``images/<first-three-article-digits>/<article_id>.jpg``.  This module maps
the article universe from ``articles.csv`` to that local image layout without
loading image pixels or adding model-provider dependencies.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from hm_recsys.core.ids import is_article_id
from hm_recsys.data.io import load_id_sequence

ARTICLE_IMAGE_INVENTORY_HEADER = (
    "article_id",
    "expected_image_relative_path",
    "image_relative_path",
    "image_exists",
)
IMAGE_DIR_NAME = "images"
IMAGE_SUFFIX = ".jpg"


@dataclass(frozen=True)
class ArticleImageInventoryRecord:
    """Image availability row for one article.

    Attributes:
        article_id: H&M article identifier preserved as a string.
        expected_image_relative_path: Canonical path relative to the raw H&M
            directory, regardless of whether the file exists.
        image_relative_path: Discovered canonical image path relative to the raw
            H&M directory, or an empty string when the image is missing.
        image_exists: Whether a canonical image was discovered for the article.
    """

    article_id: str
    expected_image_relative_path: str
    image_relative_path: str
    image_exists: bool

    def __post_init__(self) -> None:
        """Validate record identity and path consistency."""

        if not is_article_id(self.article_id):
            raise ValueError(f"invalid article_id: {self.article_id!r}")
        if not self.expected_image_relative_path:
            raise ValueError("expected_image_relative_path must not be empty")
        if self.image_exists and not self.image_relative_path:
            raise ValueError("image_relative_path must be set when image_exists is true")


@dataclass(frozen=True)
class ImagePathIssue:
    """Bounded example of an image-path issue discovered during inventory."""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class ArticleImageInventorySummary:
    """Summary of article image availability and malformed local image paths.

    Missing images are expected for some articles and do not make the summary
    invalid.  The summary is invalid only when the local ``images`` path blocks
    inventory construction, for example when it exists as a file instead of a
    directory.
    """

    generated_at_utc: str
    raw_data_dir: str
    articles_path: str
    images_dir: str
    images_dir_exists: bool
    images_dir_is_directory: bool
    valid: bool
    article_count: int
    scanned_image_file_count: int
    canonical_image_file_count: int
    matched_article_count: int
    missing_article_count: int
    extra_canonical_image_count: int
    malformed_image_file_count: int
    image_coverage: float
    failures: tuple[str, ...]
    missing_article_examples: tuple[str, ...]
    extra_image_examples: tuple[str, ...]
    malformed_image_examples: tuple[ImagePathIssue, ...]
    manifest_path: str | None = None
    report_path: str | None = None


@dataclass(frozen=True)
class ArticleImageInventory:
    """Complete image inventory records plus aggregate summary."""

    records: tuple[ArticleImageInventoryRecord, ...]
    summary: ArticleImageInventorySummary


def expected_article_image_relative_path(article_id: str) -> str:
    """Return the canonical H&M image path relative to the raw-data directory.

    Args:
        article_id: H&M article identifier.

    Returns:
        POSIX-style relative path such as ``images/010/0108775015.jpg``.

    Raises:
        ValueError: If ``article_id`` is malformed.
    """

    if not is_article_id(article_id):
        raise ValueError(f"invalid article_id: {article_id!r}")
    return f"{IMAGE_DIR_NAME}/{article_id[:3]}/{article_id}{IMAGE_SUFFIX}"


def build_article_image_inventory(
    raw_data_dir: Path | str,
    *,
    max_examples: int = 10,
) -> ArticleImageInventory:
    """Build an article-to-image inventory without reading image pixels.

    Args:
        raw_data_dir: Directory containing H&M ``articles.csv`` and optional
            ``images/`` subdirectory.
        max_examples: Maximum examples retained for each issue category.

    Returns:
        Inventory records in ``articles.csv`` order and a summary report.

    Raises:
        ValueError: If ``max_examples`` is negative or an article ID is invalid.
        hm_recsys.data.io.CsvSchemaError: If ``articles.csv`` lacks
            ``article_id``.
        hm_recsys.data.io.CsvValueError: If ``articles.csv`` contains malformed
            or duplicate article IDs.
    """

    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")

    raw_dir = Path(raw_data_dir).expanduser().resolve()
    articles_path = raw_dir / "articles.csv"
    images_dir = raw_dir / IMAGE_DIR_NAME
    article_ids = load_id_sequence(articles_path, column="article_id", require_unique=True)
    article_id_set = set(article_ids)

    failures: list[str] = []
    canonical_images: dict[str, str] = {}
    scanned_image_file_count = 0
    malformed_image_file_count = 0
    malformed_examples: list[ImagePathIssue] = []

    if images_dir.exists() and not images_dir.is_dir():
        failures.append(f"{images_dir} exists but is not a directory")
    elif images_dir.is_dir():
        for image_path in sorted(images_dir.rglob("*")):
            if not image_path.is_file():
                continue
            scanned_image_file_count += 1
            article_id, reason = _canonical_article_id_from_image_path(image_path, images_dir)
            if article_id is None:
                malformed_image_file_count += 1
                if len(malformed_examples) < max_examples:
                    malformed_examples.append(
                        ImagePathIssue(
                            relative_path=_relative_to_raw_data_dir(image_path, raw_dir),
                            reason=reason or "not a canonical H&M image path",
                        )
                    )
                continue
            canonical_images[article_id] = _relative_to_raw_data_dir(image_path, raw_dir)

    extra_image_examples: list[str] = []
    extra_canonical_image_count = 0
    for article_id, relative_path in canonical_images.items():
        if article_id in article_id_set:
            continue
        extra_canonical_image_count += 1
        if len(extra_image_examples) < max_examples:
            extra_image_examples.append(relative_path)

    records: list[ArticleImageInventoryRecord] = []
    missing_article_examples: list[str] = []
    matched_article_count = 0
    for article_id in article_ids:
        expected_relative_path = expected_article_image_relative_path(article_id)
        image_relative_path = canonical_images.get(article_id, "")
        image_exists = bool(image_relative_path)
        if image_exists:
            matched_article_count += 1
        elif len(missing_article_examples) < max_examples:
            missing_article_examples.append(article_id)
        records.append(
            ArticleImageInventoryRecord(
                article_id=article_id,
                expected_image_relative_path=expected_relative_path,
                image_relative_path=image_relative_path,
                image_exists=image_exists,
            )
        )

    article_count = len(article_ids)
    missing_article_count = article_count - matched_article_count
    image_coverage = matched_article_count / article_count if article_count else 0.0
    summary = ArticleImageInventorySummary(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_data_dir=str(raw_dir),
        articles_path=str(articles_path),
        images_dir=str(images_dir),
        images_dir_exists=images_dir.exists(),
        images_dir_is_directory=images_dir.is_dir(),
        valid=not failures,
        article_count=article_count,
        scanned_image_file_count=scanned_image_file_count,
        canonical_image_file_count=len(canonical_images),
        matched_article_count=matched_article_count,
        missing_article_count=missing_article_count,
        extra_canonical_image_count=extra_canonical_image_count,
        malformed_image_file_count=malformed_image_file_count,
        image_coverage=image_coverage,
        failures=tuple(failures),
        missing_article_examples=tuple(missing_article_examples),
        extra_image_examples=tuple(extra_image_examples),
        malformed_image_examples=tuple(malformed_examples),
    )
    return ArticleImageInventory(records=tuple(records), summary=summary)


def write_article_image_inventory(
    raw_data_dir: Path | str,
    *,
    manifest_path: Path | str,
    report_path: Path | str,
    max_examples: int = 10,
) -> ArticleImageInventorySummary:
    """Write article image manifest CSV and JSON summary report.

    Args:
        raw_data_dir: Directory containing H&M ``articles.csv`` and optional
            ``images/``.
        manifest_path: Destination CSV path for per-article image records.
        report_path: Destination JSON path for aggregate diagnostics.
        max_examples: Maximum examples retained for each issue category.

    Returns:
        Summary with manifest and report paths populated.
    """

    inventory = build_article_image_inventory(raw_data_dir, max_examples=max_examples)
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ARTICLE_IMAGE_INVENTORY_HEADER)
        for record in inventory.records:
            writer.writerow(
                (
                    record.article_id,
                    record.expected_image_relative_path,
                    record.image_relative_path,
                    str(record.image_exists).lower(),
                )
            )

    summary_with_manifest = replace(
        inventory.summary,
        manifest_path=str(resolved_manifest_path),
    )
    return write_article_image_inventory_report(summary_with_manifest, report_path)


def write_article_image_inventory_report(
    summary: ArticleImageInventorySummary,
    report_path: Path | str,
) -> ArticleImageInventorySummary:
    """Write an image-inventory JSON summary report.

    Args:
        summary: Summary to serialize.
        report_path: Destination JSON path.

    Returns:
        Summary with ``report_path`` populated.
    """

    resolved_report_path = Path(report_path).expanduser().resolve()
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_with_report = replace(summary, report_path=str(resolved_report_path))
    with resolved_report_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary_with_report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary_with_report


def _canonical_article_id_from_image_path(
    image_path: Path,
    images_dir: Path,
) -> tuple[str | None, str | None]:
    """Return the canonical article ID for an image path, or an issue reason."""

    relative_to_images = image_path.relative_to(images_dir)
    parts = relative_to_images.parts
    if len(parts) != 2:
        return None, "expected images/<first-three-article-digits>/<article_id>.jpg"

    parent_dir, file_name = parts
    if image_path.suffix != IMAGE_SUFFIX:
        return None, f"unsupported image suffix {image_path.suffix!r}; expected {IMAGE_SUFFIX!r}"

    article_id = image_path.stem
    if not is_article_id(article_id):
        return None, f"invalid article_id filename stem {article_id!r}"
    if parent_dir != article_id[:3]:
        return None, f"folder {parent_dir!r} does not match article_id prefix {article_id[:3]!r}"
    if file_name != f"{article_id}{IMAGE_SUFFIX}":
        expected_file_name = f"{article_id}{IMAGE_SUFFIX}"
        return None, f"image filename must be {expected_file_name!r}"
    return article_id, None


def _relative_to_raw_data_dir(path: Path, raw_data_dir: Path) -> str:
    """Return a POSIX path relative to raw-data dir when possible."""

    try:
        return path.relative_to(raw_data_dir).as_posix()
    except ValueError:
        return path.as_posix()
