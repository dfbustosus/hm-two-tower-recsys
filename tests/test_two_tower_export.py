import csv
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.training.two_tower_export import (
    TWO_TOWER_ARTICLE_MAPPING_HEADER,
    TWO_TOWER_CUSTOMER_MAPPING_HEADER,
    TWO_TOWER_EXAMPLE_HEADER,
    TwoTowerExampleExportConfig,
    write_two_tower_example_export,
    write_two_tower_example_export_summary,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
THIRD_CUSTOMER_ID = "c" * 64
VALIDATION_CUSTOMER_ID = "d" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
ARTICLE_3 = "0000000003"
ARTICLE_4 = "0000000004"
ARTICLE_5 = "0000000005"
VALIDATION_ONLY_ARTICLE = "0000000009"


def test_two_tower_export_is_cutoff_safe_and_preserves_mapping_integrity(
    tmp_path: Path,
) -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 3), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 4), SECOND_CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 5), THIRD_CUSTOMER_ID, ARTICLE_4),
        TransactionEvent(date(2020, 1, 6), THIRD_CUSTOMER_ID, ARTICLE_5),
        TransactionEvent(date(2020, 1, 8), VALIDATION_CUSTOMER_ID, VALIDATION_ONLY_ARTICLE),
    ]
    paths = export_paths(tmp_path)

    summary = write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-08"),
        examples_path=paths["examples"],
        customer_mapping_path=paths["customers"],
        article_mapping_path=paths["articles"],
        config=TwoTowerExampleExportConfig(
            negatives_per_positive=1,
            seed=7,
            max_positive_examples=2,
        ),
    )

    examples = read_dict_rows(paths["examples"])
    customers = read_rows(paths["customers"])
    articles = read_rows(paths["articles"])

    assert tuple(examples[0]) == TWO_TOWER_EXAMPLE_HEADER
    assert tuple(customers[0]) == TWO_TOWER_CUSTOMER_MAPPING_HEADER
    assert tuple(articles[0]) == TWO_TOWER_ARTICLE_MAPPING_HEADER
    assert VALIDATION_ONLY_ARTICLE not in {row[1] for row in articles[1:]}
    assert VALIDATION_CUSTOMER_ID not in {row[1] for row in customers[1:]}
    assert summary.train_rows_seen == 6
    assert summary.positive_examples_written == 2
    assert summary.negative_examples_written == 2
    assert summary.rows_written == 4
    assert summary.unique_customers == 1
    assert summary.unique_articles == 5

    positive_rows = [row for row in examples if row["label"] == "1"]
    negative_rows = [row for row in examples if row["label"] == "0"]

    assert {row["article_id"] for row in positive_rows} == {ARTICLE_1, ARTICLE_2}
    assert {row["positive_count"] for row in positive_rows if row["article_id"] == ARTICLE_1} == {
        "2"
    }
    assert {row["anchor_t_dat"] for row in positive_rows if row["article_id"] == ARTICLE_1} == {
        "2020-01-03"
    }
    assert all(row["article_id"] not in {ARTICLE_1, ARTICLE_2} for row in negative_rows)
    assert all(row["example_type"] == "random_negative" for row in negative_rows)
    assert_mapping_indices_exist(examples, customers, articles)


def test_two_tower_export_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 3), THIRD_CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 4), THIRD_CUSTOMER_ID, ARTICLE_4),
    ]
    first_paths = export_paths(tmp_path / "first")
    second_paths = export_paths(tmp_path / "second")
    config = TwoTowerExampleExportConfig(negatives_per_positive=2, seed=42)

    write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-05"),
        examples_path=first_paths["examples"],
        customer_mapping_path=first_paths["customers"],
        article_mapping_path=first_paths["articles"],
        config=config,
    )
    write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-05"),
        examples_path=second_paths["examples"],
        customer_mapping_path=second_paths["customers"],
        article_mapping_path=second_paths["articles"],
        config=config,
    )

    assert first_paths["examples"].read_text(encoding="utf-8") == second_paths[
        "examples"
    ].read_text(encoding="utf-8")
    assert first_paths["customers"].read_text(encoding="utf-8") == second_paths[
        "customers"
    ].read_text(encoding="utf-8")
    assert first_paths["articles"].read_text(encoding="utf-8") == second_paths[
        "articles"
    ].read_text(encoding="utf-8")


def test_two_tower_export_can_select_latest_positive_pairs(tmp_path: Path) -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 6), THIRD_CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 7), THIRD_CUSTOMER_ID, ARTICLE_4),
    ]
    paths = export_paths(tmp_path)

    summary = write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-08"),
        examples_path=paths["examples"],
        customer_mapping_path=paths["customers"],
        article_mapping_path=paths["articles"],
        config=TwoTowerExampleExportConfig(
            negatives_per_positive=0,
            max_positive_examples=2,
            positive_selection="latest",
        ),
    )

    positive_rows = [row for row in read_dict_rows(paths["examples"]) if row["label"] == "1"]
    assert summary.positive_selection == "latest"
    assert {row["article_id"] for row in positive_rows} == {ARTICLE_3, ARTICLE_4}


def test_two_tower_export_can_select_latest_positive_per_customer(tmp_path: Path) -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 5), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 4), SECOND_CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 6), THIRD_CUSTOMER_ID, ARTICLE_4),
    ]
    paths = export_paths(tmp_path)

    summary = write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-08"),
        examples_path=paths["examples"],
        customer_mapping_path=paths["customers"],
        article_mapping_path=paths["articles"],
        config=TwoTowerExampleExportConfig(
            negatives_per_positive=0,
            max_positive_examples=2,
            positive_selection="latest_customer",
        ),
    )

    positive_rows = [row for row in read_dict_rows(paths["examples"]) if row["label"] == "1"]
    assert summary.positive_selection == "latest_customer"
    assert summary.unique_customers == 2
    assert {row["customer_id"] for row in positive_rows} == {CUSTOMER_ID, THIRD_CUSTOMER_ID}
    assert {row["article_id"] for row in positive_rows} == {ARTICLE_2, ARTICLE_4}


def test_two_tower_export_reports_skipped_negatives_when_pool_is_empty(tmp_path: Path) -> None:
    events = [TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1)]
    paths = export_paths(tmp_path)

    summary = write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-02"),
        examples_path=paths["examples"],
        customer_mapping_path=paths["customers"],
        article_mapping_path=paths["articles"],
        config=TwoTowerExampleExportConfig(negatives_per_positive=1),
    )

    assert summary.positive_examples_written == 1
    assert summary.negative_examples_written == 0
    assert summary.customers_without_negative_pool == 1
    assert summary.skipped_negative_examples == 1


def test_two_tower_export_supports_popularity_weighted_negatives(tmp_path: Path) -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 1), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 3), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 4), THIRD_CUSTOMER_ID, ARTICLE_3),
        TransactionEvent(date(2020, 1, 8), VALIDATION_CUSTOMER_ID, VALIDATION_ONLY_ARTICLE),
    ]
    paths = export_paths(tmp_path)

    summary = write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-08"),
        examples_path=paths["examples"],
        customer_mapping_path=paths["customers"],
        article_mapping_path=paths["articles"],
        config=TwoTowerExampleExportConfig(
            negatives_per_positive=2,
            seed=42,
            negative_sampling="popularity",
            max_positive_examples=1,
        ),
    )

    negative_rows = [row for row in read_dict_rows(paths["examples"]) if row["label"] == "0"]
    assert summary.negative_sampling == "popularity"
    assert all(row["example_type"] == "popularity_negative" for row in negative_rows)
    assert all(row["article_id"] != ARTICLE_1 for row in negative_rows)
    assert VALIDATION_ONLY_ARTICLE not in {row["article_id"] for row in negative_rows}


def test_two_tower_export_mixed_negatives_alternate_popularity_and_random(
    tmp_path: Path,
) -> None:
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 1), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 2), THIRD_CUSTOMER_ID, ARTICLE_3),
    ]
    paths = export_paths(tmp_path)

    write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-08"),
        examples_path=paths["examples"],
        customer_mapping_path=paths["customers"],
        article_mapping_path=paths["articles"],
        config=TwoTowerExampleExportConfig(
            negatives_per_positive=2,
            seed=42,
            negative_sampling="mixed",
            max_positive_examples=1,
        ),
    )

    negative_types = [
        row["example_type"] for row in read_dict_rows(paths["examples"]) if row["label"] == "0"
    ]
    assert negative_types == ["popularity_negative", "random_negative"]


def test_two_tower_export_rejects_invalid_config(tmp_path: Path) -> None:
    paths = export_paths(tmp_path)

    with pytest.raises(ValueError, match="negatives_per_positive"):
        TwoTowerExampleExportConfig(negatives_per_positive=-1)

    with pytest.raises(ValueError, match="unsupported negative_sampling"):
        TwoTowerExampleExportConfig(negative_sampling="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="progress_interval"):
        write_two_tower_example_export(
            transaction_iter_factory=lambda: iter(()),
            split=TemporalSplit.from_isoformat("2020-01-02"),
            examples_path=paths["examples"],
            customer_mapping_path=paths["customers"],
            article_mapping_path=paths["articles"],
            progress_interval=0,
        )


def test_write_two_tower_export_summary(tmp_path: Path) -> None:
    events = [TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1)]
    paths = export_paths(tmp_path)
    summary = write_two_tower_example_export(
        transaction_iter_factory=lambda: iter(events),
        split=TemporalSplit.from_isoformat("2020-01-02"),
        examples_path=paths["examples"],
        customer_mapping_path=paths["customers"],
        article_mapping_path=paths["articles"],
        config=TwoTowerExampleExportConfig(negatives_per_positive=0),
    )

    report_path = write_two_tower_example_export_summary(summary, tmp_path / "report.json")

    assert report_path.exists()
    assert '"positive_examples_written"' in report_path.read_text(encoding="utf-8")


def export_paths(base: Path) -> dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    return {
        "examples": base / "examples.csv",
        "customers": base / "customers.csv",
        "articles": base / "articles.csv",
    }


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_rows(path: Path) -> list[tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [tuple(row) for row in csv.reader(handle)]


def assert_mapping_indices_exist(
    examples: list[dict[str, str]],
    customers: list[tuple[str, ...]],
    articles: list[tuple[str, ...]],
) -> None:
    customer_indices = {row[0] for row in customers[1:]}
    article_indices = {row[0] for row in articles[1:]}
    assert all(row["customer_index"] in customer_indices for row in examples)
    assert all(row["article_index"] in article_indices for row in examples)
