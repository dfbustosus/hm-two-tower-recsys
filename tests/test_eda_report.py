"""Unit tests for the Phase -1 exploratory data analysis report."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.eda.report import (
    DEFAULT_AGE_BUCKETS,
    REPEAT_PROXY_PURGE_DAYS,
    EdaReportConfig,
    EdaSegmentThresholds,
    _assign_age_bucket,
    _integer_percentiles,
    _top_values,
    build_eda_report_from_events,
    eda_report_to_dict,
    render_eda_report_markdown,
    write_eda_report,
    write_eda_report_markdown,
)

EMPTY_CUTOFFS_MATCH = r"rolling_cutoffs must not be empty"
INVALID_DATE_MATCH = r"Invalid isoformat string"
INVALID_PERCENTILE_MATCH = r"percentile must be between 1 and 99"
INVALID_THRESHOLD_MATCH = r"sparse_max_transactions must be >= cold_max_transactions"


CUSTOMER_ONE = "a" * 64
CUSTOMER_TWO = "b" * 64
CUSTOMER_THREE = "c" * 64
CUSTOMER_COLD = "d" * 64
ARTICLE_ONE = "0000000001"
ARTICLE_TWO = "0000000002"
ARTICLE_THREE = "0000000003"


def _basic_events() -> list[TransactionEvent]:
    return [
        TransactionEvent(date(2020, 9, 1), CUSTOMER_ONE, ARTICLE_ONE),
        TransactionEvent(date(2020, 9, 1), CUSTOMER_ONE, ARTICLE_TWO),
        TransactionEvent(date(2020, 9, 2), CUSTOMER_ONE, ARTICLE_ONE),
        TransactionEvent(date(2020, 9, 3), CUSTOMER_TWO, ARTICLE_TWO),
        TransactionEvent(date(2020, 9, 4), CUSTOMER_TWO, ARTICLE_TWO),
        TransactionEvent(date(2020, 9, 5), CUSTOMER_TWO, ARTICLE_THREE),
        TransactionEvent(date(2020, 9, 8), CUSTOMER_THREE, ARTICLE_THREE),
        TransactionEvent(date(2020, 9, 14), CUSTOMER_THREE, ARTICLE_THREE),
    ]


def _basic_channel_map() -> dict[tuple[str, str, date], int]:
    return {
        (CUSTOMER_ONE, ARTICLE_ONE, date(2020, 9, 1)): 1,
        (CUSTOMER_ONE, ARTICLE_TWO, date(2020, 9, 1)): 1,
        (CUSTOMER_ONE, ARTICLE_ONE, date(2020, 9, 2)): 1,
        (CUSTOMER_TWO, ARTICLE_TWO, date(2020, 9, 3)): 2,
        (CUSTOMER_TWO, ARTICLE_TWO, date(2020, 9, 4)): 2,
        (CUSTOMER_TWO, ARTICLE_THREE, date(2020, 9, 5)): 2,
        (CUSTOMER_THREE, ARTICLE_THREE, date(2020, 9, 8)): 1,
        (CUSTOMER_THREE, ARTICLE_THREE, date(2020, 9, 14)): 2,
    }


def _articles_rows() -> list[dict[str, str]]:
    return [
        {
            "article_id": ARTICLE_ONE,
            "product_type_name": "Trousers",
            "product_group_name": "Garment Lower body",
            "graphical_appearance_name": "Solid",
            "colour_group_name": "Black",
            "perceived_colour_value_name": "Dark",
            "perceived_colour_master_name": "Black",
            "department_name": "Jersey Basic",
            "index_name": "Ladieswear",
            "index_group_name": "Ladieswear",
            "section_name": "Womens Everyday Collection",
            "garment_group_name": "Trousers",
        },
        {
            "article_id": ARTICLE_TWO,
            "product_type_name": "Sweater",
            "product_group_name": "Garment Upper body",
            "graphical_appearance_name": "Solid",
            "colour_group_name": "Blue",
            "perceived_colour_value_name": "Medium",
            "perceived_colour_master_name": "Blue",
            "department_name": "Knit",
            "index_name": "Ladieswear",
            "index_group_name": "Ladieswear",
            "section_name": "Womens Everyday Collection",
            "garment_group_name": "Knitwear",
        },
        {
            "article_id": ARTICLE_THREE,
            "product_type_name": "Sweater",
            "product_group_name": "Garment Upper body",
            "graphical_appearance_name": "Stripe",
            "colour_group_name": "White",
            "perceived_colour_value_name": "Light",
            "perceived_colour_master_name": "White",
            "department_name": "Knit",
            "index_name": "Menswear",
            "index_group_name": "Menswear",
            "section_name": "Mens Everyday Collection",
            "garment_group_name": "Knitwear",
        },
    ]


def _customers_rows() -> list[dict[str, str]]:
    return [
        {
            "customer_id": CUSTOMER_ONE,
            "age": "25",
            "FN": "1.0",
            "Active": "1.0",
            "club_member_status": "ACTIVE",
            "fashion_news_frequency": "Regularly",
        },
        {
            "customer_id": CUSTOMER_TWO,
            "age": "45",
            "FN": "",
            "Active": "",
            "club_member_status": "ACTIVE",
            "fashion_news_frequency": "NONE",
        },
        {
            "customer_id": CUSTOMER_THREE,
            "age": "",
            "FN": "",
            "Active": "",
            "club_member_status": "PRE-CREATE",
            "fashion_news_frequency": "NONE",
        },
        {
            "customer_id": CUSTOMER_COLD,
            "age": "75",
            "FN": "1.0",
            "Active": "1.0",
            "club_member_status": "ACTIVE",
            "fashion_news_frequency": "Monthly",
        },
    ]


def test_integer_percentiles_returns_min_max_and_breakpoints() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = _integer_percentiles(values, (50, 90))
    assert result["min"] == 1
    assert result["max"] == 10
    assert result["p50"] in {5, 6}
    assert result["p90"] == 9


def test_integer_percentiles_handles_empty_input() -> None:
    result = _integer_percentiles([], (50,))
    assert result == {"min": 0, "p50": 0, "max": 0}


def test_top_values_orders_by_descending_count_then_value() -> None:
    counts = {"alpha": 2, "beta": 3, "gamma": 3, "delta": 1}
    result = _top_values(counts, limit=3)
    assert result == (("beta", 3), ("gamma", 3), ("alpha", 2))


@pytest.mark.parametrize(
    ("raw_age", "expected"),
    [
        ("", "missing"),
        ("18", "under_20"),
        ("20", "20_to_24"),
        ("24", "20_to_24"),
        ("25", "25_to_29"),
        ("30", "30_to_39"),
        ("70", "70_plus"),
        ("not-a-number", "missing"),
    ],
)
def test_assign_age_bucket_maps_known_values(raw_age: str, expected: str) -> None:
    assert _assign_age_bucket(raw_age) == expected


def test_default_age_buckets_cover_every_label_in_report() -> None:
    labels = {label for label, _, _ in DEFAULT_AGE_BUCKETS}
    assert "missing" in labels
    assert "70_plus" in labels


def test_build_eda_report_aggregates_transactions_and_segments() -> None:
    events = _basic_events()
    submission_customers = {CUSTOMER_ONE, CUSTOMER_TWO, CUSTOMER_THREE, CUSTOMER_COLD}
    config = EdaReportConfig(
        rolling_cutoffs=("2020-09-03", "2020-09-08", "2020-09-15"),
        percentiles=(50, 90),
        top_hierarchy_values=3,
        top_busy_days=3,
        segment_thresholds=EdaSegmentThresholds(
            cold_max_transactions=0,
            sparse_max_transactions=2,
        ),
    )

    report = build_eda_report_from_events(
        events=events,
        submission_customer_ids=submission_customers,
        customer_metadata_rows=_customers_rows(),
        article_metadata_rows=_articles_rows(),
        config=config,
        channel_id_provider=_basic_channel_map(),
    )

    transactions = report.transactions
    assert transactions.total_rows == len(events)
    assert transactions.distinct_customers == 3
    assert transactions.distinct_articles == 3
    assert transactions.date_min == "2020-09-01"
    assert transactions.date_max == "2020-09-14"
    assert transactions.monthly_counts == {"2020-09": 8}
    weekday_total = sum(transactions.weekday_counts.values())
    assert weekday_total == len(events)
    assert transactions.top_busy_days[0][0] == "2020-09-01"

    channels = report.channels
    assert channels.rows_by_channel == {"1": 4, "2": 4}
    assert channels.customers_with_observed_channel_count == 3
    assert channels.customers_online_dominant == 1
    assert channels.customers_store_dominant == 1
    assert channels.customers_mixed == 1

    customers = report.customers
    assert customers.submission_customers == 4
    assert customers.submission_customers_with_any_history == 3
    cold_count = customers.submission_segment_counts["cold"]
    assert cold_count == 1
    assert customers.cold_user_share_by_cutoff["2020-09-03"] == pytest.approx(3 / 4)
    assert customers.cold_user_share_by_cutoff["2020-09-08"] == pytest.approx(2 / 4)
    assert customers.cold_user_share_by_cutoff["2020-09-15"] == pytest.approx(1 / 4)
    assert customers.cold_customer_counts_by_cutoff["2020-09-03"] == 3
    assert customers.cold_customer_counts_by_cutoff["2020-09-08"] == 2
    assert customers.cold_customer_counts_by_cutoff["2020-09-15"] == 1

    articles = report.articles
    assert articles.total_articles_in_articles_csv == 3
    assert articles.articles_with_at_least_one_transaction == 3
    assert articles.hierarchy_distinct_value_counts["product_group_name"] == 2

    repeat = report.repeat_purchase
    assert repeat.distinct_customer_article_pairs == 5
    assert repeat.pairs_with_repeat == 3
    assert repeat.same_pair_returns_proxy["within_2d"] == 2
    assert repeat.same_pair_returns_proxy["within_3_to_7d"] == 1
    assert repeat.same_pair_returns_proxy["within_8_to_14d"] == 0

    metadata = report.customer_metadata
    assert metadata.total_customers_in_customers_csv == 4
    assert metadata.age_buckets["missing"] == 1
    assert metadata.age_buckets["25_to_29"] == 1
    assert metadata.categorical_distributions["club_member_status"]["ACTIVE"] == 3


def test_build_eda_report_handles_cutoff_without_history() -> None:
    events = [
        TransactionEvent(date(2020, 9, 5), CUSTOMER_ONE, ARTICLE_ONE),
    ]
    config = EdaReportConfig(rolling_cutoffs=("2020-09-01",))
    report = build_eda_report_from_events(
        events=events,
        submission_customer_ids={CUSTOMER_ONE, CUSTOMER_TWO},
        config=config,
    )
    assert report.customers.cold_user_share_by_cutoff["2020-09-01"] == pytest.approx(1.0)
    assert report.customers.cold_customer_counts_by_cutoff["2020-09-01"] == 2


def test_repeat_returns_proxy_purges_long_gap_pairs() -> None:
    cutoff_days = REPEAT_PROXY_PURGE_DAYS + 1
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ONE, ARTICLE_ONE),
        TransactionEvent(date(2020, 1, 1) + timedelta(days=cutoff_days), CUSTOMER_ONE, ARTICLE_ONE),
    ]
    config = EdaReportConfig(rolling_cutoffs=("2020-09-01",))
    report = build_eda_report_from_events(
        events=events,
        submission_customer_ids={CUSTOMER_ONE},
        config=config,
    )
    assert report.repeat_purchase.same_pair_returns_proxy["within_2d"] == 0
    assert report.repeat_purchase.same_pair_returns_proxy["within_3_to_7d"] == 0
    assert report.repeat_purchase.same_pair_returns_proxy["within_8_to_14d"] == 0


def test_eda_report_to_dict_round_trips_through_json(tmp_path: Path) -> None:
    events = _basic_events()
    report = build_eda_report_from_events(
        events=events,
        submission_customer_ids={CUSTOMER_ONE, CUSTOMER_TWO, CUSTOMER_THREE},
        customer_metadata_rows=_customers_rows(),
        article_metadata_rows=_articles_rows(),
        config=EdaReportConfig(rolling_cutoffs=("2020-09-05",)),
        channel_id_provider=_basic_channel_map(),
    )

    output_path = tmp_path / "eda_report.json"
    write_eda_report(report, output_path)
    payload = json.loads(output_path.read_text())
    assert payload["transactions"]["total_rows"] == len(events)
    assert payload["customers"]["cold_user_share_by_cutoff"]["2020-09-05"] >= 0.0
    assert isinstance(payload["transactions"]["top_busy_days"], list)
    assert payload["articles"]["hierarchy_top_values"]["product_type_name"][0][0] == "Sweater"


def test_render_eda_report_markdown_contains_section_headers(tmp_path: Path) -> None:
    events = _basic_events()
    report = build_eda_report_from_events(
        events=events,
        submission_customer_ids={CUSTOMER_ONE, CUSTOMER_TWO, CUSTOMER_THREE},
        customer_metadata_rows=_customers_rows(),
        article_metadata_rows=_articles_rows(),
        config=EdaReportConfig(rolling_cutoffs=("2020-09-05",)),
    )

    markdown_path = tmp_path / "eda_report.md"
    write_eda_report_markdown(report, markdown_path)
    markdown = markdown_path.read_text()
    for header in (
        "# H&M Exploratory Data Analysis (EDA)",
        "## Transaction volume",
        "## Sales channels",
        "## Customer history depth",
        "## Article catalog",
        "## Repeat purchases",
        "## Customer metadata",
    ):
        assert header in markdown


def test_render_eda_report_markdown_is_deterministic_for_same_input() -> None:
    events = _basic_events()
    config = EdaReportConfig(rolling_cutoffs=("2020-09-05",))
    first = build_eda_report_from_events(
        events=events,
        submission_customer_ids={CUSTOMER_ONE, CUSTOMER_TWO, CUSTOMER_THREE},
        customer_metadata_rows=_customers_rows(),
        article_metadata_rows=_articles_rows(),
        config=config,
    )
    second = build_eda_report_from_events(
        events=events,
        submission_customer_ids={CUSTOMER_ONE, CUSTOMER_TWO, CUSTOMER_THREE},
        customer_metadata_rows=_customers_rows(),
        article_metadata_rows=_articles_rows(),
        config=config,
    )
    first_dict = eda_report_to_dict(first)
    second_dict = eda_report_to_dict(second)
    first_dict.pop("generated_at_utc")
    second_dict.pop("generated_at_utc")
    assert first_dict == second_dict
    assert render_eda_report_markdown(first).replace(
        first.generated_at_utc, "T"
    ) == render_eda_report_markdown(second).replace(second.generated_at_utc, "T")


def test_eda_report_config_validates_cutoffs() -> None:
    with pytest.raises(ValueError, match=EMPTY_CUTOFFS_MATCH):
        EdaReportConfig(rolling_cutoffs=())
    with pytest.raises(ValueError, match=INVALID_DATE_MATCH):
        EdaReportConfig(rolling_cutoffs=("not-a-date",))
    with pytest.raises(ValueError, match=INVALID_PERCENTILE_MATCH):
        EdaReportConfig(percentiles=(0,))
    with pytest.raises(ValueError, match=INVALID_THRESHOLD_MATCH):
        EdaReportConfig(
            segment_thresholds=EdaSegmentThresholds(
                cold_max_transactions=5,
                sparse_max_transactions=2,
            )
        )
