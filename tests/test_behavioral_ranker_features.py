from datetime import date
from decimal import Decimal
from math import log1p
from pathlib import Path

import pytest

from hm_recsys.data.io import CsvValueError, TransactionEvent, TransactionRecord
from hm_recsys.ranking.behavioral import (
    BEHAVIORAL_FEATURE_NAMES,
    SOURCE_AND_BEHAVIORAL_FEATURE_NAMES,
    build_cutoff_behavioral_features,
    load_article_attribute_maps,
    source_and_behavioral_feature_vector,
)
from hm_recsys.ranking.deterministic import CandidateFeatures
from hm_recsys.ranking.linear import LINEAR_FEATURE_NAMES

CUSTOMER_1 = "a" * 64
CUSTOMER_2 = "b" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
ARTICLE_3 = "0000000003"
ARTICLE_4 = "0000000004"


def test_cutoff_behavioral_features_ignore_validation_window_transactions() -> None:
    cutoff = date(2020, 9, 16)
    transactions = (
        TransactionEvent(date(2020, 9, 10), CUSTOMER_1, ARTICLE_1),
        TransactionEvent(date(2020, 9, 15), CUSTOMER_1, ARTICLE_1),
        TransactionEvent(date(2020, 9, 14), CUSTOMER_1, ARTICLE_2),
        TransactionEvent(date(2020, 9, 16), CUSTOMER_1, ARTICLE_3),
        TransactionEvent(date(2020, 9, 17), CUSTOMER_1, ARTICLE_1),
    )

    features = build_cutoff_behavioral_features(transactions, cutoff)
    vector = features.vector_for(CUSTOMER_1, ARTICLE_1)

    assert len(vector) == len(BEHAVIORAL_FEATURE_NAMES)
    customer_count_index = BEHAVIORAL_FEATURE_NAMES.index("customer_transaction_count_log")
    assert vector[customer_count_index] == pytest.approx(log1p(3))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("customer_unique_article_count_log")
    ] == pytest.approx(log1p(2))
    assert vector[BEHAVIORAL_FEATURE_NAMES.index("customer_days_since_last_purchase")] == 1.0
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_all_time_purchase_count_log")
    ] == pytest.approx(log1p(2))
    article_1d_index = BEHAVIORAL_FEATURE_NAMES.index("article_1d_purchase_count_log")
    assert vector[article_1d_index] == pytest.approx(log1p(1))
    article_7d_index = BEHAVIORAL_FEATURE_NAMES.index("article_7d_purchase_count_log")
    assert vector[article_7d_index] == pytest.approx(log1p(2))
    assert vector[BEHAVIORAL_FEATURE_NAMES.index("article_days_since_last_purchase")] == 1.0
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("user_article_purchase_count_log")
    ] == pytest.approx(log1p(2))
    user_article_recency_index = BEHAVIORAL_FEATURE_NAMES.index(
        "user_article_days_since_last_purchase"
    )
    assert vector[user_article_recency_index] == 1.0


def test_cutoff_behavioral_features_include_recent_trend_ratios() -> None:
    cutoff = date(2020, 9, 16)
    transactions = (
        TransactionEvent(date(2020, 9, 15), CUSTOMER_1, ARTICLE_1),
        TransactionEvent(date(2020, 9, 14), CUSTOMER_1, ARTICLE_1),
        TransactionEvent(date(2020, 9, 1), CUSTOMER_1, ARTICLE_2),
        TransactionEvent(date(2020, 8, 1), CUSTOMER_1, ARTICLE_3),
        TransactionEvent(date(2020, 9, 10), CUSTOMER_2, ARTICLE_1),
        TransactionEvent(date(2020, 9, 16), CUSTOMER_1, ARTICLE_1),
    )

    features = build_cutoff_behavioral_features(transactions, cutoff)
    vector = features.vector_for(CUSTOMER_1, ARTICLE_1)

    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("customer_7d_transaction_count_log")
    ] == pytest.approx(log1p(2))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("customer_30d_transaction_count_log")
    ] == pytest.approx(log1p(3))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("customer_7d_to_30d_transaction_ratio")
    ] == pytest.approx(2 / 3)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("customer_30d_to_all_time_transaction_ratio")
    ] == pytest.approx(3 / 4)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_1d_to_7d_purchase_ratio")
    ] == pytest.approx(1 / 3)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_3d_to_14d_purchase_ratio")
    ] == pytest.approx(2 / 3)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_7d_to_30d_purchase_ratio")
    ] == pytest.approx(1.0)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_7d_to_all_time_purchase_ratio")
    ] == pytest.approx(1.0)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_1d_purchase_count_delta_vs_7d_rate")
    ] == pytest.approx(1 - (3 * (1 / 7)))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_3d_purchase_count_delta_vs_14d_rate")
    ] == pytest.approx(2 - (3 * (3 / 14)))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_7d_purchase_count_delta_vs_30d_rate")
    ] == pytest.approx(3 - (3 * (7 / 30)))


def test_cutoff_behavioral_features_respect_customer_and_article_scopes() -> None:
    cutoff = date(2020, 9, 16)
    transactions = (
        TransactionEvent(date(2020, 9, 15), CUSTOMER_1, ARTICLE_1),
        TransactionEvent(date(2020, 9, 15), CUSTOMER_2, ARTICLE_2),
    )

    features = build_cutoff_behavioral_features(
        transactions,
        cutoff,
        target_customer_ids=(CUSTOMER_1,),
        candidate_article_ids=(ARTICLE_2,),
        missing_days_since=777.0,
    )

    scoped_pair = features.vector_for(CUSTOMER_1, ARTICLE_2)
    out_of_scope_customer_pair = features.vector_for(CUSTOMER_2, ARTICLE_2)
    out_of_scope_article_pair = features.vector_for(CUSTOMER_1, ARTICLE_1)

    customer_count_index = BEHAVIORAL_FEATURE_NAMES.index("customer_transaction_count_log")
    assert scoped_pair[customer_count_index] == pytest.approx(log1p(1))
    assert scoped_pair[
        BEHAVIORAL_FEATURE_NAMES.index("article_all_time_purchase_count_log")
    ] == pytest.approx(log1p(1))
    user_article_count_index = BEHAVIORAL_FEATURE_NAMES.index("user_article_purchase_count_log")
    assert scoped_pair[user_article_count_index] == 0.0
    user_article_recency_index = BEHAVIORAL_FEATURE_NAMES.index(
        "user_article_days_since_last_purchase"
    )
    assert scoped_pair[user_article_recency_index] == 777.0
    assert (
        out_of_scope_customer_pair[BEHAVIORAL_FEATURE_NAMES.index("customer_transaction_count_log")]
        == 0.0
    )
    assert (
        out_of_scope_article_pair[
            BEHAVIORAL_FEATURE_NAMES.index("article_all_time_purchase_count_log")
        ]
        == 0.0
    )


def test_cutoff_behavioral_features_include_price_and_channel_stats() -> None:
    cutoff = date(2020, 9, 16)
    transactions = (
        TransactionRecord(date(2020, 9, 10), CUSTOMER_1, ARTICLE_1, Decimal("0.10"), 2),
        TransactionRecord(date(2020, 9, 15), CUSTOMER_1, ARTICLE_1, Decimal("0.20"), 1),
        TransactionRecord(date(2020, 9, 15), CUSTOMER_2, ARTICLE_1, Decimal("0.30"), 2),
        TransactionRecord(date(2020, 9, 16), CUSTOMER_1, ARTICLE_2, Decimal("0.90"), 2),
    )

    features = build_cutoff_behavioral_features(transactions, cutoff)
    vector = features.vector_for(CUSTOMER_1, ARTICLE_1)

    assert vector[BEHAVIORAL_FEATURE_NAMES.index("customer_mean_price")] == pytest.approx(0.15)
    assert vector[BEHAVIORAL_FEATURE_NAMES.index("article_mean_price")] == pytest.approx(0.20)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_mean_price_minus_customer_mean")
    ] == pytest.approx(0.05)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_mean_price_ratio_customer_mean")
    ] == pytest.approx(0.20 / 0.15)
    assert vector[BEHAVIORAL_FEATURE_NAMES.index("customer_sales_channel_2_share")] == 0.5
    assert vector[BEHAVIORAL_FEATURE_NAMES.index("article_sales_channel_2_share")] == pytest.approx(
        2 / 3
    )
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_customer_sales_channel_2_share_gap_abs")
    ] == pytest.approx((2 / 3) - 0.5)


def test_cutoff_behavioral_features_include_article_attribute_affinities() -> None:
    cutoff = date(2020, 9, 16)
    transactions = (
        TransactionEvent(date(2020, 9, 10), CUSTOMER_1, ARTICLE_1),
        TransactionEvent(date(2020, 9, 15), CUSTOMER_1, ARTICLE_2),
        TransactionEvent(date(2020, 9, 16), CUSTOMER_1, ARTICLE_4),
    )
    article_attributes = {
        ARTICLE_1: {
            "product_type_no": "10",
            "product_group_name": "Garment Upper body",
            "department_no": "100",
            "section_no": "1",
            "garment_group_no": "20",
            "colour_group_code": "9",
            "index_group_no": "3",
        },
        ARTICLE_2: {
            "product_type_no": "11",
            "product_group_name": "Garment Upper body",
            "department_no": "100",
            "section_no": "1",
            "garment_group_no": "20",
            "colour_group_code": "9",
            "index_group_no": "3",
        },
        ARTICLE_3: {
            "product_type_no": "10",
            "product_group_name": "Garment Upper body",
            "department_no": "100",
            "section_no": "1",
            "garment_group_no": "20",
            "colour_group_code": "9",
            "index_group_no": "3",
        },
        ARTICLE_4: {
            "product_type_no": "10",
            "product_group_name": "Garment Upper body",
            "department_no": "100",
            "section_no": "1",
            "garment_group_no": "20",
            "colour_group_code": "9",
            "index_group_no": "3",
        },
    }

    features = build_cutoff_behavioral_features(
        transactions,
        cutoff,
        article_attributes_by_id=article_attributes,
    )
    vector = features.vector_for(CUSTOMER_1, ARTICLE_3)

    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("user_product_type_purchase_count_log")
    ] == pytest.approx(log1p(1))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("user_product_group_purchase_count_log")
    ] == pytest.approx(log1p(2))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("user_garment_group_purchase_count_log")
    ] == pytest.approx(log1p(2))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("user_garment_group_30d_purchase_count_log")
    ] == pytest.approx(log1p(2))
    assert (
        vector[BEHAVIORAL_FEATURE_NAMES.index("user_product_type_days_since_last_purchase")] == 6.0
    )
    assert (
        vector[BEHAVIORAL_FEATURE_NAMES.index("user_garment_group_days_since_last_purchase")] == 1.0
    )


def test_cutoff_behavioral_features_include_article_attribute_trend_ratios() -> None:
    cutoff = date(2020, 9, 16)
    transactions = (
        TransactionEvent(date(2020, 9, 15), CUSTOMER_1, ARTICLE_1),
        TransactionEvent(date(2020, 8, 1), CUSTOMER_1, ARTICLE_2),
        TransactionEvent(date(2020, 9, 10), CUSTOMER_2, ARTICLE_1),
        TransactionEvent(date(2020, 9, 1), CUSTOMER_2, ARTICLE_2),
        TransactionEvent(date(2020, 9, 16), CUSTOMER_2, ARTICLE_1),
    )
    article_attributes = {
        ARTICLE_1: {
            "product_type_no": "10",
            "section_no": "1",
            "garment_group_no": "20",
        },
        ARTICLE_2: {
            "product_type_no": "10",
            "section_no": "1",
            "garment_group_no": "20",
        },
        ARTICLE_3: {
            "product_type_no": "10",
            "section_no": "1",
            "garment_group_no": "20",
        },
    }

    features = build_cutoff_behavioral_features(
        transactions,
        cutoff,
        article_attributes_by_id=article_attributes,
    )
    vector = features.vector_for(CUSTOMER_1, ARTICLE_3)

    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("user_product_type_30d_to_all_time_purchase_ratio")
    ] == pytest.approx(1 / 2)
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_product_type_7d_purchase_count_log")
    ] == pytest.approx(log1p(2))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_product_type_30d_purchase_count_log")
    ] == pytest.approx(log1p(3))
    assert vector[
        BEHAVIORAL_FEATURE_NAMES.index("article_product_type_7d_to_30d_purchase_ratio")
    ] == pytest.approx(2 / 3)


def test_cutoff_behavioral_features_reject_negative_missing_days_since() -> None:
    with pytest.raises(ValueError, match="missing_days_since"):
        build_cutoff_behavioral_features((), date(2020, 9, 16), missing_days_since=-1.0)


def test_load_article_attribute_maps_preserves_article_ids_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    articles_path = tmp_path / "articles.csv"
    articles_path.write_text(
        "article_id,product_type_no,garment_group_no\n" "0000000001,10,20\n" "0000000002,11,21\n",
        encoding="utf-8",
    )

    attributes = load_article_attribute_maps(tmp_path, columns=("product_type_no",))

    assert tuple(attributes) == ("0000000001", "0000000002")
    assert attributes["0000000001"] == {"product_type_no": "10"}

    articles_path.write_text(
        "article_id,product_type_no\n" "0000000001,10\n" "0000000001,11\n",
        encoding="utf-8",
    )
    with pytest.raises(CsvValueError, match="duplicate article_id"):
        load_article_attribute_maps(tmp_path, columns=("product_type_no",))


def test_source_and_behavioral_feature_vector_matches_combined_schema() -> None:
    cutoff = date(2020, 9, 16)
    behavioral_features = build_cutoff_behavioral_features(
        (TransactionEvent(date(2020, 9, 15), CUSTOMER_1, ARTICLE_1),),
        cutoff,
    )
    candidate_features = CandidateFeatures(
        customer_id=CUSTOMER_1,
        article_id=ARTICLE_1,
        repeat_rank=2,
        repeat_score=0.5,
        source_count=1,
        best_rank=2,
    )

    vector = source_and_behavioral_feature_vector(candidate_features, behavioral_features)

    assert (
        *LINEAR_FEATURE_NAMES,
        *BEHAVIORAL_FEATURE_NAMES,
    ) == SOURCE_AND_BEHAVIORAL_FEATURE_NAMES
    assert len(set(BEHAVIORAL_FEATURE_NAMES)) == len(BEHAVIORAL_FEATURE_NAMES)
    assert len(vector) == len(SOURCE_AND_BEHAVIORAL_FEATURE_NAMES)
    assert vector[LINEAR_FEATURE_NAMES.index("has_repeat")] == 1.0
    customer_count_index = SOURCE_AND_BEHAVIORAL_FEATURE_NAMES.index(
        "customer_transaction_count_log"
    )
    assert vector[customer_count_index] == pytest.approx(log1p(1))
