from datetime import date
from math import log1p

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.ranking.behavioral import (
    BEHAVIORAL_FEATURE_NAMES,
    SOURCE_AND_BEHAVIORAL_FEATURE_NAMES,
    build_cutoff_behavioral_features,
    source_and_behavioral_feature_vector,
)
from hm_recsys.ranking.deterministic import CandidateFeatures
from hm_recsys.ranking.linear import LINEAR_FEATURE_NAMES

CUSTOMER_1 = "a" * 64
CUSTOMER_2 = "b" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
ARTICLE_3 = "0000000003"


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


def test_cutoff_behavioral_features_reject_negative_missing_days_since() -> None:
    with pytest.raises(ValueError, match="missing_days_since"):
        build_cutoff_behavioral_features((), date(2020, 9, 16), missing_days_since=-1.0)


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
    assert len(vector) == len(SOURCE_AND_BEHAVIORAL_FEATURE_NAMES)
    assert vector[LINEAR_FEATURE_NAMES.index("has_repeat")] == 1.0
    customer_count_index = SOURCE_AND_BEHAVIORAL_FEATURE_NAMES.index(
        "customer_transaction_count_log"
    )
    assert vector[customer_count_index] == pytest.approx(log1p(1))
