import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.embeddings.contracts import ArticleEmbeddingRecord
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.content_similarity import (
    build_content_similarity_candidate_records,
    build_content_similarity_candidate_source_records,
    build_content_similarity_index,
    content_similarity_article_coverage,
)
from hm_recsys.retrieval.source_names import MULTIMODAL_SIMILARITY_SOURCE

CUSTOMER_ID = "a" * 64
OTHER_CUSTOMER_ID = "b" * 64


def test_content_similarity_candidates_use_only_pre_cutoff_history() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16", horizon_days=7)
    records = _embedding_records()
    transactions = (
        TransactionEvent(split.cutoff.replace(day=1), CUSTOMER_ID, "0100000000"),
        TransactionEvent(split.cutoff.replace(day=17), CUSTOMER_ID, "0300000000"),
        TransactionEvent(split.cutoff.replace(day=1), OTHER_CUSTOMER_ID, "0300000000"),
    )

    index = build_content_similarity_index(
        transactions,
        split,
        target_customer_ids=(CUSTOMER_ID,),
        embedding_records=records,
        max_history_items=4,
    )

    candidates = build_content_similarity_candidate_records(index, CUSTOMER_ID, k=2)

    assert index.train_rows_used == 2
    assert index.customer_histories == {CUSTOMER_ID: ("0100000000",)}
    assert tuple(candidate.article_id for candidate in candidates) == (
        "0200000000",
        "0400000000",
    )
    assert all(candidate.article_id != "0100000000" for candidate in candidates)


def test_content_similarity_source_records_are_ranker_ready() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16", horizon_days=7)
    index = build_content_similarity_index(
        (TransactionEvent(split.cutoff.replace(day=1), CUSTOMER_ID, "0100000000"),),
        split,
        target_customer_ids=(CUSTOMER_ID,),
        embedding_records=_embedding_records(),
        source_name="fashionclip_similarity",
    )

    records = build_content_similarity_candidate_source_records(index, CUSTOMER_ID, k=1)

    assert records[0].customer_id == CUSTOMER_ID
    assert records[0].article_id == "0200000000"
    assert records[0].source == "fashionclip_similarity"
    assert records[0].source_rank == 1
    assert records[0].source_score > 0.99


def test_content_similarity_handles_missing_history_and_embedding_gaps() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16", horizon_days=7)
    index = build_content_similarity_index(
        (
            TransactionEvent(split.cutoff.replace(day=1), CUSTOMER_ID, "0999999999"),
            TransactionEvent(split.cutoff.replace(day=1), OTHER_CUSTOMER_ID, "0100000000"),
        ),
        split,
        target_customer_ids=(CUSTOMER_ID, OTHER_CUSTOMER_ID),
        embedding_records=_embedding_records(),
    )

    assert CUSTOMER_ID not in index.customer_histories
    assert build_content_similarity_candidate_records(index, CUSTOMER_ID, k=2) == ()
    assert content_similarity_article_coverage(index, (CUSTOMER_ID, OTHER_CUSTOMER_ID), k=2) == 2


def test_content_similarity_can_keep_history_when_configured() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16", horizon_days=7)
    index = build_content_similarity_index(
        (TransactionEvent(split.cutoff.replace(day=1), CUSTOMER_ID, "0100000000"),),
        split,
        target_customer_ids=(CUSTOMER_ID,),
        embedding_records=_embedding_records(),
        exclude_history=False,
    )

    candidates = build_content_similarity_candidate_records(index, CUSTOMER_ID, k=1)

    assert candidates[0].article_id == "0100000000"


def test_content_similarity_rejects_invalid_configuration() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16", horizon_days=7)
    with pytest.raises(ValueError, match="source_name"):
        build_content_similarity_index((), split, (), _embedding_records(), source_name="")
    with pytest.raises(ValueError, match="max_history_items"):
        build_content_similarity_index((), split, (), _embedding_records(), max_history_items=0)
    with pytest.raises(ValueError, match="embedding_records"):
        build_content_similarity_index((), split, (), ())
    with pytest.raises(ValueError, match="same dimension"):
        build_content_similarity_index(
            (),
            split,
            (),
            (
                ArticleEmbeddingRecord("0100000000", (1.0, 0.0), "provider"),
                ArticleEmbeddingRecord("0200000000", (1.0,), "provider"),
            ),
        )
    with pytest.raises(ValueError, match="same provider"):
        build_content_similarity_index(
            (),
            split,
            (),
            (
                ArticleEmbeddingRecord("0100000000", (1.0, 0.0), "provider-a"),
                ArticleEmbeddingRecord("0200000000", (0.0, 1.0), "provider-b"),
            ),
        )
    index = build_content_similarity_index(
        (TransactionEvent(split.cutoff.replace(day=1), CUSTOMER_ID, "0100000000"),),
        split,
        target_customer_ids=(CUSTOMER_ID,),
        embedding_records=_embedding_records(),
    )
    with pytest.raises(ValueError, match="k"):
        build_content_similarity_candidate_records(index, CUSTOMER_ID, k=0)
    with pytest.raises(ValueError, match="k"):
        content_similarity_article_coverage(index, (CUSTOMER_ID,), k=0)


def test_default_content_similarity_source_name_is_canonical() -> None:
    split = TemporalSplit.from_isoformat("2020-09-16", horizon_days=7)
    index = build_content_similarity_index(
        (),
        split,
        target_customer_ids=(),
        embedding_records=_embedding_records(),
    )

    assert index.source_name == MULTIMODAL_SIMILARITY_SOURCE


def _embedding_records() -> tuple[ArticleEmbeddingRecord, ...]:
    return (
        ArticleEmbeddingRecord("0100000000", (1.0, 0.0), "fashionclip"),
        ArticleEmbeddingRecord("0200000000", (0.99, 0.01), "fashionclip"),
        ArticleEmbeddingRecord("0300000000", (0.0, 1.0), "fashionclip"),
        ArticleEmbeddingRecord("0400000000", (0.8, 0.2), "fashionclip"),
    )
