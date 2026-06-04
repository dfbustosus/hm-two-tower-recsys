import csv
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.embeddings.cache_manifest import ArticleEmbeddingCacheManifest
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.candidate_export import (
    CANDIDATE_EXPORT_HEADER,
    CandidateRecord,
    candidate_record_to_row,
    write_candidate_export_summary,
    write_validation_candidate_export,
)
from hm_recsys.retrieval.source_names import (
    AGE_SEGMENT_POPULARITY_SOURCE,
    ALL_TIME_POPULARITY_SOURCE,
    CO_VISITATION_SOURCE,
    MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE,
    RECENT_POPULARITY_SOURCE,
    REPEAT_SOURCE,
)

CUSTOMER_ID = "a" * 64
SECOND_CUSTOMER_ID = "b" * 64
THIRD_CUSTOMER_ID = "c" * 64
ARTICLE_1 = "0000000001"
ARTICLE_2 = "0000000002"
VALIDATION_ONLY_ARTICLE = "0000000003"
CONTENT_SIMILAR_ARTICLE = "0000000004"


def test_candidate_record_to_row_uses_schema_and_stable_score_format() -> None:
    row = candidate_record_to_row(
        CandidateRecord(
            customer_id=CUSTOMER_ID,
            article_id=ARTICLE_1,
            source=REPEAT_SOURCE,
            source_rank=2,
            source_score=0.5,
        )
    )

    assert row == (CUSTOMER_ID, ARTICLE_1, REPEAT_SOURCE, "2", "0.5")


def test_validation_candidate_export_is_leakage_safe_and_ranker_ready(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 3), SECOND_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, VALIDATION_ONLY_ARTICLE),
    ]
    output_path = tmp_path / "candidates.csv"

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        submission_customer_ids=(CUSTOMER_ID, SECOND_CUSTOMER_ID),
        output_path=output_path,
        k=2,
        popularity_lookback_days=7,
    )

    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))

    assert tuple(rows[0]) == CANDIDATE_EXPORT_HEADER
    assert summary.target_scope == "validation_label_customers"
    assert summary.target_customers == 1
    assert summary.rows_written == 8
    assert summary.source_row_counts == {
        ALL_TIME_POPULARITY_SOURCE: 2,
        CO_VISITATION_SOURCE: 2,
        RECENT_POPULARITY_SOURCE: 2,
        REPEAT_SOURCE: 2,
    }
    assert {row["customer_id"] for row in rows} == {CUSTOMER_ID}
    assert VALIDATION_ONLY_ARTICLE not in {row["article_id"] for row in rows}
    assert rows[0] == {
        "customer_id": CUSTOMER_ID,
        "article_id": ARTICLE_2,
        "source": REPEAT_SOURCE,
        "source_rank": "1",
        "source_score": "1",
    }


def test_validation_candidate_export_can_include_cached_content_similarity(
    tmp_path: Path,
) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, VALIDATION_ONLY_ARTICLE),
    ]
    manifest_path = write_test_embedding_cache(tmp_path)
    output_path = tmp_path / "content_candidates.csv"

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        submission_customer_ids=(CUSTOMER_ID,),
        output_path=output_path,
        k=1,
        include_co_visitation=False,
        content_similarity_manifest_path=manifest_path,
        content_similarity_source_name=MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE,
        content_similarity_popularity_prior_weight=0.3,
        content_similarity_popularity_lookback_days=7,
        content_similarity_candidate_pool_size=2,
    )

    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))
    assert summary.source_row_counts[MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE] == 1
    assert summary.content_similarity_manifest_path == str(manifest_path.resolve())
    assert summary.content_similarity_popularity_prior_weight == 0.3
    assert summary.content_similarity_popularity_lookback_days == 7
    assert summary.content_similarity_candidate_pool_size == 2
    assert {
        row["article_id"]
        for row in rows
        if row["source"] == MULTIMODAL_SIMILARITY_POPULARITY_PRIOR_SOURCE
    } == {CONTENT_SIMILAR_ARTICLE}


def test_validation_candidate_export_can_include_age_segment_popularity(
    tmp_path: Path,
) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    segment_article = "0000000005"
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 2), SECOND_CUSTOMER_ID, segment_article),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, segment_article),
    ]
    output_path = tmp_path / "segment_candidates.csv"

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        submission_customer_ids=(CUSTOMER_ID,),
        output_path=output_path,
        k=2,
        include_co_visitation=False,
        include_age_segment_popularity=True,
        customer_segment_by_id={CUSTOMER_ID: "age_30_39", SECOND_CUSTOMER_ID: "age_30_39"},
        age_segment_bucket_size=10,
        age_segment_popularity_lookback_days=7,
    )

    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))
    segment_rows = [row for row in rows if row["source"] == AGE_SEGMENT_POPULARITY_SOURCE]
    assert summary.include_age_segment_popularity is True
    assert summary.age_segment_bucket_size == 10
    assert summary.age_segment_popularity_lookback_days == 7
    assert summary.source_row_counts[AGE_SEGMENT_POPULARITY_SOURCE] == 2
    assert segment_rows[0]["article_id"] == ARTICLE_1
    assert {row["article_id"] for row in segment_rows} == {ARTICLE_1, segment_article}


def test_candidate_export_supports_deterministic_smoke_cap(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    events = [
        TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 1), THIRD_CUSTOMER_ID, ARTICLE_2),
        TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_1),
        TransactionEvent(date(2020, 1, 8), THIRD_CUSTOMER_ID, ARTICLE_2),
    ]

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter(events),
        split=split,
        submission_customer_ids=(CUSTOMER_ID, THIRD_CUSTOMER_ID),
        output_path=tmp_path / "capped.csv",
        k=1,
        max_target_customers=1,
        include_co_visitation=False,
    )

    assert summary.target_customers == 1
    assert summary.max_target_customers == 1
    assert summary.source_row_counts == {
        ALL_TIME_POPULARITY_SOURCE: 1,
        RECENT_POPULARITY_SOURCE: 1,
        REPEAT_SOURCE: 1,
    }


def test_candidate_export_rejects_invalid_limits(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")

    with pytest.raises(ValueError, match="k must be positive"):
        write_validation_candidate_export(
            transaction_iter_factory=lambda: iter(()),
            split=split,
            submission_customer_ids=(),
            output_path=tmp_path / "invalid.csv",
            k=0,
        )
    with pytest.raises(ValueError, match="customer_segment_by_id"):
        write_validation_candidate_export(
            transaction_iter_factory=lambda: iter(()),
            split=split,
            submission_customer_ids=(),
            output_path=tmp_path / "invalid_segment.csv",
            include_age_segment_popularity=True,
        )


def test_write_candidate_export_summary(tmp_path: Path) -> None:
    split = TemporalSplit.from_isoformat("2020-01-08")
    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter(
            [
                TransactionEvent(date(2020, 1, 1), CUSTOMER_ID, ARTICLE_1),
                TransactionEvent(date(2020, 1, 8), CUSTOMER_ID, ARTICLE_1),
            ]
        ),
        split=split,
        submission_customer_ids=(CUSTOMER_ID,),
        output_path=tmp_path / "candidates.csv",
        k=1,
    )

    report_path = write_candidate_export_summary(summary, tmp_path / "summary.json")

    assert report_path.exists()
    assert '"rows_written"' in report_path.read_text(encoding="utf-8")


def write_test_embedding_cache(tmp_path: Path) -> Path:
    embeddings_path = tmp_path / "embeddings.jsonl"
    mapping_path = tmp_path / "mapping.csv"
    manifest_path = tmp_path / "manifest.json"
    embeddings_path.write_text(
        f'{{"article_id":"{ARTICLE_1}","vector":[1.0,0.0]}}\n'
        f'{{"article_id":"{CONTENT_SIMILAR_ARTICLE}","vector":[0.99,0.01]}}\n',
        encoding="utf-8",
    )
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("embedding_index", "article_id"))
        writer.writerow(("0", ARTICLE_1))
        writer.writerow(("1", CONTENT_SIMILAR_ARTICLE))
    manifest = ArticleEmbeddingCacheManifest(
        generated_at_utc="2026-06-03T00:00:00+00:00",
        provider_name="test-provider",
        provider_model_id="test/model",
        provider_model_revision="test",
        embedding_kind="multimodal",
        dimension=2,
        distance_metric="cosine",
        normalized=True,
        vector_format="jsonl",
        dtype="float32",
        article_count=2,
        embedding_count=2,
        missing_embedding_count=0,
        source_article_content_path=str(tmp_path / "article_content.csv"),
        source_image_inventory_path=None,
        embeddings_path=str(embeddings_path),
        article_mapping_path=str(mapping_path),
        preprocessing="test",
        license="test",
        manifest_path=str(manifest_path),
    )
    manifest_path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    return manifest_path
