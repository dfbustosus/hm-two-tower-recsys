import csv
import json
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.embeddings.contracts import (
    ArticleEmbeddingInput,
    ArticleEmbeddingProvider,
    ArticleEmbeddingRecord,
)
from hm_recsys.embeddings.generation import (
    ArticleEmbeddingCacheWriteConfig,
    write_article_embedding_cache_from_content_export,
)
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.retrieval.content_similarity_diagnostics import (
    evaluate_cached_content_similarity,
    write_content_similarity_diagnostics_report,
)

CUSTOMER_ID = "a" * 64
OTHER_CUSTOMER_ID = "b" * 64


def test_evaluate_cached_content_similarity_reports_recall_and_coverage(tmp_path: Path) -> None:
    manifest_path = _write_cache(tmp_path)
    split = TemporalSplit.from_isoformat("2020-09-16")
    transactions = (
        TransactionEvent(date(2020, 9, 1), CUSTOMER_ID, "0100000000"),
        TransactionEvent(date(2020, 9, 17), CUSTOMER_ID, "0200000000"),
        TransactionEvent(date(2020, 9, 1), OTHER_CUSTOMER_ID, "0300000000"),
        TransactionEvent(date(2020, 9, 17), OTHER_CUSTOMER_ID, "0400000000"),
    )

    report = evaluate_cached_content_similarity(
        transaction_iter_factory=lambda: iter(transactions),
        split=split,
        submission_customer_ids=(CUSTOMER_ID, OTHER_CUSTOMER_ID),
        manifest_path=manifest_path,
        evaluation_ks=(1, 2),
    )

    assert report.target_customers == 2
    assert report.evaluated_customers == 2
    assert report.embedding_count == 4
    assert report.rows_with_embedding_history == 2
    assert report.rows_with_candidates == 2
    assert report.candidate_coverage == 1.0
    assert report.recall_at_k["1"] == 1.0
    assert report.recall_at_k["2"] == 1.0
    assert report.map_at_12 == 1.0
    assert report.candidate_count_distribution.minimum == 2


def test_content_similarity_diagnostics_supports_target_cap_and_report_write(
    tmp_path: Path,
) -> None:
    manifest_path = _write_cache(tmp_path)
    split = TemporalSplit.from_isoformat("2020-09-16")
    transactions = (
        TransactionEvent(date(2020, 9, 1), CUSTOMER_ID, "0100000000"),
        TransactionEvent(date(2020, 9, 17), CUSTOMER_ID, "0200000000"),
        TransactionEvent(date(2020, 9, 1), OTHER_CUSTOMER_ID, "0300000000"),
        TransactionEvent(date(2020, 9, 17), OTHER_CUSTOMER_ID, "0400000000"),
    )

    report = evaluate_cached_content_similarity(
        transaction_iter_factory=lambda: iter(transactions),
        split=split,
        submission_customer_ids=(CUSTOMER_ID, OTHER_CUSTOMER_ID),
        manifest_path=manifest_path,
        evaluation_ks=(2, 1),
        max_target_customers=1,
    )
    report_path = tmp_path / "content_similarity.json"
    written_path = write_content_similarity_diagnostics_report(report, report_path)

    assert report.target_customers == 1
    assert report.evaluation_ks == (1, 2)
    assert written_path == report_path.resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["target_customers"] == 1
    assert payload["candidate_count_distribution"]["minimum"] == 2


def test_content_similarity_diagnostics_rejects_invalid_config(tmp_path: Path) -> None:
    manifest_path = _write_cache(tmp_path)
    split = TemporalSplit.from_isoformat("2020-09-16")

    with pytest.raises(ValueError, match="evaluation_ks"):
        evaluate_cached_content_similarity(
            lambda: iter(()), split, (), manifest_path, evaluation_ks=()
        )
    with pytest.raises(ValueError, match="evaluation_ks"):
        evaluate_cached_content_similarity(
            lambda: iter(()), split, (), manifest_path, evaluation_ks=(0,)
        )
    with pytest.raises(ValueError, match="max_target_customers"):
        evaluate_cached_content_similarity(
            lambda: iter(()), split, (), manifest_path, max_target_customers=0
        )
    with pytest.raises(ValueError, match="source_name"):
        evaluate_cached_content_similarity(
            lambda: iter(()), split, (), manifest_path, source_name=""
        )


def _write_cache(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content_path = tmp_path / "article_content.csv"
    with content_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("article_id", "combined_text", "image_relative_path", "image_exists", "prod_name")
        )
        writer.writerow(("0100000000", "seed item", "", "false", "seed"))
        writer.writerow(("0200000000", "near item", "", "false", "near"))
        writer.writerow(("0300000000", "other seed", "", "false", "other seed"))
        writer.writerow(("0400000000", "other near", "", "false", "other near"))
    manifest_path = tmp_path / "manifest.json"
    write_article_embedding_cache_from_content_export(
        DiagnosticsProvider(),
        raw_data_dir=raw_dir,
        article_content_path=content_path,
        embeddings_path=tmp_path / "embeddings.jsonl",
        article_mapping_path=tmp_path / "mapping.csv",
        manifest_path=manifest_path,
        config=ArticleEmbeddingCacheWriteConfig(
            provider_model_id="diagnostics/model",
            provider_model_revision="test",
            embedding_kind="multimodal",
            preprocessing="test",
            license="test",
            normalized=True,
            batch_size=2,
        ),
    )
    return manifest_path


class DiagnosticsProvider(ArticleEmbeddingProvider):
    @property
    def name(self) -> str:
        return "diagnostics-provider"

    @property
    def dimension(self) -> int:
        return 2

    def embed_articles(
        self, articles: Iterable[ArticleEmbeddingInput]
    ) -> Iterator[ArticleEmbeddingRecord]:
        vectors = {
            "0100000000": (1.0, 0.0),
            "0200000000": (0.99, 0.01),
            "0300000000": (0.0, 1.0),
            "0400000000": (0.01, 0.99),
        }
        for article in articles:
            yield ArticleEmbeddingRecord(article.article_id, vectors[article.article_id], self.name)
