import csv
import json
from datetime import date

import pytest

from scripts.augment_candidates_with_content_retrieval import (
    AugmentConfig,
    _build_article_popularity_weights,
    _score_topk,
    _stream_augment,
    _write_augmented_metadata,
)


def test_popularity_prior_weights_are_log_scaled_and_multiplicative() -> None:
    import numpy as np

    weights = _build_article_popularity_weights(
        article_ids_by_row={"0000000001": 0, "0000000002": 1, "0000000003": 2},
        popularity_counts={"0000000001": 1, "0000000002": 9},
        alpha=0.5,
    )

    assert weights[1] == pytest.approx(1.0)
    assert weights[0] == pytest.approx((np.log1p(1) / np.log1p(9)) ** 0.5)
    assert weights[2] == 0.0

    raw_weights = _build_article_popularity_weights(
        article_ids_by_row={"0000000001": 0, "0000000002": 1},
        popularity_counts={},
        alpha=0.0,
    )
    assert raw_weights.tolist() == [1.0, 1.0]


def test_score_topk_applies_popularity_prior_before_ranking() -> None:
    import numpy as np

    customer_queries = {"customer": np.asarray([1.0, 0.0], dtype=np.float32)}
    article_matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )

    topk = _score_topk(
        customer_queries=customer_queries,
        article_matrix=article_matrix,
        article_ids=["0000000001", "0000000002"],
        customer_history={"customer": set()},
        k=2,
        include_history=True,
        article_popularity_weights=np.asarray([0.1, 1.0], dtype=np.float32),
    )

    assert [article_id for _, article_id, _ in topk["customer"]] == [
        "0000000002",
        "0000000001",
    ]
    assert topk["customer"][0][2] == pytest.approx(0.8)


def test_stream_augment_emits_configured_source_name(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "augmented.csv"
    candidate_csv.write_text(
        "customer_id,article_id,source,source_rank,source_score\n"
        "customer,0000000001,repeat,1,1.0\n",
        encoding="utf-8",
    )

    appended = _stream_augment(
        candidate_csv=candidate_csv,
        output_csv=output_csv,
        customer_topk={"customer": [(1, "0000000002", 0.75)]},
        source_name="item2vec_similarity",
    )

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8", newline="")))
    assert appended == 1
    assert rows[-1]["source"] == "item2vec_similarity"
    assert rows[-1]["article_id"] == "0000000002"
    assert rows[-1]["source_score"] == "0.750000"


def test_write_augmented_metadata_updates_source_counts_and_preserves_cutoff(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "augmented.csv"
    manifest_path = tmp_path / "manifest.json"
    candidate_csv.write_text("", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")
    candidate_csv.with_suffix(".json").write_text(
        json.dumps(
            {
                "cutoff": "2020-09-16",
                "horizon_days": 7,
                "validation_end_exclusive": "2020-09-23",
                "rows_written": 10,
                "source_row_counts": {"repeat": 4},
                "output_path": str(candidate_csv),
            }
        ),
        encoding="utf-8",
    )

    written = _write_augmented_metadata(
        candidate_csv=candidate_csv,
        output_csv=output_csv,
        appended_rows=6,
        embeddings_manifest_path=manifest_path,
        config=AugmentConfig(
            cutoff=date(2020, 9, 16),
            history_lookback_days=90,
            max_history_items=20,
            k=50,
            include_history=False,
            popularity_prior_alpha=0.5,
            popularity_prior_lookback_days=7,
            source_name="text_similarity_popularity_prior",
        ),
    )

    assert written == output_csv.with_suffix(".json")
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["cutoff"] == "2020-09-16"
    assert payload["rows_written"] == 16
    assert payload["source_row_counts"]["repeat"] == 4
    assert payload["source_row_counts"]["text_similarity_popularity_prior"] == 6
    assert payload["augmentation"]["kind"] == "multiplicative_content_popularity_prior"
    assert payload["augmentation"]["popularity_prior_alpha"] == 0.5
