"""Unit tests for ``cli.two_tower._load_content_embedding_matrix``.

The loader is the only piece of glue between
``generate-article-embeddings`` (which writes the manifest + JSONL on
disk) and ``train_two_tower`` (which consumes a numpy matrix). A bug in
this loader is silent: the trainer happily runs with whatever vectors it
gets, including all zeros or a swapped index space, and the only visible
symptom is that the resulting two-tower performs no better than ID-only.
These tests guard the three failure modes that cost the most debugging
time:

* manifest dim disagrees with ``--content-embedding-dim`` (fail loud);
* embeddings JSONL has rows for articles the vocabulary doesn't know
  (silently skipped — must not raise);
* vocabulary contains articles missing from the manifest (zero row, must
  not raise).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hm_recsys.cli.two_tower import _load_content_embedding_matrix
from hm_recsys.models.two_tower_dataset import IdMapper, TwoTowerVocabulary

np = pytest.importorskip("numpy")


def _make_vocabulary(article_ids: list[str]) -> TwoTowerVocabulary:
    article_mapper = IdMapper()
    for article_id in article_ids:
        article_mapper.add_or_lookup(article_id)
    customer_mapper = IdMapper()
    customer_mapper.add_or_lookup("c0")
    return TwoTowerVocabulary(
        customer_mapper=customer_mapper,
        article_mapper=article_mapper,
        article_purchase_counts=dict.fromkeys(article_ids, 1),
    )


def _write_manifest_and_jsonl(
    tmp_path: Path,
    *,
    dimension: int,
    rows: list[tuple[str, list[float]]],
    use_relative_path: bool = False,
) -> Path:
    embeddings_path = tmp_path / "vectors.jsonl"
    with embeddings_path.open("w", encoding="utf-8") as handle:
        for article_id, vector in rows:
            handle.write(json.dumps({"article_id": article_id, "vector": vector}) + "\n")
    manifest = {
        "dimension": dimension,
        "embeddings_path": (
            "vectors.jsonl" if use_relative_path else str(embeddings_path)
        ),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_loader_fills_vectors_at_correct_vocabulary_indices(tmp_path: Path) -> None:
    vocabulary = _make_vocabulary(["a0", "a1", "a2"])
    manifest_path = _write_manifest_and_jsonl(
        tmp_path,
        dimension=3,
        rows=[
            ("a0", [1.0, 0.0, 0.0]),
            ("a1", [0.0, 2.0, 0.0]),
            ("a2", [0.0, 0.0, 3.0]),
        ],
    )

    matrix = _load_content_embedding_matrix(
        manifest_path=manifest_path,
        vocabulary=vocabulary,
        expected_dim=3,
        progress=lambda _msg: None,
    )

    assert matrix.shape == (vocabulary.num_articles, 3)
    np.testing.assert_array_equal(
        matrix[vocabulary.article_mapper.index_for("a0", allow_unknown=False)],
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        matrix[vocabulary.article_mapper.index_for("a1", allow_unknown=False)],
        np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        matrix[vocabulary.article_mapper.index_for("a2", allow_unknown=False)],
        np.asarray([0.0, 0.0, 3.0], dtype=np.float32),
    )
    # Unknown index stays zero.
    np.testing.assert_array_equal(matrix[0], np.zeros(3, dtype=np.float32))


def test_loader_assigns_zero_vector_to_articles_missing_from_manifest(
    tmp_path: Path,
) -> None:
    vocabulary = _make_vocabulary(["a0", "a1", "a_missing"])
    manifest_path = _write_manifest_and_jsonl(
        tmp_path,
        dimension=3,
        rows=[
            ("a0", [1.0, 1.0, 1.0]),
            ("a1", [2.0, 2.0, 2.0]),
        ],
    )

    matrix = _load_content_embedding_matrix(
        manifest_path=manifest_path,
        vocabulary=vocabulary,
        expected_dim=3,
        progress=lambda _msg: None,
    )

    missing_index = vocabulary.article_mapper.index_for(
        "a_missing", allow_unknown=False
    )
    np.testing.assert_array_equal(matrix[missing_index], np.zeros(3, dtype=np.float32))


def test_loader_silently_skips_manifest_rows_outside_vocabulary(tmp_path: Path) -> None:
    vocabulary = _make_vocabulary(["a0"])
    manifest_path = _write_manifest_and_jsonl(
        tmp_path,
        dimension=2,
        rows=[
            ("a0", [1.0, 0.0]),
            ("not_in_vocab", [9.0, 9.0]),
        ],
    )

    matrix = _load_content_embedding_matrix(
        manifest_path=manifest_path,
        vocabulary=vocabulary,
        expected_dim=2,
        progress=lambda _msg: None,
    )

    # Only the vocabulary article gets a non-zero row.
    np.testing.assert_array_equal(
        matrix[vocabulary.article_mapper.index_for("a0", allow_unknown=False)],
        np.asarray([1.0, 0.0], dtype=np.float32),
    )


def test_loader_rejects_dim_mismatch(tmp_path: Path) -> None:
    vocabulary = _make_vocabulary(["a0"])
    manifest_path = _write_manifest_and_jsonl(
        tmp_path,
        dimension=8,
        rows=[("a0", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])],
    )

    with pytest.raises(ValueError, match="content embedding dim mismatch"):
        _load_content_embedding_matrix(
            manifest_path=manifest_path,
            vocabulary=vocabulary,
            expected_dim=16,
            progress=lambda _msg: None,
        )


def test_loader_rejects_per_row_dim_mismatch(tmp_path: Path) -> None:
    vocabulary = _make_vocabulary(["a0"])
    manifest_path = _write_manifest_and_jsonl(
        tmp_path,
        dimension=4,
        rows=[("a0", [0.1, 0.2, 0.3])],
    )

    with pytest.raises(ValueError, match="dim=3"):
        _load_content_embedding_matrix(
            manifest_path=manifest_path,
            vocabulary=vocabulary,
            expected_dim=4,
            progress=lambda _msg: None,
        )


def test_loader_resolves_relative_embeddings_path(tmp_path: Path) -> None:
    vocabulary = _make_vocabulary(["a0"])
    manifest_path = _write_manifest_and_jsonl(
        tmp_path,
        dimension=2,
        rows=[("a0", [0.5, 0.5])],
        use_relative_path=True,
    )

    matrix = _load_content_embedding_matrix(
        manifest_path=manifest_path,
        vocabulary=vocabulary,
        expected_dim=2,
        progress=lambda _msg: None,
    )
    np.testing.assert_array_equal(
        matrix[vocabulary.article_mapper.index_for("a0", allow_unknown=False)],
        np.asarray([0.5, 0.5], dtype=np.float32),
    )
