"""Tests for the two-tower embedding export and ANN index builders."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.models.two_tower import (
    ArticleTowerConfig,
    CustomerTowerConfig,
    TwoTowerConfig,
    TwoTowerTrainingConfig,
    build_torch_two_tower,
)
from hm_recsys.models.two_tower_dataset import (
    TwoTowerVocabulary,
    build_id_mappers_from_transactions,
)
from hm_recsys.models.two_tower_export import (
    build_exact_article_index,
    export_two_tower_embeddings,
    load_id_mapping,
    load_two_tower_embeddings,
)

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")


def _customer(seed: str) -> str:
    return seed * 64


def _article(seed: int) -> str:
    return f"{seed:010d}"


def _toy_vocab() -> TwoTowerVocabulary:
    cutoff = date(2020, 9, 16)
    events = [
        TransactionEvent(
            date(2020, 9, 1) + timedelta(days=i),
            _customer(letter),
            _article(article_seed),
        )
        for i, (letter, article_seed) in enumerate(
            [("a", 1), ("b", 2), ("c", 3), ("a", 4), ("b", 5)]
        )
    ]
    return build_id_mappers_from_transactions(events, cutoff)


def test_export_two_tower_embeddings_writes_npz_and_tsv(tmp_path: Path) -> None:
    vocab = _toy_vocab()
    model = build_torch_two_tower(
        TwoTowerConfig(
            customer_tower=CustomerTowerConfig(
                num_customers=vocab.num_customers,
                customer_id_embedding_dim=4,
                hidden_dims=(8,),
                output_dim=8,
                dropout=0.0,
            ),
            article_tower=ArticleTowerConfig(
                num_articles=vocab.num_articles,
                article_id_embedding_dim=4,
                content_embedding_dim=4,
                hidden_dims=(8,),
                output_dim=8,
                dropout=0.0,
            ),
            training=TwoTowerTrainingConfig(temperature=0.1),
        )
    )

    export = export_two_tower_embeddings(model=model, vocabulary=vocab, output_dir=tmp_path)

    assert export.customer_embeddings_path.exists()
    assert export.article_embeddings_path.exists()
    assert export.customer_id_mapping_path.exists()
    assert export.article_id_mapping_path.exists()

    customer_vectors = load_two_tower_embeddings(export.customer_embeddings_path)
    article_vectors = load_two_tower_embeddings(export.article_embeddings_path)
    assert len(customer_vectors) == vocab.num_customers
    assert len(article_vectors) == vocab.num_articles
    assert len(customer_vectors[0]) == 8

    reloaded_mapper = load_id_mapping(export.article_id_mapping_path)
    assert reloaded_mapper.vocab_size == vocab.num_articles
    expected_article = vocab.article_mapper.token_for(1)
    assert reloaded_mapper.token_for(1) == expected_article


def test_build_exact_article_index_skips_unknown_token() -> None:
    vocab = _toy_vocab()
    # Build a deterministic, small dimensional embedding matrix for clarity.
    embeddings = [[0.0, 0.0, 0.0]] + [
        [float(index) / 10.0, float(index) / 5.0, float(index) / 2.0]
        for index in range(1, vocab.num_articles)
    ]

    index = build_exact_article_index(
        article_embeddings=embeddings,
        article_id_mapper=vocab.article_mapper,
    )

    assert index.article_count == vocab.num_articles - 1
    top1 = index.query((1.0, 1.0, 1.0), top_k=1)
    assert top1
    assert top1[0].article_id == vocab.article_mapper.token_for(vocab.num_articles - 1)


def test_load_id_mapping_rejects_bad_header(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.tsv"
    bad_path.write_text("not_an_index_header\n1\tfoo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected header"):
        load_id_mapping(bad_path)


def test_load_id_mapping_rejects_non_contiguous(tmp_path: Path) -> None:
    path = tmp_path / "skip.tsv"
    path.write_text("index\ttoken\n1\tfoo\n3\tbar\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-contiguous"):
        load_id_mapping(path)
