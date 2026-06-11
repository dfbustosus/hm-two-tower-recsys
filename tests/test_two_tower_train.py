"""End-to-end smoke test for the two-tower trainer."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.models.two_tower import (
    ArticleTowerConfig,
    CustomerTowerConfig,
    TwoTowerConfig,
    TwoTowerTrainingConfig,
)
from hm_recsys.models.two_tower_dataset import (
    build_id_mappers_from_transactions,
    iter_positive_training_pairs,
    iter_unique_pair_batches,
)
from hm_recsys.models.two_tower_train import TwoTowerTrainerConfig, train_two_tower

torch = pytest.importorskip("torch")


def _customer(seed: str) -> str:
    return seed * 64


def _article(seed: int) -> str:
    return f"{seed:010d}"


def _make_transactions() -> list[TransactionEvent]:
    cutoff = date(2020, 9, 16)
    base = []
    # Two customers, each buys two distinct articles before cutoff.
    for day_offset in range(1, 12):
        base.append(
            TransactionEvent(
                date(2020, 9, 1) + _days(day_offset), _customer("a"), _article(day_offset)
            )
        )
        base.append(
            TransactionEvent(
                date(2020, 9, 1) + _days(day_offset), _customer("b"), _article(day_offset + 100)
            )
        )
    _ = cutoff  # documentation
    return base


def _days(n: int) -> timedelta:
    return timedelta(days=n)


def test_train_two_tower_runs_and_loss_finite() -> None:
    cutoff = date(2020, 9, 16)
    transactions = _make_transactions()
    vocab = build_id_mappers_from_transactions(transactions, cutoff)

    model_config = TwoTowerConfig(
        customer_tower=CustomerTowerConfig(
            num_customers=vocab.num_customers,
            customer_id_embedding_dim=8,
            dense_feature_dim=0,
            hidden_dims=(16,),
            output_dim=16,
            dropout=0.0,
        ),
        article_tower=ArticleTowerConfig(
            num_articles=vocab.num_articles,
            article_id_embedding_dim=8,
            content_embedding_dim=8,
            hidden_dims=(16,),
            output_dim=16,
            dropout=0.0,
        ),
        training=TwoTowerTrainingConfig(
            temperature=0.1,
            mixed_negative_count=4,
            cross_batch_negative_capacity=8,
        ),
    )
    trainer_config = TwoTowerTrainerConfig(
        epochs=2,
        learning_rate=1e-2,
        device="cpu",
        progress_every_steps=1000,
    )

    def batches() -> Iterable[tuple[list[int], list[int]]]:
        pairs = list(iter_positive_training_pairs(transactions, vocab, cutoff))
        yield from iter_unique_pair_batches(pairs, batch_size=4, drop_last=False)

    result = train_two_tower(
        model_config=model_config,
        trainer_config=trainer_config,
        vocabulary=vocab,
        batch_source=batches,
    )

    assert result.trained_successfully, result
    assert result.steps > 0
    assert result.final_loss == result.final_loss  # not NaN
    assert result.mean_loss > 0


def test_train_two_tower_loads_content_embeddings_into_article_tower() -> None:
    """Pre-loaded content vectors must end up in the frozen content slot.

    Regression guard for the
    ``train_two_tower(..., content_embeddings=...)`` path. Without this
    test it is easy to accidentally drop the injection (e.g. by re-
    ordering the build/load/optimizer lines) and silently fall back to
    random frozen content weights, which is what the production
    ``train-two-tower`` CLI used to do for months before the
    ``--content-embeddings-manifest-path`` arg was added.
    """

    cutoff = date(2020, 9, 16)
    transactions = _make_transactions()
    vocab = build_id_mappers_from_transactions(transactions, cutoff)
    content_dim = 8

    model_config = TwoTowerConfig(
        customer_tower=CustomerTowerConfig(
            num_customers=vocab.num_customers,
            customer_id_embedding_dim=8,
            dense_feature_dim=0,
            hidden_dims=(16,),
            output_dim=16,
            dropout=0.0,
        ),
        article_tower=ArticleTowerConfig(
            num_articles=vocab.num_articles,
            article_id_embedding_dim=8,
            content_embedding_dim=content_dim,
            hidden_dims=(16,),
            output_dim=16,
            dropout=0.0,
        ),
        training=TwoTowerTrainingConfig(
            temperature=0.1,
            mixed_negative_count=4,
            cross_batch_negative_capacity=8,
        ),
    )
    trainer_config = TwoTowerTrainerConfig(
        epochs=1,
        learning_rate=1e-2,
        device="cpu",
        progress_every_steps=1000,
    )

    expected = torch.arange(
        vocab.num_articles * content_dim, dtype=torch.float32
    ).reshape(vocab.num_articles, content_dim)

    def batches() -> Iterable[tuple[list[int], list[int]]]:
        pairs = list(iter_positive_training_pairs(transactions, vocab, cutoff))
        yield from iter_unique_pair_batches(pairs, batch_size=4, drop_last=False)

    result = train_two_tower(
        model_config=model_config,
        trainer_config=trainer_config,
        vocabulary=vocab,
        batch_source=batches,
        content_embeddings=expected.numpy(),
    )

    # The article tower's content slot is frozen, so the loaded values
    # must survive training byte-for-byte.
    loaded = result.model.article_tower.content_embedding.weight.detach().cpu()
    assert torch.equal(loaded, expected)
    assert not result.model.article_tower.content_embedding.weight.requires_grad


def test_train_two_tower_rejects_content_embedding_shape_mismatch() -> None:
    cutoff = date(2020, 9, 16)
    transactions = _make_transactions()
    vocab = build_id_mappers_from_transactions(transactions, cutoff)

    model_config = TwoTowerConfig(
        customer_tower=CustomerTowerConfig(
            num_customers=vocab.num_customers,
            customer_id_embedding_dim=8,
            hidden_dims=(16,),
            output_dim=16,
            dropout=0.0,
        ),
        article_tower=ArticleTowerConfig(
            num_articles=vocab.num_articles,
            article_id_embedding_dim=8,
            content_embedding_dim=8,
            hidden_dims=(16,),
            output_dim=16,
            dropout=0.0,
        ),
        training=TwoTowerTrainingConfig(
            temperature=0.1,
            mixed_negative_count=4,
            cross_batch_negative_capacity=8,
        ),
    )
    trainer_config = TwoTowerTrainerConfig(
        epochs=1,
        learning_rate=1e-2,
        device="cpu",
        progress_every_steps=1000,
    )

    def batches() -> Iterable[tuple[list[int], list[int]]]:
        pairs = list(iter_positive_training_pairs(transactions, vocab, cutoff))
        yield from iter_unique_pair_batches(pairs, batch_size=4, drop_last=False)

    bad = torch.zeros((vocab.num_articles, 4), dtype=torch.float32).numpy()
    with pytest.raises(ValueError, match="content embedding shape mismatch"):
        train_two_tower(
            model_config=model_config,
            trainer_config=trainer_config,
            vocabulary=vocab,
            batch_source=batches,
            content_embeddings=bad,
        )


def test_train_two_tower_disables_cbns_when_capacity_zero() -> None:
    cutoff = date(2020, 9, 16)
    transactions = _make_transactions()
    vocab = build_id_mappers_from_transactions(transactions, cutoff)

    model_config = TwoTowerConfig(
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
        training=TwoTowerTrainingConfig(
            temperature=0.1,
            mixed_negative_count=2,
            cross_batch_negative_capacity=0,
        ),
    )
    trainer_config = TwoTowerTrainerConfig(epochs=1, learning_rate=5e-3, device="cpu")

    def batches() -> Iterable[tuple[list[int], list[int]]]:
        pairs = list(iter_positive_training_pairs(transactions, vocab, cutoff))
        yield from iter_unique_pair_batches(pairs, batch_size=4)

    result = train_two_tower(
        model_config=model_config,
        trainer_config=trainer_config,
        vocabulary=vocab,
        batch_source=batches,
    )

    assert result.steps > 0


def test_train_two_tower_rejects_vocab_size_mismatch() -> None:
    cutoff = date(2020, 9, 16)
    transactions = _make_transactions()
    vocab = build_id_mappers_from_transactions(transactions, cutoff)

    model_config = TwoTowerConfig(
        customer_tower=CustomerTowerConfig(
            num_customers=vocab.num_customers + 1,
            output_dim=8,
        ),
        article_tower=ArticleTowerConfig(num_articles=vocab.num_articles, output_dim=8),
    )

    def batches() -> Iterable[tuple[list[int], list[int]]]:
        return iter(())

    with pytest.raises(ValueError, match="num_customers"):
        train_two_tower(
            model_config=model_config,
            trainer_config=TwoTowerTrainerConfig(device="cpu"),
            vocabulary=vocab,
            batch_source=batches,
        )
