"""Tests for the two-tower dataset glue."""

from __future__ import annotations

from datetime import date

import pytest

from hm_recsys.data.io import TransactionEvent
from hm_recsys.models.two_tower_dataset import (
    IdMapper,
    IntegerPositivePair,
    build_id_mappers_from_transactions,
    iter_positive_training_pairs,
    iter_unique_pair_batches,
)


def test_id_mapper_reserves_zero_for_unknown() -> None:
    mapper = IdMapper()

    assert mapper.index_for("missing") == IdMapper.UNKNOWN_INDEX
    assert mapper.token_for(IdMapper.UNKNOWN_INDEX) == IdMapper.UNKNOWN_TOKEN
    assert len(mapper) == 1


def test_id_mapper_assigns_stable_indices() -> None:
    mapper = IdMapper()

    first = mapper.add_or_lookup("a")
    second = mapper.add_or_lookup("b")
    again = mapper.add_or_lookup("a")

    assert first == 1
    assert second == 2
    assert again == first
    assert mapper.index_for("b", allow_unknown=False) == second
    assert tuple(mapper.items()) == (("a", 1), ("b", 2))


def test_id_mapper_strict_lookup_raises_on_missing() -> None:
    mapper = IdMapper()

    with pytest.raises(KeyError):
        mapper.index_for("missing", allow_unknown=False)


def test_id_mapper_rejects_unknown_token_registration() -> None:
    mapper = IdMapper()

    with pytest.raises(ValueError, match="UNKNOWN_TOKEN"):
        mapper.add_or_lookup(IdMapper.UNKNOWN_TOKEN)


def _customer_id(seed: str) -> str:
    return seed * 64


def _article_id(seed: int) -> str:
    return f"{seed:010d}"


def test_build_id_mappers_skips_validation_window() -> None:
    cutoff = date(2020, 9, 16)
    transactions = [
        TransactionEvent(date(2020, 9, 1), _customer_id("a"), _article_id(1)),
        TransactionEvent(date(2020, 9, 14), _customer_id("b"), _article_id(2)),
        TransactionEvent(date(2020, 9, 16), _customer_id("c"), _article_id(3)),
        TransactionEvent(date(2020, 9, 20), _customer_id("d"), _article_id(4)),
    ]

    vocab = build_id_mappers_from_transactions(transactions, cutoff)

    assert vocab.num_customers == 3  # a, b, plus unknown
    assert vocab.num_articles == 3
    assert _customer_id("c") not in dict(vocab.customer_mapper.items())
    assert _article_id(4) not in dict(vocab.article_mapper.items())


def test_article_sampling_log_probs_match_expected_distribution() -> None:
    cutoff = date(2020, 9, 16)
    transactions = [
        TransactionEvent(date(2020, 9, 1), _customer_id("a"), _article_id(1)),
        TransactionEvent(date(2020, 9, 2), _customer_id("a"), _article_id(1)),
        TransactionEvent(date(2020, 9, 3), _customer_id("b"), _article_id(2)),
    ]
    vocab = build_id_mappers_from_transactions(transactions, cutoff)

    log_probs = vocab.article_sampling_log_probs(smoothing=1.0)

    # Counts (with smoothing=1): unk=1, art1=3, art2=2 -> total 6.
    expected_p_art1 = 3 / 6
    expected_p_art2 = 2 / 6
    art1_index = vocab.article_mapper.index_for(_article_id(1), allow_unknown=False)
    art2_index = vocab.article_mapper.index_for(_article_id(2), allow_unknown=False)
    assert log_probs[art1_index] == pytest.approx(_log(expected_p_art1))
    assert log_probs[art2_index] == pytest.approx(_log(expected_p_art2))


def _log(value: float) -> float:
    import math

    return math.log(value)


def test_iter_positive_pairs_returns_integer_indices_only_pre_cutoff() -> None:
    cutoff = date(2020, 9, 16)
    transactions = [
        TransactionEvent(date(2020, 9, 1), _customer_id("a"), _article_id(1)),
        TransactionEvent(date(2020, 9, 14), _customer_id("a"), _article_id(2)),
        TransactionEvent(date(2020, 9, 16), _customer_id("a"), _article_id(3)),
    ]
    vocab = build_id_mappers_from_transactions(transactions, cutoff)

    pairs = list(iter_positive_training_pairs(transactions, vocab, cutoff))

    assert len(pairs) == 2
    assert all(isinstance(pair, IntegerPositivePair) for pair in pairs)
    customer_index = vocab.customer_mapper.index_for(_customer_id("a"), allow_unknown=False)
    assert pairs[0].customer_index == customer_index
    assert pairs[1].customer_index == customer_index


def test_unique_pair_batches_dedupe_articles_within_batch() -> None:
    pairs = [
        IntegerPositivePair(1, 10),
        IntegerPositivePair(2, 10),  # duplicate article — should defer to next batch
        IntegerPositivePair(3, 11),
        IntegerPositivePair(4, 12),
        IntegerPositivePair(5, 13),
    ]

    batches = list(iter_unique_pair_batches(pairs, batch_size=3))

    flattened_articles = [article for _, article_list in batches for article in article_list]
    assert sorted(flattened_articles) == [10, 10, 11, 12, 13]
    for _, article_list in batches:
        assert len(article_list) == len(
            set(article_list)
        ), f"batch contained duplicate articles: {article_list}"


def test_unique_pair_batches_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        list(iter_unique_pair_batches([], batch_size=0))


def test_unique_pair_batches_respects_drop_last() -> None:
    pairs = [IntegerPositivePair(i, i + 10) for i in range(5)]

    full = list(iter_unique_pair_batches(pairs, batch_size=2, drop_last=True))
    with_partial = list(iter_unique_pair_batches(pairs, batch_size=2, drop_last=False))

    assert len(full) == 2
    assert len(with_partial) == 3


def test_collate_pair_batch_as_tensors_returns_log_probs_when_provided() -> None:
    pytest.importorskip("torch")
    from hm_recsys.models.two_tower_dataset import collate_pair_batch_as_tensors

    log_probs = [-1.0, -2.0, -3.0]
    customers, articles, sampling = collate_pair_batch_as_tensors(
        ([0, 0], [1, 2]), article_sampling_log_probs=log_probs
    )

    assert customers.tolist() == [0, 0]
    assert articles.tolist() == [1, 2]
    assert sampling is not None
    assert sampling.tolist() == [-2.0, -3.0]
