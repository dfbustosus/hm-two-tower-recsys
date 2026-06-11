"""Dataset utilities that glue raw H&M transactions to the two-tower model.

The two-tower module operates on integer IDs and dense tensors. The H&M
pipeline operates on string IDs and streaming CSVs. This module bridges
the two without forcing every consumer to take a PyTorch dependency.

Components:

* :class:`IdMapper` — deterministic string->int mapping with reverse
  lookup. Used for both customer and article vocabularies.
* :func:`build_id_mappers_from_transactions` — single-pass scan that
  builds both vocabularies and counts per-article frequency so callers
  can derive sampling-bias log-probabilities for LogQ correction.
* :func:`iter_positive_training_pairs` — yields cutoff-safe positive
  ``(customer_int, article_int)`` pairs from the same transaction stream.
* :func:`PositivePairBatches` — collects positive pairs and yields fixed-
  size batches with **no in-batch duplicate positives** (a hard requirement
  for in-batch sampled softmax — duplicates break the assumption that the
  diagonal is the only positive in the candidate matrix).

All of the above are PyTorch-agnostic; PyTorch only enters the picture in
:func:`collate_pair_batch_as_tensors`, which is opt-in.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol


class _TransactionLike(Protocol):
    @property
    def t_dat(self) -> date: ...

    @property
    def customer_id(self) -> str: ...

    @property
    def article_id(self) -> str: ...


class IdMapper:
    """Bijective string<->int mapping with deterministic insertion order.

    Reserves index ``0`` for an unknown/padding token. Callers must
    register a string with :meth:`add_or_lookup` before requesting its
    index via :meth:`__getitem__`.

    Attributes:
        unknown_index: Index returned for strings not in the vocabulary
            when ``allow_unknown=True``.
    """

    UNKNOWN_TOKEN: str = "<unk>"
    UNKNOWN_INDEX: int = 0

    def __init__(self) -> None:
        self._token_to_index: dict[str, int] = {self.UNKNOWN_TOKEN: self.UNKNOWN_INDEX}
        self._index_to_token: list[str] = [self.UNKNOWN_TOKEN]

    def __len__(self) -> int:
        """Return the vocabulary size, including the unknown token."""

        return len(self._index_to_token)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size (including the unknown token)."""

        return len(self)

    def add_or_lookup(self, token: str) -> int:
        """Insert ``token`` if absent and return its stable integer index.

        Args:
            token: String to register.

        Returns:
            Stable integer index for ``token``.

        Raises:
            ValueError: If ``token`` equals :attr:`UNKNOWN_TOKEN`.
        """

        if token == self.UNKNOWN_TOKEN:
            raise ValueError("UNKNOWN_TOKEN is reserved")
        index = self._token_to_index.get(token)
        if index is None:
            index = len(self._index_to_token)
            self._token_to_index[token] = index
            self._index_to_token.append(token)
        return index

    def index_for(self, token: str, *, allow_unknown: bool = True) -> int:
        """Return the integer for ``token``.

        Args:
            token: String to look up.
            allow_unknown: When ``True`` (default) unknown strings return
                :attr:`UNKNOWN_INDEX`; when ``False`` they raise.

        Returns:
            Integer index assigned to ``token``.

        Raises:
            KeyError: If ``allow_unknown=False`` and ``token`` is missing.
        """

        index = self._token_to_index.get(token)
        if index is None:
            if not allow_unknown:
                raise KeyError(token)
            return self.UNKNOWN_INDEX
        return index

    def token_for(self, index: int) -> str:
        """Return the string registered under ``index``.

        Raises:
            IndexError: If ``index`` is out of range.
        """

        return self._index_to_token[index]

    def items(self) -> Iterable[tuple[str, int]]:
        """Yield registered ``(token, index)`` pairs excluding the unknown."""

        for token, index in self._token_to_index.items():
            if token == self.UNKNOWN_TOKEN:
                continue
            yield token, index


@dataclass(frozen=True)
class TwoTowerVocabulary:
    """Combined customer/article vocabularies plus item-frequency counts."""

    customer_mapper: IdMapper
    article_mapper: IdMapper
    article_purchase_counts: dict[str, int]

    @property
    def num_customers(self) -> int:
        """Return the customer vocabulary size."""

        return self.customer_mapper.vocab_size

    @property
    def num_articles(self) -> int:
        """Return the article vocabulary size."""

        return self.article_mapper.vocab_size

    @property
    def total_positive_interactions(self) -> int:
        """Return total pre-cutoff positive interactions across all articles.

        Equals the number of ``IntegerPositivePair`` events that
        :func:`iter_positive_training_pairs` will emit for the same
        transaction stream. Useful for ETA / progress reporting.
        """

        return sum(self.article_purchase_counts.values())

    def article_sampling_log_probs(self, smoothing: float = 1.0) -> list[float]:
        """Return per-article log-probabilities for LogQ correction.

        The probability assigned to article ``i`` is
        ``(c_i + s) / (sum_j c_j + s * V)``, where ``s`` is the
        smoothing constant. The returned list is indexed by the integer
        vocabulary order, so element ``0`` corresponds to the unknown
        token and gets the same smoothed mass as any other token.

        Args:
            smoothing: Add-``smoothing`` smoothing to avoid ``log(0)``.

        Returns:
            List of length ``num_articles`` of log-probabilities.

        Raises:
            ValueError: If ``smoothing`` is negative.
        """

        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        counts = [
            (
                float(smoothing)
                if index == IdMapper.UNKNOWN_INDEX
                else float(
                    self.article_purchase_counts.get(self.article_mapper.token_for(index), 0)
                )
                + smoothing
            )
            for index in range(self.num_articles)
        ]
        total = sum(counts)
        if total <= 0:
            raise ValueError("article counts must sum to a positive number")
        return [math.log(count / total) for count in counts]


def build_id_mappers_from_transactions(
    transactions: Iterable[_TransactionLike],
    cutoff: date,
) -> TwoTowerVocabulary:
    """Build customer/article vocabularies from pre-cutoff transactions.

    Args:
        transactions: Stream of transaction events. Events with
            ``t_dat >= cutoff`` are ignored, preserving leakage-safety.
        cutoff: Exclusive cutoff date.

    Returns:
        :class:`TwoTowerVocabulary` ready to feed the two-tower model.
    """

    customer_mapper = IdMapper()
    article_mapper = IdMapper()
    counts: Counter[str] = Counter()
    for transaction in transactions:
        if transaction.t_dat >= cutoff:
            continue
        customer_mapper.add_or_lookup(transaction.customer_id)
        article_mapper.add_or_lookup(transaction.article_id)
        counts[transaction.article_id] += 1
    return TwoTowerVocabulary(
        customer_mapper=customer_mapper,
        article_mapper=article_mapper,
        article_purchase_counts=dict(counts),
    )


@dataclass(frozen=True)
class IntegerPositivePair:
    """One positive ``(customer_int, article_int)`` training pair."""

    customer_index: int
    article_index: int


def iter_positive_training_pairs(
    transactions: Iterable[_TransactionLike],
    vocabulary: TwoTowerVocabulary,
    cutoff: date,
) -> Iterator[IntegerPositivePair]:
    """Yield cutoff-safe positive pairs as integer indices.

    Args:
        transactions: Stream of transaction events.
        vocabulary: Vocabulary already constructed via
            :func:`build_id_mappers_from_transactions`.
        cutoff: Exclusive cutoff date.

    Yields:
        :class:`IntegerPositivePair` for every event with
        ``t_dat < cutoff`` whose customer and article are in vocabulary.
    """

    customer_mapper = vocabulary.customer_mapper
    article_mapper = vocabulary.article_mapper
    for transaction in transactions:
        if transaction.t_dat >= cutoff:
            continue
        customer_index = customer_mapper.index_for(transaction.customer_id, allow_unknown=False)
        article_index = article_mapper.index_for(transaction.article_id, allow_unknown=False)
        yield IntegerPositivePair(customer_index=customer_index, article_index=article_index)


def iter_unique_pair_batches(
    pairs: Iterable[IntegerPositivePair],
    *,
    batch_size: int,
    drop_last: bool = False,
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield ``(customer_indices, article_indices)`` batches with unique articles.

    In-batch sampled softmax assumes the diagonal is the only positive in
    each row. Two positives with the same article in one batch would make
    the negatives of one row a positive in another — silently inflating
    the loss. We therefore guarantee the article index appears at most
    once per emitted batch by buffering collisions and emitting them in a
    later batch.

    Args:
        pairs: Stream of positive pairs.
        batch_size: Target batch size. Must be positive.
        drop_last: When ``True``, the trailing partial batch is dropped.

    Yields:
        Tuple of ``(customer_indices, article_indices)`` lists of length
        ``batch_size`` (or less if ``drop_last`` is ``False``).

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    deferred: list[IntegerPositivePair] = []
    batch_customers: list[int] = []
    batch_articles: list[int] = []
    articles_in_batch: set[int] = set()

    def _flush_if_full() -> Iterator[tuple[list[int], list[int]]]:
        if len(batch_customers) >= batch_size:
            yield batch_customers.copy(), batch_articles.copy()
            batch_customers.clear()
            batch_articles.clear()
            articles_in_batch.clear()

    for pair in pairs:
        if pair.article_index in articles_in_batch:
            deferred.append(pair)
            continue
        batch_customers.append(pair.customer_index)
        batch_articles.append(pair.article_index)
        articles_in_batch.add(pair.article_index)
        yield from _flush_if_full()

    for pair in deferred:
        if pair.article_index in articles_in_batch:
            # Defer again — extremely rare but keeps the invariant.
            continue
        batch_customers.append(pair.customer_index)
        batch_articles.append(pair.article_index)
        articles_in_batch.add(pair.article_index)
        yield from _flush_if_full()

    if batch_customers and not drop_last:
        yield batch_customers.copy(), batch_articles.copy()


def collate_pair_batch_as_tensors(
    batch: tuple[list[int], list[int]],
    article_sampling_log_probs: list[float] | None = None,
) -> tuple[Any, Any, Any | None]:
    """Convert a positive-pair batch to PyTorch tensors.

    Args:
        batch: ``(customer_indices, article_indices)`` tuple.
        article_sampling_log_probs: Optional list of length
            ``num_articles`` used to gather per-positive log-probabilities
            for LogQ correction. When ``None``, the log-probs tensor is
            ``None``.

    Returns:
        Tuple ``(customer_tensor, article_tensor, sampling_log_probs_tensor)``.

    Raises:
        ImportError: If PyTorch is not installed.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "collate_pair_batch_as_tensors requires PyTorch; " "install via `pip install torch`."
        ) from exc

    customer_indices, article_indices = batch
    customer_tensor = torch.as_tensor(customer_indices, dtype=torch.long)
    article_tensor = torch.as_tensor(article_indices, dtype=torch.long)
    if article_sampling_log_probs is None:
        return customer_tensor, article_tensor, None
    log_prob_tensor = torch.as_tensor(
        [article_sampling_log_probs[index] for index in article_indices],
        dtype=torch.float32,
    )
    return customer_tensor, article_tensor, log_prob_tensor


__all__ = (
    "IdMapper",
    "IntegerPositivePair",
    "TwoTowerVocabulary",
    "build_id_mappers_from_transactions",
    "collate_pair_batch_as_tensors",
    "iter_positive_training_pairs",
    "iter_unique_pair_batches",
)
