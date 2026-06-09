"""``Ranker`` protocol shared by every learn-to-rank submission builder.

The protocol decouples *what to rank* (candidate features per customer) from
*how to rank* (linear model, deterministic weights, LightGBM behavioural model,
future CatBoost/DCNv2/two-tower scorers). Concrete adapters live next to their
training/inference modules, while the common submission scaffolding in
:mod:`hm_recsys.ranking.submission` consumes the protocol uniformly.

Design intent (Phase 0 refactor, SDD plan P0.4):
* **SRP / ISP.** Adapters expose exactly the methods the scaffolding needs and
  nothing more, so plugging in a new ranker is a single-class change.
* **No leakage of training-time knobs.** All adapters operate on candidate
  features for a single inference cutoff and never touch transactions.
* **Deterministic ordering.** Implementations must return predictions in
  descending-quality order and must be stable across runs given identical
  inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from hm_recsys.ranking.deterministic import CandidateFeatures

__all__ = (
    "PerCustomerCandidateFeatures",
    "PerCustomerRankedArticles",
    "Ranker",
)

PerCustomerCandidateFeatures = Mapping[str, Mapping[str, CandidateFeatures]]
"""Mapping ``{customer_id: {article_id: CandidateFeatures}}`` ready to rank."""

PerCustomerRankedArticles = Mapping[str, tuple[str, ...]]
"""Mapping ``{customer_id: ranked_article_ids}`` ordered by descending score."""


@runtime_checkable
class Ranker(Protocol):
    """Protocol implemented by every concrete ranker adapter.

    Implementations score candidate features for one or more customers and
    return the top-``k`` article identifiers per customer ordered by descending
    score. Implementations must be deterministic given identical inputs.

    Attributes:
        name: Stable short identifier used in JSON reports, e.g.
            ``"linear"``, ``"deterministic"``, ``"lightgbm"``.

    Example:
        >>> class FirstCandidateRanker:
        ...     name = "first"
        ...     def rank_customer_batch(self, features, *, k):
        ...         return {
        ...             customer_id: tuple(sorted(per_article))[:k]
        ...             for customer_id, per_article in features.items()
        ...         }
    """

    name: str

    def rank_customer_batch(
        self,
        features_by_customer: PerCustomerCandidateFeatures,
        *,
        k: int,
    ) -> PerCustomerRankedArticles:
        """Rank candidate articles for each customer in the batch.

        Args:
            features_by_customer: Aggregated candidate features keyed by
                customer ID, then by article ID.
            k: Recommendation depth. Must be positive.

        Returns:
            Mapping from customer ID to a tuple of at most ``k`` article IDs
            ordered by descending model score.
        """
