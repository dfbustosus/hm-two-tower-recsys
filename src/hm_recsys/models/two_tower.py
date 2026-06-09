"""Multimodal two-tower model with in-batch softmax, LogQ, MNS and CBNS.

The implementation here is deliberately compact and tested on CPU so the
suite runs quickly. Architecture-wise it follows the recipes that Google
and YouTube publish for very large-scale retrieval:

* customer tower — categorical embeddings + dense behavioral features
  projected through an MLP to a shared embedding dimension;
* article tower — ID embedding combined with a frozen content embedding
  (e.g. Marqo-FashionSigLIP) projected through an MLP to the same shared
  embedding dimension;
* sampled-softmax loss with LogQ correction over the in-batch positives
  combined with two additional negative pools — uniformly sampled
  "mixed" negatives (MNS) and a rolling cross-batch queue (CBNS).

The PyTorch dependency is imported lazily so consumers that only need the
pure-Python configuration object (e.g. test discovery, CLI argument
validation, config-hash artifacts) do not pay the cost of importing
``torch``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

EmbeddingDim = int


@dataclass(frozen=True)
class CustomerTowerConfig:
    """Configuration for the customer (query) tower."""

    num_customers: int
    customer_id_embedding_dim: EmbeddingDim = 64
    dense_feature_dim: int = 0
    hidden_dims: tuple[int, ...] = (256, 128)
    output_dim: EmbeddingDim = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_customers <= 0:
            raise ValueError("num_customers must be positive")
        if self.customer_id_embedding_dim <= 0:
            raise ValueError("customer_id_embedding_dim must be positive")
        if self.dense_feature_dim < 0:
            raise ValueError("dense_feature_dim must be non-negative")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")


@dataclass(frozen=True)
class ArticleTowerConfig:
    """Configuration for the article (item) tower."""

    num_articles: int
    article_id_embedding_dim: EmbeddingDim = 64
    content_embedding_dim: EmbeddingDim = 768
    hidden_dims: tuple[int, ...] = (256, 128)
    output_dim: EmbeddingDim = 128
    dropout: float = 0.1
    freeze_content_embedding: bool = True

    def __post_init__(self) -> None:
        if self.num_articles <= 0:
            raise ValueError("num_articles must be positive")
        if self.article_id_embedding_dim <= 0:
            raise ValueError("article_id_embedding_dim must be positive")
        if self.content_embedding_dim <= 0:
            raise ValueError("content_embedding_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")


@dataclass(frozen=True)
class TwoTowerTrainingConfig:
    """Configuration for the sampled-softmax loss and negative pools."""

    temperature: float = 0.05
    mixed_negative_count: int = 256
    cross_batch_negative_capacity: int = 8192
    log_q_correction: bool = True
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.mixed_negative_count < 0:
            raise ValueError("mixed_negative_count must be non-negative")
        if self.cross_batch_negative_capacity < 0:
            raise ValueError("cross_batch_negative_capacity must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")


@dataclass(frozen=True)
class TwoTowerConfig:
    """Aggregate configuration for the two-tower retrieval model."""

    customer_tower: CustomerTowerConfig
    article_tower: ArticleTowerConfig
    training: TwoTowerTrainingConfig = field(default_factory=TwoTowerTrainingConfig)

    def __post_init__(self) -> None:
        if self.customer_tower.output_dim != self.article_tower.output_dim:
            raise ValueError(
                "customer_tower.output_dim must equal article_tower.output_dim "
                "for dot-product retrieval"
            )

    @property
    def embedding_dim(self) -> EmbeddingDim:
        """Return the shared retrieval embedding dimensionality."""

        return self.customer_tower.output_dim


def apply_log_q_correction(logits: Any, sampling_log_probs: Any, eps: float = 1e-12) -> Any:
    """Subtract the log sampling probability from logits.

    See "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
    Recommendations" (Yi et al., 2019). The corrected logit is the
    score the model would assign if items were sampled uniformly.

    Args:
        logits: Raw similarity tensor, shape ``(batch, candidates)``.
        sampling_log_probs: Log-probability tensor, same shape as logits.
        eps: Numerical floor to avoid ``log(0)`` blowups upstream.

    Returns:
        Logits with LogQ correction applied (same shape as input).
    """

    import torch

    safe = torch.clamp(sampling_log_probs, min=math.log(eps))
    return logits - safe


def compute_in_batch_logits(query: Any, candidate: Any, temperature: float) -> Any:
    """Dense in-batch dot-product logits scaled by an inverse temperature."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return query @ candidate.transpose(0, 1) / temperature


def sampled_softmax_loss(
    query: Any,
    positive: Any,
    extra_negatives: Any | None = None,
    sampling_log_probs: Any | None = None,
    *,
    temperature: float = 0.05,
    label_smoothing: float = 0.0,
    log_q_correction: bool = True,
) -> Any:
    """Cross-entropy sampled-softmax loss with LogQ correction support.

    Negatives are drawn from the union of (a) the other positives in the
    same batch and (b) ``extra_negatives`` if provided (for MNS/CBNS).

    Args:
        query: Customer embeddings of shape ``(batch, dim)``.
        positive: Per-row positive article embeddings of shape ``(batch, dim)``.
        extra_negatives: Optional ``(num_extra, dim)`` tensor of additional
            negative item embeddings.
        sampling_log_probs: Optional concatenated log-probability tensor of
            shape ``(num_candidates,)`` covering positives then extras.
        temperature: Softmax temperature applied to logits.
        label_smoothing: Optional label smoothing in ``[0, 1)``.
        log_q_correction: Whether to apply the LogQ correction when
            ``sampling_log_probs`` is provided.

    Returns:
        Scalar cross-entropy loss tensor.
    """

    import torch

    if extra_negatives is None:
        candidates = positive
    else:
        candidates = torch.cat([positive, extra_negatives], dim=0)
    logits = compute_in_batch_logits(query, candidates, temperature)
    if log_q_correction and sampling_log_probs is not None:
        broadcast = sampling_log_probs.to(logits.dtype).unsqueeze(0).expand_as(logits)
        logits = apply_log_q_correction(logits, broadcast)

    batch_size = query.shape[0]
    targets = torch.arange(batch_size, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, targets, label_smoothing=label_smoothing)


class CrossBatchNegativeQueue:
    """Rolling FIFO queue of detached item embeddings for CBNS.

    The queue stores L2-normalized item embeddings produced by previous
    training batches. Sampling returns the entire queue (cheap matmul on
    modern GPUs/MPS) rather than a random subset to maximise gradient
    signal density.
    """

    def __init__(self, capacity: int, dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._capacity = capacity
        self._dim = dim
        self._buffer: Any | None = None
        self._size = 0

    @property
    def capacity(self) -> int:
        """Return the maximum number of items stored."""

        return self._capacity

    @property
    def size(self) -> int:
        """Return the current number of items in the queue."""

        return self._size

    def enqueue(self, item_embeddings: Any) -> None:
        """Add a batch of detached item embeddings to the queue.

        Args:
            item_embeddings: Tensor of shape ``(batch, dim)``. The tensor is
                detached before storage.

        Raises:
            ValueError: If ``item_embeddings`` has a mismatched feature dim.
        """

        import torch

        if item_embeddings.dim() != 2 or item_embeddings.shape[1] != self._dim:
            raise ValueError(
                f"item_embeddings must have shape (*, {self._dim}); "
                f"got {tuple(item_embeddings.shape)}"
            )
        detached = item_embeddings.detach()
        if self._buffer is None:
            self._buffer = detached.clone()
        else:
            self._buffer = torch.cat([self._buffer, detached], dim=0)
        if self._buffer.shape[0] > self._capacity:
            self._buffer = self._buffer[-self._capacity :].contiguous()
        self._size = int(self._buffer.shape[0])

    def sample(self) -> Any | None:
        """Return the current queue contents or ``None`` if empty."""

        return self._buffer


def sample_mixed_negatives(
    article_embedding_table: Any,
    *,
    count: int,
    item_sampling_log_probs: Any | None = None,
) -> tuple[Any, Any | None]:
    """Sample uniform mixed negatives from the full article embedding table.

    Args:
        article_embedding_table: Tensor of shape ``(num_articles, dim)`` —
            the article tower's output for every catalog item.
        count: Number of negatives to draw.
        item_sampling_log_probs: Optional log-probability of each item
            being drawn (used for LogQ correction). For uniform MNS this
            is ``-log(num_articles)``.

    Returns:
        Tuple ``(negatives, log_probs)`` where ``negatives`` has shape
        ``(count, dim)`` and ``log_probs`` has shape ``(count,)`` (or
        ``None`` if ``item_sampling_log_probs`` was not provided).
    """

    import torch

    if count <= 0:
        raise ValueError("count must be positive")
    num_articles = int(article_embedding_table.shape[0])
    indices = torch.randint(
        low=0,
        high=num_articles,
        size=(count,),
        device=article_embedding_table.device,
    )
    negatives = article_embedding_table.index_select(0, indices)
    if item_sampling_log_probs is None:
        return negatives, None
    log_probs = item_sampling_log_probs.index_select(0, indices)
    return negatives, log_probs


def build_torch_two_tower(config: TwoTowerConfig) -> Any:
    """Build a PyTorch :class:`torch.nn.Module` implementing the two towers.

    The returned module exposes ``encode_customer`` and ``encode_article``
    methods that return L2-normalized embeddings.

    Args:
        config: Two-tower configuration.

    Returns:
        A PyTorch module ready for training on CPU/MPS/CUDA.

    Raises:
        ImportError: If PyTorch is not installed.
    """

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "build_torch_two_tower requires PyTorch; install via `pip install torch`."
        ) from exc

    class _MLP(nn.Module):
        def __init__(
            self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float
        ) -> None:
            super().__init__()
            layers: list[Any] = []
            prev = in_dim
            for width in hidden_dims:
                layers.append(nn.Linear(prev, width))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = width
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class _CustomerTower(nn.Module):
        def __init__(self, tower_config: CustomerTowerConfig) -> None:
            super().__init__()
            self.id_embedding = nn.Embedding(
                tower_config.num_customers, tower_config.customer_id_embedding_dim
            )
            input_dim = tower_config.customer_id_embedding_dim + tower_config.dense_feature_dim
            self.mlp = _MLP(
                input_dim, tower_config.hidden_dims, tower_config.output_dim, tower_config.dropout
            )

        def forward(self, customer_ids: Any, dense_features: Any | None = None) -> Any:
            embedded = self.id_embedding(customer_ids)
            if dense_features is not None:
                embedded = torch.cat([embedded, dense_features], dim=-1)
            projected = self.mlp(embedded)
            return torch.nn.functional.normalize(projected, dim=-1)

    class _ArticleTower(nn.Module):
        def __init__(self, tower_config: ArticleTowerConfig) -> None:
            super().__init__()
            self.id_embedding = nn.Embedding(
                tower_config.num_articles, tower_config.article_id_embedding_dim
            )
            self.content_embedding = nn.Embedding(
                tower_config.num_articles, tower_config.content_embedding_dim
            )
            if tower_config.freeze_content_embedding:
                self.content_embedding.weight.requires_grad_(False)
            input_dim = tower_config.article_id_embedding_dim + tower_config.content_embedding_dim
            self.mlp = _MLP(
                input_dim,
                tower_config.hidden_dims,
                tower_config.output_dim,
                tower_config.dropout,
            )

        def forward(self, article_ids: Any) -> Any:
            id_part = self.id_embedding(article_ids)
            content_part = self.content_embedding(article_ids)
            combined = torch.cat([id_part, content_part], dim=-1)
            projected = self.mlp(combined)
            return torch.nn.functional.normalize(projected, dim=-1)

        def load_content_embeddings(self, weights: Any) -> None:
            """Replace the content embedding weights with a precomputed cache."""

            if weights.shape != self.content_embedding.weight.shape:
                raise ValueError(
                    f"content embedding shape mismatch: expected "
                    f"{tuple(self.content_embedding.weight.shape)}, "
                    f"got {tuple(weights.shape)}"
                )
            with torch.no_grad():
                self.content_embedding.weight.copy_(weights)

    class _TwoTower(nn.Module):
        def __init__(self, full_config: TwoTowerConfig) -> None:
            super().__init__()
            self.customer_tower = _CustomerTower(full_config.customer_tower)
            self.article_tower = _ArticleTower(full_config.article_tower)

        def encode_customer(self, customer_ids: Any, dense_features: Any | None = None) -> Any:
            return self.customer_tower(customer_ids, dense_features)

        def encode_article(self, article_ids: Any) -> Any:
            return self.article_tower(article_ids)

        def forward(
            self,
            customer_ids: Any,
            positive_article_ids: Any,
            dense_features: Any | None = None,
        ) -> tuple[Any, Any]:
            return (
                self.encode_customer(customer_ids, dense_features),
                self.encode_article(positive_article_ids),
            )

    return _TwoTower(config)


__all__ = (
    "ArticleTowerConfig",
    "CrossBatchNegativeQueue",
    "CustomerTowerConfig",
    "TwoTowerConfig",
    "TwoTowerTrainingConfig",
    "apply_log_q_correction",
    "build_torch_two_tower",
    "compute_in_batch_logits",
    "sample_mixed_negatives",
    "sampled_softmax_loss",
)
