"""Minimal, testable trainer for the multimodal two-tower model.

The trainer is deliberately compact: one process, one optimizer step per
batch, optional CBNS/MNS/LogQ as specified in :class:`TwoTowerTrainingConfig`.
Heavy concerns (distributed training, mixed precision, gradient
accumulation) are out of scope here — get a working signal first, then
scale.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from hm_recsys.models.two_tower import (
    CrossBatchNegativeQueue,
    TwoTowerConfig,
    build_torch_two_tower,
    sampled_softmax_loss,
)
from hm_recsys.models.two_tower_dataset import TwoTowerVocabulary, collate_pair_batch_as_tensors

BatchSource = Callable[[], Iterable[tuple[list[int], list[int]]]]


@dataclass(frozen=True)
class TwoTowerTrainerConfig:
    """Configuration for :func:`train_two_tower`.

    Hyperparameters relevant to stability:

    * ``warmup_steps`` — linear LR ramp from 0 to ``learning_rate``. Strongly
      recommended for sampled-softmax training with large in-batch +
      CBNS negative pools; without it, the first few hundred steps can
      push the towers into a degenerate fixed point where every
      embedding collapses to the same point (saturated dot products).
    * ``grad_clip_norm`` — global gradient L2 norm cap. Set to a positive
      number (e.g. ``1.0``) to enable; ``None`` disables clipping.
    """

    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    log_q_correction: bool = True
    use_cross_batch_negatives: bool = True
    use_mixed_negatives: bool = True
    progress_every_steps: int = 50
    expected_total_steps: int | None = None
    warmup_steps: int = 0
    grad_clip_norm: float | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.progress_every_steps <= 0:
            raise ValueError("progress_every_steps must be positive")
        if self.expected_total_steps is not None and self.expected_total_steps <= 0:
            raise ValueError("expected_total_steps must be positive when provided")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when provided")


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short human string."""

    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{int(remainder):02d}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{int(minutes):02d}m"


@dataclass(frozen=True)
class TwoTowerTrainingResult:
    """Result of one :func:`train_two_tower` invocation."""

    model: Any
    steps: int
    final_loss: float
    mean_loss: float

    @property
    def trained_successfully(self) -> bool:
        """``True`` when at least one optimizer step was taken."""

        return self.steps > 0


def _resolve_device(preference: str) -> str:
    if preference in {"cpu", "mps", "cuda"}:
        return preference
    if preference != "auto":
        raise ValueError(f"unsupported device: {preference!r}")
    import torch

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_two_tower(
    *,
    model_config: TwoTowerConfig,
    trainer_config: TwoTowerTrainerConfig,
    vocabulary: TwoTowerVocabulary,
    batch_source: BatchSource,
    progress: Callable[[str], None] | None = None,
    content_embeddings: Any | None = None,
) -> TwoTowerTrainingResult:
    """Train the two-tower model on a stream of positive pair batches.

    Each call to ``batch_source()`` must yield positive pair batches for
    one epoch; the trainer calls it once per epoch so callers can re-seed
    or reshuffle between epochs.

    Args:
        model_config: Architecture configuration. ``vocabulary.num_customers``
            and ``vocabulary.num_articles`` must match the tower configs.
        trainer_config: Training hyperparameters.
        vocabulary: ID mappers and article frequencies. The article
            frequencies feed the LogQ correction.
        batch_source: Callable that returns an iterable of
            ``(customer_indices, article_indices)`` batches.
        progress: Optional callback for progress strings. Useful when
            running from CLI or notebooks.
        content_embeddings: Optional precomputed content embedding matrix
            of shape ``(num_articles, content_embedding_dim)`` aligned to
            the vocabulary's article index space. When provided, it is
            loaded into the article tower's frozen ``content_embedding``
            slot via :meth:`ArticleTower.load_content_embeddings` before
            the first optimizer step. Without this, the slot trains from
            random weights (frozen by default), so the content dimensions
            contribute only noise to the article representation.

    Returns:
        :class:`TwoTowerTrainingResult` with the trained model and loss
        statistics.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If the vocabulary disagrees with the model config or
            if ``content_embeddings`` has an incompatible shape.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "train_two_tower requires PyTorch; install via `pip install torch`."
        ) from exc

    if vocabulary.num_customers != model_config.customer_tower.num_customers:
        raise ValueError(
            "vocabulary.num_customers does not match model_config.customer_tower.num_customers"
        )
    if vocabulary.num_articles != model_config.article_tower.num_articles:
        raise ValueError(
            "vocabulary.num_articles does not match model_config.article_tower.num_articles"
        )

    device = _resolve_device(trainer_config.device)
    model = build_torch_two_tower(model_config).to(device)
    if content_embeddings is not None:
        # Inject the precomputed (frozen) content vectors into the article
        # tower BEFORE optimizer construction so any future Adam state for
        # those parameters reflects the loaded weights, not the random
        # initialization. ``load_content_embeddings`` validates the shape.
        content_tensor = torch.as_tensor(content_embeddings, dtype=torch.float32, device=device)
        model.article_tower.load_content_embeddings(content_tensor)
        if progress is not None:
            progress(
                f"loaded content embeddings: shape={tuple(content_tensor.shape)} device={device}"
            )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
    )
    article_log_probs = (
        torch.as_tensor(vocabulary.article_sampling_log_probs(), dtype=torch.float32, device=device)
        if trainer_config.log_q_correction
        else None
    )
    cbns_queue: CrossBatchNegativeQueue | None = None
    if (
        trainer_config.use_cross_batch_negatives
        and model_config.training.cross_batch_negative_capacity > 0
    ):
        cbns_queue = CrossBatchNegativeQueue(
            capacity=model_config.training.cross_batch_negative_capacity,
            dim=model_config.embedding_dim,
        )

    log_probs_list = (
        vocabulary.article_sampling_log_probs() if trainer_config.log_q_correction else None
    )
    loss_history: list[float] = []
    step_count = 0
    num_articles = vocabulary.num_articles
    expected_total_steps = trainer_config.expected_total_steps
    base_lr = trainer_config.learning_rate
    warmup_steps = trainer_config.warmup_steps
    grad_clip_norm = trainer_config.grad_clip_norm
    start_time = time.perf_counter()
    for epoch_index in range(trainer_config.epochs):
        for batch in batch_source():
            customer_tensor, article_tensor, log_prob_tensor = collate_pair_batch_as_tensors(
                batch,
                article_sampling_log_probs=log_probs_list,
            )
            customer_tensor = customer_tensor.to(device)
            article_tensor = article_tensor.to(device)
            if log_prob_tensor is not None:
                log_prob_tensor = log_prob_tensor.to(device)
            optimizer.zero_grad()
            query_emb, positive_emb = model(customer_tensor, article_tensor)

            extra_negatives_parts: list[Any] = []
            extra_log_probs_parts: list[Any] = []
            if (
                trainer_config.use_mixed_negatives
                and model_config.training.mixed_negative_count > 0
            ):
                # Sample indices first, then encode only the chosen IDs. This
                # is O(mixed_negative_count) per step instead of O(num_articles)
                # and matches the published MNS recipe.
                mns_indices = torch.randint(
                    low=0,
                    high=num_articles,
                    size=(model_config.training.mixed_negative_count,),
                    device=device,
                )
                mns_negatives = model.encode_article(mns_indices)
                extra_negatives_parts.append(mns_negatives)
                if article_log_probs is not None:
                    extra_log_probs_parts.append(article_log_probs.index_select(0, mns_indices))
            if cbns_queue is not None:
                cbns_sample = cbns_queue.sample()
                if cbns_sample is not None:
                    extra_negatives_parts.append(cbns_sample)
                    if article_log_probs is not None:
                        # CBNS entries are detached previous-batch positives. We
                        # do not know their original sampling probability — use
                        # the dataset uniform as a robust prior to avoid NaNs.
                        uniform = torch.full(
                            (cbns_sample.shape[0],),
                            -float(torch.log(torch.tensor(vocabulary.num_articles))),
                            device=device,
                        )
                        extra_log_probs_parts.append(uniform)

            extra_negatives = (
                torch.cat(extra_negatives_parts, dim=0) if extra_negatives_parts else None
            )
            if log_prob_tensor is not None and extra_log_probs_parts:
                sampling_log_probs = torch.cat([log_prob_tensor, *extra_log_probs_parts], dim=0)
            elif log_prob_tensor is not None and extra_negatives is None:
                sampling_log_probs = log_prob_tensor
            else:
                sampling_log_probs = None

            loss = sampled_softmax_loss(
                query_emb,
                positive_emb,
                extra_negatives,
                sampling_log_probs=sampling_log_probs,
                temperature=model_config.training.temperature,
                label_smoothing=model_config.training.label_smoothing,
                log_q_correction=trainer_config.log_q_correction and sampling_log_probs is not None,
            )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            # Linear LR warmup. After warmup_steps the LR stays at base_lr
            # (no decay schedule is wired here on purpose — keep the trainer
            # minimal, layer cosine decay in a follow-up if needed).
            if warmup_steps > 0 and step_count < warmup_steps:
                lr_scale = (step_count + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr * lr_scale
            elif warmup_steps > 0 and step_count == warmup_steps:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr
            optimizer.step()
            if cbns_queue is not None:
                cbns_queue.enqueue(positive_emb.detach())

            loss_value = float(loss.detach())
            loss_history.append(loss_value)
            step_count += 1
            if progress is not None and step_count % trainer_config.progress_every_steps == 0:
                elapsed = time.perf_counter() - start_time
                steps_per_sec = step_count / elapsed if elapsed > 0 else 0.0
                # Smoothed loss across the last reporting window dampens
                # batch-to-batch noise without hiding the trend.
                window_start = max(0, step_count - trainer_config.progress_every_steps)
                window_loss = sum(loss_history[window_start:]) / (step_count - window_start)
                parts = [
                    f"epoch {epoch_index + 1}/{trainer_config.epochs}",
                    f"step {step_count}",
                ]
                if expected_total_steps is not None:
                    pct = 100.0 * step_count / expected_total_steps
                    remaining_steps = max(0, expected_total_steps - step_count)
                    eta_seconds = (
                        remaining_steps / steps_per_sec if steps_per_sec > 0 else float("inf")
                    )
                    parts.append(f"of {expected_total_steps} ({pct:5.1f}%)")
                    parts.append(f"eta {_format_duration(eta_seconds)}")
                parts.append(f"elapsed {_format_duration(elapsed)}")
                parts.append(f"rate {steps_per_sec:5.2f} steps/s")
                parts.append(f"loss {loss_value:.4f} (avg50 {window_loss:.4f})")
                progress(" ".join(parts))

    if not loss_history:
        return TwoTowerTrainingResult(model=model, steps=0, final_loss=0.0, mean_loss=0.0)
    return TwoTowerTrainingResult(
        model=model,
        steps=step_count,
        final_loss=loss_history[-1],
        mean_loss=sum(loss_history) / len(loss_history),
    )


__all__ = (
    "BatchSource",
    "TwoTowerTrainerConfig",
    "TwoTowerTrainingResult",
    "train_two_tower",
)
