"""Tests for the multimodal two-tower module."""

from __future__ import annotations

import pytest

from hm_recsys.models.two_tower import (
    ArticleTowerConfig,
    CustomerTowerConfig,
    TwoTowerConfig,
    TwoTowerTrainingConfig,
)

torch = pytest.importorskip("torch")


def _make_config(num_customers: int = 16, num_articles: int = 32, dim: int = 16) -> TwoTowerConfig:
    return TwoTowerConfig(
        customer_tower=CustomerTowerConfig(
            num_customers=num_customers,
            customer_id_embedding_dim=8,
            dense_feature_dim=4,
            hidden_dims=(16,),
            output_dim=dim,
            dropout=0.0,
        ),
        article_tower=ArticleTowerConfig(
            num_articles=num_articles,
            article_id_embedding_dim=8,
            content_embedding_dim=16,
            hidden_dims=(16,),
            output_dim=dim,
            dropout=0.0,
        ),
        training=TwoTowerTrainingConfig(
            temperature=0.1,
            mixed_negative_count=4,
            cross_batch_negative_capacity=8,
        ),
    )


def test_config_rejects_mismatched_tower_output_dims() -> None:
    with pytest.raises(ValueError, match="output_dim"):
        TwoTowerConfig(
            customer_tower=CustomerTowerConfig(num_customers=4, output_dim=16),
            article_tower=ArticleTowerConfig(num_articles=4, output_dim=32),
        )


@pytest.mark.parametrize(
    ("field", "kwargs", "match"),
    [
        ("num_customers", {"num_customers": 0}, "num_customers"),
        ("output_dim", {"output_dim": 0}, "output_dim"),
        ("dropout", {"dropout": 1.5}, "dropout"),
        ("hidden_dims", {"hidden_dims": (0,)}, "hidden_dims"),
    ],
)
def test_customer_tower_config_rejects_invalid(
    field: str, kwargs: dict[str, object], match: str
) -> None:
    base: dict[str, object] = {"num_customers": 4, "output_dim": 16}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CustomerTowerConfig(**base)


def test_training_config_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError, match="cross_batch_negative_capacity"):
        TwoTowerTrainingConfig(cross_batch_negative_capacity=-1)


def test_build_torch_two_tower_produces_normalized_embeddings() -> None:
    from hm_recsys.models.two_tower import build_torch_two_tower

    config = _make_config()
    model = build_torch_two_tower(config)
    customer_ids = torch.arange(4)
    article_ids = torch.arange(4)
    dense_features = torch.zeros(4, config.customer_tower.dense_feature_dim)

    customer_embeds = model.encode_customer(customer_ids, dense_features)
    article_embeds = model.encode_article(article_ids)

    assert customer_embeds.shape == (4, config.embedding_dim)
    assert article_embeds.shape == (4, config.embedding_dim)
    customer_norms = customer_embeds.norm(dim=-1)
    article_norms = article_embeds.norm(dim=-1)
    assert torch.allclose(customer_norms, torch.ones_like(customer_norms), atol=1e-5)
    assert torch.allclose(article_norms, torch.ones_like(article_norms), atol=1e-5)


def test_sampled_softmax_loss_decreases_on_simple_dataset() -> None:
    from hm_recsys.models.two_tower import (
        build_torch_two_tower,
        sample_mixed_negatives,
        sampled_softmax_loss,
    )

    torch.manual_seed(0)
    config = _make_config(num_customers=8, num_articles=16, dim=8)
    model = build_torch_two_tower(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    customer_ids = torch.arange(8)
    positive_article_ids = torch.arange(8)
    dense_features = torch.zeros(8, config.customer_tower.dense_feature_dim)

    initial_loss = None
    for step in range(20):
        optimizer.zero_grad()
        customer_emb, positive_emb = model(customer_ids, positive_article_ids, dense_features)
        all_article_emb = model.encode_article(torch.arange(config.article_tower.num_articles))
        mixed_neg, _ = sample_mixed_negatives(
            all_article_emb, count=config.training.mixed_negative_count
        )
        loss = sampled_softmax_loss(
            customer_emb,
            positive_emb,
            mixed_neg,
            temperature=config.training.temperature,
        )
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.detach())

    final_loss = float(loss.detach())
    assert initial_loss is not None
    assert final_loss < initial_loss, f"loss did not decrease: {initial_loss} -> {final_loss}"


def test_cross_batch_negative_queue_respects_capacity() -> None:
    from hm_recsys.models.two_tower import CrossBatchNegativeQueue

    queue = CrossBatchNegativeQueue(capacity=4, dim=3)
    queue.enqueue(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    queue.enqueue(torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))

    assert queue.size == 4
    sample = queue.sample()
    assert sample is not None
    assert sample.shape == (4, 3)
    expected_first = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(sample[0], expected_first)


def test_cross_batch_negative_queue_rejects_bad_shape() -> None:
    from hm_recsys.models.two_tower import CrossBatchNegativeQueue

    queue = CrossBatchNegativeQueue(capacity=4, dim=3)
    with pytest.raises(ValueError, match="must have shape"):
        queue.enqueue(torch.zeros((2, 5)))


def test_apply_log_q_correction_increases_low_probability_logits() -> None:
    from hm_recsys.models.two_tower import apply_log_q_correction

    logits = torch.tensor([[1.0, 2.0]])
    sampling_log_probs = torch.tensor([[-1.0, -3.0]])

    corrected = apply_log_q_correction(logits, sampling_log_probs)

    assert torch.allclose(corrected, torch.tensor([[2.0, 5.0]]))


def test_sampled_softmax_loss_applies_log_q_when_enabled() -> None:
    from hm_recsys.models.two_tower import sampled_softmax_loss

    torch.manual_seed(0)
    query = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    positive = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    extra = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    log_probs = torch.tensor([-2.0, -2.0, -1.0, -1.0, -1.0])

    without = sampled_softmax_loss(
        query, positive, extra, sampling_log_probs=log_probs, log_q_correction=False
    )
    with_correction = sampled_softmax_loss(
        query, positive, extra, sampling_log_probs=log_probs, log_q_correction=True
    )

    assert not torch.allclose(without, with_correction)
