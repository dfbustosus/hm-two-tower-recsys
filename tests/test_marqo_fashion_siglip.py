"""Tests for the Marqo-FashionSigLIP embedding provider scaffolding."""

from __future__ import annotations

import math

import pytest

from hm_recsys.embeddings.marqo_fashion_siglip import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_MARQO_MODEL_ID,
    DEFAULT_PROVIDER_NAME,
    MarqoFashionSigLipConfig,
    compose_article_text,
    mean_of_normalized,
    select_device,
)


def test_config_defaults_match_marqo_model_card() -> None:
    config = MarqoFashionSigLipConfig()

    assert config.model_id == DEFAULT_MARQO_MODEL_ID
    assert config.provider_name == DEFAULT_PROVIDER_NAME
    assert config.embedding_dimension == DEFAULT_EMBEDDING_DIMENSION
    assert config.batch_size > 0


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("model_id", "", "model_id"),
        ("revision", "", "revision"),
        ("provider_name", "", "provider_name"),
    ],
)
def test_config_rejects_empty_strings(field: str, value: str, error_match: str) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=error_match):
        MarqoFashionSigLipConfig(**kwargs)


def test_config_rejects_non_positive_numeric_fields() -> None:
    with pytest.raises(ValueError, match="embedding_dimension"):
        MarqoFashionSigLipConfig(embedding_dimension=0)
    with pytest.raises(ValueError, match="batch_size"):
        MarqoFashionSigLipConfig(batch_size=0)


def test_compose_article_text_skips_blanks_and_orders_fields() -> None:
    pairs = (
        ("prod_name", "Slim Jeans"),
        ("product_type_name", ""),
        ("product_group_name", "Garment Lower body"),
        ("detail_desc", " 5-pocket jeans in stretch denim. "),
    )

    composed = compose_article_text(pairs)

    assert composed == "Slim Jeans | Garment Lower body | 5-pocket jeans in stretch denim."


def test_compose_article_text_empty_when_all_blank() -> None:
    assert compose_article_text((("prod_name", ""),)) == ""


def test_mean_of_normalized_produces_unit_vector_of_correct_dim() -> None:
    text_vector = (1.0, 0.0, 0.0)
    image_vector = (0.0, 1.0, 0.0)

    combined = mean_of_normalized([text_vector, image_vector])

    assert len(combined) == 3
    assert math.isclose(sum(component * component for component in combined), 1.0, abs_tol=1e-9)


def test_mean_of_normalized_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same dimensionality"):
        mean_of_normalized([(1.0, 0.0), (1.0,)])


def test_mean_of_normalized_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean_of_normalized([])


def test_select_device_explicit_passthrough() -> None:
    assert select_device("cpu") == "cpu"
    assert select_device("mps") == "mps"
    assert select_device("cuda") == "cuda"


def test_select_device_rejects_unknown_preference() -> None:
    with pytest.raises(ValueError, match="unsupported device preference"):
        select_device("tpu")


def test_select_device_auto_returns_known_value() -> None:
    """``auto`` must resolve to one of the supported devices in all envs."""

    assert select_device("auto") in {"cpu", "mps", "cuda"}


def test_provider_requires_optional_dependencies() -> None:
    """Constructing the provider without torch/transformers raises ImportError."""

    torch_module = pytest.importorskip("torch", reason="torch not installed", exc_type=ImportError)
    if torch_module is None:  # pragma: no cover - unreachable
        pytest.skip("torch not installed")
    pytest.importorskip("transformers", reason="transformers not installed")

    from hm_recsys.embeddings.marqo_fashion_siglip import MarqoFashionSigLipProvider

    provider = MarqoFashionSigLipProvider(MarqoFashionSigLipConfig(device="cpu", batch_size=1))
    assert provider.name == DEFAULT_PROVIDER_NAME
    assert provider.dimension == DEFAULT_EMBEDDING_DIMENSION
    assert provider.device == "cpu"
