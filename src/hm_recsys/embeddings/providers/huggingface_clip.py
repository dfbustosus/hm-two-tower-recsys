"""Optional HuggingFace CLIP-style article embedding provider.

This provider is intended for open-source models with ``get_text_features`` and
``get_image_features`` methods, including FashionCLIP-style checkpoints exposed
through ``transformers``.  Heavy dependencies are imported lazily only when the
provider is instantiated.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from hm_recsys.embeddings.article_content import combine_article_text_fields
from hm_recsys.embeddings.cache_manifest import EmbeddingCacheKind
from hm_recsys.embeddings.contracts import (
    ArticleEmbeddingInput,
    ArticleEmbeddingRecord,
    EmbeddingVector,
)

DEFAULT_FASHIONCLIP_MODEL_ID = "patrickjohncyh/fashion-clip"


class HuggingFaceClipArticleEmbeddingProvider:
    """Article embedding provider backed by a HuggingFace CLIP-style model."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_FASHIONCLIP_MODEL_ID,
        revision: str = "main",
        embedding_kind: EmbeddingCacheKind = "multimodal",
        device: str = "auto",
        normalize: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        """Load an optional HuggingFace image/text encoder.

        Raises:
            ImportError: If ``torch``, ``transformers``, or ``Pillow`` is not
                installed.
            ValueError: If the loaded model does not expose CLIP-style feature
                methods or the configuration is invalid.
        """

        if not model_id:
            raise ValueError("model_id must not be empty")
        if not revision:
            raise ValueError("revision must not be empty")
        if embedding_kind not in {"image", "text", "multimodal"}:
            raise ValueError("embedding_kind must be image, text, or multimodal")
        self._model_id = model_id
        self._revision = revision
        self._embedding_kind = embedding_kind
        self._normalize = normalize
        self._torch = _import_optional_dependency("torch", "pip install torch")
        transformers = _import_optional_dependency("transformers", "pip install transformers")
        self._image_module = _import_optional_dependency("PIL.Image", "pip install pillow")
        self._device = _resolve_device(self._torch, device)
        self._processor = transformers.AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self._model = transformers.AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        self._model.eval()
        self._validate_model_methods()
        self._dimension = _infer_projection_dimension(self._model)
        self._text_max_length = _infer_text_max_length(self._model)

    @property
    def name(self) -> str:
        """Return the stable provider registry name."""

        return "hf_clip"

    @property
    def dimension(self) -> int:
        """Return the emitted embedding dimensionality."""

        return self._dimension

    @property
    def model_id(self) -> str:
        """Return the HuggingFace model identifier."""

        return self._model_id

    @property
    def revision(self) -> str:
        """Return the HuggingFace model revision."""

        return self._revision

    @property
    def embedding_kind(self) -> EmbeddingCacheKind:
        """Return the configured embedding kind."""

        return self._embedding_kind

    def embed_articles(
        self,
        articles: Iterable[ArticleEmbeddingInput],
    ) -> Iterator[ArticleEmbeddingRecord]:
        """Embed article text and/or images.

        Articles missing all required modality inputs are skipped rather than
        failing the full batch; the cache writer records skipped rows as missing
        embeddings in the manifest.
        """

        article_batch = tuple(articles)
        if not article_batch:
            return

        text_vectors: dict[str, EmbeddingVector] = {}
        image_vectors: dict[str, EmbeddingVector] = {}
        if self.embedding_kind in {"text", "multimodal"}:
            text_vectors = self._embed_text_batch(article_batch)
        if self.embedding_kind in {"image", "multimodal"}:
            image_vectors = self._embed_image_batch(article_batch)

        for article in article_batch:
            vector = _combine_modal_vectors(
                text_vectors.get(article.article_id),
                image_vectors.get(article.article_id),
                self.embedding_kind,
            )
            if vector is None:
                continue
            yield (
                ArticleEmbeddingRecord(
                    article_id=article.article_id,
                    vector=vector,
                    provider_name=self.name,
                )
            )

    def _embed_text_batch(
        self,
        articles: tuple[ArticleEmbeddingInput, ...],
    ) -> dict[str, EmbeddingVector]:
        article_ids: list[str] = []
        prompts: list[str] = []
        for article in articles:
            prompt = combine_article_text_fields(article.text_fields)
            if not prompt:
                continue
            article_ids.append(article.article_id)
            prompts.append(prompt)
        if not prompts:
            return {}
        encoded = self._processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._text_max_length,
        )
        encoded = _move_batch_to_device(encoded, self._device)
        with self._torch.no_grad():
            features = self._model.get_text_features(**encoded)
        features = _extract_feature_tensor(features)
        return dict(zip(article_ids, self._tensor_to_vectors(features), strict=True))

    def _embed_image_batch(
        self,
        articles: tuple[ArticleEmbeddingInput, ...],
    ) -> dict[str, EmbeddingVector]:
        article_ids: list[str] = []
        images: list[Any] = []
        for article in articles:
            if article.image_path is None or not article.image_path.exists():
                continue
            article_ids.append(article.article_id)
            images.append(_load_rgb_image(self._image_module, article.image_path))
        if not images:
            return {}
        encoded = self._processor(images=images, return_tensors="pt")
        encoded = _move_batch_to_device(encoded, self._device)
        with self._torch.no_grad():
            features = self._model.get_image_features(**encoded)
        features = _extract_feature_tensor(features)
        return dict(zip(article_ids, self._tensor_to_vectors(features), strict=True))

    def _tensor_to_vectors(self, tensor: Any) -> tuple[EmbeddingVector, ...]:
        if self._normalize:
            tensor = self._torch.nn.functional.normalize(tensor, dim=-1)
        values = tensor.detach().cpu().float().tolist()
        return tuple(tuple(float(value) for value in row) for row in values)

    def _validate_model_methods(self) -> None:
        if self.embedding_kind in {"text", "multimodal"} and not hasattr(
            self._model, "get_text_features"
        ):
            raise ValueError(f"{self.model_id!r} does not expose get_text_features")
        if self.embedding_kind in {"image", "multimodal"} and not hasattr(
            self._model, "get_image_features"
        ):
            raise ValueError(f"{self.model_id!r} does not expose get_image_features")


def _import_optional_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Optional embedding provider dependency {module_name!r} is missing. "
            f"Install it with: {install_hint}"
        ) from exc


def _resolve_device(torch_module: Any, device: str) -> Any:
    if device != "auto":
        return torch_module.device(device)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def _infer_projection_dimension(model: Any) -> int:
    config = getattr(model, "config", None)
    for candidate in (
        getattr(config, "projection_dim", None),
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "projection_dim", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
    ):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    raise ValueError("could not infer embedding dimension from model config")


def _infer_text_max_length(model: Any) -> int | None:
    """Infer max text length for CLIP-style tokenizers when available."""

    config = getattr(model, "config", None)
    for candidate in (
        getattr(getattr(config, "text_config", None), "max_position_embeddings", None),
        getattr(config, "max_position_embeddings", None),
    ):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    return None


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    return {key: value.to(device) for key, value in batch.items()}


def _extract_feature_tensor(features: Any) -> Any:
    """Return a tensor from CLIP/SigLIP feature outputs."""

    if hasattr(features, "detach"):
        return features
    for attribute in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(features, attribute, None)
        if value is not None:
            return value
    if isinstance(features, tuple) and features:
        return features[0]
    raise ValueError("could not extract tensor from provider feature output")


def _load_rgb_image(image_module: Any, image_path: Path) -> Any:
    with image_module.open(image_path) as image:
        return image.convert("RGB").copy()


def _combine_modal_vectors(
    text_vector: EmbeddingVector | None,
    image_vector: EmbeddingVector | None,
    embedding_kind: EmbeddingCacheKind,
) -> EmbeddingVector | None:
    if embedding_kind == "text":
        return text_vector
    if embedding_kind == "image":
        return image_vector
    if text_vector is None:
        return image_vector
    if image_vector is None:
        return text_vector
    return tuple(
        (text_value + image_value) / 2.0
        for text_value, image_value in zip(text_vector, image_vector, strict=True)
    )
