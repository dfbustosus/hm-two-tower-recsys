"""Marqo-FashionSigLIP embedding provider for H&M article content.

This module wires the open-vocabulary `Marqo/marqo-fashionSigLIP` model
into the existing :class:`ArticleEmbeddingProvider` protocol so that the
catalog can be embedded with a single, modality-rich vector per article.

Design notes:

* The provider lazily imports ``torch`` / ``transformers`` so the rest of
  the package keeps a small dependency surface. When these are missing,
  :class:`MarqoFashionSigLipProvider` raises a clear ``ImportError`` on
  construction rather than at first batch.
* Embeddings are L2-normalized on emission so downstream consumers can use
  cosine similarity as a dot-product.
* Text-only and multimodal inputs are both supported: when ``image_path``
  is provided and the file exists, the image branch is used and combined
  with the text branch via element-wise mean of L2-normalized features.
  This matches the FashionSigLIP recipe for content retrieval and keeps
  the implementation deterministic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from hm_recsys.embeddings.contracts import (
    ArticleEmbeddingInput,
    ArticleEmbeddingProvider,
    ArticleEmbeddingRecord,
    EmbeddingVector,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_MARQO_MODEL_ID: str = "Marqo/marqo-fashionSigLIP"
DEFAULT_MARQO_REVISION: str = "main"
DEFAULT_PROVIDER_NAME: str = "marqo_fashion_siglip"
DEFAULT_EMBEDDING_DIMENSION: int = 768
DEFAULT_BATCH_SIZE: int = 32
TEXT_FIELD_ORDER: tuple[str, ...] = (
    "prod_name",
    "product_type_name",
    "product_group_name",
    "detail_desc",
)


@dataclass(frozen=True)
class MarqoFashionSigLipConfig:
    """Configuration for :class:`MarqoFashionSigLipProvider`."""

    model_id: str = DEFAULT_MARQO_MODEL_ID
    revision: str = DEFAULT_MARQO_REVISION
    provider_name: str = DEFAULT_PROVIDER_NAME
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"
    use_image_when_available: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any numeric setting is out of range or a string
                identifier is empty.
        """

        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.revision:
            raise ValueError("revision must not be empty")
        if not self.provider_name:
            raise ValueError("provider_name must not be empty")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def compose_article_text(text_fields: Iterable[tuple[str, str]]) -> str:
    """Compose a single text prompt from ordered article fields.

    Args:
        text_fields: Sequence of ``(field_name, value)`` pairs in the
            preferred order. Empty values are skipped.

    Returns:
        A pipe-delimited string usable as input to the FashionSigLIP text
        encoder. Returns an empty string when no fields are populated.
    """

    parts = [value.strip() for _, value in text_fields if value and value.strip()]
    return " | ".join(parts)


def select_device(preference: str) -> str:
    """Resolve the device to run the model on.

    The function is import-light: it only checks for ``torch`` when the
    caller has not requested CPU explicitly. ``"auto"`` prefers Apple MPS
    (M-series GPUs), then CUDA, then CPU.

    Args:
        preference: One of ``"auto"``, ``"cpu"``, ``"mps"``, or ``"cuda"``.

    Returns:
        The resolved device string.
    """

    if preference in {"cpu", "mps", "cuda"}:
        return preference
    if preference != "auto":
        raise ValueError(f"unsupported device preference: {preference!r}")
    try:
        import torch
    except ImportError:
        return "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in values))
    if norm == 0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def mean_of_normalized(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return the L2-normalized mean of L2-normalized input vectors.

    Used to combine text and image branches into a single multimodal vector.

    Args:
        vectors: Two or more dense vectors of equal length.

    Returns:
        The averaged then renormalized vector.

    Raises:
        ValueError: If ``vectors`` is empty or contains vectors of unequal
            length.
    """

    if not vectors:
        raise ValueError("vectors must not be empty")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("all vectors must share the same dimensionality")
    summed = [0.0] * dimension
    for vector in vectors:
        normalized = _l2_normalize(vector)
        for index, value in enumerate(normalized):
            summed[index] += value
    return _l2_normalize(tuple(value / len(vectors) for value in summed))


class MarqoFashionSigLipProvider:
    """Embed H&M articles with the Marqo-FashionSigLIP model.

    The provider conforms to :class:`ArticleEmbeddingProvider` and can be
    registered with :class:`EmbeddingProviderFactory` for use by the cache
    generation pipeline.

    Heavy dependencies (``torch``, ``transformers``, ``PIL``) are imported
    lazily on first call. Constructing the provider without the underlying
    model installed raises :class:`ImportError` with an actionable message.
    """

    def __init__(self, config: MarqoFashionSigLipConfig | None = None) -> None:
        """Initialize the provider.

        Args:
            config: Optional provider configuration. Defaults to
                :class:`MarqoFashionSigLipConfig`.

        Raises:
            ImportError: If ``transformers``/``torch`` are unavailable. The
                error message lists the required packages.
        """

        self._config = config or MarqoFashionSigLipConfig()
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency probe
            raise ImportError(
                "MarqoFashionSigLipProvider requires the optional 'torch' and "
                "'transformers' packages. Install them via "
                "`pip install torch transformers pillow`."
            ) from exc

        self._device = select_device(self._config.device)
        self._auto_model_cls = AutoModel
        self._auto_processor_cls = AutoProcessor
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def name(self) -> str:
        """Return the stable provider registry name."""

        return self._config.provider_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""

        return self._config.embedding_dimension

    @property
    def device(self) -> str:
        """Return the resolved device string."""

        return self._device

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        processor_factory: Any = self._auto_processor_cls.from_pretrained
        model_factory: Any = self._auto_model_cls.from_pretrained
        self._processor = processor_factory(
            self._config.model_id,
            revision=self._config.revision,
            trust_remote_code=True,
        )
        self._model = model_factory(
            self._config.model_id,
            revision=self._config.revision,
            trust_remote_code=True,
        )
        self._model.eval()
        self._model.to(self._device)

    def embed_articles(
        self, articles: Iterable[ArticleEmbeddingInput]
    ) -> Iterator[ArticleEmbeddingRecord]:
        """Embed article inputs into dense multimodal vectors.

        Args:
            articles: Article content records to embed.

        Yields:
            One :class:`ArticleEmbeddingRecord` per article. The vector is
            L2-normalized so cosine similarity equals dot product.
        """

        import torch

        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        batch: list[ArticleEmbeddingInput] = []
        for article in articles:
            batch.append(article)
            if len(batch) >= self._config.batch_size:
                yield from self._embed_batch(batch, torch)
                batch.clear()
        if batch:
            yield from self._embed_batch(batch, torch)

    def _embed_batch(
        self, batch: Sequence[ArticleEmbeddingInput], torch_module: Any
    ) -> Iterator[ArticleEmbeddingRecord]:
        from PIL import Image

        assert self._model is not None
        assert self._processor is not None
        texts = [
            compose_article_text(
                (field, article.text_fields.get(field, "")) for field in TEXT_FIELD_ORDER
            )
            for article in batch
        ]
        image_indices: list[int] = []
        images: list[Any] = []
        if self._config.use_image_when_available:
            for index, article in enumerate(batch):
                if article.image_path is None:
                    continue
                path = Path(article.image_path)
                if not path.exists():
                    continue
                try:
                    images.append(Image.open(path).convert("RGB"))
                    image_indices.append(index)
                except OSError as exc:  # pragma: no cover - I/O guard
                    _LOGGER.warning("failed to open image for %s: %s", article.article_id, exc)

        text_inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        text_inputs = {key: value.to(self._device) for key, value in text_inputs.items()}
        with torch_module.inference_mode():
            text_features = self._model.get_text_features(**text_inputs)
            text_features = torch_module.nn.functional.normalize(text_features, dim=-1)
            image_features = None
            if images:
                image_inputs = self._processor(images=images, return_tensors="pt")
                image_inputs = {key: value.to(self._device) for key, value in image_inputs.items()}
                image_features = self._model.get_image_features(**image_inputs)
                image_features = torch_module.nn.functional.normalize(image_features, dim=-1)

        text_vectors = text_features.detach().to("cpu").tolist()
        image_vectors = (
            image_features.detach().to("cpu").tolist() if image_features is not None else []
        )
        image_lookup = dict(zip(image_indices, image_vectors, strict=True))

        for index, article in enumerate(batch):
            text_vector = text_vectors[index]
            if index in image_lookup:
                combined = mean_of_normalized([text_vector, image_lookup[index]])
            else:
                combined = _l2_normalize(text_vector)
            yield ArticleEmbeddingRecord(
                article_id=article.article_id,
                vector=combined,
                provider_name=self.name,
            )


def build_marqo_fashion_siglip_provider() -> ArticleEmbeddingProvider:
    """Default factory used when registering with the provider registry."""

    return MarqoFashionSigLipProvider()


__all__ = (
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_MARQO_MODEL_ID",
    "DEFAULT_MARQO_REVISION",
    "DEFAULT_PROVIDER_NAME",
    "TEXT_FIELD_ORDER",
    "MarqoFashionSigLipConfig",
    "MarqoFashionSigLipProvider",
    "build_marqo_fashion_siglip_provider",
    "compose_article_text",
    "mean_of_normalized",
    "select_device",
)


def _placeholder_for_embedding_vector_alias() -> EmbeddingVector:
    """Internal helper to keep the ``EmbeddingVector`` import live for typing."""

    return ()
