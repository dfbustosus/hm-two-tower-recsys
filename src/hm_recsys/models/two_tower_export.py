"""Export trained two-tower embeddings and provide ANN retrieval helpers.

The export pipeline persists two artifacts to disk:

* ``customer_embeddings.npz`` — customer-int -> embedding tensor.
* ``article_embeddings.npz`` — article-int -> embedding tensor.

These pure-tensor artifacts are storage-cheap and avoid coupling the
retrieval pipeline to PyTorch checkpoints. They can be loaded by either
the included :class:`ExactVectorIndex` (correctness baseline) or, when
FAISS is installed, an HNSW index for production-grade ANN.

The module is structured so that ``faiss`` is an *optional* dependency:
all FAISS-specific code is contained in :func:`build_faiss_hnsw_index`
and gated behind a try/except import.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hm_recsys.embeddings.contracts import ArticleEmbeddingRecord
from hm_recsys.indexing.contracts import DistanceMetric, VectorIndexConfig
from hm_recsys.indexing.exact import ExactVectorIndex
from hm_recsys.models.two_tower_dataset import IdMapper, TwoTowerVocabulary


@dataclass(frozen=True)
class TwoTowerEmbeddingExport:
    """File paths produced by :func:`export_two_tower_embeddings`."""

    customer_embeddings_path: Path
    article_embeddings_path: Path
    customer_id_mapping_path: Path
    article_id_mapping_path: Path


def export_two_tower_embeddings(
    *,
    model: Any,
    vocabulary: TwoTowerVocabulary,
    output_dir: Path | str,
    chunk_size: int = 8192,
) -> TwoTowerEmbeddingExport:
    """Compute and persist customer/article embeddings from a trained model.

    The customer and article tensors are written using ``numpy.savez``
    (compressed=False to keep load latency low). The corresponding string
    ID mappings are written as plain ``.tsv`` files (index<TAB>token) so
    they can be inspected without code.

    Args:
        model: The trained two-tower module returned by
            :func:`hm_recsys.models.two_tower.build_torch_two_tower`.
        vocabulary: Vocabulary used during training. Must match the
            model's embedding tables.
        output_dir: Destination directory. Created if missing.
        chunk_size: Number of IDs encoded per forward pass.

    Returns:
        :class:`TwoTowerEmbeddingExport` with the on-disk paths.

    Raises:
        ImportError: If PyTorch or NumPy is not installed.
    """

    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "export_two_tower_embeddings requires torch and numpy; "
            "install via `pip install torch numpy`."
        ) from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    customer_embeddings = _encode_chunks(
        torch=torch,
        encode=lambda indices: model.encode_customer(indices),
        num_entries=vocabulary.num_customers,
        device=device,
        chunk_size=chunk_size,
    )
    article_embeddings = _encode_chunks(
        torch=torch,
        encode=lambda indices: model.encode_article(indices),
        num_entries=vocabulary.num_articles,
        device=device,
        chunk_size=chunk_size,
    )

    customer_path = output_path / "customer_embeddings.npz"
    article_path = output_path / "article_embeddings.npz"
    np.savez(customer_path, embeddings=customer_embeddings.cpu().numpy())
    np.savez(article_path, embeddings=article_embeddings.cpu().numpy())

    customer_mapping_path = output_path / "customer_id_mapping.tsv"
    article_mapping_path = output_path / "article_id_mapping.tsv"
    _write_id_mapping(customer_mapping_path, vocabulary.customer_mapper)
    _write_id_mapping(article_mapping_path, vocabulary.article_mapper)

    return TwoTowerEmbeddingExport(
        customer_embeddings_path=customer_path,
        article_embeddings_path=article_path,
        customer_id_mapping_path=customer_mapping_path,
        article_id_mapping_path=article_mapping_path,
    )


def build_exact_article_index(
    *,
    article_embeddings: Sequence[Sequence[float]],
    article_id_mapper: IdMapper,
    provider_name: str = "two_tower",
    metric: DistanceMetric = "dot",
) -> ExactVectorIndex:
    """Build a deterministic :class:`ExactVectorIndex` over article vectors.

    Skips the unknown token row at index ``0`` to keep the index aligned
    with the H&M article-ID universe.
    """

    if not article_embeddings:
        raise ValueError("article_embeddings must not be empty")
    dimension = len(article_embeddings[0])
    config = VectorIndexConfig(name=provider_name, dimension=dimension, metric=metric)
    index = ExactVectorIndex(config)
    records: list[ArticleEmbeddingRecord] = []
    for integer_index, vector in enumerate(article_embeddings):
        if integer_index == IdMapper.UNKNOWN_INDEX:
            continue
        article_id = article_id_mapper.token_for(integer_index)
        records.append(
            ArticleEmbeddingRecord(
                article_id=article_id,
                vector=tuple(vector),
                provider_name=provider_name,
            )
        )
    index.build(records)
    return index


def build_faiss_hnsw_index(
    *,
    article_embeddings: Sequence[Sequence[float]],
    m_neighbors: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> Any:
    """Build a FAISS HNSW index for production-grade ANN retrieval.

    Args:
        article_embeddings: Sequence of L2-normalized article vectors. The
            unknown-token slot at index 0 is included unchanged so the
            integer index matches the row of the embedding tensor.
        m_neighbors: HNSW graph degree. Larger values trade memory for
            recall.
        ef_construction: HNSW build-time exploration parameter.
        ef_search: HNSW query-time exploration parameter.

    Returns:
        A FAISS ``IndexHNSWFlat`` instance with all article embeddings
        already added.

    Raises:
        ImportError: If ``faiss`` is not installed.
        ImportError: If NumPy is not installed.
    """

    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "build_faiss_hnsw_index requires faiss-cpu (or faiss-gpu) and numpy."
        ) from exc

    if not article_embeddings:
        raise ValueError("article_embeddings must not be empty")
    dimension = len(article_embeddings[0])
    matrix = np.asarray(article_embeddings, dtype="float32")
    index = faiss.IndexHNSWFlat(dimension, m_neighbors)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(matrix)
    return index


def _encode_chunks(
    *,
    torch: Any,
    encode: Any,
    num_entries: int,
    device: Any,
    chunk_size: int,
) -> Any:
    chunks: list[Any] = []
    with torch.inference_mode():
        for start in range(0, num_entries, chunk_size):
            stop = min(start + chunk_size, num_entries)
            indices = torch.arange(start, stop, device=device)
            chunks.append(encode(indices).detach())
    return torch.cat(chunks, dim=0)


def _write_id_mapping(path: Path, mapper: IdMapper) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("index\ttoken\n")
        for token, index in mapper.items():
            handle.write(f"{index}\t{token}\n")


def load_two_tower_embeddings(path: Path | str) -> Sequence[Sequence[float]]:
    """Load an exported embedding matrix as a sequence of float vectors."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError("load_two_tower_embeddings requires numpy.") from exc

    data = np.load(Path(path), allow_pickle=False)
    matrix = data["embeddings"]
    rows: list[tuple[float, ...]] = [tuple(float(value) for value in row) for row in matrix]
    return rows


def load_id_mapping(path: Path | str) -> IdMapper:
    """Load an ID mapping TSV file into a fresh :class:`IdMapper`."""

    mapper = IdMapper()
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if header.strip() != "index\ttoken":
            raise ValueError(f"unexpected header in {path}: {header!r}")
        rows: list[tuple[int, str]] = []
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            index_str, token = line.split("\t", maxsplit=1)
            rows.append((int(index_str), token))
    rows.sort(key=lambda pair: pair[0])
    next_expected = 1  # index 0 is reserved for UNKNOWN_TOKEN
    for index, token in rows:
        if index != next_expected:
            raise ValueError(
                f"non-contiguous ID mapping at {path}: expected {next_expected}, got {index}"
            )
        mapper.add_or_lookup(token)
        next_expected += 1
    return mapper


__all__ = (
    "TwoTowerEmbeddingExport",
    "build_exact_article_index",
    "build_faiss_hnsw_index",
    "export_two_tower_embeddings",
    "load_id_mapping",
    "load_two_tower_embeddings",
)
