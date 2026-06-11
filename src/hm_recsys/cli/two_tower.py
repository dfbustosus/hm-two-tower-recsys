"""CLI subcommands for the multimodal two-tower workflow.

This module is the first command group migrated out of the legacy
``cli/_legacy.py`` monolith. It registers two subcommands:

* ``train-two-tower`` — train the in-batch sampled-softmax two-tower on
  cutoff-safe positive pairs and persist customer/article embeddings.
* ``score-two-tower-candidates`` — load an exported two-tower index and
  append a ``two_tower_score`` column to an existing candidate-export CSV
  so downstream rankers can use it as a feature.

Both commands degrade gracefully when their optional dependencies
(PyTorch, NumPy) are missing: argument parsing still works (so help is
discoverable), and an actionable :class:`ImportError` is raised only when
the handler actually needs the dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from datetime import date as _date
from pathlib import Path
from typing import Any

from hm_recsys.data.io import TransactionEvent, iter_csv_rows
from hm_recsys.evaluation.temporal import TemporalSplit
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.models.two_tower import (
    ArticleTowerConfig,
    CustomerTowerConfig,
    TwoTowerConfig,
    TwoTowerTrainingConfig,
)
from hm_recsys.models.two_tower_dataset import (
    TwoTowerVocabulary,
    build_id_mappers_from_transactions,
    iter_positive_training_pairs,
    iter_unique_pair_batches,
)
from hm_recsys.models.two_tower_export import (
    export_two_tower_embeddings,
    load_id_mapping,
    load_two_tower_embeddings,
)
from hm_recsys.models.two_tower_train import TwoTowerTrainerConfig, train_two_tower

DEFAULT_OUTPUT_SUBDIR = "two-tower-exports"


def register_subcommands(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> None:
    """Register the two-tower subcommands onto an existing subparser group."""

    _register_train_two_tower(subparsers)
    _register_score_two_tower_candidates(subparsers)


def _register_train_two_tower(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> None:
    parser = subparsers.add_parser(
        "train-two-tower",
        help=(
            "Train the multimodal two-tower on cutoff-safe positive pairs and "
            "persist customer/article embeddings for downstream scoring."
        ),
    )
    parser.add_argument("--cutoff", required=True, help="Exclusive training cutoff (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the exported NPZ + TSV artifacts "
            "(default: artifacts/two-tower-exports/<cutoff>)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--customer-id-embedding-dim", type=int, default=64)
    parser.add_argument("--article-id-embedding-dim", type=int, default=64)
    parser.add_argument("--content-embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--mixed-negative-count", type=int, default=256)
    parser.add_argument("--cross-batch-negative-capacity", type=int, default=8192)
    parser.add_argument(
        "--no-log-q-correction",
        action="store_true",
        help="Disable LogQ correction (default: enabled).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Compute device (default: auto-detects MPS, then CUDA, then CPU).",
    )
    parser.add_argument(
        "--max-positive-pairs",
        type=int,
        default=None,
        help="Optional cap on positive pairs used for training (random first N).",
    )
    parser.add_argument(
        "--progress-every-steps",
        type=int,
        default=50,
        help="Emit a progress line every N optimizer steps (default: 50).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Linear LR warmup over N steps (default: 0 / disabled). Strongly "
            "recommended (try 500-2000) for sampled-softmax training to prevent "
            "early collapse to a degenerate fixed point."
        ),
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help=(
            "Global gradient L2 norm cap (default: disabled). Try 1.0 to "
            "stabilize training when the loss bounces or saturates."
        ),
    )
    parser.add_argument(
        "--content-embeddings-manifest-path",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON manifest produced by 'generate-article-"
            "embeddings'. When provided, the manifest's JSONL vectors are "
            "loaded into the article tower's frozen content slot, so the "
            "content dimensions carry real FashionCLIP signal instead of "
            "random noise. Vector dim must equal --content-embedding-dim; "
            "articles missing from the manifest get zero vectors."
        ),
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    parser.set_defaults(handler=_handle_train_two_tower)


def _register_score_two_tower_candidates(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
) -> None:
    parser = subparsers.add_parser(
        "score-two-tower-candidates",
        help=(
            "Append a 'two_tower_score' column to an existing candidate export "
            "using the exported customer/article embeddings."
        ),
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding customer_embeddings.npz, article_embeddings.npz, "
            "and their TSV mappings."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Destination CSV. Defaults to input CSV with .two_tower.csv suffix.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.set_defaults(handler=_handle_score_two_tower_candidates)


def _handle_train_two_tower(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    cutoff = _date.fromisoformat(args.cutoff)
    split = TemporalSplit.from_isoformat(args.cutoff)
    output_dir = args.output_dir or paths.artifacts_dir / DEFAULT_OUTPUT_SUBDIR / args.cutoff
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[train-two-tower] cutoff={cutoff.isoformat()} "
        f"horizon_days={split.horizon_days} output={output_dir}",
        flush=True,
    )

    def _events() -> Iterator[TransactionEvent]:
        return _iter_transaction_events(paths.raw_data_dir / "transactions_train.csv")

    print("[train-two-tower] scanning transactions to build vocabularies...", flush=True)
    vocabulary = build_id_mappers_from_transactions(_events(), cutoff)
    total_positive_interactions = vocabulary.total_positive_interactions
    if args.max_positive_pairs is not None:
        total_positive_interactions = min(total_positive_interactions, args.max_positive_pairs)
    # The unique-batch constraint can yield a few partial batches when
    # the trailing positives all collide on article ids; using ceil here
    # is a strict upper bound on actual step count.
    expected_steps_per_epoch = max(
        1, -(-total_positive_interactions // max(1, args.batch_size))
    )
    expected_total_steps = expected_steps_per_epoch * args.epochs
    print(
        f"[train-two-tower] vocabulary: customers={vocabulary.num_customers} "
        f"articles={vocabulary.num_articles} "
        f"positives={total_positive_interactions} "
        f"steps_per_epoch~{expected_steps_per_epoch} "
        f"total_steps~{expected_total_steps}",
        flush=True,
    )

    model_config = TwoTowerConfig(
        customer_tower=CustomerTowerConfig(
            num_customers=vocabulary.num_customers,
            customer_id_embedding_dim=args.customer_id_embedding_dim,
            dense_feature_dim=0,
            hidden_dims=tuple(args.hidden_dims),
            output_dim=args.embedding_dim,
            dropout=args.dropout,
        ),
        article_tower=ArticleTowerConfig(
            num_articles=vocabulary.num_articles,
            article_id_embedding_dim=args.article_id_embedding_dim,
            content_embedding_dim=args.content_embedding_dim,
            hidden_dims=tuple(args.hidden_dims),
            output_dim=args.embedding_dim,
            dropout=args.dropout,
        ),
        training=TwoTowerTrainingConfig(
            temperature=args.temperature,
            mixed_negative_count=args.mixed_negative_count,
            cross_batch_negative_capacity=args.cross_batch_negative_capacity,
        ),
    )
    trainer_config = TwoTowerTrainerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_q_correction=not args.no_log_q_correction,
        progress_every_steps=args.progress_every_steps,
        expected_total_steps=expected_total_steps,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
    )

    def _batches() -> Iterator[tuple[list[int], list[int]]]:
        pair_stream = iter_positive_training_pairs(_events(), vocabulary, cutoff)
        if args.max_positive_pairs is not None:
            pair_stream = _take(pair_stream, args.max_positive_pairs)
        yield from iter_unique_pair_batches(
            pair_stream, batch_size=args.batch_size, drop_last=False
        )

    content_embeddings = None
    if args.content_embeddings_manifest_path is not None:
        content_embeddings = _load_content_embedding_matrix(
            manifest_path=Path(args.content_embeddings_manifest_path),
            vocabulary=vocabulary,
            expected_dim=args.content_embedding_dim,
            progress=lambda message: print(f"[train-two-tower] {message}", flush=True),
        )

    result = train_two_tower(
        model_config=model_config,
        trainer_config=trainer_config,
        vocabulary=vocabulary,
        batch_source=_batches,
        progress=lambda message: print(f"[train-two-tower] {message}", flush=True),
        content_embeddings=content_embeddings,
    )
    print(
        f"[train-two-tower] training done: steps={result.steps} "
        f"final_loss={result.final_loss:.4f} mean_loss={result.mean_loss:.4f}",
        flush=True,
    )

    export = export_two_tower_embeddings(
        model=result.model, vocabulary=vocabulary, output_dir=output_dir
    )
    print(
        "[train-two-tower] embeddings exported to: "
        f"{export.customer_embeddings_path}, {export.article_embeddings_path}",
        flush=True,
    )
    return 0


def _handle_score_two_tower_candidates(args: argparse.Namespace) -> int:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "score-two-tower-candidates requires numpy; install via `pip install numpy`."
        ) from exc

    export_dir = Path(args.export_dir)
    customer_vectors = load_two_tower_embeddings(export_dir / "customer_embeddings.npz")
    article_vectors = load_two_tower_embeddings(export_dir / "article_embeddings.npz")
    customer_mapper = load_id_mapping(export_dir / "customer_id_mapping.tsv")
    article_mapper = load_id_mapping(export_dir / "article_id_mapping.tsv")
    customer_matrix = np.asarray(customer_vectors, dtype=np.float32)
    article_matrix = np.asarray(article_vectors, dtype=np.float32)

    candidate_csv = Path(args.candidate_csv)
    output_csv = (
        Path(args.output_csv) if args.output_csv else candidate_csv.with_suffix(".two_tower.csv")
    )
    customer_id_column = "customer_id"
    article_id_column = "article_id"
    additional_column = "two_tower_score"

    print(
        f"[score-two-tower-candidates] reading {candidate_csv}",
        flush=True,
    )
    # Pass 1: collect unique (customer_index, article_index) pairs so we
    # only run one dot per pair regardless of how many source rows it has.
    with candidate_csv.open("r", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None or customer_id_column not in reader.fieldnames:
            raise ValueError(f"input CSV {candidate_csv} missing {customer_id_column!r} header")
        if article_id_column not in reader.fieldnames:
            raise ValueError(f"input CSV {candidate_csv} missing {article_id_column!r} header")
        fieldnames = list(reader.fieldnames)
        unique_pairs: dict[tuple[int, int], float] = {}
        row_pairs: list[tuple[int, int]] = []
        for row in reader:
            customer_idx = customer_mapper.index_for(row[customer_id_column])
            article_idx = article_mapper.index_for(row[article_id_column])
            row_pairs.append((customer_idx, article_idx))
            unique_pairs.setdefault((customer_idx, article_idx), 0.0)

    print(
        f"[score-two-tower-candidates] rows={len(row_pairs)} "
        f"unique_pairs={len(unique_pairs)} "
        f"customers={customer_matrix.shape} articles={article_matrix.shape}",
        flush=True,
    )

    # Vectorized scoring: stack indices, gather embeddings, single einsum.
    pair_keys = list(unique_pairs.keys())
    customer_indices = np.fromiter(
        (key[0] for key in pair_keys), dtype=np.int64, count=len(pair_keys)
    )
    article_indices = np.fromiter(
        (key[1] for key in pair_keys), dtype=np.int64, count=len(pair_keys)
    )
    valid_mask = (customer_indices != 0) & (article_indices != 0)
    scores = np.zeros(len(pair_keys), dtype=np.float32)
    if valid_mask.any():
        # einsum is faster than (a*b).sum(axis=1) and avoids the temporary.
        scores[valid_mask] = np.einsum(
            "ij,ij->i",
            customer_matrix[customer_indices[valid_mask]],
            article_matrix[article_indices[valid_mask]],
        )
    pair_score_map = {pair: float(score) for pair, score in zip(pair_keys, scores, strict=True)}

    # Pass 2: write the original CSV plus the new column, looking up the
    # cached score for each row by its (customer, article) pair.
    output_fieldnames = [*fieldnames, additional_column]
    with (
        candidate_csv.open("r", encoding="utf-8") as input_handle,
        output_csv.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=output_fieldnames)
        writer.writeheader()
        scored_rows = 0
        for index, row in enumerate(reader):
            row[additional_column] = f"{pair_score_map[row_pairs[index]]:.6f}"
            writer.writerow(row)
            scored_rows += 1

    print(
        f"[score-two-tower-candidates] scored {scored_rows} rows; wrote {output_csv}",
        flush=True,
    )
    return 0


def _iter_transaction_events(path: Path) -> Iterator[TransactionEvent]:
    for row in iter_csv_rows(path, ("t_dat", "customer_id", "article_id")):
        yield TransactionEvent(
            t_dat=_date.fromisoformat(row["t_dat"]),
            customer_id=row["customer_id"],
            article_id=row["article_id"],
        )


def _take(stream: Iterator[Any], count: int) -> Iterator[Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    for index, item in enumerate(stream):
        if index >= count:
            return
        yield item


def _load_content_embedding_matrix(
    *,
    manifest_path: Path,
    vocabulary: TwoTowerVocabulary,
    expected_dim: int,
    progress: Any,
) -> Any:
    """Build a ``(num_articles, expected_dim)`` matrix from a CLIP manifest.

    The manifest is the JSON file produced by
    ``generate-article-embeddings``. We expect alongside it:

    * ``embeddings_path`` (JSONL with one ``{"article_id", "vector"}`` per
      line), and
    * ``dimension`` matching ``expected_dim``.

    Articles present in the two-tower vocabulary but absent from the
    manifest get zero vectors; articles in the manifest but absent from
    the vocabulary are silently skipped. The index-0 unknown row is
    always zero.

    Raises:
        FileNotFoundError: If manifest or embeddings file is missing.
        ValueError: If the manifest dimension does not match
            ``expected_dim``.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "Loading content embeddings requires numpy; install via `pip install numpy`."
        ) from exc

    progress(f"loading content-embedding manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_dim = int(manifest.get("dimension", -1))
    if manifest_dim != expected_dim:
        raise ValueError(
            f"content embedding dim mismatch: manifest has dim={manifest_dim}, "
            f"--content-embedding-dim={expected_dim}. Re-run generate-article-"
            f"embeddings with a matching model OR adjust --content-embedding-dim."
        )
    embeddings_path = Path(manifest["embeddings_path"])
    if not embeddings_path.is_absolute():
        # Manifests written from a different CWD may store relative paths;
        # resolve them against the manifest's own directory so callers
        # don't have to set CWD before running the trainer.
        embeddings_path = (manifest_path.parent / embeddings_path).resolve()
    matrix = np.zeros((vocabulary.num_articles, expected_dim), dtype=np.float32)
    loaded = 0
    skipped_unknown = 0
    progress(f"reading embeddings from: {embeddings_path}")
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            article_id = row["article_id"]
            try:
                index = vocabulary.article_mapper.index_for(
                    article_id, allow_unknown=False
                )
            except KeyError:
                skipped_unknown += 1
                continue
            vector = row["vector"]
            if len(vector) != expected_dim:
                raise ValueError(
                    f"vector for article {article_id} has dim={len(vector)} "
                    f"but manifest claims dim={expected_dim}"
                )
            matrix[index] = np.asarray(vector, dtype=np.float32)
            loaded += 1
    progress(
        f"content embeddings loaded: matched={loaded}/{vocabulary.num_articles - 1} "
        f"manifest_skipped_unknown_articles={skipped_unknown} "
        f"missing={vocabulary.num_articles - 1 - loaded}"
    )
    return matrix


__all__ = ("register_subcommands",)
