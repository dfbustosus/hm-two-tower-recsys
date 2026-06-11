"""Augment a candidate CSV with a ``content_user_cosine`` column.

Pipeline:

1. Load FashionCLIP (or any HF-CLIP) per-article embeddings from a
   JSONL manifest produced by ``generate-article-embeddings``.
2. Stream the transactions CSV up to ``--cutoff`` (exclusive) and build,
   for each customer that appears in the candidate CSV, a customer
   content embedding equal to the L2-normalized MEAN of the FashionCLIP
   vectors of the customer's last ``--max-history-items`` purchases
   within ``--history-lookback-days`` of the cutoff. Customers with no
   covered history get a zero vector (cold start) and contribute
   ``content_user_cosine = 0.0`` to every candidate.
3. Stream the candidate CSV (canonical or already-augmented) row-by-row
   and write an augmented CSV that adds ``content_user_cosine`` to the
   trailing optional columns, preserving any columns already present.

The script is deliberately a standalone Python ``scripts/`` entry (NOT
a `hm_recsys` CLI subcommand) because the candidate-augmentation seam
is intentionally informal: experiments add new optional columns
frequently, and keeping them out of the public CLI surface avoids
churning the published interface every time we try a new feature.
Stable / production-grade variants can graduate to ``hm-recsys
score-content-similarity-candidates`` later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from hm_recsys.retrieval.candidate_export import (
    CANDIDATE_EXPORT_HEADER,
    CONTENT_USER_COSINE_COLUMN,
    KNOWN_AUGMENTED_COLUMNS,
    TWO_TOWER_SCORE_COLUMN,
)


@dataclass(frozen=True)
class ScoreConfig:
    """Configuration knobs for content-similarity augmentation."""

    cutoff: date
    history_lookback_days: int
    max_history_items: int


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = ScoreConfig(
        cutoff=date.fromisoformat(args.cutoff),
        history_lookback_days=args.history_lookback_days,
        max_history_items=args.max_history_items,
    )
    started = time.perf_counter()
    _log(f"loading article embeddings manifest: {args.embeddings_manifest_path}")
    article_vectors, dim = _load_article_embeddings(Path(args.embeddings_manifest_path))
    _log(f"loaded {len(article_vectors)} article vectors (dim={dim})")

    _log(f"collecting target customers from {args.candidate_csv}")
    target_customers = _collect_target_customers(Path(args.candidate_csv))
    _log(f"target customer count: {len(target_customers)}")

    _log("building per-customer content embeddings from transactions")
    customer_vectors = _build_customer_embeddings(
        transactions_csv=Path(args.transactions_csv),
        target_customers=target_customers,
        article_vectors=article_vectors,
        config=config,
    )
    nonzero = sum(1 for v in customer_vectors.values() if v is not None)
    _log(
        f"customer embeddings ready: matched={nonzero}/{len(target_customers)} "
        f"(cold-start={len(target_customers) - nonzero})"
    )

    _log(f"writing augmented CSV: {args.output_csv}")
    rows_written, scored_rows = _stream_augment(
        candidate_csv=Path(args.candidate_csv),
        output_csv=Path(args.output_csv),
        customer_vectors=customer_vectors,
        article_vectors=article_vectors,
    )
    elapsed = time.perf_counter() - started
    _log(
        f"done in {elapsed:.1f}s: rows={rows_written} "
        f"non_zero_scored={scored_rows} ({100*scored_rows/max(1,rows_written):.2f}%)"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--embeddings-manifest-path", type=Path, required=True)
    parser.add_argument("--transactions-csv", type=Path, required=True)
    parser.add_argument(
        "--cutoff",
        required=True,
        help="Exclusive cutoff date for customer history (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--history-lookback-days",
        type=int,
        default=90,
        help=(
            "Only include customer transactions within this many days BEFORE "
            "cutoff in the customer-content embedding (default: 90)."
        ),
    )
    parser.add_argument(
        "--max-history-items",
        type=int,
        default=20,
        help=(
            "Cap on number of most-recent in-window purchases used per customer "
            "(default: 20)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------


def _load_article_embeddings(manifest_path: Path) -> tuple[dict[str, list[float]], int]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "score_content_similarity_candidates requires numpy; "
            "install via `pip install numpy`."
        ) from exc

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dim = int(manifest["dimension"])
    embeddings_path = Path(manifest["embeddings_path"])
    if not embeddings_path.is_absolute():
        embeddings_path = (manifest_path.parent / embeddings_path).resolve()

    article_vectors: dict[str, list[float]] = {}
    with embeddings_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            vector = row["vector"]
            if len(vector) != dim:
                raise ValueError(
                    f"vector for article {row['article_id']} has dim={len(vector)} "
                    f"but manifest claims dim={dim}"
                )
            # Eager L2 normalization at load time: the manifest claims
            # ``normalized=true`` but defending against a future manifest
            # variant costs almost nothing.
            arr = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            article_vectors[row["article_id"]] = arr.tolist()
    return article_vectors, dim


# ---------------------------------------------------------------------------
# Candidate-customer set + transaction streaming
# ---------------------------------------------------------------------------


def _collect_target_customers(candidate_csv: Path) -> set[str]:
    customers: set[str] = set()
    with candidate_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "customer_id" not in reader.fieldnames:
            raise ValueError(f"candidate CSV missing customer_id column: {candidate_csv}")
        for row in reader:
            customers.add(row["customer_id"])
    return customers


def _iter_transactions(
    path: Path,
) -> Iterator[tuple[date, str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield (
                date.fromisoformat(row["t_dat"]),
                row["customer_id"],
                row["article_id"],
            )


def _build_customer_embeddings(
    *,
    transactions_csv: Path,
    target_customers: set[str],
    article_vectors: dict[str, list[float]],
    config: ScoreConfig,
) -> dict[str, Any]:
    """Return ``{customer_id -> np.ndarray | None}``.

    ``None`` represents a cold-start customer with no in-window history.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency probe
        raise ImportError(
            "score_content_similarity_candidates requires numpy; "
            "install via `pip install numpy`."
        ) from exc

    # First pass: collect recent (date, article_id) per target customer up
    # to ``max_history_items`` * 3 to allow for in-window filtering after
    # sorting. Using a bounded deque keyed by raw insertion order keeps
    # memory linear in the customer count.
    history_cap = max(1, config.max_history_items * 3)
    customer_history: dict[str, deque[tuple[date, str]]] = defaultdict(
        lambda: deque(maxlen=history_cap)
    )
    lookback_start = date.fromordinal(
        config.cutoff.toordinal() - config.history_lookback_days
    )
    processed_rows = 0
    for t_dat, customer_id, article_id in _iter_transactions(transactions_csv):
        processed_rows += 1
        if processed_rows % 5_000_000 == 0:
            _log(f"  scanned {processed_rows} transaction rows...")
        if t_dat >= config.cutoff or t_dat < lookback_start:
            continue
        if customer_id not in target_customers:
            continue
        if article_id not in article_vectors:
            continue
        customer_history[customer_id].append((t_dat, article_id))

    # Second pass: build embeddings. Sort each customer's history by
    # ``t_dat`` (descending), keep the top ``max_history_items``, then
    # mean-and-normalize the corresponding article vectors.
    embeddings: dict[str, Any] = {}
    for customer_id in target_customers:
        history = customer_history.get(customer_id)
        if not history:
            embeddings[customer_id] = None
            continue
        sorted_items = sorted(history, key=lambda item: item[0], reverse=True)
        recent_articles = [a for _, a in sorted_items[: config.max_history_items]]
        if not recent_articles:
            embeddings[customer_id] = None
            continue
        stacked = np.asarray(
            [article_vectors[a] for a in recent_articles], dtype=np.float32
        )
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-12:
            embeddings[customer_id] = None
            continue
        embeddings[customer_id] = (mean / norm).astype(np.float32)
        if len(embeddings) % 5000 == 0:
            _log(f"  built embeddings for {len(embeddings)} customers...")
    return embeddings


# ---------------------------------------------------------------------------
# Augmenting writer
# ---------------------------------------------------------------------------


def _stream_augment(
    *,
    candidate_csv: Path,
    output_csv: Path,
    customer_vectors: dict[str, Any],
    article_vectors: dict[str, list[float]],
) -> tuple[int, int]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "score_content_similarity_candidates requires numpy; "
            "install via `pip install numpy`."
        ) from exc

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Convert article_vectors to numpy arrays once for fast dot.
    article_np: dict[str, Any] = {
        a: np.asarray(v, dtype=np.float32) for a, v in article_vectors.items()
    }
    rows_written = 0
    scored_rows = 0
    with (
        candidate_csv.open("r", encoding="utf-8", newline="") as input_handle,
        output_csv.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty candidate CSV: {candidate_csv}")
        input_fields = list(reader.fieldnames)
        _validate_header(input_fields)
        output_fields = _build_output_fields(input_fields)
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()
        for row in reader:
            customer_vector = customer_vectors.get(row["customer_id"])
            article_vector = article_np.get(row["article_id"])
            if customer_vector is None or article_vector is None:
                score = 0.0
            else:
                # Both vectors are pre-normalized to unit length, so the
                # dot product is the cosine similarity.
                score = float(np.dot(customer_vector, article_vector))
                if math.isnan(score):
                    score = 0.0
                else:
                    scored_rows += 1
            row[CONTENT_USER_COSINE_COLUMN] = f"{score:.6f}"
            writer.writerow(row)
            rows_written += 1
    return rows_written, scored_rows


def _validate_header(input_fields: list[str]) -> None:
    canonical = list(CANDIDATE_EXPORT_HEADER)
    if input_fields[: len(canonical)] != canonical:
        raise ValueError(
            f"input CSV header must start with {','.join(canonical)}; "
            f"got {','.join(input_fields)}"
        )
    trailing = input_fields[len(canonical) :]
    for column in trailing:
        if column not in KNOWN_AUGMENTED_COLUMNS:
            raise ValueError(
                f"input CSV has unknown trailing column {column!r}; "
                f"known augmented columns are: {','.join(KNOWN_AUGMENTED_COLUMNS)}"
            )
    if CONTENT_USER_COSINE_COLUMN in trailing:
        raise ValueError(
            f"input CSV already contains {CONTENT_USER_COSINE_COLUMN!r}; "
            f"refuse to overwrite \u2014 delete the column first or use a "
            f"different --output-csv to keep both"
        )


def _build_output_fields(input_fields: list[str]) -> list[str]:
    """Return input fields with ``content_user_cosine`` appended in canonical position."""

    canonical = list(CANDIDATE_EXPORT_HEADER)
    trailing = input_fields[len(canonical) :]
    # Re-emit trailing optional columns in the canonical order declared by
    # KNOWN_AUGMENTED_COLUMNS, inserting content_user_cosine at its slot.
    present = set(trailing) | {CONTENT_USER_COSINE_COLUMN}
    new_trailing = [c for c in KNOWN_AUGMENTED_COLUMNS if c in present]
    return canonical + new_trailing


def _log(message: str) -> None:
    print(f"[content-sim] {message}", flush=True)


if __name__ == "__main__":
    sys.exit(main())


# ``TWO_TOWER_SCORE_COLUMN`` is imported above to anchor the module's
# dependency on the augmented-column schema (used during header validation
# via ``KNOWN_AUGMENTED_COLUMNS``); a no-op reference keeps linters from
# flagging it as unused while making the intent visible.
_ = TWO_TOWER_SCORE_COLUMN
