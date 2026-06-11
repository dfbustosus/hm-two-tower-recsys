"""Train Item2Vec (skip-gram Word2Vec) embeddings over H&M purchase sessions.

A "session" is one (customer, t_dat) tuple's set of purchased articles.
Each session becomes one shuffled "sentence"; gensim's Word2Vec learns dense
article embeddings such that articles co-purchased in the same session (or
nearby in the same customer's history when ``--session-mode customer``)
end up close in cosine space.

Why Item2Vec (vs FashionCLIP text/image embeddings):
- FashionCLIP captures visual/textual similarity. Two black hoodies are
  close even if no one ever buys them together. That signal poorly
  predicts "what does this customer buy NEXT" - confirmed empirically on
  this dataset (FashionCLIP-as-candidate-source produced LB 0.01581).
- Item2Vec captures CO-PURCHASE patterns directly. If two items get
  bought together often (same basket, same customer's recent history),
  they end up near each other in embedding space. That signal is
  empirically purchase-predictive (cf. Senkin13 1st-place writeup, where
  trained item embeddings drove the strongest candidate source).

Output:
- ``<output_dir>/item_embeddings.jsonl`` - one JSON per article with
  ``article_id`` and ``vector`` (L2-normalized float list), matching the
  schema consumed by ``augment_candidates_with_content_retrieval.py`` and
  ``generate_content_similarity_submission.py``.
- ``<output_dir>/item_manifest.json`` - manifest with ``dimension``,
  ``embedding_count``, ``embeddings_path``, ``training_config``, etc.,
  matching the FashionCLIP manifest schema.

Session modes:
- ``basket`` (default): one sentence per (customer, t_dat). Cleanest
  co-purchase signal; ignores cross-day sequence structure.
- ``customer``: one sentence per customer = full ordered purchase history.
  Captures cross-day sequential patterns; longer sentences mean each
  window sees more context. Slower but typically stronger.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    cutoff: date | None
    session_mode: str
    embedding_dim: int
    window: int
    min_count: int
    epochs: int
    workers: int
    seed: int


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = TrainConfig(
        cutoff=date.fromisoformat(args.cutoff) if args.cutoff else None,
        session_mode=args.session_mode,
        embedding_dim=args.embedding_dim,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )

    started = time.perf_counter()
    _log(
        f"building sessions: mode={config.session_mode} cutoff={config.cutoff}"
    )
    sessions = list(_build_sessions(Path(args.transactions_csv), config))
    n_sessions = len(sessions)
    n_total_tokens = sum(len(s) for s in sessions)
    _log(
        f"  -> {n_sessions} sessions, {n_total_tokens} token positions, "
        f"avg session length {n_total_tokens / max(1, n_sessions):.2f}"
    )

    _log("training gensim Word2Vec (skip-gram, negative sampling)")
    model = _train_word2vec(sessions, config)
    vocab = sorted(model.wv.key_to_index.keys())
    _log(f"  -> learned embeddings for {len(vocab)} articles")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "item_embeddings.jsonl"
    manifest_path = output_dir / "item_manifest.json"

    _log(f"writing embeddings: {embeddings_path}")
    _write_embeddings_jsonl(model, vocab, embeddings_path)

    elapsed = time.perf_counter() - started
    manifest = {
        "article_count": len(vocab),
        "dimension": config.embedding_dim,
        "embedding_count": len(vocab),
        "embeddings_path": embeddings_path.name,
        "distance_metric": "cosine",
        "dtype": "float32",
        "embedding_kind": "item2vec_skipgram",
        "manifest_path": str(manifest_path),
        "missing_embedding_count": 0,
        "normalized": True,
        "training_config": {
            "session_mode": config.session_mode,
            "cutoff": config.cutoff.isoformat() if config.cutoff else None,
            "embedding_dim": config.embedding_dim,
            "window": config.window,
            "min_count": config.min_count,
            "epochs": config.epochs,
            "workers": config.workers,
            "seed": config.seed,
            "session_count": n_sessions,
            "token_count": n_total_tokens,
            "runtime_seconds": elapsed,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"wrote manifest: {manifest_path}")
    _log(f"done in {elapsed:.1f}s")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cutoff",
        default=None,
        help="Optional YYYY-MM-DD; if set, only transactions with t_dat < cutoff "
        "contribute to training.",
    )
    parser.add_argument(
        "--session-mode",
        choices=("basket", "customer"),
        default="basket",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------


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


def _build_sessions(
    transactions_csv: Path, config: TrainConfig
) -> Iterator[list[str]]:
    """Yield one list[article_id] per session.

    - ``basket`` mode: emit one session per (customer, t_dat) tuple. Articles
      within the basket are shuffled (deterministic seed) since order within
      a single day is not meaningful for co-purchase signal.
    - ``customer`` mode: emit one session per customer = ordered purchase
      history (chronological). Articles within the same day are kept in
      insertion order to avoid spurious random noise.
    """

    rng = random.Random(config.seed)
    if config.session_mode == "basket":
        baskets: dict[tuple[str, date], list[str]] = defaultdict(list)
        processed = 0
        for t_dat, customer_id, article_id in _iter_transactions(transactions_csv):
            processed += 1
            if processed % 5_000_000 == 0:
                _log(f"    scanned {processed} transaction rows...")
            if config.cutoff is not None and t_dat >= config.cutoff:
                continue
            baskets[(customer_id, t_dat)].append(article_id)
        for items in baskets.values():
            if len(items) < 2:
                # Single-item baskets carry no co-purchase signal; the
                # min_count filter inside Word2Vec already handles
                # unique-article rarity, but skipping these here speeds
                # training and avoids meaningless 1-token "sessions".
                continue
            rng.shuffle(items)
            yield items
    else:
        customer_history: dict[str, list[tuple[date, str]]] = defaultdict(list)
        processed = 0
        for t_dat, customer_id, article_id in _iter_transactions(transactions_csv):
            processed += 1
            if processed % 5_000_000 == 0:
                _log(f"    scanned {processed} transaction rows...")
            if config.cutoff is not None and t_dat >= config.cutoff:
                continue
            customer_history[customer_id].append((t_dat, article_id))
        for items in customer_history.values():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x[0])
            yield [aid for _, aid in items]


# ---------------------------------------------------------------------------
# Word2Vec training
# ---------------------------------------------------------------------------


def _train_word2vec(sessions: list[list[str]], config: TrainConfig):
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=sessions,
        vector_size=config.embedding_dim,
        window=config.window,
        min_count=config.min_count,
        sg=1,  # skip-gram
        negative=5,
        ns_exponent=0.75,
        workers=config.workers,
        epochs=config.epochs,
        seed=config.seed,
    )
    return model


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_embeddings_jsonl(model, vocab: list[str], path: Path) -> None:
    import numpy as np

    with path.open("w", encoding="utf-8") as handle:
        for article_id in vocab:
            vector = model.wv[article_id].astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            handle.write(
                json.dumps(
                    {"article_id": article_id, "vector": vector.tolist()},
                    separators=(",", ":"),
                )
                + "\n"
            )


def _log(message: str) -> None:
    print(f"[item2vec] {message}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
