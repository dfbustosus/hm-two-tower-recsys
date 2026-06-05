"""Cutoff-safe training example export for two-tower retrieval challengers."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from hm_recsys.data.io import TransactionEvent
from hm_recsys.evaluation.temporal import TemporalSplit

TWO_TOWER_EXAMPLE_HEADER = (
    "customer_index",
    "article_index",
    "label",
    "customer_id",
    "article_id",
    "example_type",
    "anchor_t_dat",
    "positive_count",
    "negative_sample_rank",
    "positive_article_id_anchor",
)
TWO_TOWER_CUSTOMER_MAPPING_HEADER = ("customer_index", "customer_id")
TWO_TOWER_ARTICLE_MAPPING_HEADER = ("article_index", "article_id")
TwoTowerExportNegativeSampling = Literal["random"]
TwoTowerPositiveSelection = Literal["first", "latest", "latest_customer"]


@dataclass(frozen=True)
class TwoTowerExampleExportConfig:
    """Configuration for two-tower example export.

    Attributes:
        negatives_per_positive: Number of random negatives sampled per positive pair.
        seed: Non-negative random seed used for deterministic negative sampling.
        negative_sampling: Negative sampling strategy. The initial export supports
            random negatives only; harder strategies can be added after this data
            contract is stable.
        positive_selection: Strategy for positive exports. ``first`` keeps the
            earliest unique pairs in transaction-file order; ``latest`` keeps the
            most recently observed unique pairs before the cutoff;
            ``latest_customer`` keeps at most one latest positive pair per
            customer before applying the optional cap.
        max_positive_examples: Optional deterministic cap for smoke exports. When
            provided, the first unique pre-cutoff positive pairs in transaction-file
            order are exported, while negatives still exclude all pre-cutoff positives
            for selected customers.
    """

    negatives_per_positive: int = 1
    seed: int = 42
    negative_sampling: TwoTowerExportNegativeSampling = "random"
    positive_selection: TwoTowerPositiveSelection = "first"
    max_positive_examples: int | None = None

    def __post_init__(self) -> None:
        """Validate export configuration values.

        Raises:
            ValueError: If any numeric value or sampling strategy is invalid.
        """

        if self.negatives_per_positive < 0:
            raise ValueError("negatives_per_positive must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.negative_sampling != "random":
            raise ValueError(f"unsupported negative_sampling: {self.negative_sampling!r}")
        if self.positive_selection not in {"first", "latest", "latest_customer"}:
            raise ValueError(f"unsupported positive_selection: {self.positive_selection!r}")
        if self.max_positive_examples is not None and self.max_positive_examples <= 0:
            raise ValueError("max_positive_examples must be positive when provided")


@dataclass(frozen=True)
class PositivePairStats:
    """Aggregated positive-pair statistics for a customer/article pair.

    Attributes:
        count: Number of pre-cutoff purchases for this pair.
        last_seen: Latest pre-cutoff purchase date for this pair.
    """

    count: int
    last_seen: date


@dataclass(frozen=True)
class TwoTowerExampleExportSummary:
    """Summary metadata for a two-tower example export.

    Attributes:
        generated_at_utc: UTC timestamp for the export run.
        cutoff: Exclusive training cutoff date.
        validation_end_exclusive: Exclusive validation end implied by the split.
        horizon_days: Validation horizon in days.
        negative_sampling: Negative sampling strategy used by the export.
        negatives_per_positive: Requested negative rows per positive row.
        seed: Deterministic random seed.
        max_positive_examples: Optional positive-pair cap used for smoke exports.
        positive_selection: Positive-pair selection strategy used when capped.
        mapping_sort: Stable sort policy used for customer/article mappings.
        train_rows_seen: Number of pre-cutoff transaction rows scanned.
        positive_pairs_selected: Number of unique positive pairs exported.
        positive_examples_written: Positive rows written to the examples CSV.
        negative_examples_written: Negative rows written to the examples CSV.
        rows_written: Total data rows written to the examples CSV.
        unique_customers: Number of customers in the customer mapping.
        unique_articles: Number of known pre-cutoff articles in the article mapping.
        customers_without_negative_pool: Customers that could not receive all
            requested negatives because every known article was positive or already
            emitted as a negative.
        skipped_negative_examples: Requested negatives that could not be emitted.
        examples_path: Resolved examples CSV path.
        customer_mapping_path: Resolved customer mapping CSV path.
        article_mapping_path: Resolved article mapping CSV path.
        runtime_seconds: Wall-clock runtime for the export.
    """

    generated_at_utc: str
    cutoff: str
    validation_end_exclusive: str
    horizon_days: int
    negative_sampling: str
    negatives_per_positive: int
    seed: int
    max_positive_examples: int | None
    positive_selection: str
    mapping_sort: str
    train_rows_seen: int
    positive_pairs_selected: int
    positive_examples_written: int
    negative_examples_written: int
    rows_written: int
    unique_customers: int
    unique_articles: int
    customers_without_negative_pool: int
    skipped_negative_examples: int
    examples_path: str
    customer_mapping_path: str
    article_mapping_path: str
    runtime_seconds: float


def write_two_tower_example_export(
    transaction_iter_factory: Callable[[], Iterable[TransactionEvent]],
    split: TemporalSplit,
    examples_path: Path | str,
    customer_mapping_path: Path | str,
    article_mapping_path: Path | str,
    config: TwoTowerExampleExportConfig | None = None,
    progress_interval: int | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> TwoTowerExampleExportSummary:
    """Write cutoff-safe two-tower examples and stable ID mappings.

    Positive examples are unique pre-cutoff ``(customer_id, article_id)`` pairs
    with an explicit ``positive_count`` preserving repeat-purchase frequency.
    Random negatives are sampled from articles known before the cutoff, excluding
    every pre-cutoff positive article for the same selected customer. Transactions
    at or after ``split.cutoff`` never affect positives, mappings, or negatives.

    Args:
        transaction_iter_factory: Callable returning a fresh transaction iterable.
            A second pass is used only for capped smoke exports to collect complete
            positive exclusion sets for selected customers.
        split: Temporal split whose cutoff is the exclusive training boundary.
        examples_path: Destination examples CSV path.
        customer_mapping_path: Destination customer mapping CSV path.
        article_mapping_path: Destination article mapping CSV path.
        config: Export configuration. Defaults to one random negative per positive
            with seed 42.
        progress_interval: Optional transaction-row interval for progress callbacks.
        progress_callback: Optional callback receiving a phase name and scanned rows.

    Returns:
        Export summary with row counts, config, split, and artifact paths.

    Raises:
        ValueError: If progress configuration is invalid.
    """

    export_config = config or TwoTowerExampleExportConfig()
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError("progress_interval must be positive when provided")

    started_at = perf_counter()
    scan = _scan_pre_cutoff_transactions(
        transactions=transaction_iter_factory(),
        split=split,
        config=export_config,
        progress_interval=progress_interval,
        progress_callback=progress_callback,
    )
    customer_positive_items = scan.customer_positive_items
    if (
        export_config.max_positive_examples is not None
        or export_config.positive_selection == "latest_customer"
    ):
        customer_positive_items = _collect_selected_customer_positive_items(
            transactions=transaction_iter_factory(),
            split=split,
            selected_customers=set(scan.selected_customer_ids),
            progress_interval=progress_interval,
            progress_callback=progress_callback,
        )

    selected_customer_ids = tuple(sorted(scan.selected_customer_ids))
    known_article_ids = tuple(sorted(scan.known_article_ids))
    customer_to_index = {
        customer_id: index for index, customer_id in enumerate(selected_customer_ids)
    }
    article_to_index = {article_id: index for index, article_id in enumerate(known_article_ids)}

    resolved_examples_path = Path(examples_path).expanduser().resolve()
    resolved_customer_mapping_path = Path(customer_mapping_path).expanduser().resolve()
    resolved_article_mapping_path = Path(article_mapping_path).expanduser().resolve()
    _write_mapping(
        path=resolved_customer_mapping_path,
        header=TWO_TOWER_CUSTOMER_MAPPING_HEADER,
        ids=selected_customer_ids,
    )
    _write_mapping(
        path=resolved_article_mapping_path,
        header=TWO_TOWER_ARTICLE_MAPPING_HEADER,
        ids=known_article_ids,
    )
    write_result = _write_examples(
        path=resolved_examples_path,
        positive_pair_stats=scan.positive_pair_stats,
        customer_ids=selected_customer_ids,
        article_ids=known_article_ids,
        customer_to_index=customer_to_index,
        article_to_index=article_to_index,
        customer_positive_items=customer_positive_items,
        config=export_config,
    )

    return TwoTowerExampleExportSummary(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        cutoff=split.cutoff.isoformat(),
        validation_end_exclusive=split.validation_end.isoformat(),
        horizon_days=split.horizon_days,
        negative_sampling=export_config.negative_sampling,
        negatives_per_positive=export_config.negatives_per_positive,
        seed=export_config.seed,
        max_positive_examples=export_config.max_positive_examples,
        positive_selection=export_config.positive_selection,
        mapping_sort="lexicographic_id",
        train_rows_seen=scan.train_rows_seen,
        positive_pairs_selected=len(scan.positive_pair_stats),
        positive_examples_written=write_result.positive_examples_written,
        negative_examples_written=write_result.negative_examples_written,
        rows_written=write_result.rows_written,
        unique_customers=len(selected_customer_ids),
        unique_articles=len(known_article_ids),
        customers_without_negative_pool=write_result.customers_without_negative_pool,
        skipped_negative_examples=write_result.skipped_negative_examples,
        examples_path=str(resolved_examples_path),
        customer_mapping_path=str(resolved_customer_mapping_path),
        article_mapping_path=str(resolved_article_mapping_path),
        runtime_seconds=perf_counter() - started_at,
    )


def two_tower_example_export_summary_to_dict(
    summary: TwoTowerExampleExportSummary,
) -> dict[str, Any]:
    """Convert a two-tower export summary to serializable primitives.

    Args:
        summary: Summary object to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """

    return asdict(summary)


def write_two_tower_example_export_summary(
    summary: TwoTowerExampleExportSummary,
    path: Path | str,
) -> Path:
    """Write a two-tower example export summary as deterministic JSON.

    Args:
        summary: Export summary object to serialize.
        path: Destination JSON path.

    Returns:
        Resolved report path written to disk.
    """

    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(two_tower_example_export_summary_to_dict(summary), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report_path


@dataclass(frozen=True)
class _PreCutoffScan:
    """In-memory state collected from pre-cutoff transactions."""

    train_rows_seen: int
    known_article_ids: set[str]
    selected_customer_ids: set[str]
    positive_pair_stats: dict[tuple[str, str], PositivePairStats]
    customer_positive_items: dict[str, set[str]]


@dataclass(frozen=True)
class _ExampleWriteResult:
    """Row-count result from writing examples."""

    positive_examples_written: int
    negative_examples_written: int
    rows_written: int
    customers_without_negative_pool: int
    skipped_negative_examples: int


def _scan_pre_cutoff_transactions(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    config: TwoTowerExampleExportConfig,
    progress_interval: int | None,
    progress_callback: Callable[[str, int], None] | None,
) -> _PreCutoffScan:
    """Collect known articles and selected positive pairs before cutoff."""

    train_rows_seen = 0
    scanned_rows = 0
    known_article_ids: set[str] = set()
    selected_customer_ids: set[str] = set()
    positive_pair_stats: OrderedDict[tuple[str, str], PositivePairStats] = OrderedDict()
    customer_positive_items: dict[str, set[str]] = {}
    latest_customer_selection = config.positive_selection == "latest_customer"
    full_export = config.max_positive_examples is None and not latest_customer_selection
    latest_capped_export = (
        config.max_positive_examples is not None and config.positive_selection == "latest"
    )
    latest_customer_pair_stats: dict[str, tuple[str, PositivePairStats]] = {}

    for scanned_rows, transaction in enumerate(transactions, start=1):
        if transaction.t_dat >= split.cutoff:
            _maybe_report_progress(
                "two_tower_scan",
                scanned_rows,
                progress_interval,
                progress_callback,
            )
            continue
        train_rows_seen += 1
        known_article_ids.add(transaction.article_id)
        key = (transaction.customer_id, transaction.article_id)
        if latest_customer_selection:
            _update_latest_customer_positive_pair_stats(
                latest_customer_pair_stats,
                transaction.customer_id,
                transaction.article_id,
                transaction.t_dat,
            )
        elif latest_capped_export:
            _update_latest_positive_pair_stats(
                positive_pair_stats,
                key,
                transaction.t_dat,
                max_positive_examples=config.max_positive_examples or 0,
            )
        else:
            should_select = full_export or key in positive_pair_stats
            if not should_select and len(positive_pair_stats) < (config.max_positive_examples or 0):
                should_select = True
            if should_select:
                selected_customer_ids.add(transaction.customer_id)
                _update_positive_pair_stats(positive_pair_stats, key, transaction.t_dat)
        if full_export:
            customer_positive_items.setdefault(transaction.customer_id, set()).add(
                transaction.article_id
            )
        _maybe_report_progress("two_tower_scan", scanned_rows, progress_interval, progress_callback)
    _maybe_report_final_progress(
        "two_tower_scan",
        scanned_rows,
        progress_interval,
        progress_callback,
    )
    if latest_capped_export:
        selected_customer_ids = {customer_id for customer_id, _ in positive_pair_stats}
    if latest_customer_selection:
        positive_pair_stats = _select_latest_customer_positive_pair_stats(
            latest_customer_pair_stats,
            max_positive_examples=config.max_positive_examples,
        )
        selected_customer_ids = {customer_id for customer_id, _ in positive_pair_stats}

    return _PreCutoffScan(
        train_rows_seen=train_rows_seen,
        known_article_ids=known_article_ids,
        selected_customer_ids=selected_customer_ids,
        positive_pair_stats=positive_pair_stats,
        customer_positive_items=customer_positive_items,
    )


def _collect_selected_customer_positive_items(
    transactions: Iterable[TransactionEvent],
    split: TemporalSplit,
    selected_customers: set[str],
    progress_interval: int | None,
    progress_callback: Callable[[str, int], None] | None,
) -> dict[str, set[str]]:
    """Collect complete pre-cutoff positive item sets for selected customers."""

    customer_positive_items: dict[str, set[str]] = {
        customer_id: set() for customer_id in selected_customers
    }
    scanned_rows = 0
    for scanned_rows, transaction in enumerate(transactions, start=1):
        if transaction.t_dat < split.cutoff and transaction.customer_id in selected_customers:
            customer_positive_items[transaction.customer_id].add(transaction.article_id)
        _maybe_report_progress(
            "two_tower_selected_customer_scan",
            scanned_rows,
            progress_interval,
            progress_callback,
        )
    _maybe_report_final_progress(
        "two_tower_selected_customer_scan",
        scanned_rows,
        progress_interval,
        progress_callback,
    )
    return customer_positive_items


def _update_positive_pair_stats(
    positive_pair_stats: dict[tuple[str, str], PositivePairStats],
    key: tuple[str, str],
    seen_date: date,
) -> None:
    """Update count and recency for one selected positive pair."""

    current = positive_pair_stats.get(key)
    if current is None:
        positive_pair_stats[key] = PositivePairStats(count=1, last_seen=seen_date)
        return
    positive_pair_stats[key] = PositivePairStats(
        count=current.count + 1,
        last_seen=max(current.last_seen, seen_date),
    )


def _update_latest_positive_pair_stats(
    positive_pair_stats: OrderedDict[tuple[str, str], PositivePairStats],
    key: tuple[str, str],
    seen_date: date,
    max_positive_examples: int,
) -> None:
    """Update bounded latest-positive state, keeping newest unique pairs."""

    current = positive_pair_stats.pop(key, None)
    count = (current.count + 1) if current is not None else 1
    positive_pair_stats[key] = PositivePairStats(count=count, last_seen=seen_date)
    while len(positive_pair_stats) > max_positive_examples:
        positive_pair_stats.popitem(last=False)


def _update_latest_customer_positive_pair_stats(
    latest_customer_pair_stats: dict[str, tuple[str, PositivePairStats]],
    customer_id: str,
    article_id: str,
    seen_date: date,
) -> None:
    """Track the latest observed positive pair for one customer."""

    current = latest_customer_pair_stats.get(customer_id)
    if current is None:
        latest_customer_pair_stats[customer_id] = (
            article_id,
            PositivePairStats(count=1, last_seen=seen_date),
        )
        return

    current_article_id, current_stats = current
    if article_id == current_article_id:
        latest_customer_pair_stats[customer_id] = (
            article_id,
            PositivePairStats(
                count=current_stats.count + 1,
                last_seen=max(current_stats.last_seen, seen_date),
            ),
        )
        return
    if seen_date >= current_stats.last_seen:
        latest_customer_pair_stats[customer_id] = (
            article_id,
            PositivePairStats(count=1, last_seen=seen_date),
        )


def _select_latest_customer_positive_pair_stats(
    latest_customer_pair_stats: dict[str, tuple[str, PositivePairStats]],
    max_positive_examples: int | None,
) -> OrderedDict[tuple[str, str], PositivePairStats]:
    """Return selected latest-customer positive pairs with deterministic ordering."""

    ranked = sorted(
        latest_customer_pair_stats.items(),
        key=lambda item: (-item[1][1].last_seen.toordinal(), item[0], item[1][0]),
    )
    if max_positive_examples is not None:
        ranked = ranked[:max_positive_examples]
    selected = OrderedDict()
    for customer_id, (article_id, stats) in sorted(
        ranked,
        key=lambda item: (item[0], item[1][0]),
    ):
        selected[(customer_id, article_id)] = stats
    return selected


def _write_mapping(path: Path, header: tuple[str, str], ids: tuple[str, ...]) -> None:
    """Write an index-to-ID mapping CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, identifier in enumerate(ids):
            writer.writerow((index, identifier))


def _write_examples(
    path: Path,
    positive_pair_stats: dict[tuple[str, str], PositivePairStats],
    customer_ids: tuple[str, ...],
    article_ids: tuple[str, ...],
    customer_to_index: dict[str, int],
    article_to_index: dict[str, int],
    customer_positive_items: dict[str, set[str]],
    config: TwoTowerExampleExportConfig,
) -> _ExampleWriteResult:
    """Write positive and negative example rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    positive_examples_written = 0
    negative_examples_written = 0
    skipped_negative_examples = 0
    customers_without_negative_pool: set[str] = set()
    positive_articles_by_customer: dict[str, list[str]] = {
        customer_id: [] for customer_id in customer_ids
    }
    for customer_id, article_id in sorted(positive_pair_stats):
        positive_articles_by_customer.setdefault(customer_id, []).append(article_id)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TWO_TOWER_EXAMPLE_HEADER)
        for customer_id in customer_ids:
            positive_articles = tuple(positive_articles_by_customer.get(customer_id, ()))
            emitted_negative_articles: set[str] = set()
            for article_id in positive_articles:
                stats = positive_pair_stats[(customer_id, article_id)]
                writer.writerow(
                    _positive_example_row(
                        customer_id=customer_id,
                        article_id=article_id,
                        customer_to_index=customer_to_index,
                        article_to_index=article_to_index,
                        stats=stats,
                    )
                )
                positive_examples_written += 1
                negative_article_ids = _sample_negative_article_ids(
                    article_ids=article_ids,
                    customer_positive_items=customer_positive_items.get(customer_id, set()),
                    emitted_negative_articles=emitted_negative_articles,
                    customer_id=customer_id,
                    positive_article_id=article_id,
                    seed=config.seed,
                    count=config.negatives_per_positive,
                )
                if len(negative_article_ids) < config.negatives_per_positive:
                    customers_without_negative_pool.add(customer_id)
                    skipped_negative_examples += config.negatives_per_positive - len(
                        negative_article_ids
                    )
                for negative_rank, negative_article_id in enumerate(
                    negative_article_ids,
                    start=1,
                ):
                    emitted_negative_articles.add(negative_article_id)
                    writer.writerow(
                        _negative_example_row(
                            customer_id=customer_id,
                            article_id=negative_article_id,
                            positive_article_id=article_id,
                            anchor_t_dat=stats.last_seen,
                            negative_rank=negative_rank,
                            customer_to_index=customer_to_index,
                            article_to_index=article_to_index,
                        )
                    )
                    negative_examples_written += 1

    rows_written = positive_examples_written + negative_examples_written
    return _ExampleWriteResult(
        positive_examples_written=positive_examples_written,
        negative_examples_written=negative_examples_written,
        rows_written=rows_written,
        customers_without_negative_pool=len(customers_without_negative_pool),
        skipped_negative_examples=skipped_negative_examples,
    )


def _positive_example_row(
    customer_id: str,
    article_id: str,
    customer_to_index: dict[str, int],
    article_to_index: dict[str, int],
    stats: PositivePairStats,
) -> tuple[str, ...]:
    """Build one positive example CSV row."""

    return (
        str(customer_to_index[customer_id]),
        str(article_to_index[article_id]),
        "1",
        customer_id,
        article_id,
        "positive",
        stats.last_seen.isoformat(),
        str(stats.count),
        "",
        "",
    )


def _negative_example_row(
    customer_id: str,
    article_id: str,
    positive_article_id: str,
    anchor_t_dat: date,
    negative_rank: int,
    customer_to_index: dict[str, int],
    article_to_index: dict[str, int],
) -> tuple[str, ...]:
    """Build one random-negative example CSV row."""

    return (
        str(customer_to_index[customer_id]),
        str(article_to_index[article_id]),
        "0",
        customer_id,
        article_id,
        "random_negative",
        anchor_t_dat.isoformat(),
        "0",
        str(negative_rank),
        positive_article_id,
    )


def _sample_negative_article_ids(
    article_ids: tuple[str, ...],
    customer_positive_items: set[str],
    emitted_negative_articles: set[str],
    customer_id: str,
    positive_article_id: str,
    seed: int,
    count: int,
) -> tuple[str, ...]:
    """Sample deterministic random negatives for one positive anchor."""

    if count == 0 or not article_ids:
        return ()
    selected: list[str] = []
    selected_set: set[str] = set()
    rng = random.Random(_stable_pair_seed(seed, customer_id, positive_article_id))
    max_attempts = max(100, count * 100)
    attempts = 0
    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        candidate = article_ids[rng.randrange(len(article_ids))]
        if _is_forbidden_negative(
            candidate,
            customer_positive_items,
            emitted_negative_articles,
            selected_set,
        ):
            continue
        selected.append(candidate)
        selected_set.add(candidate)

    if len(selected) < count:
        start_index = rng.randrange(len(article_ids))
        for offset in range(len(article_ids)):
            candidate = article_ids[(start_index + offset) % len(article_ids)]
            if _is_forbidden_negative(
                candidate,
                customer_positive_items,
                emitted_negative_articles,
                selected_set,
            ):
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if len(selected) == count:
                break
    return tuple(selected)


def _is_forbidden_negative(
    article_id: str,
    customer_positive_items: set[str],
    emitted_negative_articles: set[str],
    selected_negative_articles: set[str],
) -> bool:
    """Return whether an article is invalid as a negative for the current customer."""

    return (
        article_id in customer_positive_items
        or article_id in emitted_negative_articles
        or article_id in selected_negative_articles
    )


def _stable_pair_seed(seed: int, customer_id: str, article_id: str) -> int:
    """Build a deterministic integer seed for a customer/positive pair."""

    payload = f"{seed}|{customer_id}|{article_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _maybe_report_progress(
    phase: str,
    scanned_rows: int,
    progress_interval: int | None,
    progress_callback: Callable[[str, int], None] | None,
) -> None:
    """Invoke a progress callback at configured transaction-row intervals."""

    if (
        progress_callback is not None
        and progress_interval is not None
        and scanned_rows % progress_interval == 0
    ):
        progress_callback(phase, scanned_rows)


def _maybe_report_final_progress(
    phase: str,
    scanned_rows: int,
    progress_interval: int | None,
    progress_callback: Callable[[str, int], None] | None,
) -> None:
    """Invoke a final progress callback when the last row is not on an interval."""

    if (
        progress_callback is not None
        and progress_interval is not None
        and scanned_rows
        and scanned_rows % progress_interval != 0
    ):
        progress_callback(phase, scanned_rows)
