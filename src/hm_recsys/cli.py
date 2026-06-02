from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hm_recsys.data.contracts import validate_hm_data_contract, write_data_contract_report
from hm_recsys.data.io import (
    iter_transaction_events,
    load_article_ids,
    load_submission_customer_ids,
)
from hm_recsys.evaluation.submission import (
    validate_submission_file,
    write_submission_validation_report,
)
from hm_recsys.evaluation.temporal import (
    TemporalSplit,
    summarize_temporal_split,
    write_temporal_split_summary,
)
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.retrieval.baselines import (
    evaluate_repeat_popularity_baseline,
    write_baseline_evaluation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hm-recsys")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-data-contract",
        help="Validate local H&M raw data files and write a JSON report.",
    )
    validate_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to auto-detection from the current directory.",
    )
    validate_parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help=(
            "Raw H&M data directory. Defaults to "
            "data/raw/h-and-m-personalized-fashion-recommendations/."
        ),
    )
    validate_parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Report path. Defaults to artifacts/data-contract/data_contract_report.json.",
    )
    validate_parser.set_defaults(handler=_handle_validate_data_contract)

    split_parser = subparsers.add_parser(
        "summarize-temporal-split",
        help="Summarize train/validation row counts for a cutoff-based split.",
    )
    split_parser.add_argument("--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD.")
    split_parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="Validation horizon in days. Defaults to 7.",
    )
    split_parser.add_argument("--project-root", type=Path, default=None)
    split_parser.add_argument("--raw-data-dir", type=Path, default=None)
    split_parser.add_argument("--report-path", type=Path, default=None)
    split_parser.set_defaults(handler=_handle_summarize_temporal_split)

    submission_parser = subparsers.add_parser(
        "validate-submission",
        help="Validate a Kaggle submission CSV against sample_submission and articles.",
    )
    submission_parser.add_argument("--submission-path", type=Path, required=True)
    submission_parser.add_argument("--project-root", type=Path, default=None)
    submission_parser.add_argument("--raw-data-dir", type=Path, default=None)
    submission_parser.add_argument("--report-path", type=Path, default=None)
    submission_parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow fewer than 12 predictions per customer.",
    )
    submission_parser.set_defaults(handler=_handle_validate_submission)

    baseline_parser = subparsers.add_parser(
        "evaluate-baseline",
        help="Evaluate repeat-purchase plus recent-popularity baseline on a temporal split.",
    )
    baseline_parser.add_argument(
        "--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD."
    )
    baseline_parser.add_argument("--horizon-days", type=int, default=7)
    baseline_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    baseline_parser.add_argument("--k", type=int, default=12)
    baseline_parser.add_argument("--project-root", type=Path, default=None)
    baseline_parser.add_argument("--raw-data-dir", type=Path, default=None)
    baseline_parser.add_argument("--report-path", type=Path, default=None)
    baseline_parser.set_defaults(handler=_handle_evaluate_baseline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_validate_data_contract(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    report_path = _resolve_report_path(paths, args.report_path)
    report = validate_hm_data_contract(paths.raw_data_dir)
    written_report_path = write_data_contract_report(report, report_path)

    print(f"Raw data directory: {paths.raw_data_dir}")
    print(f"Data contract valid: {report.valid}")
    print(f"Report written to: {written_report_path}")
    for result in report.files:
        status = "valid" if result.valid else "invalid"
        print(f"- {result.file_name}: {status}, rows={result.row_count}")
        for failure in result.failures:
            print(f"  - {failure}")
    if report.optional_images["exists"] and not report.optional_images["is_directory"]:
        print(f"- images: invalid optional path {report.optional_images['path']}")
    return 0 if report.valid else 1


def _handle_summarize_temporal_split(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    report_path = args.report_path or paths.temporal_split_report_path(args.cutoff)
    if not report_path.is_absolute():
        report_path = paths.root / report_path

    summary = summarize_temporal_split(iter_transaction_events(paths.raw_data_dir), split)
    written_report_path = write_temporal_split_summary(summary, report_path)
    print(f"Raw data directory: {paths.raw_data_dir}")
    print(f"Cutoff: {summary.cutoff}")
    print(f"Validation end exclusive: {summary.validation_end_exclusive}")
    print(f"Train rows: {summary.train_rows}")
    print(f"Validation rows: {summary.validation_rows}")
    print(f"Future rows: {summary.future_rows}")
    print(f"Report written to: {written_report_path}")
    return 0


def _handle_validate_submission(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    report_path = args.report_path or paths.submission_validation_report_path(args.submission_path)
    if not report_path.is_absolute():
        report_path = paths.root / report_path

    result = validate_submission_file(
        submission_path=args.submission_path,
        expected_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        valid_article_ids=load_article_ids(paths.raw_data_dir),
        require_full_length=not args.allow_short,
    )
    written_report_path = write_submission_validation_report(result, report_path)
    print(f"Submission valid: {result.valid}")
    print(f"Rows: {result.row_count}")
    print(f"Expected customers: {result.expected_customer_count}")
    print(f"Report written to: {written_report_path}")
    for failure in result.failures:
        print(f"- {failure}")
    return 0 if result.valid else 1


def _handle_evaluate_baseline(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    report_path = args.report_path or paths.baseline_report_path(
        cutoff=args.cutoff,
        lookback_days=args.popularity_lookback_days,
        k=args.k,
    )
    if not report_path.is_absolute():
        report_path = paths.root / report_path

    report = evaluate_repeat_popularity_baseline(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        k=args.k,
        popularity_lookback_days=args.popularity_lookback_days,
    )
    written_report_path = write_baseline_evaluation_report(report, report_path)
    print(f"Cutoff: {report.cutoff}")
    print(f"Validation end exclusive: {report.validation_end_exclusive}")
    print(f"MAP@{report.k}: {report.map_at_k:.8f}")
    print(f"Recall@{report.k}: {report.recall_at_k:.8f}")
    print(f"Evaluated customers: {report.diagnostics.evaluated_customers}")
    print(
        "Full-length prediction coverage: "
        f"{report.diagnostics.customers_with_full_length_predictions}/"
        f"{report.diagnostics.evaluated_customers}"
    )
    print(f"Duplicate prediction rows: {report.diagnostics.duplicate_prediction_rows}")
    print(f"Report written to: {written_report_path}")
    return 0


def _resolve_report_path(paths: ProjectPaths, report_path: Path | None) -> Path:
    if report_path is None:
        return paths.data_contract_report_path
    expanded = report_path.expanduser()
    if expanded.is_absolute():
        return expanded
    return paths.root / expanded


if __name__ == "__main__":
    raise SystemExit(main())
