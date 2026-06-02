"""Command-line interface for H&M recommender validation utilities."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hm_recsys.data.contracts import validate_hm_data_contract, write_data_contract_report
from hm_recsys.data.io import (
    iter_transaction_events,
    load_article_ids,
    load_submission_customer_ids,
    load_submission_customer_ids_in_order,
)
from hm_recsys.evaluation.submission import (
    validate_submission_file,
    write_submission_file,
    write_submission_validation_report,
)
from hm_recsys.evaluation.temporal import (
    TemporalSplit,
    summarize_temporal_split,
    write_temporal_split_summary,
)
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.retrieval.baselines import (
    build_repeat_popularity_submission_baseline,
    evaluate_repeat_popularity_baseline,
    write_baseline_evaluation_report,
)
from hm_recsys.retrieval.candidate_diagnostics import (
    DEFAULT_EVALUATION_KS,
    evaluate_baseline_candidate_diagnostics,
    write_candidate_diagnostics_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``hm-recsys`` argument parser.

    Returns:
        Configured ``ArgumentParser`` with all supported subcommands.
    """

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

    baseline_submission_parser = subparsers.add_parser(
        "generate-baseline-submission",
        help="Generate and validate a repeat-purchase plus popularity submission CSV.",
    )
    baseline_submission_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    baseline_submission_parser.add_argument("--k", type=int, default=12)
    baseline_submission_parser.add_argument("--project-root", type=Path, default=None)
    baseline_submission_parser.add_argument("--raw-data-dir", type=Path, default=None)
    baseline_submission_parser.add_argument("--output-path", type=Path, default=None)
    baseline_submission_parser.add_argument("--validation-report-path", type=Path, default=None)
    baseline_submission_parser.set_defaults(handler=_handle_generate_baseline_submission)

    candidate_parser = subparsers.add_parser(
        "candidate-diagnostics",
        help="Evaluate repeat and popularity candidate-source diagnostics.",
    )
    candidate_parser.add_argument(
        "--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD."
    )
    candidate_parser.add_argument("--horizon-days", type=int, default=7)
    candidate_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    candidate_parser.add_argument(
        "--evaluation-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVALUATION_KS),
        help="Candidate recall cutoffs. Defaults to 12 50 100.",
    )
    candidate_parser.add_argument("--project-root", type=Path, default=None)
    candidate_parser.add_argument("--raw-data-dir", type=Path, default=None)
    candidate_parser.add_argument("--report-path", type=Path, default=None)
    candidate_parser.set_defaults(handler=_handle_candidate_diagnostics)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code from the selected command handler.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_validate_data_contract(args: argparse.Namespace) -> int:
    """Handle the ``validate-data-contract`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` when the data contract is valid; otherwise ``1``.
    """

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
    """Handle the ``summarize-temporal-split`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the temporal split report.
    """

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
    """Handle the ``validate-submission`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` when the submission is valid; otherwise ``1``.
    """

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
    """Handle the ``evaluate-baseline`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after evaluating the baseline and writing its report.
    """

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
        target_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        k=args.k,
        popularity_lookback_days=args.popularity_lookback_days,
    )
    written_report_path = write_baseline_evaluation_report(report, report_path)
    print(f"Cutoff: {report.cutoff}")
    print(f"Validation end exclusive: {report.validation_end_exclusive}")
    print(f"MAP@{report.k}: {report.map_at_k:.8f}")
    print(f"Recall@{report.k}: {report.recall_at_k:.8f}")
    print(f"Prediction target customers: {report.diagnostics.target_customers}")
    print(f"Evaluated label customers: {report.diagnostics.evaluated_customers}")
    print(
        "Full-length prediction coverage: "
        f"{report.diagnostics.customers_with_full_length_predictions}/"
        f"{report.diagnostics.target_customers}"
    )
    print(f"Duplicate prediction rows: {report.diagnostics.duplicate_prediction_rows}")
    print(f"Report written to: {written_report_path}")
    return 0


def _handle_generate_baseline_submission(args: argparse.Namespace) -> int:
    """Handle the ``generate-baseline-submission`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` when the generated submission passes validation; otherwise ``1``.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    output_path = args.output_path or paths.baseline_submission_path(
        lookback_days=args.popularity_lookback_days,
        k=args.k,
    )
    output_path = _resolve_path_under_root(paths, output_path)

    target_customer_ids = load_submission_customer_ids_in_order(paths.raw_data_dir)
    submission = build_repeat_popularity_submission_baseline(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        target_customer_ids=target_customer_ids,
        k=args.k,
        popularity_lookback_days=args.popularity_lookback_days,
    )
    written_submission_path = write_submission_file(
        predictions_by_customer=submission.predictions.predictions,
        customer_ids=target_customer_ids,
        path=output_path,
        max_predictions=args.k,
    )

    validation_report_path = args.validation_report_path or paths.submission_validation_report_path(
        written_submission_path
    )
    validation_report_path = _resolve_path_under_root(paths, validation_report_path)
    validation_result = validate_submission_file(
        submission_path=written_submission_path,
        expected_customer_ids=set(target_customer_ids),
        valid_article_ids=load_article_ids(paths.raw_data_dir),
        max_predictions=args.k,
        require_full_length=True,
    )
    written_validation_report_path = write_submission_validation_report(
        validation_result,
        validation_report_path,
    )

    print(f"Max transaction date: {submission.max_transaction_date.isoformat()}")
    print(f"Training cutoff: {submission.training_cutoff.isoformat()}")
    print(f"Prediction target customers: {submission.diagnostics.target_customers}")
    print(
        "Full-length prediction coverage: "
        f"{submission.diagnostics.customers_with_full_length_predictions}/"
        f"{submission.diagnostics.target_customers}"
    )
    print(f"Duplicate prediction rows: {submission.diagnostics.duplicate_prediction_rows}")
    print(f"Submission written to: {written_submission_path}")
    print(f"Submission valid: {validation_result.valid}")
    print(f"Validation report written to: {written_validation_report_path}")
    for failure in validation_result.failures:
        print(f"- {failure}")
    return 0 if validation_result.valid else 1


def _handle_candidate_diagnostics(args: argparse.Namespace) -> int:
    """Handle the ``candidate-diagnostics`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the candidate diagnostics report.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    max_k = max(set(args.evaluation_ks))
    report_path = args.report_path or paths.candidate_diagnostics_report_path(
        cutoff=args.cutoff,
        lookback_days=args.popularity_lookback_days,
        max_k=max_k,
    )
    report_path = _resolve_path_under_root(paths, report_path)

    report = evaluate_baseline_candidate_diagnostics(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        target_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        popularity_lookback_days=args.popularity_lookback_days,
        evaluation_ks=args.evaluation_ks,
    )
    written_report_path = write_candidate_diagnostics_report(report, report_path)
    print(f"Cutoff: {report.cutoff}")
    print(f"Validation end exclusive: {report.validation_end_exclusive}")
    print(f"Target customers: {report.target_customers}")
    print(f"Evaluated label customers: {report.evaluated_customers}")
    for source in report.sources:
        recall_summary = ", ".join(
            f"recall@{k}={value:.6f}" for k, value in source.recall_at_k.items()
        )
        print(
            f"- {source.source}: MAP@12={source.map_at_12:.8f}, "
            f"{recall_summary}, coverage={source.candidate_coverage:.6f}"
        )
    print(f"Report written to: {written_report_path}")
    return 0


def _resolve_report_path(paths: ProjectPaths, report_path: Path | None) -> Path:
    """Resolve an optional report path against project paths.

    Args:
        paths: Canonical project paths.
        report_path: Optional CLI-provided report path.

    Returns:
        Absolute report path. Defaults to the data-contract report path when no
        override is provided.
    """

    if report_path is None:
        return paths.data_contract_report_path
    expanded = report_path.expanduser()
    if expanded.is_absolute():
        return expanded
    return paths.root / expanded


def _resolve_path_under_root(paths: ProjectPaths, path: Path) -> Path:
    """Resolve a CLI path against the project root when it is relative.

    Args:
        paths: Canonical project paths.
        path: Candidate CLI path.

    Returns:
        Absolute path for writing or reading.
    """

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return paths.root / expanded


if __name__ == "__main__":
    raise SystemExit(main())
