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
    summarize_temporal_split_with_labels,
    write_temporal_split_summary,
)
from hm_recsys.infrastructure.paths import ProjectPaths
from hm_recsys.ranking.deterministic import (
    evaluate_deterministic_ranker_from_csv,
    write_deterministic_ranker_report,
)
from hm_recsys.ranking.linear import (
    LinearRankerConfig,
    build_learned_linear_ranker_report,
    evaluate_linear_ranker_from_csv,
    previous_window_split,
    train_linear_ranker_from_csv,
    write_learned_linear_ranker_report,
)
from hm_recsys.ranking.rolling import (
    build_rolling_ranker_validation_report,
    write_rolling_ranker_validation_report,
)
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
from hm_recsys.retrieval.candidate_export import (
    CandidateExportSummary,
    select_validation_label_customer_ids,
    write_candidate_export_summary,
    write_validation_candidate_export,
)
from hm_recsys.retrieval.co_visitation import (
    DEFAULT_MAX_HISTORY_ITEMS,
    DEFAULT_MAX_NEIGHBORS_PER_ITEM,
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
    candidate_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate diagnostics.",
    )
    candidate_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    candidate_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    candidate_parser.add_argument("--project-root", type=Path, default=None)
    candidate_parser.add_argument("--raw-data-dir", type=Path, default=None)
    candidate_parser.add_argument("--report-path", type=Path, default=None)
    candidate_parser.set_defaults(handler=_handle_candidate_diagnostics)

    candidate_export_parser = subparsers.add_parser(
        "export-candidates",
        help="Export ranker-ready candidate source rows for validation-label customers.",
    )
    candidate_export_parser.add_argument(
        "--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD."
    )
    candidate_export_parser.add_argument("--horizon-days", type=int, default=7)
    candidate_export_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    candidate_export_parser.add_argument("--k", type=int, default=12)
    candidate_export_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    candidate_export_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    candidate_export_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    candidate_export_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke exports.",
    )
    candidate_export_parser.add_argument("--project-root", type=Path, default=None)
    candidate_export_parser.add_argument("--raw-data-dir", type=Path, default=None)
    candidate_export_parser.add_argument("--output-path", type=Path, default=None)
    candidate_export_parser.add_argument("--report-path", type=Path, default=None)
    candidate_export_parser.set_defaults(handler=_handle_export_candidates)

    ranker_parser = subparsers.add_parser(
        "evaluate-ranker-baseline",
        help="Evaluate deterministic source-feature ranker on temporal validation.",
    )
    ranker_parser.add_argument(
        "--cutoff", required=True, help="Validation cutoff date, YYYY-MM-DD."
    )
    ranker_parser.add_argument("--horizon-days", type=int, default=7)
    ranker_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    ranker_parser.add_argument("--candidate-k", type=int, default=12)
    ranker_parser.add_argument("--k", type=int, default=12)
    ranker_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    ranker_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    ranker_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    ranker_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap for smoke evaluations.",
    )
    ranker_parser.add_argument("--project-root", type=Path, default=None)
    ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    ranker_parser.add_argument("--candidate-output-path", type=Path, default=None)
    ranker_parser.add_argument("--candidate-report-path", type=Path, default=None)
    ranker_parser.add_argument("--report-path", type=Path, default=None)
    ranker_parser.set_defaults(handler=_handle_evaluate_ranker_baseline)

    learned_ranker_parser = subparsers.add_parser(
        "evaluate-learned-ranker-baseline",
        help="Train on the previous window and evaluate a learned linear ranker.",
    )
    learned_ranker_parser.add_argument(
        "--cutoff", required=True, help="Evaluation cutoff date, YYYY-MM-DD."
    )
    learned_ranker_parser.add_argument(
        "--train-cutoff",
        default=None,
        help="Optional training-label cutoff. Defaults to previous non-overlapping window.",
    )
    learned_ranker_parser.add_argument("--horizon-days", type=int, default=7)
    learned_ranker_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    learned_ranker_parser.add_argument("--candidate-k", type=int, default=12)
    learned_ranker_parser.add_argument("--k", type=int, default=12)
    learned_ranker_parser.add_argument("--epochs", type=int, default=3)
    learned_ranker_parser.add_argument("--learning-rate", type=float, default=0.01)
    learned_ranker_parser.add_argument("--l2", type=float, default=0.001)
    learned_ranker_parser.add_argument("--positive-weight", type=float, default=None)
    learned_ranker_parser.add_argument("--max-auto-positive-weight", type=float, default=10.0)
    learned_ranker_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    learned_ranker_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    learned_ranker_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    learned_ranker_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap applied separately to train and eval windows.",
    )
    learned_ranker_parser.add_argument("--project-root", type=Path, default=None)
    learned_ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    learned_ranker_parser.add_argument("--train-candidate-output-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--eval-candidate-output-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--train-candidate-report-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--eval-candidate-report-path", type=Path, default=None)
    learned_ranker_parser.add_argument("--report-path", type=Path, default=None)
    learned_ranker_parser.set_defaults(handler=_handle_evaluate_learned_ranker_baseline)

    rolling_ranker_parser = subparsers.add_parser(
        "rolling-ranker-validation",
        help="Evaluate learned and deterministic rankers across rolling temporal windows.",
    )
    rolling_ranker_parser.add_argument(
        "--cutoffs",
        nargs="+",
        required=True,
        help="Evaluation cutoff dates, e.g. 2020-09-02 2020-09-09 2020-09-16.",
    )
    rolling_ranker_parser.add_argument("--horizon-days", type=int, default=7)
    rolling_ranker_parser.add_argument("--popularity-lookback-days", type=int, default=7)
    rolling_ranker_parser.add_argument("--candidate-k", type=int, default=12)
    rolling_ranker_parser.add_argument("--k", type=int, default=12)
    rolling_ranker_parser.add_argument("--epochs", type=int, default=3)
    rolling_ranker_parser.add_argument("--learning-rate", type=float, default=0.01)
    rolling_ranker_parser.add_argument("--l2", type=float, default=0.001)
    rolling_ranker_parser.add_argument("--positive-weight", type=float, default=None)
    rolling_ranker_parser.add_argument("--max-auto-positive-weight", type=float, default=10.0)
    rolling_ranker_parser.add_argument(
        "--no-co-visitation",
        action="store_true",
        help="Disable co-visitation candidate rows.",
    )
    rolling_ranker_parser.add_argument(
        "--co-visitation-max-history-items",
        type=int,
        default=DEFAULT_MAX_HISTORY_ITEMS,
        help=(
            "Recent unique articles per customer for co-visitation. "
            f"Defaults to {DEFAULT_MAX_HISTORY_ITEMS}."
        ),
    )
    rolling_ranker_parser.add_argument(
        "--co-visitation-max-neighbors-per-item",
        type=int,
        default=DEFAULT_MAX_NEIGHBORS_PER_ITEM,
        help=(
            "Neighbors retained per source article. "
            f"Defaults to {DEFAULT_MAX_NEIGHBORS_PER_ITEM}."
        ),
    )
    rolling_ranker_parser.add_argument(
        "--max-target-customers",
        type=int,
        default=None,
        help="Optional deterministic cap applied separately to all rolling windows.",
    )
    rolling_ranker_parser.add_argument("--project-root", type=Path, default=None)
    rolling_ranker_parser.add_argument("--raw-data-dir", type=Path, default=None)
    rolling_ranker_parser.add_argument("--report-path", type=Path, default=None)
    rolling_ranker_parser.set_defaults(handler=_handle_rolling_ranker_validation)
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
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
    )
    report_path = _resolve_path_under_root(paths, report_path)

    report = evaluate_baseline_candidate_diagnostics(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        target_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        popularity_lookback_days=args.popularity_lookback_days,
        evaluation_ks=args.evaluation_ks,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
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


def _handle_export_candidates(args: argparse.Namespace) -> int:
    """Handle the ``export-candidates`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the candidate CSV and summary JSON.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    output_path = args.output_path or paths.candidate_export_path(
        cutoff=args.cutoff,
        k=args.k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        max_target_customers=args.max_target_customers,
    )
    output_path = _resolve_path_under_root(paths, output_path)
    report_path = args.report_path or paths.candidate_export_report_path(output_path)
    report_path = _resolve_path_under_root(paths, report_path)

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        submission_customer_ids=load_submission_customer_ids(paths.raw_data_dir),
        output_path=output_path,
        k=args.k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        max_target_customers=args.max_target_customers,
    )
    written_report_path = write_candidate_export_summary(summary, report_path)
    print(f"Cutoff: {summary.cutoff}")
    print(f"Validation end exclusive: {summary.validation_end_exclusive}")
    print(f"Target scope: {summary.target_scope}")
    print(f"Target customers: {summary.target_customers}")
    print(f"Rows written: {summary.rows_written}")
    for source, row_count in summary.source_row_counts.items():
        print(f"- {source}: rows={row_count}")
    print(f"Candidate CSV written to: {summary.output_path}")
    print(f"Summary report written to: {written_report_path}")
    return 0


def _handle_evaluate_ranker_baseline(args: argparse.Namespace) -> int:
    """Handle the ``evaluate-ranker-baseline`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing candidate export artifacts and ranker report.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    candidate_output_path = args.candidate_output_path or paths.candidate_export_path(
        cutoff=args.cutoff,
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        max_target_customers=args.max_target_customers,
    )
    candidate_output_path = _resolve_path_under_root(paths, candidate_output_path)
    candidate_report_path = args.candidate_report_path or paths.candidate_export_report_path(
        candidate_output_path
    )
    candidate_report_path = _resolve_path_under_root(paths, candidate_report_path)
    ranker_report_path = args.report_path or paths.ranker_baseline_report_path(
        cutoff=args.cutoff,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
    )
    ranker_report_path = _resolve_path_under_root(paths, ranker_report_path)

    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    export_summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        submission_customer_ids=submission_customer_ids,
        output_path=candidate_output_path,
        k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        max_target_customers=args.max_target_customers,
    )
    written_candidate_report_path = write_candidate_export_summary(
        export_summary, candidate_report_path
    )

    validation_data = summarize_temporal_split_with_labels(
        iter_transaction_events(paths.raw_data_dir), split
    )
    target_customer_ids = select_validation_label_customer_ids(
        validation_labels=validation_data.validation_labels,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_target_customers,
    )
    validation_labels = {
        customer_id: validation_data.validation_labels[customer_id]
        for customer_id in target_customer_ids
    }
    ranker_report = evaluate_deterministic_ranker_from_csv(
        candidate_path=candidate_output_path,
        validation_labels=validation_labels,
        split=split,
        k=args.k,
    )
    written_ranker_report_path = write_deterministic_ranker_report(
        ranker_report, ranker_report_path
    )
    print(f"Cutoff: {ranker_report.cutoff}")
    print(f"Validation end exclusive: {ranker_report.validation_end_exclusive}")
    print(f"Candidate rows: {ranker_report.candidate_rows}")
    print(f"Unique candidate pairs: {ranker_report.unique_candidate_pairs}")
    print(f"Evaluated customers: {ranker_report.evaluated_customers}")
    print(f"Ranker MAP@{ranker_report.k}: {ranker_report.map_at_k:.8f}")
    print(f"Baseline MAP@{ranker_report.k}: {ranker_report.baseline_map_at_k:.8f}")
    print(f"Delta MAP@{ranker_report.k}: {ranker_report.delta_map_at_k:.8f}")
    print(f"Ranker recall@{ranker_report.k}: {ranker_report.recall_at_k:.8f}")
    print(f"Candidate CSV written to: {export_summary.output_path}")
    print(f"Candidate summary written to: {written_candidate_report_path}")
    print(f"Ranker report written to: {written_ranker_report_path}")
    return 0


def _handle_evaluate_learned_ranker_baseline(args: argparse.Namespace) -> int:
    """Handle the ``evaluate-learned-ranker-baseline`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after training/evaluating and writing artifacts.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    evaluation_split = TemporalSplit.from_isoformat(args.cutoff, horizon_days=args.horizon_days)
    train_split = (
        TemporalSplit.from_isoformat(args.train_cutoff, horizon_days=args.horizon_days)
        if args.train_cutoff is not None
        else previous_window_split(evaluation_split)
    )
    if train_split.validation_end > evaluation_split.cutoff:
        raise ValueError("training label window must end before the evaluation cutoff")

    train_candidate_path = args.train_candidate_output_path or paths.candidate_export_path(
        cutoff=train_split.cutoff.isoformat(),
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        max_target_customers=args.max_target_customers,
    )
    eval_candidate_path = args.eval_candidate_output_path or paths.candidate_export_path(
        cutoff=args.cutoff,
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        max_target_customers=args.max_target_customers,
    )
    train_candidate_path = _resolve_path_under_root(paths, train_candidate_path)
    eval_candidate_path = _resolve_path_under_root(paths, eval_candidate_path)
    train_candidate_report_path = (
        args.train_candidate_report_path or paths.candidate_export_report_path(train_candidate_path)
    )
    eval_candidate_report_path = (
        args.eval_candidate_report_path or paths.candidate_export_report_path(eval_candidate_path)
    )
    train_candidate_report_path = _resolve_path_under_root(paths, train_candidate_report_path)
    eval_candidate_report_path = _resolve_path_under_root(paths, eval_candidate_report_path)
    report_path = args.report_path or paths.learned_ranker_baseline_report_path(
        train_cutoff=train_split.cutoff.isoformat(),
        evaluation_cutoff=args.cutoff,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
        config_slug=_learned_ranker_config_slug(args),
    )
    report_path = _resolve_path_under_root(paths, report_path)

    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    train_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=train_split,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_target_customers,
    )
    eval_labels = _validation_labels_for_split(
        raw_data_dir=paths.raw_data_dir,
        split=evaluation_split,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=args.max_target_customers,
    )

    train_export_summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=train_split,
        submission_customer_ids=submission_customer_ids,
        output_path=train_candidate_path,
        k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        max_target_customers=args.max_target_customers,
    )
    eval_export_summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=evaluation_split,
        submission_customer_ids=submission_customer_ids,
        output_path=eval_candidate_path,
        k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        max_target_customers=args.max_target_customers,
    )
    written_train_candidate_report_path = write_candidate_export_summary(
        train_export_summary, train_candidate_report_path
    )
    written_eval_candidate_report_path = write_candidate_export_summary(
        eval_export_summary, eval_candidate_report_path
    )

    config = LinearRankerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        positive_weight=args.positive_weight,
        max_auto_positive_weight=args.max_auto_positive_weight,
    )
    training_result = train_linear_ranker_from_csv(
        candidate_path=train_candidate_path,
        validation_labels=train_labels,
        config=config,
    )
    evaluation = evaluate_linear_ranker_from_csv(
        candidate_path=eval_candidate_path,
        validation_labels=eval_labels,
        model=training_result.model,
        k=args.k,
    )
    report = build_learned_linear_ranker_report(
        train_split=train_split,
        evaluation_split=evaluation_split,
        k=args.k,
        candidate_k=args.candidate_k,
        config=config,
        training_result=training_result,
        evaluation=evaluation,
    )
    written_report_path = write_learned_linear_ranker_report(report, report_path)

    print(f"Training cutoff: {report.train_cutoff}")
    print(f"Training label end exclusive: {report.train_validation_end_exclusive}")
    print(f"Evaluation cutoff: {report.evaluation_cutoff}")
    print(f"Evaluation end exclusive: {report.evaluation_end_exclusive}")
    print(f"Training pairs: {report.training.unique_candidate_pairs}")
    print(f"Training positives: {report.training.positive_pairs}")
    print(f"Evaluation pairs: {report.evaluation.unique_candidate_pairs}")
    print(f"Evaluated customers: {report.evaluation.evaluated_customers}")
    print(f"Learned MAP@{report.k}: {report.evaluation.map_at_k:.8f}")
    print(f"Deterministic MAP@{report.k}: {report.evaluation.deterministic_map_at_k:.8f}")
    print(f"Source-order MAP@{report.k}: {report.evaluation.baseline_map_at_k:.8f}")
    print(
        f"Delta vs deterministic MAP@{report.k}: "
        f"{report.evaluation.delta_vs_deterministic_map_at_k:.8f}"
    )
    print(f"Train candidate CSV written to: {train_export_summary.output_path}")
    print(f"Eval candidate CSV written to: {eval_export_summary.output_path}")
    print(f"Train candidate summary written to: {written_train_candidate_report_path}")
    print(f"Eval candidate summary written to: {written_eval_candidate_report_path}")
    print(f"Learned ranker report written to: {written_report_path}")
    return 0


def _handle_rolling_ranker_validation(args: argparse.Namespace) -> int:
    """Handle the ``rolling-ranker-validation`` subcommand.

    Args:
        args: Parsed command arguments.

    Returns:
        ``0`` after writing the rolling validation report and candidate artifacts.
    """

    paths = ProjectPaths.from_root(root=args.project_root, raw_data_dir=args.raw_data_dir)
    cutoffs = tuple(args.cutoffs)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("rolling evaluation cutoffs must be unique")
    if tuple(sorted(cutoffs)) != cutoffs:
        raise ValueError("rolling evaluation cutoffs must be in ascending order")
    report_path = args.report_path or paths.rolling_ranker_validation_report_path(
        cutoffs=cutoffs,
        k=args.k,
        candidate_k=args.candidate_k,
        max_target_customers=args.max_target_customers,
        config_slug=_learned_ranker_config_slug(args),
    )
    report_path = _resolve_path_under_root(paths, report_path)

    submission_customer_ids = load_submission_customer_ids(paths.raw_data_dir)
    config = LinearRankerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        positive_weight=args.positive_weight,
        max_auto_positive_weight=args.max_auto_positive_weight,
    )

    candidate_export_cache: dict[Path, tuple[CandidateExportSummary, Path]] = {}
    labels_cache: dict[tuple[str, int], dict[str, tuple[str, ...]]] = {}
    learned_reports = []

    print(
        "Starting rolling ranker validation. "
        f"cutoffs={', '.join(cutoffs)}, "
        f"max_target_customers={args.max_target_customers}, "
        f"include_co_visitation={not args.no_co_visitation}. "
        "This command scans transactions for each unique train/eval window.",
        flush=True,
    )

    for window_index, cutoff in enumerate(cutoffs, start=1):
        evaluation_split = TemporalSplit.from_isoformat(cutoff, horizon_days=args.horizon_days)
        train_split = previous_window_split(evaluation_split)
        if train_split.validation_end > evaluation_split.cutoff:
            raise ValueError("training label window must end before the evaluation cutoff")
        print(
            f"[{window_index}/{len(cutoffs)}] evaluation={evaluation_split.cutoff.isoformat()} "
            f"train={train_split.cutoff.isoformat()}.."
            f"{train_split.validation_end.isoformat()} exclusive",
            flush=True,
        )

        train_candidate_path = _ranker_candidate_export_path(
            paths=paths,
            split=train_split,
            args=args,
        )
        eval_candidate_path = _ranker_candidate_export_path(
            paths=paths,
            split=evaluation_split,
            args=args,
        )
        print(
            f"[{window_index}/{len(cutoffs)}] writing/reusing train candidates: "
            f"{train_candidate_path}",
            flush=True,
        )
        _write_cached_ranker_candidate_export(
            cache=candidate_export_cache,
            paths=paths,
            split=train_split,
            submission_customer_ids=submission_customer_ids,
            output_path=train_candidate_path,
            args=args,
        )
        print(
            f"[{window_index}/{len(cutoffs)}] writing/reusing eval candidates: "
            f"{eval_candidate_path}",
            flush=True,
        )
        _write_cached_ranker_candidate_export(
            cache=candidate_export_cache,
            paths=paths,
            split=evaluation_split,
            submission_customer_ids=submission_customer_ids,
            output_path=eval_candidate_path,
            args=args,
        )
        print(
            f"[{window_index}/{len(cutoffs)}] collecting labels and training linear ranker",
            flush=True,
        )
        train_labels = _cached_validation_labels_for_split(
            cache=labels_cache,
            raw_data_dir=paths.raw_data_dir,
            split=train_split,
            submission_customer_ids=submission_customer_ids,
            max_target_customers=args.max_target_customers,
        )
        eval_labels = _cached_validation_labels_for_split(
            cache=labels_cache,
            raw_data_dir=paths.raw_data_dir,
            split=evaluation_split,
            submission_customer_ids=submission_customer_ids,
            max_target_customers=args.max_target_customers,
        )

        training_result = train_linear_ranker_from_csv(
            candidate_path=train_candidate_path,
            validation_labels=train_labels,
            config=config,
        )
        print(
            f"[{window_index}/{len(cutoffs)}] evaluating learned, deterministic, "
            "and source-order rankers",
            flush=True,
        )
        evaluation = evaluate_linear_ranker_from_csv(
            candidate_path=eval_candidate_path,
            validation_labels=eval_labels,
            model=training_result.model,
            k=args.k,
        )
        learned_reports.append(
            build_learned_linear_ranker_report(
                train_split=train_split,
                evaluation_split=evaluation_split,
                k=args.k,
                candidate_k=args.candidate_k,
                config=config,
                training_result=training_result,
                evaluation=evaluation,
            )
        )
        print(
            f"[{window_index}/{len(cutoffs)}] done: "
            f"learned_MAP@{args.k}={evaluation.map_at_k:.8f}, "
            f"deterministic_MAP@{args.k}={evaluation.deterministic_map_at_k:.8f}, "
            f"source_order_MAP@{args.k}={evaluation.baseline_map_at_k:.8f}",
            flush=True,
        )

    candidate_summary_paths = tuple(
        str(summary_path)
        for _, summary_path in sorted(
            candidate_export_cache.values(), key=lambda cached: str(cached[1])
        )
    )
    rolling_report = build_rolling_ranker_validation_report(
        learned_reports,
        max_target_customers=args.max_target_customers,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        candidate_summary_paths=candidate_summary_paths,
    )
    written_report_path = write_rolling_ranker_validation_report(rolling_report, report_path)

    print(f"Rolling cutoffs: {', '.join(rolling_report.cutoffs)}")
    print(f"Windows evaluated: {rolling_report.window_count}")
    print(
        f"Mean learned MAP@{rolling_report.k}: "
        f"{rolling_report.aggregate.mean_learned_map_at_k:.8f}"
    )
    print(
        f"Mean deterministic MAP@{rolling_report.k}: "
        f"{rolling_report.aggregate.mean_deterministic_map_at_k:.8f}"
    )
    print(
        f"Mean source-order MAP@{rolling_report.k}: "
        f"{rolling_report.aggregate.mean_source_order_map_at_k:.8f}"
    )
    print(
        f"Mean delta vs deterministic MAP@{rolling_report.k}: "
        f"{rolling_report.aggregate.mean_delta_learned_vs_deterministic_map_at_k:.8f}"
    )
    print(
        "Windows improved vs deterministic: "
        f"{rolling_report.aggregate.windows_improved_vs_deterministic}/"
        f"{rolling_report.window_count}"
    )
    for window in rolling_report.windows:
        print(
            f"- {window.evaluation_cutoff}: learned={window.learned_map_at_k:.8f}, "
            f"deterministic={window.deterministic_map_at_k:.8f}, "
            f"source_order={window.source_order_map_at_k:.8f}, "
            f"delta_det={window.delta_learned_vs_deterministic_map_at_k:.8f}"
        )
    print(f"Candidate summary reports written: {len(candidate_summary_paths)}")
    print(f"Rolling ranker report written to: {written_report_path}")
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


def _ranker_candidate_export_path(
    paths: ProjectPaths,
    split: TemporalSplit,
    args: argparse.Namespace,
) -> Path:
    """Return the default candidate export path for ranker CLI commands.

    Args:
        paths: Canonical project paths.
        split: Temporal split whose validation-label customers are exported.
        args: Parsed ranker command arguments.

    Returns:
        Absolute candidate export path under the project artifacts directory.
    """

    candidate_path = paths.candidate_export_path(
        cutoff=split.cutoff.isoformat(),
        k=args.candidate_k,
        lookback_days=args.popularity_lookback_days,
        co_visitation_history_items=(
            None if args.no_co_visitation else args.co_visitation_max_history_items
        ),
        co_visitation_neighbors_per_item=(
            None if args.no_co_visitation else args.co_visitation_max_neighbors_per_item
        ),
        max_target_customers=args.max_target_customers,
    )
    return _resolve_path_under_root(paths, candidate_path)


def _write_cached_ranker_candidate_export(
    cache: dict[Path, tuple[CandidateExportSummary, Path]],
    paths: ProjectPaths,
    split: TemporalSplit,
    submission_customer_ids: set[str],
    output_path: Path,
    args: argparse.Namespace,
) -> tuple[CandidateExportSummary, Path]:
    """Write or reuse a candidate export and its JSON summary.

    Args:
        cache: In-memory cache keyed by resolved candidate CSV path.
        paths: Canonical project paths.
        split: Temporal split to export.
        submission_customer_ids: Authoritative H&M submission customer universe.
        output_path: Candidate CSV path.
        args: Parsed ranker command arguments.

    Returns:
        Candidate export summary and written JSON summary path.
    """

    cached = cache.get(output_path)
    if cached is not None:
        return cached

    summary = write_validation_candidate_export(
        transaction_iter_factory=lambda: iter_transaction_events(paths.raw_data_dir),
        split=split,
        submission_customer_ids=submission_customer_ids,
        output_path=output_path,
        k=args.candidate_k,
        popularity_lookback_days=args.popularity_lookback_days,
        include_co_visitation=not args.no_co_visitation,
        co_visitation_max_history_items=args.co_visitation_max_history_items,
        co_visitation_max_neighbors_per_item=args.co_visitation_max_neighbors_per_item,
        max_target_customers=args.max_target_customers,
    )
    candidate_report_path = _resolve_path_under_root(
        paths, paths.candidate_export_report_path(output_path)
    )
    written_report_path = write_candidate_export_summary(summary, candidate_report_path)
    result = (summary, written_report_path)
    cache[output_path] = result
    return result


def _cached_validation_labels_for_split(
    cache: dict[tuple[str, int], dict[str, tuple[str, ...]]],
    raw_data_dir: Path,
    split: TemporalSplit,
    submission_customer_ids: set[str],
    max_target_customers: int | None,
) -> dict[str, tuple[str, ...]]:
    """Return cached validation labels for a split and customer cap.

    Args:
        cache: In-memory labels cache keyed by cutoff and horizon.
        raw_data_dir: Directory containing H&M raw CSV files.
        split: Temporal split whose labels should be collected.
        submission_customer_ids: Authoritative H&M submission customer universe.
        max_target_customers: Optional deterministic cap.

    Returns:
        Validation labels keyed by selected target customer ID.
    """

    cache_key = (split.cutoff.isoformat(), split.horizon_days)
    labels = cache.get(cache_key)
    if labels is None:
        labels = _validation_labels_for_split(
            raw_data_dir=raw_data_dir,
            split=split,
            submission_customer_ids=submission_customer_ids,
            max_target_customers=max_target_customers,
        )
        cache[cache_key] = labels
    return labels


def _validation_labels_for_split(
    raw_data_dir: Path,
    split: TemporalSplit,
    submission_customer_ids: set[str],
    max_target_customers: int | None,
) -> dict[str, tuple[str, ...]]:
    """Collect validation labels filtered to submission customers and smoke cap.

    Args:
        raw_data_dir: Directory containing H&M raw CSV files.
        split: Temporal split whose labels should be collected.
        submission_customer_ids: Authoritative submission customer universe.
        max_target_customers: Optional deterministic cap.

    Returns:
        Validation labels keyed by selected target customer ID.
    """

    validation_data = summarize_temporal_split_with_labels(
        iter_transaction_events(raw_data_dir), split
    )
    target_customer_ids = select_validation_label_customer_ids(
        validation_labels=validation_data.validation_labels,
        submission_customer_ids=submission_customer_ids,
        max_target_customers=max_target_customers,
    )
    return {
        customer_id: validation_data.validation_labels[customer_id]
        for customer_id in target_customer_ids
    }


def _learned_ranker_config_slug(args: argparse.Namespace) -> str:
    """Build a compact filesystem-safe descriptor for learned-ranker config.

    Args:
        args: Parsed CLI arguments for the learned-ranker command.

    Returns:
        Compact config slug used in default artifact paths.
    """

    positive_weight = "auto" if args.positive_weight is None else _float_slug(args.positive_weight)
    return (
        f"e{args.epochs}_lr{_float_slug(args.learning_rate)}_"
        f"l2{_float_slug(args.l2)}_pw{positive_weight}_"
        f"maxpw{_float_slug(args.max_auto_positive_weight)}"
    )


def _float_slug(value: float) -> str:
    """Convert a float into a compact path-safe token."""

    return f"{value:g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
