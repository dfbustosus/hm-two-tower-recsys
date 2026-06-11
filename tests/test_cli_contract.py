"""Regression test that locks down the public CLI subcommand contract.

The Phase-0 refactor moves command handlers out of the 5000+ line ``cli.py``
monolith into the ``hm_recsys.cli`` subpackage modules
(``data``, ``eda``, ``retrieval``, ``ranking``, ``submission``, ``two_tower``).
Mistakes in this refactor would silently drop a subcommand and break Makefile
targets / Kaggle pipelines. This test asserts the full surface up-front so
any drop is caught immediately by ``pytest``.
"""

from __future__ import annotations

import argparse

import pytest

from hm_recsys.cli import build_parser, main

EXPECTED_SUBCOMMANDS = frozenset(
    {
        "candidate-diagnostics",
        "compute-perfect-ranker-ceiling",
        "content-similarity-diagnostics",
        "eda-report",
        "evaluate-baseline",
        "evaluate-learned-ranker-baseline",
        "evaluate-lightgbm-behavioral-ranker",
        "evaluate-ranker-baseline",
        "evaluate-two-tower-retrieval",
        "export-article-content",
        "export-candidates",
        "export-two-tower-examples",
        "generate-article-embeddings",
        "generate-baseline-submission",
        "generate-deterministic-ranker-submission",
        "generate-learned-ranker-submission",
        "generate-lightgbm-behavioral-submission",
        "inventory-article-images",
        "pin-baseline-champion",
        "rolling-ranker-validation",
        "score-two-tower-candidates",
        "summarize-temporal-split",
        "train-two-tower",
        "tune-deterministic-ranker",
        "validate-data-contract",
        "validate-submission",
    }
)


def _subcommands(parser: argparse.ArgumentParser) -> frozenset[str]:
    subparsers_actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers_actions) == 1, "build_parser must define exactly one subparser group"
    return frozenset(subparsers_actions[0].choices.keys())


def test_build_parser_exposes_all_expected_subcommands() -> None:
    parser = build_parser()
    assert _subcommands(parser) == EXPECTED_SUBCOMMANDS


def test_every_subcommand_has_a_handler() -> None:
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    for command_name, subparser in subparsers_action.choices.items():
        defaults = subparser._defaults
        assert "handler" in defaults, f"subcommand {command_name!r} missing default handler"
        assert callable(defaults["handler"]), (
            f"handler for subcommand {command_name!r} must be callable"
        )


def test_build_parser_does_not_require_runtime_io() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_help_runs_without_side_effects(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "validate-data-contract" in captured.out
    for command_name in sorted(EXPECTED_SUBCOMMANDS):
        assert command_name in captured.out, f"help output missing {command_name!r}"
