"""H&M recommender command-line dispatcher.

This module is the public entry point for the ``hm_recsys.cli`` package. The
dispatcher itself is intentionally tiny so we can reason about it in isolation;
all 24 subcommand parsers and handlers live next to the domain logic they
exercise:

* :mod:`hm_recsys.cli._legacy` - staging ground for parsers/handlers that have
  not yet been split into a domain module. New code MUST NOT add to this
  module; instead, add a new ``hm_recsys.cli.<domain>.py`` module and register
  it from :func:`build_parser`.

Subcommand-level guarantees live in ``tests/test_cli_contract.py``, which
asserts the full surface every commit so a missing or renamed command is
caught immediately by ``pytest``.
"""

from __future__ import annotations

import argparse

from hm_recsys.cli import _legacy
from hm_recsys.cli import two_tower as _two_tower

__all__ = ("build_parser", "main")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with every registered subcommand."""

    parser = _legacy.build_parser()
    subparsers_actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    if len(subparsers_actions) != 1:
        raise RuntimeError("legacy build_parser must define exactly one subparser group")
    _two_tower.register_subcommands(subparsers_actions[0])
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Uses the augmented parser built here."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("no handler registered for selected subcommand")
    return int(handler(args))
