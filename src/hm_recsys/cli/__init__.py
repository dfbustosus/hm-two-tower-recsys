"""H&M recommender command-line dispatcher.

This module is the public entry point for the ``hm_recsys.cli`` package. The
dispatcher itself is intentionally tiny so we can reason about it in isolation;
all 24 subcommand parsers and handlers live next to the domain logic they
exercise:

* :mod:`hm_recsys.cli._legacy` – staging ground for parsers/handlers that have
  not yet been split into a domain module. New code MUST NOT add to this
  module; instead, add a new ``hm_recsys.cli.<domain>.py`` module and register
  it from :func:`build_parser`.

Subcommand-level guarantees live in ``tests/test_cli_contract.py``, which
asserts the full surface every commit so a missing or renamed command is
caught immediately by ``pytest``.
"""

from __future__ import annotations

from hm_recsys.cli._legacy import build_parser, main

__all__ = ("build_parser", "main")
