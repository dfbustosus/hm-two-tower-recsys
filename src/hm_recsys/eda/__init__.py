"""Exploratory data analysis for the H&M Kaggle dataset.

This package produces a deterministic, leakage-safe descriptive report
covering transaction volume, channel mix, customer history depth,
article hierarchy fanout, repeat-purchase structure, and cold-user
share across rolling validation cutoffs. The output drives Phase -1
decisions in the gap-closure plan.
"""

from hm_recsys.eda.report import (
    DEFAULT_PERCENTILES,
    DEFAULT_ROLLING_CUTOFFS,
    DEFAULT_TOP_BUSY_DAYS,
    DEFAULT_TOP_HIERARCHY_VALUES,
    EdaReport,
    EdaReportConfig,
    EdaSegmentThresholds,
    build_eda_report,
    eda_report_to_dict,
    render_eda_report_markdown,
    write_eda_report,
    write_eda_report_markdown,
)

__all__ = [
    "DEFAULT_PERCENTILES",
    "DEFAULT_ROLLING_CUTOFFS",
    "DEFAULT_TOP_BUSY_DAYS",
    "DEFAULT_TOP_HIERARCHY_VALUES",
    "EdaReport",
    "EdaReportConfig",
    "EdaSegmentThresholds",
    "build_eda_report",
    "eda_report_to_dict",
    "render_eda_report_markdown",
    "write_eda_report",
    "write_eda_report_markdown",
]
