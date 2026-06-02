"""Identifier validation helpers for the H&M recommendation data contract."""

from __future__ import annotations

import re

CUSTOMER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARTICLE_ID_PATTERN = re.compile(r"^[0-9]{10}$")


def is_customer_id(value: str) -> bool:
    """Return whether ``value`` is a canonical H&M ``customer_id``.

    Args:
        value: Candidate customer identifier loaded as a string.

    Returns:
        ``True`` when the value is a 64-character lowercase hexadecimal ID;
        otherwise ``False``.
    """

    return CUSTOMER_ID_PATTERN.fullmatch(value) is not None


def is_article_id(value: str) -> bool:
    """Return whether ``value`` is a canonical H&M ``article_id``.

    Args:
        value: Candidate article identifier loaded as a string.

    Returns:
        ``True`` when the value is exactly ten decimal digits; otherwise
        ``False``.
    """

    return ARTICLE_ID_PATTERN.fullmatch(value) is not None
