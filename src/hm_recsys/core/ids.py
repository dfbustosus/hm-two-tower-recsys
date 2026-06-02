from __future__ import annotations

import re

CUSTOMER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARTICLE_ID_PATTERN = re.compile(r"^[0-9]{10}$")


def is_customer_id(value: str) -> bool:
    return CUSTOMER_ID_PATTERN.fullmatch(value) is not None


def is_article_id(value: str) -> bool:
    return ARTICLE_ID_PATTERN.fullmatch(value) is not None
