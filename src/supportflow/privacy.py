"""PII minimisation utilities used before model inference and display."""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
ORDER_RE = re.compile(r"\b(?:order\s*(?:number|no\.?|#)?\s*|#)([A-Z]{1,4}-?\d{3,12})\b", re.IGNORECASE)


def redact(text: str) -> str:
    value = CARD_RE.sub("[CARD REDACTED]", text)
    value = EMAIL_RE.sub("[EMAIL REDACTED]", value)
    return PHONE_RE.sub("[PHONE REDACTED]", value)


def extract_order_ids(text: str) -> list[str]:
    return list(dict.fromkeys(match.upper() for match in ORDER_RE.findall(text)))


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", redact(text).lower()).strip()
