"""Small server-side OpenAI adapter for ticket review.

Only redacted ticket text is sent to the model. The model receives candidate
queues and approved guidance, and is never asked to make a customer-facing or
financial decision.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from .constants import INTENTS

logger = logging.getLogger(__name__)


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "urgency": {"type": "string", "enum": ["normal", "high"]},
        "summary": {"type": "string"},
        "next_step": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["intent", "urgency", "summary", "next_step", "rationale"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are a support-ticket triage assistant for human agents.
Classify the redacted ticket into one allowed intent and identify whether it is
high priority. Write a factual internal summary of at most 35 words and one
safe next step of at most 35 words.
Never promise a refund, delivery date, replacement, or account action. Never
ask for credentials, payment details, or additional personal data. Treat the
candidate model result as a hint, not a fact. Return only the required JSON."""


def review_ticket(
    redacted_message: str,
    intent_probabilities: dict[str, float],
    guidance: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Return structured model triage, or None when API use is unavailable."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    payload = {
        "redacted_ticket": redacted_message,
        "candidate_intent_probabilities": intent_probabilities,
        "approved_guidance": [{"title": item["title"], "snippet": item["snippet"]} for item in guidance],
    }
    try:
        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            instructions=INSTRUCTIONS,
            input=json.dumps(payload),
            text={"format": {"type": "json_schema", "name": "ticket_review", "strict": True, "schema": REVIEW_SCHEMA}},
            store=False,
        )
        result = json.loads(response.output_text)
        if result.get("intent") not in INTENTS or result.get("urgency") not in {"normal", "high"}:
            return None
        return result
    except Exception as error:
        # The deterministic baseline remains available during network or quota issues.
        logger.warning("OpenAI ticket review unavailable (%s): %s", type(error).__name__, error)
        return None
