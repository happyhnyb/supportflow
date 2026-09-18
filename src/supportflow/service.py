from __future__ import annotations

import re

import pandas as pd

from .constants import INTENT_NAMES, RECOMMENDATIONS
from .openai_review import review_ticket
from .privacy import extract_order_ids, redact

URGENT_RE = re.compile(
    r"\b(urgent|asap|today|tomorrow|charged twice|fraud|locked out|deadline|wedding|medical)\b",
    re.IGNORECASE,
)
NEGATIVE_RE = re.compile(
    r"\b(angry|cancel|terrible|unacceptable|complaint|manager)\b",
    re.IGNORECASE,
)


def urgency(message: str, intent: str) -> tuple[str, str]:
    if URGENT_RE.search(message) or (intent == "payment_problem" and "twice" in message.lower()):
        return "high", "Time-sensitive or potentially financial-risk language detected."
    if NEGATIVE_RE.search(message) or intent in {"damaged_item", "account_access"}:
        return "normal", "Route to the relevant specialist queue within the standard service target."
    return "normal", "No immediate escalation cue detected."


def summary(message: str, intent: str, order_ids: list[str]) -> str:
    if order_ids:
        order_note = f" Order reference: {', '.join(order_ids)}."
    else:
        order_note = " No order reference was supplied."

    compact = " ".join(redact(message).split())[:220]
    return f"Customer reports a {INTENT_NAMES[intent].lower()} issue.{order_note} Ticket: {compact}"


def analyse(model, knowledge_base, message: str) -> dict:
    redacted_message = redact(message)
    probabilities = model.predict_proba(pd.Series([redacted_message])).iloc[0]
    probability_map = {
        intent: round(float(probabilities[intent]), 4)
        for intent in model.intents
    }
    baseline_intent = str(probabilities.idxmax())
    baseline_guidance = knowledge_base.search(redacted_message, preferred_intent=baseline_intent)
    ai_review = review_ticket(redacted_message, probability_map, baseline_guidance)

    intent = baseline_intent
    level, rationale = urgency(message, intent)

    if ai_review:
        intent = ai_review["intent"]
        level = ai_review["urgency"]
        rationale = ai_review["rationale"]

    order_ids = extract_order_ids(redacted_message)
    guidance = (
        baseline_guidance
        if intent == baseline_intent
        else knowledge_base.search(redacted_message, preferred_intent=intent)
    )

    return {
        "redacted_message": redacted_message,
        "intent": intent,
        "intent_name": INTENT_NAMES[intent],
        "confidence": round(float(probabilities[intent]), 4),
        "intent_probabilities": probability_map,
        "urgency": level,
        "urgency_rationale": rationale,
        "order_ids": order_ids,
        "agent_summary": ai_review["summary"] if ai_review else summary(redacted_message, intent, order_ids),
        "recommended_next_step": ai_review["next_step"] if ai_review else RECOMMENDATIONS[intent],
        "knowledge_matches": guidance,
        "analysis_source": "openai" if ai_review else "baseline",
    }
