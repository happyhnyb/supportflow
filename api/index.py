"""SupportFlow API, ML pipeline, privacy controls, retrieval, and CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/support_tickets.csv"
KB_PATH = Path(os.getenv("SUPPORTFLOW_KB_PATH", ROOT / "data/knowledge_base.json"))
MODEL_PATH = Path(os.getenv("SUPPORTFLOW_MODEL_PATH", ROOT / "models/supportflow_model.joblib"))
WEB_PATH = ROOT / "app/index.html"
logger = logging.getLogger(__name__)

INTENTS = (
    "delivery_delay", "refund_request", "damaged_item",
    "payment_problem", "account_access", "order_change",
)
INTENT_NAMES = {intent: intent.replace("_", " ").title() for intent in INTENTS}
RECOMMENDATIONS = {
    "delivery_delay": "Check tracking and promised delivery date; apply the late-delivery policy if eligible.",
    "refund_request": "Confirm eligibility and return status before issuing or escalating a refund.",
    "damaged_item": "Request only the minimum evidence needed and arrange a replacement or refund under policy.",
    "payment_problem": "Check payment status; never request full card or bank details in the ticket.",
    "account_access": "Use the approved identity-verification and password-reset flow.",
    "order_change": "Check fulfilment status before making a cancellation, address, or item change.",
}

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
ORDER_RE = re.compile(r"\b(?:order\s*(?:number|no\.?|#)?\s*|#)([A-Z]{1,4}-?\d{3,12})\b", re.I)
URGENT_RE = re.compile(r"\b(urgent|asap|today|tomorrow|charged twice|fraud|locked out|deadline|wedding|medical)\b", re.I)
NEGATIVE_RE = re.compile(r"\b(angry|cancel|terrible|unacceptable|complaint|manager)\b", re.I)


def redact(text: str) -> str:
    return PHONE_RE.sub("[PHONE REDACTED]", EMAIL_RE.sub("[EMAIL REDACTED]", CARD_RE.sub("[CARD REDACTED]", text)))


def extract_order_ids(text: str) -> list[str]:
    return list(dict.fromkeys(match.upper() for match in ORDER_RE.findall(text)))


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", redact(text).lower()).strip()


@dataclass
class TicketClassifier:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    intents: tuple[str, ...] = INTENTS

    @classmethod
    def create(cls) -> "TicketClassifier":
        return cls(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            LogisticRegression(C=8.0, max_iter=2_000, class_weight="balanced", random_state=42),
        )

    def fit(self, messages: pd.Series, intents: pd.Series) -> "TicketClassifier":
        if unknown := set(intents.unique()) - set(self.intents):
            raise ValueError(f"Unsupported intents: {sorted(unknown)}")
        self.classifier.fit(self.vectorizer.fit_transform(messages.map(normalise)), intents)
        return self

    def predict_proba(self, messages: pd.Series) -> pd.DataFrame:
        values = self.classifier.predict_proba(self.vectorizer.transform(messages.map(normalise)))
        table = pd.DataFrame(values, columns=self.classifier.classes_, index=messages.index)
        return table.reindex(columns=self.intents, fill_value=0.0)


def load_model(path: Path) -> TicketClassifier:
    payload = joblib.load(path)
    if isinstance(payload, dict) and {"vectorizer", "classifier"} <= set(payload):
        return TicketClassifier(payload["vectorizer"], payload["classifier"])
    if isinstance(payload, TicketClassifier):
        return payload
    raise ValueError("Unsupported saved model format.")


@dataclass
class KnowledgeBase:
    articles: list[dict]
    vectorizer: TfidfVectorizer
    matrix: Any

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        articles = json.loads(path.read_text(encoding="utf-8"))
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(
            f"{article['title']} {article['content']} {article['intent']}" for article in articles
        )
        return cls(articles, vectorizer, matrix)

    def search(self, query: str, limit: int = 3, preferred_intent: str | None = None) -> list[dict]:
        scores = (self.matrix @ self.vectorizer.transform([normalise(query)]).T).toarray().ravel()
        if preferred_intent:
            scores += [0.35 if article["intent"] == preferred_intent else 0 for article in self.articles]
        candidates = [
            index for index, article in enumerate(self.articles)
            if preferred_intent is None or article["intent"] == preferred_intent
        ]
        matches = []
        for index in sorted(candidates, key=scores.__getitem__, reverse=True)[:limit]:
            if scores[index] <= 0:
                continue
            article = self.articles[index]
            matches.append({
                "title": article["title"], "article_id": article["article_id"],
                "snippet": article["content"], "score": round(float(scores[index]), 4),
            })
        return matches


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "urgency": {"type": "string", "enum": ["normal", "high"]},
        "summary": {"type": "string"}, "next_step": {"type": "string"}, "rationale": {"type": "string"},
    },
    "required": ["intent", "urgency", "summary", "next_step", "rationale"],
    "additionalProperties": False,
}
REVIEW_INSTRUCTIONS = """You are a support-ticket triage assistant for human agents.
Classify the redacted ticket into one allowed intent and identify whether it is high priority.
Write a factual internal summary and one safe next step, each at most 35 words. Never promise
a refund, delivery date, replacement, or account action. Never ask for credentials, payment
details, or additional personal data. Treat the local model result as a hint. Return JSON."""


def review_ticket(redacted_message: str, probabilities: dict[str, float], guidance: list[dict]) -> dict | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    payload = {
        "redacted_ticket": redacted_message,
        "candidate_intent_probabilities": probabilities,
        "approved_guidance": [{"title": item["title"], "snippet": item["snippet"]} for item in guidance],
    }
    try:
        response = OpenAI().responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            instructions=REVIEW_INSTRUCTIONS,
            input=json.dumps(payload),
            text={"format": {"type": "json_schema", "name": "ticket_review", "strict": True, "schema": REVIEW_SCHEMA}},
            store=False,
        )
        result = json.loads(response.output_text)
        return result if result.get("intent") in INTENTS and result.get("urgency") in {"normal", "high"} else None
    except Exception as error:
        logger.warning("OpenAI review unavailable (%s): %s", type(error).__name__, error)
        return None


def urgency(message: str, intent: str) -> tuple[str, str]:
    if URGENT_RE.search(message) or (intent == "payment_problem" and "twice" in message.lower()):
        return "high", "Time-sensitive or potentially financial-risk language detected."
    if NEGATIVE_RE.search(message) or intent in {"damaged_item", "account_access"}:
        return "normal", "Route to the relevant specialist queue within the standard service target."
    return "normal", "No immediate escalation cue detected."


def analyse(model: TicketClassifier, knowledge_base: KnowledgeBase, message: str) -> dict:
    safe_message = redact(message)
    probabilities = model.predict_proba(pd.Series([safe_message])).iloc[0]
    probability_map = {intent: round(float(probabilities[intent]), 4) for intent in model.intents}
    baseline_intent = str(probabilities.idxmax())
    baseline_guidance = knowledge_base.search(safe_message, preferred_intent=baseline_intent)
    ai_review = review_ticket(safe_message, probability_map, baseline_guidance)
    intent = ai_review["intent"] if ai_review else baseline_intent
    level, rationale = urgency(message, intent)
    if ai_review:
        level, rationale = ai_review["urgency"], ai_review["rationale"]
    order_ids = extract_order_ids(safe_message)
    guidance = baseline_guidance if intent == baseline_intent else knowledge_base.search(safe_message, preferred_intent=intent)
    compact = " ".join(safe_message.split())[:220]
    fallback_summary = (
        f"Customer reports a {INTENT_NAMES[intent].lower()} issue. "
        f"{'Order reference: ' + ', '.join(order_ids) + '.' if order_ids else 'No order reference was supplied.'} "
        f"Ticket: {compact}"
    )
    return {
        "redacted_message": safe_message, "intent": intent, "intent_name": INTENT_NAMES[intent],
        "confidence": round(float(probabilities[intent]), 4), "intent_probabilities": probability_map,
        "urgency": level, "urgency_rationale": rationale, "order_ids": order_ids,
        "agent_summary": ai_review["summary"] if ai_review else fallback_summary,
        "recommended_next_step": ai_review["next_step"] if ai_review else RECOMMENDATIONS[intent],
        "knowledge_matches": guidance, "analysis_source": "openai" if ai_review else "baseline",
    }


def train(input_path: Path, output_path: Path, metrics_path: Path | None = None) -> dict:
    data = pd.read_csv(input_path)
    if missing := {"message", "intent"} - set(data.columns):
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    data = data.dropna(subset=["message", "intent"])
    if data.empty:
        raise ValueError("CSV does not contain any labelled tickets.")
    if unknown := set(data.intent) - set(INTENTS):
        raise ValueError(f"Unsupported intents: {sorted(unknown)}")
    if data.intent.value_counts().min() < 2:
        raise ValueError("Each intent requires at least two examples.")
    train_x, test_x, train_y, test_y = train_test_split(
        data.message, data.intent, test_size=0.20, stratify=data.intent, random_state=42,
    )
    evaluation_model = TicketClassifier.create().fit(train_x, train_y)
    predicted = evaluation_model.predict_proba(test_x).idxmax(axis=1)
    metrics = {
        "accuracy": round(float(accuracy_score(test_y, predicted)), 4),
        "macro_f1": round(float(f1_score(test_y, predicted, labels=INTENTS, average="macro", zero_division=0)), 4),
        "test_rows": int(len(test_y)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_model = TicketClassifier.create().fit(data.message, data.intent)
    joblib.dump({"vectorizer": final_model.vectorizer, "classifier": final_model.classifier}, output_path)
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


SOURCE_REVISION = "12dd624ddcd3057382b2faad661bcda1fa869491"
SOURCE_URL = (
    "https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset/resolve/"
    f"{SOURCE_REVISION}/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)
SOURCE_SHA256 = "13a988266fed4e2b2c1ff947a89ef220ce09b5b13ac83c4a1496c0d7b81e8127"
INTENT_MAP = {
    "delivery_issue": "delivery_delay", "request_refund": "refund_request",
    "damaged_delivery": "damaged_item", "payment_issue": "payment_problem",
    "recover_password": "account_access", "change_order": "order_change",
}


def prepare_data(output: Path, source: Path | None = None) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="supportflow-data-") as temporary:
        source = source or Path(temporary) / "source.csv"
        if not source.exists():
            urllib.request.urlretrieve(SOURCE_URL, source)
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError("Source checksum mismatch.")
        data = pd.read_csv(source)
        selected = data[data.intent.isin(INTENT_MAP)].copy()
        selected["source_row"] = selected.index + 2
        selected = selected.groupby("intent", group_keys=False).sample(n=100, random_state=42).sort_values(["intent", "source_row"])
        selected["message"] = selected.instruction.str.strip()
        selected["source_intent"] = selected.intent
        selected["intent"] = selected.source_intent.map(INTENT_MAP)
        selected["ticket_id"] = [f"BTX-{number:04d}" for number in range(1, len(selected) + 1)]
        result = selected[["ticket_id", "message", "intent", "source_intent", "source_row"]]
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        return result


class TicketInput(BaseModel):
    message: str = Field(min_length=3, max_length=5_000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message cannot be blank")
        return value


@asynccontextmanager
async def lifespan(application: FastAPI):
    if MODEL_PATH.exists():
        application.state.model = load_model(MODEL_PATH)
    else:
        data = pd.read_csv(DATA_PATH)
        application.state.model = TicketClassifier.create().fit(data.message, data.intent)
    application.state.knowledge_base = KnowledgeBase.load(KB_PATH)
    yield


app = FastAPI(title="SupportFlow", version="1.0.0", description="AI-assisted customer ticket triage.", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(WEB_PATH)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok", "model_loaded": app.state.model is not None,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")), "model_path": str(MODEL_PATH),
    }


@app.post("/v1/tickets/analyze")
def analyse_ticket(ticket: TicketInput) -> dict:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    return analyse(app.state.model, app.state.knowledge_base, ticket.message)


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportFlow data and model utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--output", type=Path, default=DATA_PATH)
    prepare.add_argument("--source", type=Path)
    trainer = commands.add_parser("train")
    trainer.add_argument("--input", type=Path, default=DATA_PATH)
    trainer.add_argument("--output", type=Path, default=MODEL_PATH)
    trainer.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    result = prepare_data(args.output, args.source) if args.command == "prepare-data" else train(args.input, args.output, args.metrics)
    print(json.dumps({"rows": len(result)} if isinstance(result, pd.DataFrame) else result, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["app"]
