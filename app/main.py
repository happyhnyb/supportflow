from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from supportflow.knowledge import KnowledgeBase
from supportflow.model import TicketClassifier
from supportflow.service import analyse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env.local")

MODEL_PATH = Path(
    os.getenv("SUPPORTFLOW_MODEL_PATH", PROJECT_ROOT / "models/supportflow_model.joblib")
)
KB_PATH = Path(
    os.getenv("SUPPORTFLOW_KB_PATH", PROJECT_ROOT / "data/knowledge_base.json")
)
STATIC_DIR = Path(__file__).parent / "static"


class TicketInput(BaseModel):
    message: str = Field(min_length=3, max_length=5_000, examples=["Order #AB123 has not arrived and was due Friday."])

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message cannot be blank")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
    else:
        demo_data = pd.read_csv(PROJECT_ROOT / "data/support_tickets.csv")
        app.state.model = TicketClassifier.create().fit(demo_data.message, demo_data.intent)
    app.state.knowledge_base = KnowledgeBase.load(KB_PATH)
    yield


app = FastAPI(
    title="SupportFlow",
    version="1.0.0",
    description="AI-assisted customer ticket triage.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "model_path": str(MODEL_PATH),
    }


@app.post("/v1/tickets/analyze")
def analyse_ticket(ticket: TicketInput) -> dict:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Run the documented training command first.")
    return analyse(app.state.model, app.state.knowledge_base, ticket.message)
