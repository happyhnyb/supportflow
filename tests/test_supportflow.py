from pathlib import Path

import joblib
import pandas as pd
import pytest

from supportflow.knowledge import KnowledgeBase
from supportflow.privacy import extract_order_ids, redact
from supportflow.service import analyse
from supportflow.train import train


def test_redaction_and_order_extraction():
    text = "Order #AB123 - email jo@example.com or call +1 415 555 1234"
    assert "jo@example.com" not in redact(text)
    assert "+1 415" not in redact(text)
    assert extract_order_ids(text) == ["AB123"]


def test_analysis_returns_matching_guidance():
    model = joblib.load("models/supportflow_model.joblib")
    kb = KnowledgeBase.load(Path("data/knowledge_base.json"))
    output = analyse(model, kb, "My new headphones arrived broken and I need a replacement.")
    assert output["intent"] == "damaged_item"
    assert output["knowledge_matches"][0]["article_id"] == "KB-301"


def test_training_reports_missing_columns(tmp_path):
    source = tmp_path / "tickets.csv"
    pd.DataFrame({"message": ["Where is my order?"]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="intent"):
        train(source, tmp_path / "model.joblib")
