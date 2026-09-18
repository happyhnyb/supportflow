from pathlib import Path

import pandas as pd
import pytest

from api.index import KnowledgeBase, analyse, extract_order_ids, load_model, redact, train


def test_redaction_and_order_extraction():
    text = "Order #AB123 - email jo@example.com or call +1 415 555 1234"
    assert "jo@example.com" not in redact(text)
    assert "+1 415" not in redact(text)
    assert extract_order_ids(text) == ["AB123"]


def test_analysis_returns_matching_guidance(tmp_path):
    model_path = tmp_path / "supportflow_model.joblib"
    train(Path("data/support_tickets.csv"), model_path)
    model = load_model(model_path)
    kb = KnowledgeBase.load(Path("data/knowledge_base.json"))
    output = analyse(model, kb, "My new headphones arrived broken and I need a replacement.")
    assert output["intent"] == "damaged_item"
    assert output["knowledge_matches"][0]["article_id"] == "KB-301"


def test_training_reports_missing_columns(tmp_path):
    source = tmp_path / "tickets.csv"
    pd.DataFrame({"message": ["Where is my order?"]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="intent"):
        train(source, tmp_path / "model.joblib")
