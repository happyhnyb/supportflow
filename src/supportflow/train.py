from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .constants import INTENTS
from .model import TicketClassifier


def train(input_path: Path, output_path: Path, metrics_path: Path | None = None) -> dict:
    data = pd.read_csv(input_path)
    required_columns = {"message", "intent"}
    if missing := required_columns - set(data.columns):
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_list}")

    data = data.dropna(subset=["message", "intent"])
    if data.empty:
        raise ValueError("CSV does not contain any labelled tickets.")
    if unknown_intents := set(data.intent) - set(INTENTS):
        raise ValueError(f"Unsupported intents: {sorted(unknown_intents)}")
    if data.intent.value_counts().min() < 2:
        raise ValueError("Each intent requires at least two examples.")

    train_x, test_x, train_y, test_y = train_test_split(
        data.message,
        data.intent,
        test_size=0.20,
        stratify=data.intent,
        random_state=42,
    )
    evaluation_model = TicketClassifier.create().fit(train_x, train_y)
    predicted = evaluation_model.predict_proba(test_x).idxmax(axis=1)
    metrics = {
        "accuracy": round(float(accuracy_score(test_y, predicted)), 4),
        "macro_f1": round(
            float(
                f1_score(
                    test_y,
                    predicted,
                    labels=INTENTS,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "test_rows": int(len(test_y)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(TicketClassifier.create().fit(data.message, data.intent), output_path)

    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/supportflow_model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/supportflow_metrics.json"))
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output, args.metrics), indent=2))
