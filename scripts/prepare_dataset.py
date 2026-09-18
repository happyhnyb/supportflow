"""Build the checked-in SupportFlow training subset from a public dataset."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd


SOURCE_REVISION = "12dd624ddcd3057382b2faad661bcda1fa869491"
SOURCE_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-retail-ecommerce-llm-chatbot-training-dataset/resolve/"
    f"{SOURCE_REVISION}/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)
SOURCE_SHA256 = "13a988266fed4e2b2c1ff947a89ef220ce09b5b13ac83c4a1496c0d7b81e8127"
ROWS_PER_CLASS = 100
RANDOM_STATE = 42

INTENT_MAP = {
    "delivery_issue": "delivery_delay",
    "request_refund": "refund_request",
    "damaged_delivery": "damaged_item",
    "payment_issue": "payment_problem",
    "recover_password": "account_access",
    "change_order": "order_change",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: Path, source: Path | None = None) -> pd.DataFrame:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if source is None:
        temporary = tempfile.TemporaryDirectory(prefix="supportflow-data-")
        source = Path(temporary.name) / "source.csv"
        urllib.request.urlretrieve(SOURCE_URL, source)

    actual_hash = sha256(source)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(
            "Source checksum mismatch: "
            f"expected {SOURCE_SHA256}, received {actual_hash}."
        )

    source_data = pd.read_csv(source)
    selected = source_data[source_data["intent"].isin(INTENT_MAP)].copy()
    selected["source_row"] = selected.index + 2  # one-based CSV line, including header
    selected = (
        selected.groupby("intent", group_keys=False)
        .sample(n=ROWS_PER_CLASS, random_state=RANDOM_STATE)
        .sort_values(["intent", "source_row"])
    )
    selected["message"] = selected["instruction"].str.strip()
    selected["source_intent"] = selected["intent"]
    selected["intent"] = selected["source_intent"].map(INTENT_MAP)
    selected["ticket_id"] = [f"BTX-{number:04d}" for number in range(1, len(selected) + 1)]
    result = selected[["ticket_id", "message", "intent", "source_intent", "source_row"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    if temporary is not None:
        temporary.cleanup()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/support_tickets.csv"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    data = build(args.output, args.source)
    print(f"Wrote {len(data)} rows to {args.output}")

