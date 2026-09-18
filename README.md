# SupportFlow

SupportFlow is a compact AI-assisted customer-support triage application. It
classifies tickets, detects urgency, redacts common PII, extracts order IDs,
retrieves approved guidance, and optionally asks OpenAI for a structured
second review. A human agent remains responsible for every action.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest -q tests
uvicorn api.index:app --reload
```

Open <http://127.0.0.1:8000>. The JSON endpoint is
`POST /v1/tickets/analyze`. The application trains an in-memory baseline from
the checked-in data when no saved model exists. To create a saved model:

```bash
python3 api/index.py train --output models/supportflow_model.joblib
```

`OPENAI_API_KEY` is optional and must only be provided as a server environment
variable. The browser never receives it, and the deterministic baseline works
without it.

## Public dataset

`data/support_tickets.csv` is a deterministic, balanced 600-row subset of
[Bitext's public Retail (eCommerce) dataset](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset),
published under CDLA-Sharing-1.0. It uses source revision
`12dd624ddcd3057382b2faad661bcda1fa869491` and source CSV SHA-256
`13a988266fed4e2b2c1ff947a89ef220ce09b5b13ac83c4a1496c0d7b81e8127`.

The reproducible selection uses seed 42 and maps six source intents:
`delivery_issue -> delivery_delay`, `request_refund -> refund_request`,
`damaged_delivery -> damaged_item`, `payment_issue -> payment_problem`,
`recover_password -> account_access`, and `change_order -> order_change`.
Every output row retains its source intent and source row. Rebuild it with:

```bash
python3 api/index.py prepare-data
```

## Minimal layout

```text
api/index.py              API, ML, privacy, retrieval, OpenAI layer, and CLI
app/index.html            self-contained dashboard (HTML, CSS, and JavaScript)
data/                     public-derived tickets and approved knowledge base
tests/test_supportflow.py automated checks
report.md                 assessment report
```

Repository: <https://github.com/happyhnyb/supportflow>
