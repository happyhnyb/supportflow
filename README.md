# SupportFlow

SupportFlow is an AI-assisted customer-support ticket triage application for an e-commerce business. It classifies a customer message, detects urgent cases, redacts common PII, extracts order IDs, retrieves approved help-centre articles, and prepares an agent-ready summary. The application is decision support: a trained agent remains responsible for the reply and final action.

## Repository layout

```
app/                    FastAPI service and browser-based agent dashboard
src/supportflow/        training, redaction, retrieval and evaluation code
data/support_tickets.csv public-derived labelled training subset
data/knowledge_base.json approved policy articles used for retrieval
report.md               submission report
tests/                  automated checks
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python3 scripts/prepare_dataset.py
python3 -m supportflow.train --input data/support_tickets.csv --output models/supportflow_model.joblib
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the agent dashboard. Or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/v1/tickets/analyze \
  -H 'content-type: application/json' \
  -d '{"message":"Order #AB123 has not arrived and it was due Friday. Please help."}'
```

The endpoint returns intent probabilities, urgency, safely redacted text, extracted order IDs, a concise summary, and approved policy articles ranked by relevance.

Set `OPENAI_API_KEY` only as a server environment variable. The server sends redacted ticket text to the OpenAI Responses API with `store=False` for the structured agent review; the browser never receives the key.

## Data contract and responsible use

Training requires a CSV with `message` and `intent`. Allowed intents are `delivery_delay`, `refund_request`, `damaged_item`, `payment_problem`, `account_access`, and `order_change`; `ticket_id` is optional. The checked-in training subset is reproducibly derived from [Bitext's public retail e-commerce dataset](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset); see [data/README.md](data/README.md) for source revision, checksum, licence, mapping, and rebuild instructions. Use a chronological holdout in production, not a random split alone.

Do not train on payment card numbers, authentication secrets, government IDs, or unnecessary customer-profile data. Redaction reduces exposure but is not a substitute for data minimisation, access control, and retention policies. See [the full report](report.md).

## GitHub handoff

```bash
git add .
git commit -m "Initial SupportFlow application"
git branch -M main
git remote add origin https://github.com/happyhnyb/supportflow.git
git push -u origin main
```
