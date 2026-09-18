# SupportFlow — submission handoff

## Live application

- Production dashboard: https://supportflow-ticket-triage-fdvznvmp6-happyhnybs-projects.vercel.app
- Health check: https://supportflow-ticket-triage-fdvznvmp6-happyhnybs-projects.vercel.app/health
- Source repository: https://github.com/happyhnyb/supportflow

## Included work

- FastAPI customer-support triage application and browser dashboard (`app/`)
- Vercel serverless deployment entry point (`api/index.py`, `vercel.json`)
- Classifier, redaction, retrieval, privacy, and optional AI-review modules (`src/supportflow/`)
- Public-derived training subset, provenance, and approved knowledge base (`data/`)
- Automated checks (`tests/`)
- Written report (`report.md`) and rendered PDF (`output/pdf/supportflow_report.pdf`)

## Reviewer quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`, or submit a message to `POST /v1/tickets/analyze`.

## Release verification

- Automated tests: 3 passed
- Training evaluation: 120-row holdout; accuracy 1.0000; macro F1 1.0000 (see report for limitations)
- Deployment target: Vercel production

`OPENAI_API_KEY` is optional and must be configured only as a server environment variable. The demo works without it; no secret is included in this package.
