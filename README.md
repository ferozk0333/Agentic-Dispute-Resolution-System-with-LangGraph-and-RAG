# Agentic Fraud Detection System with LangGraph and RAG

A three-agent pipeline that resolves credit card disputes end-to-end — from raw claim text to a compliance-grounded decision — with full audit logging and a live streaming UI.

---

## Demo

> 📹 *(video)*

---

## How it works

A dispute flows through three specialized agents orchestrated by LangGraph:

| Agent | Model | Role |
|---|---|---|
| Triage | `gpt-4o-mini` | Extracts entities, masks PII, classifies dispute type |
| Investigator | `gpt-4o` | Queries transaction DB, runs fraud classifier, retrieves Visa policy via RAG |
| Auditor | `gpt-4o-mini` | Checks compliance rules, issues final decision, halts on policy violation |

Each agent receives only the prior agent's structured output — no chain-of-thought leakage across boundaries.

---

## Key features

- **RAG over Visa Core Rules** — ChromaDB retrieval grounded in the real Visa rulebook (April 2026 edition), with source section + page returned alongside each chunk
- **Decision Tree fraud classifier** — trained on IEEE-CIS calibrated synthetic data, decision path visible in UI
- **Policy guardrails** — Auditor halts and logs violation type if compliance rules are breached
- **Full audit trail** — every run writes a structured `AuditRecord` to `audit_log.jsonl` and a SQLite `audit_runs` table
- **Streaming UI** — SSE events push per-agent latency, token usage, and cost to the frontend in real time
- **Audit replay panel** — `audit.html` replays any stored run showing DT path, RAG chunks, LLM reasoning, and policy checks

---

## Stack

`LangGraph` · `FastAPI` · `ChromaDB` · `SQLite` · `OpenAI API` · `scikit-learn` · `Vanilla JS`

---

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env          # add OPENAI_API_KEY

python backend/data/seed_demo_data.py      # seed 25 demo disputes
python backend/data/ingest_ieee.py         # load IEEE-CIS transactions (eval only)
python backend/data/ingest_visa_rules.py   # embed Visa rulebook into ChromaDB

uvicorn backend.main:app --reload
open frontend/index.html
```

---

## Evaluation

| Component | Metric | Value |
|---|---|---|
| Fraud classifier | F1 | 0.055 *(baseline — retraining in progress)* |
| RAG retrieval | Hit@3 | 0.25 |
| RAG retrieval | MRR | 0.133 |
