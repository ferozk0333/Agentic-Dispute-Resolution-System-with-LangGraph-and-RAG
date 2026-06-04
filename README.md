# Agentic Fraud Detection System with LangGraph and RAG

A three-agent LangGraph pipeline that resolves credit card disputes end-to-end. A raw dispute claim enters Agent 1, which fetches transaction records, masks PII, and extracts structured entities. Agent 2 runs four parallel checks, a Decision Tree fraud classifier on the transaction, velocity and risk signals pulled from disparate data sources, and a RAG retrieval over the Visa Core Rules to surface the exact policy clause governing the claim. Agent 3 applies deterministic compliance checks (example, 120-day dispute window) and issues a final decision, halting the pipeline if a rule is violated.

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
| Auditor | `gpt-4o-mini` | Checks compliance rules, issues final decision (Reject/Allow claim, Escalate to human, Policy Violation) |


---

## Key features

- **RAG over Visa Core Rules:** ChromaDB retrieval grounded in the real Visa rulebook (April 2026 edition), with source section + page returned alongside each chunk
- **Decision Tree fraud classifier:** Trained on IEEE-CIS calibrated data, decision path visible in UI
- **Policy guardrails:** Auditor halts and logs violation type if compliance rules are breached
- **Full audit trail:** Rvery run writes a structured `AuditRecord` to `audit_log.jsonl` as part of Governance layer
- **Streaming UI:** SSE events push per-agent latency, token usage, and cost to the frontend in real time

---

## Stack

`LangGraph` · `FastAPI` · `ChromaDB` · `SQLite` · `OpenAI API` · `scikit-learn` · `Vanilla JS`

---

## Setup

```bash
pip install -r requirements.txt

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
