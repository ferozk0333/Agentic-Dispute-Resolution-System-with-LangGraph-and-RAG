# Agentic Dispute Resolution System with LangGraph and RAG

**[Read the full explainer →](https://ferozk0333.github.io/Agentic-Fraud-Detection-System-with-LangGraph-and-RAG/)**

---

<img width="800" height="703" alt="demo" src="https://github.com/user-attachments/assets/4629a71f-fdaa-4d38-8cf6-667ffb6e4cdd" />


*Input a dispute, watch three agents reason through it in real time.*

---

## What it does

A three-agent pipeline that resolves credit card disputes end-to-end: extracts entities and masks PII, investigates using live tool calls and a RAG-grounded Visa rulebook lookup, then runs deterministic policy checks before issuing a verdict.

## Agents

| Agent | Model | Role |
|---|---|---|
| Triage | gpt-4o-mini | NER, PII masking, urgency classification |
| Investigator | gpt-4o | 5-tool loop, fraud classifier, Visa RAG retrieval |
| Auditor | gpt-4o-mini | 6 deterministic policy checks, independent of Agent 2's reasoning |

## Stack

LangGraph + LangChain · OpenAI API · ChromaDB · SQL · FastAPI

## Key design decisions

- **Interpretable classifier** — Decision Tree over XGBoost so every prediction exposes a human-readable decision path, not just a score
- **Hierarchical RAG chunking** — chunks prefixed with ancestor breadcrumbs to retrieve specific Visa rule clauses; Hit@3 improved from 0.25 to 0.893
- **Agent isolation** — Auditor receives only Agent 2's structured output, never its reasoning trace, enforcing independent adjudication
- **Deterministic guardrails** — 4 of 6 policy checks run in Python before any LLM call; confidence floor of 0.85 required to approve or reject

## Eval

| Metric | Value |
|---|---|
| RAG Hit@3 | 0.893 |
| RAG MRR | 0.738 |
| Classifier CV F1 | 0.853 ± 0.023 |
| Classifier ROC-AUC | 0.966 ± 0.009 |
| Audit completeness | 100% |

## Quickstart

```bash
pip install -r requirements.txt
python backend/data/seed_demo_data.py
python backend/data/ingest_visa_rules.py
uvicorn backend.main:app --reload
open frontend/index.html
```

Set `OPENAI_API_KEY` and `CHROMA_PERSIST_DIR` in a `.env` file before running.
