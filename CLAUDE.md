# CLAUDE.md — Multi-Agent Dispute Resolution System

## What this project is

A portfolio-grade AI system that resolves credit card disputes using a three-agent pipeline grounded in real fraud data (IEEE-CIS) and real compliance rules (Visa public rulebook). Built to demonstrate multi-agent orchestration, RAG-grounded reasoning, deterministic guardrails, and cost-aware model routing.

## Repo structure

```
dispute-agent/
├── CLAUDE.md
├── docs/
│   ├── ARCHITECTURE.md   agent design, LangGraph state, model routing
│   ├── FRONTEND.md       HTML/CSS/JS stepper UI spec
│   ├── DATA.md           IEEE-CIS setup, SQLite schema, RAG ingestion
│   ├── METRICS.md        latency, cost, F1, RAG precision instrumentation
│   └── AGENTS.md         per-agent prompts, tool schemas, output contracts
├── backend/
│   ├── main.py           FastAPI entry point
│   ├── agents/
│   │   ├── triage.py
│   │   ├── investigator.py
│   │   └── auditor.py
│   ├── tools/
│   │   ├── sql_tool.py
│   │   ├── rag_tool.py
│   │   ├── risk_tool.py
│   │   └── velocity_tool.py
│   │   └── classifier_tool.py
│   ├── classifier/
│   │   ├── calibrate.py
│   │   ├── generate_training_data.py
│   │   ├── train.py
│   │   ├── fraud_dt.joblib        (generated)
│   │   └── model_metadata.json   (generated)
│   ├── governance/
│   │   └── audit_writer.py
│   └── data/
│       ├── ingest_ieee.py
│       └── ingest_visa_rules.py
└── frontend/
    ├── index.html
    ├── audit.html
    ├── style.css
    ├── app.js
    └── audit.js
```

## Build order

Follow this sequence exactly. Do not skip ahead.

1. `docs/DATA.md` — set up data layer first (SQLite + ChromaDB). Nothing else works without it.
2. `docs/AGENTS.md` — implement agents with tool schemas before wiring the graph.
3. `docs/ARCHITECTURE.md` — wire LangGraph state graph once agents are individually tested.
4. `docs/METRICS.md` — instrument after pipeline runs end-to-end.
5. `docs/FRONTEND.md` — build UI last, consuming the FastAPI `/resolve` endpoint.
6. `docs/CLASSIFIER.md` — generate synthetic data, train DT, verify inference tool works standalone.
7. `docs/GOVERNANCE.md` — add audit writer, audit endpoints, integrate into LangGraph graph.

## Stack

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph (StateGraph) |
| LLM provider | OpenAI API |
| Backend | FastAPI + uvicorn |
| Vector store | ChromaDB (local persistent) |
| Transaction DB | SQLite (IEEE-CIS data) |
| Frontend | Vanilla HTML + CSS + JS (no framework) |
| Embeddings | OpenAI Embeddings |

## Model routing

| Agent | Model | Justification |
|---|---|---|
| Triage | `gpt-4o-mini` | Simple NER + PII masking, minimize cost |
| Investigator | `gpt-4o` | Multi-tool reasoning, RAG synthesis, highest accuracy needed |
| Auditor | `gpt-4o-mini` | Deterministic rule checks only, no creativity required |

## Environment variables

```
OPENAI_API_KEY=
CHROMA_PERSIST_DIR=./data/chroma
SQLITE_DB_PATH=./data/disputes.db
IEEE_CSV_PATH=./data/train_transaction.csv
VISA_RULES_PDF_PATH=./data/visa_core_rules.pdf
```

## Key constraints

- Agent 3 (Auditor) must receive only Agent 2's structured output — not its chain of thought or tool call history. Enforced via LangGraph state typing.
- Every dispute run must produce a structured audit log entry written to `audit_log.jsonl`.
- All token counts and costs must come from `response.usage` — never estimated.
- Frontend communicates with backend via a single POST `/resolve` endpoint that streams SSE events per agent step.
- No external CSS frameworks. No React. Vanilla JS only.

## How to run

```bash
pip install -r requirements.txt
python backend/data/seed_demo_data.py     # seed 25 hand-crafted disputes into SQLite
python backend/data/ingest_ieee.py        # load IEEE-CIS into SQLite (eval only)
python backend/data/ingest_visa_rules.py  # chunk + embed Visa PDF into ChromaDB
uvicorn backend.main:app --reload         # start API on :8000
open frontend/index.html                  # open UI in browser
```

## Data split — important

- `demo_disputes` table → live demo tool calls. Has real merchant names, dates, narratives.
- `ieee_transactions` table → batch eval F1 scoring only. Never queried by agents during demo.
- Do NOT mix them. Agents never touch `ieee_transactions` at runtime.

## Definition of done

 Single dispute resolves end-to-end through all three agents
 UI shows per-agent latency, token usage, and cost at each step
 Full AuditRecord written to audit_log.jsonl AND audit_runs SQLite table on every run
 Batch eval script produces F1 score on IEEE-CIS test set
 RAG tool returns source clause (section + page) alongside retrieved text
 Policy violation correctly halts pipeline and logs violation type
 DT classifier runs as tool in Agent 2, decision path visible in UI
 GET /audit/runs returns paginated run list
 GET /audit/runs/{run_id} returns full record for replay
 audit.html renders replay panel from stored AuditRecord
 Replay panel shows DT path, RAG chunks, LLM reasoning, and policy checks
 Replay banner distinguishes stored replay from live run