# GOVERNANCE.md — Explainability Audit Trail & Replay Dashboard

## Purpose

Every dispute resolution produces a complete, replayable reasoning trace —
DT decision path + RAG clause + LLM reasoning + Agent 3 policy checks —
all stored and linked by `transaction_id`. A separate `/audit` page lets
you replay any past dispute and see exactly why it was decided.

This directly mirrors JPMC's Model Risk Management (MRM) function, which
requires every model-assisted decision to be auditable, explainable, and
replayable by a reviewer who was not present at decision time.

---

## What gets stored per run

Every completed pipeline run writes one entry to two places:
1. `audit_log.jsonl` — append-only flat file (already exists, extend it)
2. `audit_runs` SQLite table — queryable, powers the replay dashboard

### Full audit record schema

```python
@dataclass
class AuditRecord:
    # Identity
    run_id:              str    # uuid4, generated at pipeline start
    transaction_id:      str
    timestamp:           str    # ISO 8601

    # Input
    raw_input:           dict   # original dispute form submission

    # Agent 1 — Triage
    triage_entities:     dict   # extracted entities
    triage_masked_stmt:  str    # PII-masked statement
    triage_latency_ms:   float
    triage_tokens_in:    int
    triage_tokens_out:   int
    triage_cost_usd:     float

    # Agent 2 — Investigator
    inv_tool_calls:      list[dict]   # [{tool, args, result, status}]
    inv_dt_fraud_prob:   float        # DT classifier output
    inv_dt_prediction:   str          # "fraud" | "legitimate"
    inv_dt_decision_path: list[str]   # full DT path
    inv_dt_top_features: list[dict]   # feature importances
    inv_rag_query:       str          # query sent to ChromaDB
    inv_rag_chunks:      list[dict]   # [{text, section, page, relevance}]
    inv_rag_precision:   float
    inv_recommendation:  str          # "approve" | "reject" | "escalate"
    inv_confidence:      float
    inv_fraud_signals:   list[str]
    inv_llm_reasoning:   str          # full LLM chain-of-thought (if available)
    inv_latency_ms:      float
    inv_tokens_in:       int
    inv_tokens_out:      int
    inv_cost_usd:        float

    # Agent 3 — Auditor
    aud_rules_checked:   list[dict]   # [{rule, passed, detail}]
    aud_violations:      list[str]
    aud_verdict:         str
    aud_latency_ms:      float
    aud_tokens_in:       int
    aud_tokens_out:      int
    aud_cost_usd:        float

    # Pipeline totals
    total_latency_ms:    float
    total_tokens:        int
    total_cost_usd:      float

    # Governance metadata
    model_triage:        str    # e.g. "gpt-4o-mini"
    model_investigator:  str    # e.g. "gpt-4o"
    model_auditor:       str    # e.g. "gpt-4o-mini"
    dt_model_version:    str    # e.g. "dt_v1"
    pipeline_version:    str    # e.g. "v1.0.0" from env
```

---

## SQLite schema

Add to `backend/data/seed_demo_data.py` (create table on first run):

```sql
CREATE TABLE IF NOT EXISTS audit_runs (
    run_id               TEXT PRIMARY KEY,
    transaction_id       TEXT NOT NULL,
    timestamp            TEXT NOT NULL,
    final_verdict        TEXT NOT NULL,    -- approve|reject|escalate|policy_violation
    inv_recommendation   TEXT,
    inv_confidence       REAL,
    inv_dt_fraud_prob    REAL,
    inv_dt_prediction    TEXT,
    aud_violations_count INTEGER DEFAULT 0,
    total_latency_ms     REAL,
    total_tokens         INTEGER,
    total_cost_usd       REAL,
    rag_precision        REAL,
    full_record_json     TEXT NOT NULL     -- complete AuditRecord as JSON blob
);

CREATE INDEX IF NOT EXISTS idx_audit_txn ON audit_runs(transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts  ON audit_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_verdict ON audit_runs(final_verdict);
```

The top-level columns enable fast filtering in the dashboard.
`full_record_json` holds the complete record for replay — never query-parse it,
only read it when loading a specific run for replay.

---

## Audit writer

**File:** `backend/governance/audit_writer.py`

```python
import json, sqlite3, os
from datetime import datetime, timezone
from dataclasses import asdict
from uuid import uuid4

JSONL_PATH = os.getenv("AUDIT_LOG_PATH",  "./data/audit_log.jsonl")
DB_PATH    = os.getenv("SQLITE_DB_PATH",  "./data/disputes.db")

def write_audit_record(record: AuditRecord) -> str:
    """
    Write audit record to both audit_log.jsonl and audit_runs SQLite table.
    Returns run_id.
    Call this at the END of run_auditor(), after verdict is determined.
    """
    record_dict = asdict(record)

    # 1. Append to JSONL (append-only, never overwrite)
    with open(JSONL_PATH, "a") as fh:
        fh.write(json.dumps(record_dict) + "\n")

    # 2. Write to SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO audit_runs (
            run_id, transaction_id, timestamp, final_verdict,
            inv_recommendation, inv_confidence, inv_dt_fraud_prob,
            inv_dt_prediction, aud_violations_count,
            total_latency_ms, total_tokens, total_cost_usd,
            rag_precision, full_record_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record.run_id,
        record.transaction_id,
        record.timestamp,
        record.aud_verdict,
        record.inv_recommendation,
        record.inv_confidence,
        record.inv_dt_fraud_prob,
        record.inv_dt_prediction,
        len(record.aud_violations),
        record.total_latency_ms,
        record.total_tokens,
        record.total_cost_usd,
        record.inv_rag_precision,
        json.dumps(record_dict),
    ))
    conn.commit()
    conn.close()

    return record.run_id


def build_audit_record(state: dict, run_id: str) -> AuditRecord:
    """
    Construct AuditRecord from completed LangGraph state.
    Call after all three agents have completed.
    """
    t   = state["triage"]
    inv = state["investigator"]
    aud = state["auditor"]

    return AuditRecord(
        run_id            = run_id,
        transaction_id    = t.transaction_id,
        timestamp         = datetime.now(timezone.utc).isoformat(),
        raw_input         = state["raw_input"],

        triage_entities     = t.entities,
        triage_masked_stmt  = t.masked_statement,
        triage_latency_ms   = t.metrics.latency_ms,
        triage_tokens_in    = t.metrics.tokens_in,
        triage_tokens_out   = t.metrics.tokens_out,
        triage_cost_usd     = t.metrics.cost_usd,

        inv_tool_calls      = inv.tool_calls_made,
        inv_dt_fraud_prob   = inv.dt_output.get("fraud_probability", 0.0),
        inv_dt_prediction   = inv.dt_output.get("prediction", "unknown"),
        inv_dt_decision_path = inv.dt_output.get("decision_path", []),
        inv_dt_top_features = inv.dt_output.get("top_features", []),
        inv_rag_query       = inv.rag_query,
        inv_rag_chunks      = inv.rag_chunks,
        inv_rag_precision   = inv.rag_precision,
        inv_recommendation  = inv.recommendation,
        inv_confidence      = inv.confidence,
        inv_fraud_signals   = inv.fraud_signals,
        inv_llm_reasoning   = inv.llm_reasoning,
        inv_latency_ms      = inv.metrics.latency_ms,
        inv_tokens_in       = inv.metrics.tokens_in,
        inv_tokens_out      = inv.metrics.tokens_out,
        inv_cost_usd        = inv.metrics.cost_usd,

        aud_rules_checked   = [asdict(r) for r in aud.rules_checked],
        aud_violations      = aud.violations,
        aud_verdict         = aud.verdict,
        aud_latency_ms      = aud.metrics.latency_ms,
        aud_tokens_in       = aud.metrics.tokens_in,
        aud_tokens_out      = aud.metrics.tokens_out,
        aud_cost_usd        = aud.metrics.cost_usd,

        total_latency_ms    = t.metrics.latency_ms + inv.metrics.latency_ms + aud.metrics.latency_ms,
        total_tokens        = (t.metrics.tokens_in + t.metrics.tokens_out +
                               inv.metrics.tokens_in + inv.metrics.tokens_out +
                               aud.metrics.tokens_in + aud.metrics.tokens_out),
        total_cost_usd      = t.metrics.cost_usd + inv.metrics.cost_usd + aud.metrics.cost_usd,

        model_triage        = "gpt-4o-mini",
        model_investigator  = "gpt-4o",
        model_auditor       = "gpt-4o-mini",
        dt_model_version    = inv.dt_output.get("model_version", "dt_v1"),
        pipeline_version    = os.getenv("PIPELINE_VERSION", "v1.0.0"),
    )
```

---

## FastAPI audit endpoints

Add to `backend/main.py`:

```python
# GET /audit/runs
# Returns paginated list of past runs for the dashboard table
@app.get("/audit/runs")
def list_audit_runs(
    limit:   int = 20,
    offset:  int = 0,
    verdict: str = None,      # filter by verdict
    txn_id:  str = None,      # filter by transaction_id
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    where_clauses = []
    params = []
    if verdict:
        where_clauses.append("final_verdict = ?")
        params.append(verdict)
    if txn_id:
        where_clauses.append("transaction_id LIKE ?")
        params.append(f"%{txn_id}%")

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    rows = conn.execute(f"""
        SELECT run_id, transaction_id, timestamp, final_verdict,
               inv_dt_fraud_prob, inv_recommendation, inv_confidence,
               aud_violations_count, total_latency_ms, total_cost_usd,
               rag_precision
        FROM audit_runs
        {where}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, params + [limit, offset]).fetchall()

    total = conn.execute(f"SELECT COUNT(*) FROM audit_runs {where}", params).fetchone()[0]
    conn.close()

    return {
        "total": total,
        "runs": [dict(r) for r in rows],
    }


# GET /audit/runs/{run_id}
# Returns full AuditRecord for replay
@app.get("/audit/runs/{run_id}")
def get_audit_run(run_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT full_record_json FROM audit_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    return json.loads(row[0])


# GET /audit/stats
# Aggregate stats for dashboard header cards
@app.get("/audit/stats")
def get_audit_stats():
    conn = sqlite3.connect(DB_PATH)
    stats = conn.execute("""
        SELECT
            COUNT(*)                                        AS total_runs,
            SUM(CASE WHEN final_verdict='approve' THEN 1 ELSE 0 END)  AS approved,
            SUM(CASE WHEN final_verdict='reject'  THEN 1 ELSE 0 END)  AS rejected,
            SUM(CASE WHEN final_verdict='escalate' THEN 1 ELSE 0 END) AS escalated,
            SUM(CASE WHEN final_verdict='policy_violation' THEN 1 ELSE 0 END) AS violations,
            ROUND(AVG(total_latency_ms), 0)                AS avg_latency_ms,
            ROUND(SUM(total_cost_usd), 6)                  AS total_cost_usd,
            ROUND(AVG(inv_dt_fraud_prob), 3)               AS avg_fraud_prob,
            ROUND(AVG(rag_precision), 3)                   AS avg_rag_precision
        FROM audit_runs
    """).fetchone()
    conn.close()

    keys = ["total_runs","approved","rejected","escalated","violations",
            "avg_latency_ms","total_cost_usd","avg_fraud_prob","avg_rag_precision"]
    return dict(zip(keys, stats))
```

---

## Audit dashboard — `/audit` page

**File:** `frontend/audit.html`

A separate page from `index.html`. Linked via a small "Audit log →" button
in the top-right corner of the main demo page.

### Layout

```
┌─ AUDIT DASHBOARD ──────────────────────────────────────────────┐
│  [Total runs: 24]  [Approved: 14]  [Rejected: 5]               │
│  [Escalated: 3]    [Violations: 2] [Avg cost: $0.0082]         │
├────────────────────────────────────────────────────────────────┤
│  Filter: [All verdicts ▼]  [Search TXN ID...]   [Refresh]      │
├────────────────────────────────────────────────────────────────┤
│  TXN ID              │ Verdict  │ DT Prob │ Confidence │ Cost  │
│  TXN-2024-0318-8821  │ APPROVE  │  0.87   │   high     │$0.008 │
│  TXN-2024-0316-9934  │ VIOLATION│  0.94   │   high     │$0.009 │
│  TXN-2024-0317-4499  │ ESCALATE │  0.61   │  medium    │$0.008 │
│  ...                 │          │         │            │       │
├────────────────────────────────────────────────────────────────┤
│  [← Prev]  Page 1 of 2  [Next →]                               │
└────────────────────────────────────────────────────────────────┘
```

Clicking any row opens the **Replay Panel** (see below).

### Replay panel

When a row is clicked, fetch `GET /audit/runs/{run_id}` and render the full
AuditRecord as a read-only version of the main stepper UI — same five panels,
same layout, but populated from stored data instead of a live SSE stream.

Key differences from the live demo view:
- Banner at top: `"REPLAY — Run from {timestamp}"` in amber
- All panels are immediately visible (no animation delay)
- A "What changed vs. policy" diff section if `aud_violations` is non-empty
- DT decision path rendered exactly as in live demo
- Full RAG chunks shown (not just the top result)
- LLM reasoning shown verbatim in a collapsible block

### `audit.js` — key functions

```javascript
// Load stats for header cards
async function loadStats() {
  const res = await fetch("http://localhost:8000/audit/stats");
  const stats = await res.json();
  document.getElementById("total-runs").textContent  = stats.total_runs;
  document.getElementById("approved").textContent    = stats.approved;
  document.getElementById("rejected").textContent    = stats.rejected;
  document.getElementById("escalated").textContent   = stats.escalated;
  document.getElementById("violations").textContent  = stats.violations;
  document.getElementById("avg-cost").textContent    = fmt.cost(stats.total_cost_usd);
}

// Load paginated run list
async function loadRuns(page = 0, verdict = null, txnSearch = null) {
  const params = new URLSearchParams({
    limit: 20,
    offset: page * 20,
    ...(verdict   && { verdict }),
    ...(txnSearch && { txn_id: txnSearch }),
  });
  const res  = await fetch(`http://localhost:8000/audit/runs?${params}`);
  const data = await res.json();
  renderTable(data.runs);
  renderPagination(data.total, page);
}

// Load and render a specific run for replay
async function replayRun(runId) {
  const res    = await fetch(`http://localhost:8000/audit/runs/${runId}`);
  const record = await res.json();
  renderReplayPanel(record);
}

// Render the replay stepper from stored AuditRecord
function renderReplayPanel(record) {
  // Show replay banner
  document.getElementById("replay-banner").textContent =
    `REPLAY — Run ${record.run_id.slice(0,8)} · ${new Date(record.timestamp).toLocaleString()}`;

  // Populate all five panels from record fields
  populateTriagePanel(record);
  populateInvestigatorPanel(record);
  populateAuditorPanel(record);
  populateResultsPanel(record);

  // Show all panels simultaneously (no animation)
  document.querySelectorAll(".panel").forEach(p => p.classList.add("visible"));
}
```

---

## Verdict badge color rules (consistent across demo + audit pages)

```javascript
const VERDICT_COLORS = {
  "approve":          { bg: "#E1F5EE", text: "#085041", label: "Approved"  },
  "reject":           { bg: "#FCEBEB", text: "#791F1F", label: "Rejected"  },
  "escalate":         { bg: "#FFF8E7", text: "#633806", label: "Escalate"  },
  "policy_violation": { bg: "#FAECE7", text: "#712B13", label: "Violation" },
};

function verdictBadge(verdict) {
  const c = VERDICT_COLORS[verdict] || { bg: "#eee", text: "#333", label: verdict };
  return `<span class="badge" style="background:${c.bg};color:${c.text}">${c.label}</span>`;
}
```

---

## LLM reasoning capture

To populate `inv_llm_reasoning`, update the Investigator agent to capture
the final LLM message content before JSON parsing:

```python
# In run_investigator(), after the tool call loop exits:
raw_reasoning = response.choices[0].message.content  # full text before json.loads()
output = json.loads(raw_reasoning)
output["_llm_reasoning_raw"] = raw_reasoning   # attach before returning
```

Then in `build_audit_record()`:
```python
inv_llm_reasoning = inv.dt_output.get("_llm_reasoning_raw", "")
```

This gives the replay panel the complete LLM output — useful for an MRM reviewer
to see whether the model's written reasoning matches its structured verdict.

---

## Interview framing for this module

> *"Every dispute in the system produces a complete audit record — the DT decision
> path, the RAG clause that was retrieved, the LLM's full reasoning, and the
> independent policy checks — all stored and linked by transaction ID. There's a
> separate audit dashboard where any past run can be replayed in full. This mirrors
> how JPMC's Model Risk Management function operates: any model-assisted decision
> needs to be replayable and explainable to a reviewer who wasn't present at
> decision time, with no dependency on the original model being available."*

Key point to emphasize: **the replay works even if the model is retrained or
replaced** — because the reasoning trace is stored as data, not recomputed.

---

## Files created by this module

```
backend/governance/
└── audit_writer.py        AuditRecord dataclass + write_audit_record()

frontend/
├── audit.html             audit dashboard page
└── audit.js               dashboard + replay logic
```

## Integration checklist

- [ ] `AuditRecord` dataclass defined in `audit_writer.py`
- [ ] `build_audit_record()` called at end of `run_auditor()` in LangGraph graph
- [ ] `write_audit_record()` called immediately after build
- [ ] `audit_runs` table created in `seed_demo_data.py`
- [ ] Three audit endpoints added to `main.py`
- [ ] `audit.html` linked from `index.html` top-right corner
- [ ] Replay panel populates all five sections from stored record
- [ ] Replay banner distinguishes live run from replay visually
- [ ] `inv_llm_reasoning` captured from raw LLM message content
