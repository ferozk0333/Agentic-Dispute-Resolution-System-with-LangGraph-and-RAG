"""
FastAPI entry point.

POST /resolve  — accepts DisputeInput JSON, streams SSE events per agent step.
GET  /health   — liveness check.

Each SSE event is a JSON object:
  { "agent": "triage"|"investigator"|"auditor",
    "output": {...},
    "metrics": {latency_ms, prompt_tokens, completion_tokens, cost_usd},
    "halted": bool,          # auditor only
    "halt_reason": str|null  # auditor only, first violation if halted
  }
A final `event: done` signals the stream is closed.
"""
import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from backend.agents.triage import DisputeInput
from backend.graph import DisputeState, _initial_state, build_graph

_AUDIT_LOG = os.getenv("AUDIT_LOG_PATH", "./data/audit_log.jsonl")
_DB_PATH   = os.getenv("SQLITE_DB_PATH",  "./data/disputes.db")


class FlagRequest(BaseModel):
    transaction_id: str
    reason: str = "Flagged for manual review via UI"

load_dotenv()

app = FastAPI(title="Dispute Resolution API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _serialise(obj) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


async def _stream_pipeline(dispute: DisputeInput) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph pipeline in a thread pool and yield SSE-formatted strings
    as each agent completes. The sync graph.stream() call blocks its thread;
    events are passed back to the async generator via an asyncio.Queue.
    """
    graph = build_graph()
    initial = _initial_state(dispute)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            # Accumulate state across steps so the auditor event can reference
            # triage/investigator metrics for pipeline_totals.
            accumulated: dict = {}

            for step in graph.stream(initial):
                for agent_name, updates in step.items():
                    accumulated.update(updates)
                    payload: dict = {"agent": agent_name}

                    if agent_name == "triage" and updates.get("triage_output"):
                        payload["output"]  = _serialise(updates["triage_output"])
                        payload["metrics"] = _serialise(updates["triage_metrics"])

                    elif agent_name == "investigator" and updates.get("investigator_output"):
                        payload["output"]  = _serialise(updates["investigator_output"])
                        payload["metrics"] = _serialise(updates["investigator_metrics"])

                    elif agent_name == "auditor" and updates.get("auditor_output"):
                        payload["output"]      = _serialise(updates["auditor_output"])
                        payload["metrics"]     = _serialise(updates["auditor_metrics"])
                        payload["halted"]      = updates.get("halted", False)
                        payload["halt_reason"] = updates.get("halt_reason")

                        t_m = accumulated.get("triage_metrics")
                        i_m = accumulated.get("investigator_metrics")
                        a_m = updates["auditor_metrics"]
                        if t_m and i_m and a_m:
                            payload["pipeline_totals"] = {
                                "total_latency_ms": round(
                                    t_m.latency_ms + i_m.latency_ms + a_m.latency_ms, 1
                                ),
                                "total_tokens_in":  t_m.tokens_in  + i_m.tokens_in  + a_m.tokens_in,
                                "total_tokens_out": t_m.tokens_out + i_m.tokens_out + a_m.tokens_out,
                                "total_tokens": (
                                    t_m.tokens_in + t_m.tokens_out
                                    + i_m.tokens_in + i_m.tokens_out
                                    + a_m.tokens_in + a_m.tokens_out
                                ),
                                "total_cost_usd": round(
                                    t_m.cost_usd + i_m.cost_usd + a_m.cost_usd, 6
                                ),
                            }

                    loop.call_soon_threadsafe(queue.put_nowait, payload)

        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"error": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # end sentinel

    future = loop.run_in_executor(None, _run)

    while True:
        item = await queue.get()
        if item is None:
            yield "event: done\ndata: {}\n\n"
            break
        yield f"data: {json.dumps(item)}\n\n"

    await future  # surface any thread exception


@app.post("/resolve")
async def resolve_dispute(dispute: DisputeInput):
    return StreamingResponse(
        _stream_pipeline(dispute),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/flag")
async def flag_for_review(body: FlagRequest):
    """Append a manual-review flag entry to the audit log."""
    entry = {
        "type":           "manual_review_flag",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "transaction_id": body.transaction_id,
        "reason":         body.reason,
    }
    os.makedirs(os.path.dirname(_AUDIT_LOG) or ".", exist_ok=True)
    with open(_AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"flagged": True, "transaction_id": body.transaction_id}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Audit endpoints ───────────────────────────────────────────────────────────

@app.get("/audit/runs")
def list_audit_runs(
    limit:   int            = 20,
    offset:  int            = 0,
    verdict: Optional[str]  = None,
    txn_id:  Optional[str]  = None,
):
    """Paginated list of past runs for the audit dashboard table."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row

        where_clauses: list[str] = []
        params: list = []
        if verdict:
            where_clauses.append("final_verdict = ?")
            params.append(verdict)
        if txn_id:
            where_clauses.append("transaction_id LIKE ?")
            params.append(f"%{txn_id}%")

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        rows = conn.execute(
            f"""
            SELECT run_id, transaction_id, timestamp, final_verdict,
                   inv_dt_fraud_prob, inv_recommendation, inv_confidence,
                   aud_violations_count, total_latency_ms, total_cost_usd,
                   rag_precision
            FROM audit_runs
            {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

        total = conn.execute(
            f"SELECT COUNT(*) FROM audit_runs {where}", params
        ).fetchone()[0]
        conn.close()

        return {"total": total, "runs": [dict(r) for r in rows]}
    except sqlite3.OperationalError:
        return {"total": 0, "runs": []}


@app.get("/audit/runs/{run_id}")
def get_audit_run(run_id: str):
    """Full AuditRecord for replay — loads from full_record_json column."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        row  = conn.execute(
            "SELECT full_record_json FROM audit_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        conn.close()
    except sqlite3.OperationalError:
        raise HTTPException(status_code=404, detail="Run not found")

    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    return json.loads(row[0])


@app.get("/audit/stats")
def get_audit_stats():
    """Aggregate stats for the audit dashboard header cards."""
    try:
        conn  = sqlite3.connect(_DB_PATH)
        stats = conn.execute("""
            SELECT
                COUNT(*)                                                     AS total_runs,
                SUM(CASE WHEN final_verdict='approve'          THEN 1 ELSE 0 END) AS approved,
                SUM(CASE WHEN final_verdict='reject'           THEN 1 ELSE 0 END) AS rejected,
                SUM(CASE WHEN final_verdict='escalate'         THEN 1 ELSE 0 END) AS escalated,
                SUM(CASE WHEN final_verdict='policy_violation' THEN 1 ELSE 0 END) AS violations,
                ROUND(AVG(total_latency_ms), 0)                              AS avg_latency_ms,
                ROUND(SUM(total_cost_usd), 6)                                AS total_cost_usd,
                ROUND(AVG(inv_dt_fraud_prob), 3)                             AS avg_fraud_prob,
                ROUND(AVG(rag_precision), 3)                                 AS avg_rag_precision
            FROM audit_runs
        """).fetchone()
        conn.close()
    except sqlite3.OperationalError:
        return {k: 0 for k in ["total_runs","approved","rejected","escalated","violations",
                                "avg_latency_ms","total_cost_usd","avg_fraud_prob","avg_rag_precision"]}

    keys = ["total_runs","approved","rejected","escalated","violations",
            "avg_latency_ms","total_cost_usd","avg_fraud_prob","avg_rag_precision"]
    return dict(zip(keys, stats))


@app.get("/eval-results")
async def eval_results():
    path = "./data/eval_results.json"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Eval results not found. Run: python3 -m backend.eval.run_eval",
        )
    with open(path) as f:
        return json.load(f)
