"""
Audit trail writer.

Every completed pipeline run is written to:
  1. audit_log.jsonl  — append-only flat file
  2. audit_runs       — SQLite table, powers the replay dashboard

Call write_audit_record() once at the end of run_auditor().
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4

JSONL_PATH = os.getenv("AUDIT_LOG_PATH", "./data/audit_log.jsonl")
DB_PATH    = os.getenv("SQLITE_DB_PATH",  "./data/disputes.db")

_DDL = """
CREATE TABLE IF NOT EXISTS audit_runs (
    run_id               TEXT PRIMARY KEY,
    transaction_id       TEXT NOT NULL,
    timestamp            TEXT NOT NULL,
    final_verdict        TEXT NOT NULL,
    inv_recommendation   TEXT,
    inv_confidence       REAL,
    inv_dt_fraud_prob    REAL,
    inv_dt_prediction    TEXT,
    aud_violations_count INTEGER DEFAULT 0,
    total_latency_ms     REAL,
    total_tokens         INTEGER,
    total_cost_usd       REAL,
    rag_precision        REAL,
    full_record_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_txn     ON audit_runs(transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts      ON audit_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_verdict ON audit_runs(final_verdict);
"""


@dataclass
class AuditRecord:
    # Identity
    run_id:              str
    transaction_id:      str
    timestamp:           str

    # Raw input
    raw_input:           dict

    # Agent 1 — Triage
    triage_entities:     dict
    triage_masked_stmt:  str
    triage_latency_ms:   float
    triage_tokens_in:    int
    triage_tokens_out:   int
    triage_cost_usd:     float

    # Agent 2 — Investigator
    inv_tool_calls_log:  list       # [{tool, args, result, status}]
    inv_dt_fraud_prob:   float
    inv_dt_prediction:   str
    inv_dt_confidence:   float
    inv_dt_decision_path: list
    inv_dt_top_features: list
    inv_rag_query:       str
    inv_rag_chunks:      list
    inv_rag_precision:   float
    inv_recommendation:  str
    inv_confidence:      float
    inv_fraud_signals:   list
    inv_llm_reasoning:   str
    inv_latency_ms:      float
    inv_tokens_in:       int
    inv_tokens_out:      int
    inv_cost_usd:        float

    # Agent 3 — Auditor
    aud_rules_checked:   list       # [{rule, passed, detail}]
    aud_violations:      list
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
    model_triage:        str
    model_investigator:  str
    model_auditor:       str
    dt_model_version:    str
    pipeline_version:    str


def write_audit_record(record: AuditRecord) -> str:
    """Write record to audit_log.jsonl and audit_runs SQLite table. Returns run_id."""
    record_dict = asdict(record)
    # final_verdict alias keeps backward-compat with _check_no_duplicate in auditor.py
    record_dict["final_verdict"] = record.aud_verdict

    # 1. Append-only JSONL
    os.makedirs(os.path.dirname(JSONL_PATH) or ".", exist_ok=True)
    with open(JSONL_PATH, "a") as fh:
        fh.write(json.dumps(record_dict) + "\n")

    # 2. SQLite — ensure table exists then upsert
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_DDL)
    conn.execute(
        """
        INSERT OR REPLACE INTO audit_runs (
            run_id, transaction_id, timestamp, final_verdict,
            inv_recommendation, inv_confidence, inv_dt_fraud_prob,
            inv_dt_prediction, aud_violations_count,
            total_latency_ms, total_tokens, total_cost_usd,
            rag_precision, full_record_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
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
        ),
    )
    conn.commit()
    conn.close()

    return record.run_id
