"""
Governance layer evaluation — checks AuditRecord completeness and
LLM reasoning / verdict consistency for all stored audit runs.
"""

import json
import sqlite3

DB_PATH = "data/disputes.db"

REQUIRED_FIELDS = [
    "run_id", "transaction_id", "timestamp",
    "triage_entities", "triage_masked_stmt",
    "inv_tool_calls_log", "inv_dt_fraud_prob", "inv_dt_decision_path",
    "inv_rag_query", "inv_rag_chunks", "inv_recommendation", "inv_confidence",
    "aud_rules_checked", "aud_violations", "aud_verdict",
    "total_latency_ms", "total_tokens", "total_cost_usd",
]


def eval_governance() -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT run_id, full_record_json FROM audit_runs"
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        print("  WARNING: audit_runs table not found. Run pipeline eval first.")
        return {"error": "no_records", "total_records": 0,
                "completeness_rate": 0, "consistency_rate": 0,
                "completeness_failures": [], "reasoning_mismatches": []}

    if not rows:
        print("  WARNING: No audit records found. Run pipeline eval first.")
        return {"error": "no_records", "total_records": 0,
                "completeness_rate": 0, "consistency_rate": 0,
                "completeness_failures": [], "reasoning_mismatches": []}

    completeness_failures = []
    reasoning_mismatches  = []

    for run_id, record_json in rows:
        record = json.loads(record_json)

        for field in REQUIRED_FIELDS:
            if record.get(field) is None:
                completeness_failures.append({
                    "run_id":        run_id,
                    "missing_field": field,
                })

        reasoning = record.get("inv_llm_reasoning", "")
        verdict   = record.get("inv_recommendation", "")
        if reasoning and verdict:
            rl = reasoning.lower()
            if verdict == "approve" and "reject" in rl and "approve" not in rl:
                reasoning_mismatches.append({
                    "run_id":  run_id,
                    "verdict": verdict,
                    "issue":   "reasoning says reject, verdict=approve",
                })
            elif verdict == "reject" and "approve" in rl and "reject" not in rl:
                reasoning_mismatches.append({
                    "run_id":  run_id,
                    "verdict": verdict,
                    "issue":   "reasoning says approve, verdict=reject",
                })

    total             = len(rows)
    failed_run_ids    = {f["run_id"] for f in completeness_failures}
    complete          = total - len(failed_run_ids)
    completeness_rate = complete / total if total > 0 else 0.0
    consistency_rate  = (total - len(reasoning_mismatches)) / total if total > 0 else 0.0

    print(f"\n── GOVERNANCE EVALUATION ───────────────────────────")
    print(f"Audit records:         {total}")
    print(f"Completeness:          {complete}/{total} ({completeness_rate * 100:.1f}%)")
    print(f"Reasoning consistency: {total - len(reasoning_mismatches)}/{total} ({consistency_rate * 100:.1f}%)")

    if completeness_failures:
        print(f"\nCompleteness failures:")
        for f in completeness_failures[:5]:
            print(f"  {f['run_id'][:8]}: missing {f['missing_field']}")

    if reasoning_mismatches:
        print(f"\nReasoning mismatches:")
        for m in reasoning_mismatches:
            print(f"  {m['run_id'][:8]}: {m['issue']}")

    return {
        "total_records":          total,
        "completeness_rate":      round(completeness_rate, 4),
        "consistency_rate":       round(consistency_rate, 4),
        "completeness_failures":  completeness_failures,
        "reasoning_mismatches":   reasoning_mismatches,
    }
