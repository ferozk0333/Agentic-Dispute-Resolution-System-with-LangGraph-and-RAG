"""
Pipeline end-to-end evaluation — runs all 25 demo disputes through the full
three-agent pipeline and checks verdict accuracy against ground truth.

Also runs a 5-run consistency check on one sample dispute.
"""

import os
import sqlite3
import tempfile
from datetime import date

from backend.agents.auditor import run_auditor
from backend.agents.investigator import run_investigator
from backend.agents.triage import DisputeInput, run_triage

DB_PATH = "data/disputes.db"

_CONSISTENCY_SAMPLE = DisputeInput(
    transaction_id="TXN-2024-0318-8821",
    amount=1240.00,
    merchant="ElectroMart Inc.",
    reason_code="10.4",
    customer_statement="I did not authorize this charge.",
)


def _run_dispute(dispute: dict) -> str:
    """Run one dispute through the full pipeline and return the final verdict."""
    inp = DisputeInput(
        transaction_id=dispute["transaction_id"],
        amount=dispute["transaction_amt"],
        merchant=dispute["merchant_name"],
        reason_code="10.4",
        customer_statement=(
            f"I did not authorize this charge at {dispute['merchant_name']}."
        ),
    )
    triage_out, triage_m     = run_triage(inp)
    inv_out,    inv_m        = run_investigator(triage_out)
    aud_out,    _            = run_auditor(
        dispute=inp,
        triage=triage_out,
        investigator=inv_out,
        triage_metrics=triage_m,
        investigator_metrics=inv_m,
        filing_date=date.today(),
    )
    return aud_out.verdict


def eval_pipeline() -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    disputes = conn.execute(
        "SELECT * FROM demo_disputes ORDER BY transaction_id"
    ).fetchall()
    conn.close()

    correct            = 0
    violations_seeded  = 0
    violations_caught  = 0
    results            = []

    # Use a temporary audit log so prior UI-demo runs don't contaminate the
    # no_duplicate check. Within the eval run, ordering is preserved — TXN-8821
    # is written before TXN-8821-DUP is checked.
    tmp_log = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp_log.close()
    os.environ["AUDIT_LOG_PATH"] = tmp_log.name

    total = len(disputes)
    for i, row in enumerate(disputes, 1):
        dispute  = dict(row)
        expected = dispute["dispute_outcome"]

        print(f"  [{i:02d}/{total}] {dispute['transaction_id']} (expected={expected})", end="", flush=True)
        try:
            predicted = _run_dispute(dispute)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            predicted = "error"

        match = predicted == expected

        if expected == "policy_violation":
            violations_seeded += 1
            if predicted == "policy_violation":
                violations_caught += 1

        results.append({
            "transaction_id": dispute["transaction_id"],
            "scenario":       dispute["dispute_scenario"],
            "expected":       expected,
            "predicted":      predicted,
            "correct":        match,
        })

        marker = "✓" if match else f"✗ got={predicted}"
        print(f"  {marker}")
        if match:
            correct += 1

    # Clean up temp audit log
    os.unlink(tmp_log.name)
    os.environ.pop("AUDIT_LOG_PATH", None)

    accuracy   = correct / total
    catch_rate = violations_caught / max(violations_seeded, 1)

    print(f"\n── PIPELINE EVALUATION ─────────────────────────────")
    print(f"Verdict accuracy:   {correct}/{total} ({accuracy * 100:.1f}%)")
    print(f"Violation catch:    {violations_caught}/{violations_seeded} ({catch_rate * 100:.1f}%)")

    mismatches = [r for r in results if not r["correct"]]
    if mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print(f"  {m['transaction_id']} — expected={m['expected']}, got={m['predicted']}")

    return {
        "verdict_accuracy":     round(accuracy, 4),
        "correct":              correct,
        "total":                total,
        "violation_catch_rate": round(catch_rate, 4),
        "results":              results,
    }


def eval_consistency(n_runs: int = 5) -> dict:
    """Run the same dispute n_runs times and verify verdict stability."""
    verdicts = []
    for i in range(n_runs):
        print(f"  Consistency run {i + 1}/{n_runs}...", end="", flush=True)
        try:
            triage_out, triage_m = run_triage(_CONSISTENCY_SAMPLE)
            inv_out,    inv_m    = run_investigator(triage_out)
            aud_out,    _        = run_auditor(
                dispute=_CONSISTENCY_SAMPLE,
                triage=triage_out,
                investigator=inv_out,
                triage_metrics=triage_m,
                investigator_metrics=inv_m,
                filing_date=date.today(),
            )
            verdicts.append(aud_out.verdict)
            print(f"  {aud_out.verdict}")
        except Exception as exc:
            verdicts.append("error")
            print(f"  ERROR: {exc}")

    unique_verdicts = list(set(verdicts))
    consistent      = len(unique_verdicts) == 1

    print(f"\n── CONSISTENCY TEST ────────────────────────────────")
    print(f"Verdicts across {n_runs} runs: {verdicts}")
    print(f"Consistent: {consistent}")
    if not consistent:
        print(f"  WARNING: non-deterministic verdicts detected: {unique_verdicts}")
        print(f"  Fix: set temperature=0 in all OpenAI calls")

    return {
        "consistent":      consistent,
        "verdicts":        verdicts,
        "unique_verdicts": unique_verdicts,
        "n_runs":          n_runs,
    }
