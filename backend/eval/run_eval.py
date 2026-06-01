"""
Batch F1 evaluator — runs N disputes from the IEEE-CIS eval set through the
full three-agent pipeline and computes fraud classification metrics.

Data source: data/ieee_eval_set.csv  (20% split produced by ingest_ieee.py)
Never uses demo_disputes — this is the only context where IEEE data is used.

Estimated cost: ~$0.009 per dispute (dominated by gpt-4o Investigator calls).
  50 samples ≈ $0.45    ← default
 200 samples ≈ $1.80
 500 samples ≈ $4.50

Usage:
  python3 -m backend.eval.run_eval                  # 50 samples
  python3 -m backend.eval.run_eval --n 200
  python3 -m backend.eval.run_eval --n 200 --seed 7
"""
import argparse
import json
import os
import sys
import time
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score

load_dotenv()

EVAL_CSV = os.getenv("EVAL_CSV_PATH", "./data/ieee_eval_set.csv")

# ProductCD → merchant label for readable dispute statements
_PRODUCT_LABELS = {
    "W": "Electronics & Gadgets",
    "H": "Hotel & Lodging",
    "C": "Cash Advance Service",
    "S": "General Services",
    "R": "Retail Store",
}

# All disputes are treated as unauthorized (reason code 10.4) for eval
_REASON_CODE = "10.4"


def _row_to_dispute(row: pd.Series):
    """Convert one IEEE-CIS row to a DisputeInput for the pipeline."""
    from backend.agents.triage import DisputeInput

    txn_id   = f"IEEE-{int(row['TransactionID'])}"
    amount   = float(row["TransactionAmt"])
    product  = str(row.get("ProductCD", "W")).strip()
    merchant = _PRODUCT_LABELS.get(product, "Online Merchant")
    email    = row.get("P_emaildomain", "unknown.com") or "unknown.com"

    statement = (
        f"I did not authorise a ${amount:.2f} charge from {merchant} "
        f"on my card. The purchase was linked to email domain '{email}'. "
        "Please investigate this transaction."
    )

    return DisputeInput(
        transaction_id=txn_id,
        amount=amount,
        merchant=merchant,
        reason_code=_REASON_CODE,
        customer_statement=statement,
    )


def _verdict_to_label(verdict: str) -> int:
    """Map pipeline verdict to binary fraud label (1=fraud, 0=not fraud)."""
    return 1 if verdict in ("reject", "policy_violation") else 0


def run_pipeline_sync(dispute) -> dict:
    """Run the full three-agent pipeline synchronously; return a summary dict."""
    from backend.agents.auditor import run_auditor
    from backend.agents.investigator import run_investigator
    from backend.agents.triage import run_triage

    triage_out,  triage_m  = run_triage(dispute)
    inv_out,     inv_m     = run_investigator(triage_out)
    aud_out,     aud_m     = run_auditor(
        dispute=dispute,
        triage=triage_out,
        investigator=inv_out,
        triage_metrics=triage_m,
        investigator_metrics=inv_m,
        filing_date=date.today(),
    )

    total_cost = round(triage_m.cost_usd + inv_m.cost_usd + aud_m.cost_usd, 6)
    total_ms   = round(triage_m.latency_ms + inv_m.latency_ms + aud_m.latency_ms, 1)

    return {
        "verdict":         aud_out.verdict,
        "recommendation":  inv_out.recommendation,
        "confidence":      inv_out.confidence,
        "total_cost_usd":  total_cost,
        "total_latency_ms": total_ms,
    }


def run_batch_eval(n_samples: int = 50, seed: int = 42) -> dict:
    if not os.path.exists(EVAL_CSV):
        sys.exit(
            f"Eval CSV not found at {EVAL_CSV!r}.\n"
            "Run:  python3 backend/data/ingest_ieee.py  first."
        )

    df = pd.read_csv(EVAL_CSV)
    if "isFraud" not in df.columns:
        sys.exit("isFraud column missing from eval CSV.")

    df = df.dropna(subset=["isFraud", "TransactionAmt"]).sample(
        min(n_samples, len(df)), random_state=seed
    )
    actual_n = len(df)

    est_cost = round(actual_n * 0.009, 2)
    print(f"\nRunning batch eval: {actual_n} samples (seed={seed})")
    print(f"Estimated API cost: ~${est_cost:.2f}")
    print("-" * 50)

    y_true, y_pred = [], []
    costs, latencies = [], []
    errors = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        dispute = _row_to_dispute(row)
        true_label = int(row["isFraud"])

        try:
            result = run_pipeline_sync(dispute)
            pred_label = _verdict_to_label(result["verdict"])
            costs.append(result["total_cost_usd"])
            latencies.append(result["total_latency_ms"])
        except Exception as exc:
            print(f"  [{i}/{actual_n}] ERROR on {dispute.transaction_id}: {exc}")
            pred_label = 0
            errors += 1

        y_true.append(true_label)
        y_pred.append(pred_label)

        if i % 10 == 0:
            running_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            print(
                f"  [{i}/{actual_n}]  running F1={running_f1:.3f}  "
                f"cost so far=${sum(costs):.3f}"
            )

    print("\n" + "=" * 50)
    print(classification_report(y_true, y_pred, target_names=["not_fraud", "fraud"]))

    f1       = f1_score(y_true, y_pred, average="binary", zero_division=0)
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    avg_lat  = sum(latencies) / len(latencies) if latencies else 0.0
    total_c  = sum(costs)

    print(f"F1 (binary):       {f1:.4f}")
    print(f"Avg cost/dispute:  ${avg_cost:.5f}")
    print(f"Avg latency:       {avg_lat:.0f}ms")
    print(f"Total eval cost:   ${total_c:.4f}")
    print(f"Errors:            {errors}/{actual_n}")

    results = {
        "f1":               round(f1, 4),
        "n_samples":        actual_n,
        "seed":             seed,
        "avg_cost_usd":     round(avg_cost, 5),
        "avg_latency_ms":   round(avg_lat, 1),
        "total_cost_usd":   round(total_c, 4),
        "errors":           errors,
    }

    # Write results to file for later reference
    out_path = "./data/eval_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written → {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE-CIS batch F1 evaluator")
    parser.add_argument("--n",    type=int, default=50,  help="Number of samples (default: 50)")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed (default: 42)")
    args = parser.parse_args()

    run_batch_eval(n_samples=args.n, seed=args.seed)
