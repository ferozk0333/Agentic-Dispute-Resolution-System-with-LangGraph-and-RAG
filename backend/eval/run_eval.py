"""
Master evaluation runner — four surfaces, one command.

Usage:
  python -m backend.eval.run_eval --mode all
  python -m backend.eval.run_eval --mode classifier
  python -m backend.eval.run_eval --mode rag
  python -m backend.eval.run_eval --mode pipeline
  python -m backend.eval.run_eval --mode governance

Outputs:
  eval_results.json  — machine-readable (all modes)
  eval_report.md     — human-readable, appended (all modes)
"""

import argparse
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

BANNER = """
══ DISPUTE AGENT — FULL EVALUATION REPORT ══════════════════════════
Visa Core Rules and Visa Product and Service Rules · 18 April 2026
"""


def _print_summary(results: dict) -> None:
    print(BANNER)

    if "classifier" in results:
        c = results["classifier"]
        print("── 1. ML CLASSIFIER ────────────────────────────────────────")
        print(f"  F1:       {c['f1']:.4f}   Precision: {c['precision']:.4f}   Recall: {c['recall']:.4f}")
        print(f"  ROC-AUC:  {c['roc_auc']:.4f}   Eval rows: {c['n_eval']:,}")

    if "rag" in results:
        r = results["rag"]
        print("\n── 2. RAG RETRIEVAL ────────────────────────────────────────")
        print(f"  Hit@1:  {r['hit_at_1']:.3f}   Hit@3:  {r['hit_at_3']:.3f}   MRR: {r['mrr']:.3f}")
        print(f"  Failed queries: {len(r['failed_queries'])}/{r['n_queries']}")
        for fq in r["failed_queries"]:
            print(f"    [{fq['id']}] §{fq['expected_section']} — {fq['query'][:60]}")

    if "pipeline" in results:
        p  = results["pipeline"]
        co = results.get("consistency", {})
        print("\n── 3. PIPELINE END-TO-END ──────────────────────────────────")
        print(f"  Verdict accuracy:    {p['correct']}/{p['total']} ({p['verdict_accuracy'] * 100:.1f}%)")
        print(f"  Violation catch rate: {p['violation_catch_rate'] * 100:.1f}%")
        if co:
            status = "PASS" if co["consistent"] else f"FAIL — {co['unique_verdicts']}"
            print(f"  Consistency (5 runs): {status}")

    if "governance" in results:
        g = results["governance"]
        if g.get("error"):
            print("\n── 4. GOVERNANCE ───────────────────────────────────────────")
            print("  No audit records found — run pipeline eval first.")
        else:
            print("\n── 4. GOVERNANCE ───────────────────────────────────────────")
            print(f"  Audit completeness:    {g['completeness_rate'] * 100:.1f}%")
            print(f"  Reasoning consistency: {g['consistency_rate'] * 100:.1f}%")

    print(f"\n══ GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ══════════════════════")


def _save(results: dict) -> None:
    from backend.eval.report_writer import write_report

    write_report(results)

    out_path = "eval_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Results saved → {out_path}")


def run_all() -> dict:
    from backend.eval.eval_classifier import eval_classifier
    from backend.eval.eval_governance import eval_governance
    from backend.eval.eval_pipeline   import eval_consistency, eval_pipeline
    from backend.eval.eval_rag        import eval_rag

    results: dict = {}
    results["run_id"]      = datetime.now(timezone.utc).isoformat()
    results["classifier"]  = eval_classifier()
    results["rag"]         = eval_rag()
    results["pipeline"]    = eval_pipeline()
    results["consistency"] = eval_consistency(n_runs=5)
    results["governance"]  = eval_governance()

    _print_summary(results)
    _save(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dispute Agent system evaluator")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "classifier", "rag", "pipeline", "governance"],
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_all()

    elif args.mode == "classifier":
        from backend.eval.eval_classifier import eval_classifier
        results = {"run_id": datetime.now(timezone.utc).isoformat(), "classifier": eval_classifier()}
        _print_summary(results)
        _save(results)

    elif args.mode == "rag":
        from backend.eval.eval_rag import eval_rag
        results = {"run_id": datetime.now(timezone.utc).isoformat(), "rag": eval_rag()}
        _print_summary(results)
        _save(results)

    elif args.mode == "pipeline":
        from backend.eval.eval_pipeline import eval_consistency, eval_pipeline
        results = {
            "run_id":      datetime.now(timezone.utc).isoformat(),
            "pipeline":    eval_pipeline(),
            "consistency": eval_consistency(n_runs=5),
        }
        _print_summary(results)
        _save(results)

    elif args.mode == "governance":
        from backend.eval.eval_governance import eval_governance
        results = {"run_id": datetime.now(timezone.utc).isoformat(), "governance": eval_governance()}
        _print_summary(results)
        _save(results)
