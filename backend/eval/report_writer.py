"""
Writes human-readable eval report to eval_report.md.
Appends each run — never overwrites — so history is preserved.
"""

from datetime import datetime, timezone

REPORT_PATH = "eval_report.md"


def write_report(results: dict) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"\n---\n")
    lines.append(f"## Eval run — {timestamp}\n")

    # ── 1. Classifier ────────────────────────────────────────────────────
    if "classifier" in results:
        c = results["classifier"]
        lines.append("### 1. ML Classifier (Decision Tree)\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| F1 | {c['f1']:.4f} |")
        lines.append(f"| Precision | {c['precision']:.4f} |")
        lines.append(f"| Recall | {c['recall']:.4f} |")
        lines.append(f"| ROC-AUC | {c['roc_auc']:.4f} |")
        lines.append(f"| Eval rows | {c['n_eval']:,} |")
        lines.append("")

    # ── 2. RAG ───────────────────────────────────────────────────────────
    if "rag" in results:
        r = results["rag"]
        lines.append("### 2. RAG Retrieval\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Hit@1 | {r['hit_at_1']:.3f} |")
        lines.append(f"| Hit@3 (runtime precision) | {r['hit_at_3']:.3f} |")
        lines.append(f"| MRR | {r['mrr']:.3f} |")
        lines.append(f"| Queries evaluated | {r['n_queries']} |")
        lines.append(f"| Failed queries | {len(r['failed_queries'])} |")

        if r["failed_queries"]:
            lines.append("\n**Failed queries:**\n")
            for fq in r["failed_queries"]:
                lines.append(
                    f"- `[{fq['id']}]` §{fq['expected_section']} — "
                    f"`{fq['query'][:70]}`"
                )
        lines.append("")

    # ── 3. Pipeline ──────────────────────────────────────────────────────
    if "pipeline" in results:
        p  = results["pipeline"]
        co = results.get("consistency", {})
        lines.append("### 3. Pipeline End-to-End\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(
            f"| Verdict accuracy | {p['correct']}/{p['total']} "
            f"({p['verdict_accuracy'] * 100:.1f}%) |"
        )
        lines.append(f"| Violation catch rate | {p['violation_catch_rate'] * 100:.1f}% |")
        if co:
            status = "PASS" if co["consistent"] else f"FAIL — {co['unique_verdicts']}"
            lines.append(f"| Consistency ({co['n_runs']} runs) | {status} |")

        if "results" in p:
            scenario_map: dict[str, dict] = {}
            for r in p["results"]:
                s = r["scenario"]
                if s not in scenario_map:
                    scenario_map[s] = {"correct": 0, "total": 0}
                scenario_map[s]["total"] += 1
                if r["correct"]:
                    scenario_map[s]["correct"] += 1

            lines.append("\n**Per-scenario breakdown:**\n")
            lines.append("| Scenario | Correct | Total | Accuracy |")
            lines.append("|---|---|---|---|")
            for scenario, counts in sorted(scenario_map.items()):
                acc  = counts["correct"] / counts["total"]
                flag = "" if acc == 1.0 else " ⚠️"
                lines.append(
                    f"| `{scenario}` | {counts['correct']} | "
                    f"{counts['total']} | {acc * 100:.0f}%{flag} |"
                )

        mismatches = [r for r in p.get("results", []) if not r["correct"]]
        if mismatches:
            lines.append("\n**Mismatches:**\n")
            for m in mismatches:
                lines.append(
                    f"- `{m['transaction_id']}` — "
                    f"expected `{m['expected']}`, got `{m['predicted']}`"
                )
        lines.append("")

    # ── 4. Governance ────────────────────────────────────────────────────
    if "governance" in results:
        g = results["governance"]
        if g.get("error"):
            lines.append("### 4. Governance Layer\n")
            lines.append(f"> No audit records found — run pipeline eval first.\n")
        else:
            lines.append("### 4. Governance Layer\n")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| Audit records | {g['total_records']} |")
            lines.append(f"| Completeness | {g['completeness_rate'] * 100:.1f}% |")
            lines.append(f"| Reasoning consistency | {g['consistency_rate'] * 100:.1f}% |")

            if g.get("completeness_failures"):
                lines.append("\n**Completeness failures:**\n")
                for f in g["completeness_failures"][:5]:
                    lines.append(f"- Run `{f['run_id'][:8]}`: missing field `{f['missing_field']}`")

            if g.get("reasoning_mismatches"):
                lines.append("\n**Reasoning mismatches:**\n")
                for m in g["reasoning_mismatches"]:
                    lines.append(f"- Run `{m['run_id'][:8]}`: {m['issue']}")
        lines.append("")

    # ── Warnings ─────────────────────────────────────────────────────────
    warnings = _collect_warnings(results)
    if warnings:
        lines.append("### Warnings\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("### All checks passed\n")

    with open(REPORT_PATH, "a") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nReport appended → {REPORT_PATH}")


def _collect_warnings(results: dict) -> list[str]:
    warnings = []

    if "rag" in results:
        r = results["rag"]
        if r["hit_at_1"] < 0.60:
            warnings.append(
                f"RAG Hit@1 = {r['hit_at_1']:.2f} < 0.60 — "
                "improve chunking: section headings and time limit tables "
                "are likely being split across chunks"
            )
        if r["hit_at_3"] < 0.80:
            warnings.append(
                f"RAG Hit@3 = {r['hit_at_3']:.2f} < 0.80 — "
                "check query formulation in Investigator system prompt; "
                "queries must include reason code number and 'time limit'"
            )
        if r["mrr"] < 0.65:
            warnings.append(
                f"MRR = {r['mrr']:.2f} < 0.65 — ranking is poor even when "
                "correct chunk is retrieved; try prefixing chunks with "
                "section number during ingestion"
            )

    if "classifier" in results:
        c = results["classifier"]
        if c["f1"] < 0.70:
            warnings.append(
                f"Classifier F1 = {c['f1']:.4f} < 0.70 — "
                "increase fraud_velocity_lambda from 3.4 to 4.5 in "
                "generate_training_data.py CALIBRATION dict and retrain"
            )

    if "pipeline" in results:
        p = results["pipeline"]
        if p["verdict_accuracy"] < 0.88:
            warnings.append(
                f"Verdict accuracy = {p['verdict_accuracy'] * 100:.1f}% < 88% — "
                "check AGENTS.md Investigator system prompt; confidence "
                "floor (0.85) may be too aggressive for ambiguous scenarios"
            )
        if p["violation_catch_rate"] < 1.0:
            warnings.append(
                f"CRITICAL: Violation catch rate = "
                f"{p['violation_catch_rate'] * 100:.1f}% < 100% — "
                "Agent 3 missed a policy violation. Review auditor.py "
                "pre_check_rules() and system prompt immediately."
            )

    if "consistency" in results:
        co = results["consistency"]
        if not co.get("consistent"):
            warnings.append(
                f"Non-deterministic verdicts detected: {co['unique_verdicts']} — "
                "set temperature=0 in ALL OpenAI client calls"
            )

    if "governance" in results:
        g = results["governance"]
        if not g.get("error") and g.get("completeness_rate", 1.0) < 1.0:
            warnings.append(
                f"Audit completeness = {g['completeness_rate'] * 100:.1f}% — "
                "some AuditRecord fields are None; check write_audit_record() "
                "in audit_writer.py"
            )

    return warnings
