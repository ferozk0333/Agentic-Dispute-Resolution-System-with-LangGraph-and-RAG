# EVAL.md — Full System Evaluation Specification

## Source document
Visa Core Rules and Visa Product and Service Rules — 18 April 2026 edition
923 pages. All rule citations below reference this document.

## Overview — four evaluation surfaces

| Surface | Method | Script | Output |
|---|---|---|---|
| 1. ML Classifier (DT) | Held-out test set + CV | `run_eval.py --mode classifier` | F1, AUC, per-scenario breakdown |
| 2. RAG Retrieval | Offline golden set (20 pairs) | `run_eval.py --mode rag` | Hit@1, Hit@3, MRR |
| 3. Pipeline end-to-end | All 25 demo disputes | `run_eval.py --mode pipeline` | Verdict accuracy, catch rate, consistency |
| 4. Governance layer | Audit record checks | `run_eval.py --mode governance` | Completeness, replay fidelity, reasoning consistency |
| **All** | Combined | `run_eval.py --mode all` | Full report → `eval_results.json` |

Run `--mode all` to generate the report screenshot for your portfolio README.

---

## Evaluation 1 — ML Classifier

**File:** `backend/eval/eval_classifier.py`

```python
import pandas as pd, numpy as np, json, joblib
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

MODEL_PATH = "backend/classifier/fraud_dt.joblib"
META_PATH  = "backend/classifier/model_metadata.json"
EVAL_PATH  = "data/ieee_eval_set.csv"

FEATURES = [
    "transaction_amt", "amt_vs_typical_ratio", "velocity_24h",
    "velocity_amt_24h", "geo_mismatch", "merchant_risk_score",
    "is_high_risk_category", "email_domain_risk", "unique_locations_24h",
]

def eval_classifier() -> dict:
    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(EVAL_PATH)

    # Build features from IEEE-CIS columns where possible
    # Map available columns to our feature set
    df["transaction_amt"]       = df["TransactionAmt"]
    df["amt_vs_typical_ratio"]  = (df["TransactionAmt"] / 500).clip(0, 10)  # normalise
    df["velocity_24h"]          = df["C1"].fillna(1).clip(1, 20).astype(int)
    df["velocity_amt_24h"]      = df["C2"].fillna(df["TransactionAmt"])
    df["geo_mismatch"]          = (df["addr1"] != df["addr2"]).astype(int)
    df["merchant_risk_score"]   = df["V12"].fillna(0.5).clip(0, 1)
    df["is_high_risk_category"] = (df["ProductCD"].isin(["S", "H"])).astype(int)
    df["email_domain_risk"]     = df["P_emaildomain"].apply(
        lambda x: 1 if pd.notna(x) and any(r in str(x) for r in
                  ["protonmail", "temp", "mail.com", "guerrilla"]) else 0
    )
    df["unique_locations_24h"]  = df["D4"].fillna(1).clip(1, 5).astype(int)

    df = df.dropna(subset=FEATURES + ["isFraud"])
    X  = df[FEATURES]
    y  = df["isFraud"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Overall metrics
    f1        = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall    = recall_score(y, y_pred)
    auc       = roc_auc_score(y, y_prob)

    print("\n── CLASSIFIER EVALUATION ──────────────────────────")
    print(classification_report(y, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"ROC-AUC: {auc:.4f}")

    return {
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "roc_auc":   round(auc, 4),
        "n_eval":    len(df),
    }
```

### Per-scenario breakdown

After running the full pipeline eval (Section 3), join verdicts back to
`demo_disputes.dispute_scenario` and compute F1 per scenario group:

| Scenario | Expected F1 range |
|---|---|
| `clean_unauthorized` | 0.85 – 0.95 |
| `geo_mismatch` | 0.80 – 0.92 |
| `high_velocity` | 0.82 – 0.93 |
| `policy_violation` | N/A (caught by Agent 3, not DT) |
| `escalate` (ambiguous) | 0.60 – 0.75 — expected to be weakest |

If `escalate` F1 drops below 0.55, reduce the DT `min_samples_leaf` from 10 to 5.

---

## Evaluation 2 — RAG Retrieval

### Golden eval set — 20 query → expected chunk pairs

Derived directly from the Visa Core Rules and Visa Product and Service Rules
(18 April 2026 edition). Each entry has the query Agent 2 would actually send,
the expected section that should be retrieved, and a relevance label.

**File:** `backend/eval/rag_golden_set.py`

```python
RAG_GOLDEN_SET = [

    # ── Dispute time limits ───────────────────────────────────────────────

    {
        "id": "rag_001",
        "query": "dispute condition 10.4 card absent environment time limit",
        "expected_section": "11.7.5.4",
        "expected_text_fragment": "120 calendar days from the Transaction Processing Date",
        "page": 705,
        "notes": "Core time window check for unauthorized CNP disputes. "
                 "Table 11-29. Applies to all regions.",
    },
    {
        "id": "rag_002",
        "query": "reason code 10.4 unauthorized transaction card not present",
        "expected_section": "11.7.5.1",
        "expected_text_fragment": "Cardholder denies authorization of or participation in a "
                                  "Transaction conducted in a Card-Absent Environment",
        "page": 698,
        "notes": "Dispute reason definition for 10.4. Table 11-26.",
    },
    {
        "id": "rag_003",
        "query": "dispute condition 10.3 card present fraud time limit 120 days",
        "expected_section": "11.7.4.4",
        "expected_text_fragment": "120 calendar days from the Transaction Processing Date",
        "page": 695,
        "notes": "Card-present fraud time limit. Table 11-23.",
    },
    {
        "id": "rag_004",
        "query": "dispute condition 13.1 merchandise not received time limit",
        "expected_section": "11.10.2.4",
        "expected_text_fragment": "120 calendar days",
        "page": 747,
        "notes": "Not-received dispute time limit. Table 11-92.",
    },
    {
        "id": "rag_005",
        "query": "dispute condition 12.5 incorrect amount time limit",
        "expected_section": "11.9.4",
        "expected_text_fragment": "120 calendar days from either",
        "page": 733,
        "notes": "Incorrect amount dispute. Table 11-74.",
    },
    {
        "id": "rag_006",
        "query": "duplicate processing paid by other means 12.6 time limit",
        "expected_section": "11.9.5",
        "expected_text_fragment": "120 calendar days from either",
        "page": 737,
        "notes": "Duplicate dispute time limit. Table 11-80.",
    },
    {
        "id": "rag_007",
        "query": "card recovery bulletin dispute condition 11.1 time limit 75 days",
        "expected_section": "11.8.1.3",
        "expected_text_fragment": "75 calendar days from the Transaction Processing Date",
        "page": 712,
        "notes": "Shorter 75-day window for CRB disputes. Table 11-38.",
    },

    # ── Dispute rights and conditions ─────────────────────────────────────

    {
        "id": "rag_008",
        "query": "issuer must attempt to settle before initiating dispute",
        "expected_section": "1.10.1.1",
        "expected_text_fragment": "Before initiating a Dispute, the Issuer must attempt to honor the Transaction",
        "page": 147,
        "notes": "Core rule: attempt to settle first. ID# 0003287.",
    },
    {
        "id": "rag_009",
        "query": "cardholder must not be credited twice same transaction dispute",
        "expected_section": "1.10.1.1",
        "expected_text_fragment": "Issuer must not be reimbursed twice for the same Transaction",
        "page": 147,
        "notes": "Anti-double-credit rule. Critical for duplicate dispute detection.",
    },
    {
        "id": "rag_010",
        "query": "dispute invalid 10.4 more than 35 disputes same account 120 days",
        "expected_section": "11.7.5.3",
        "expected_text_fragment": "Transaction on an Account Number for which the Issuer has initiated "
                                  "more than 35 Disputes within the previous 120 calendar days",
        "page": 699,
        "notes": "High-velocity dispute abuse check. Table 11-28. Important fraud signal.",
    },
    {
        "id": "rag_011",
        "query": "10.4 invalid dispute cryptocurrency non-fungible token NFT cardholder deceived",
        "expected_section": "11.7.5.3",
        "expected_text_fragment": "Transaction for the acquisition of non-fiat currency",
        "page": 700,
        "notes": "Crypto/NFT carve-out from 10.4 dispute rights. Table 11-28.",
    },
    {
        "id": "rag_012",
        "query": "compelling evidence card absent environment pre-arbitration",
        "expected_section": "11.5.1",
        "expected_text_fragment": "Acquirer may submit Compelling Evidence with a pre-Arbitration attempt",
        "page": 677,
        "notes": "Compelling evidence rules. Table 11-6. Relevant to reject decisions.",
    },
    {
        "id": "rag_013",
        "query": "EMV liability shift counterfeit fraud dispute condition 10.1",
        "expected_section": "11.7.2",
        "expected_text_fragment": "EMV Liability Shift Counterfeit Fraud",
        "page": 688,
        "notes": "10.1 dispute condition definition.",
    },

    # ── Dispute amounts ───────────────────────────────────────────────────

    {
        "id": "rag_014",
        "query": "minimum dispute amount T&E transactions USD 25",
        "expected_section": "11.4.3",
        "expected_text_fragment": "USD 25 (or local currency equivalent)",
        "page": 676,
        "notes": "Minimum dispute threshold. Table 11-5.",
    },
    {
        "id": "rag_015",
        "query": "dispute amount must not exceed transaction amount",
        "expected_section": "11.4.1",
        "expected_text_fragment": "Dispute amount must not exceed the Transaction amount",
        "page": 675,
        "notes": "Dispute amount ceiling. ID# 0030217.",
    },

    # ── Compliance and arbitration ────────────────────────────────────────

    {
        "id": "rag_016",
        "query": "arbitration compliance decision financial liability visa",
        "expected_section": "1.10.2.3",
        "expected_text_fragment": "responsible Member is financially liable",
        "page": 149,
        "notes": "Financial liability assignment. ID# 0003623.",
    },
    {
        "id": "rag_017",
        "query": "non-compliance assessment tier 1 violation USD 25000",
        "expected_section": "1.11.2.2",
        "expected_text_fragment": "Level 1 non-compliance assessment of USD 25,000",
        "page": 152,
        "notes": "Tier 1 non-compliance schedule. Table 1-13.",
    },
    {
        "id": "rag_018",
        "query": "enforcement appeal member 30 days new evidence violation",
        "expected_section": "1.11.2.7",
        "expected_text_fragment": "appeal letter must be received by Visa within 30 calendar days",
        "page": 157,
        "notes": "Appeal window for compliance violations. ID# 0025975.",
    },

    # ── Dispute processing requirements ──────────────────────────────────

    {
        "id": "rag_019",
        "query": "dispute 10.4 processing requirements cardholder certification denies authorization",
        "expected_section": "11.7.5.5",
        "expected_text_fragment": "Certification that the Cardholder denies authorization of or "
                                  "participation in the Transaction",
        "page": 705,
        "notes": "Required documentation for 10.4 disputes. Table 11-30.",
    },
    {
        "id": "rag_020",
        "query": "13.1 not received processing requirements cardholder attempted resolve merchant",
        "expected_section": "11.10.2.5",
        "expected_text_fragment": "Cardholder attempted to resolve the Dispute with the Merchant",
        "page": 748,
        "notes": "Required documentation for 13.1 disputes. Table 11-93.",
    },
    
]
```

### RAG eval script

```python
# backend/eval/eval_rag.py

import numpy as np
from backend.tools.rag_tool import search_compliance_docs
from backend.eval.rag_golden_set import RAG_GOLDEN_SET

def eval_rag() -> dict:
    hit_at_1  = []
    hit_at_3  = []
    rr_scores = []  # reciprocal rank for MRR

    failed_queries = []

    # In eval_rag.py — replace the hit detection logic

    for item in RAG_GOLDEN_SET:
        results = search_compliance_docs(item["query"], top_k=3)

        # Runtime precision = max() — at least one chunk is relevant
        chunk_scores = []
        for i, chunk in enumerate(results):
            fragment_found = item["expected_text_fragment"].lower() in chunk["text"].lower()
            section_found  = item["expected_section"] in chunk.get("section", "")
            chunk_scores.append(1.0 if (fragment_found or section_found) else 0.0)

        # Hit@1 = was the BEST-RANKED chunk correct
        h1 = chunk_scores[0] == 1.0 if chunk_scores else False

        # Hit@3 = your runtime metric — any chunk correct (matches your max() fix)
        h3 = max(chunk_scores) == 1.0 if chunk_scores else False

        # Runtime precision as you've implemented it
        runtime_precision = max(chunk_scores) if chunk_scores else 0.0

        # Reciprocal rank
        rr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / rank
                break

        hit_at_1.append(int(h1))
        hit_at_3.append(int(h3))
        rr_scores.append(rr)

        if not h3:
            failed_queries.append({
                "id":    item["id"],
                "query": item["query"],
                "expected_section": item["expected_section"],
                "top_result": results[0]["text"][:120] if results else "NO RESULTS",
            })

    hit1  = round(np.mean(hit_at_1), 3)
    hit3  = round(np.mean(hit_at_3), 3)
    mrr   = round(np.mean(rr_scores), 3)

    print("\n── RAG RETRIEVAL EVALUATION ────────────────────────")
    print(f"Hit@1:  {hit1:.3f}  ({sum(hit_at_1)}/{len(hit_at_1)} queries)")
    print(f"Hit@3:  {hit3:.3f}  ({sum(hit_at_3)}/{len(hit_at_3)} queries)")
    print(f"MRR:    {mrr:.3f}")

    if failed_queries:
        print(f"\nFailed queries ({len(failed_queries)}):")
        for fq in failed_queries:
            print(f"  [{fq['id']}] {fq['query']}")
            print(f"    expected: §{fq['expected_section']}")
            print(f"    got:      {fq['top_result']}")

    return {
        "hit_at_1":       hit1,
        "hit_at_3":       hit3,
        "mrr":            mrr,
        "n_queries":      len(RAG_GOLDEN_SET),
        "failed_queries": failed_queries,
    }
```

### Diagnosing RAG failures

If Hit@1 < 0.60, work through this checklist:

**Chunking problem** (most common): The Visa PDF has dense tables and
section headers that split rules across chunks. Verify that each chunk
contains a complete rule unit — the time limit table should be in the
same chunk as its section heading.

Fix: reduce chunk overlap from 50 to 100 tokens, set `min_chunk_length=200`.

**Query formulation problem**: Agent 2 may be sending queries like
"dispute window" instead of "dispute condition 10.4 time limit".
Fix: add to Investigator system prompt — `search_compliance_docs` query
must include the reason code number and the word "time limit" or
"dispute rights".

**Embedding mismatch**: `text-embedding-3-small` may struggle with
legal clause numbering (e.g. "11.7.5.4"). Try prefixing chunks with
their section number during ingestion: `f"Section {section}: {text}"`.

---

## Evaluation 3 — Pipeline end-to-end

### Verdict accuracy test

Run all 25 demo disputes through the full live pipeline and compare
`final_verdict` against `dispute_outcome` ground truth.

```python
# backend/eval/eval_pipeline.py

import sqlite3, json, asyncio
from backend.main import build_graph

DB_PATH = "data/disputes.db"

async def eval_pipeline() -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    disputes = conn.execute(
        "SELECT * FROM demo_disputes ORDER BY transaction_id"
    ).fetchall()
    conn.close()

    correct    = 0
    results    = []
    violations_seeded  = 0
    violations_caught  = 0

    for row in disputes:
        dispute = dict(row)
        graph   = build_graph()

        state = await graph.ainvoke({
            "raw_input": {
                "transaction_id":    dispute["transaction_id"],
                "amount":            dispute["transaction_amt"],
                "merchant":          dispute["merchant_name"],
                "reason_code":       "10.4",
                "customer_statement": f"I did not authorize this charge at {dispute['merchant_name']}.",
            }
        })

        predicted = state["auditor"].verdict
        expected  = dispute["dispute_outcome"]
        match     = predicted == expected

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

        if match:
            correct += 1
        else:
            print(f"  MISMATCH: {dispute['transaction_id']} — "
                  f"expected={expected}, got={predicted}")

    accuracy        = correct / len(disputes)
    catch_rate      = violations_caught / max(violations_seeded, 1)

    print(f"\n── PIPELINE EVALUATION ─────────────────────────────")
    print(f"Verdict accuracy:   {correct}/{len(disputes)} ({accuracy*100:.1f}%)")
    print(f"Violation catch:    {violations_caught}/{violations_seeded} ({catch_rate*100:.1f}%)")

    return {
        "verdict_accuracy":    round(accuracy, 4),
        "correct":             correct,
        "total":               len(disputes),
        "violation_catch_rate": round(catch_rate, 4),
        "results":             results,
    }
```

### Consistency test

Run the same dispute 5 times. Verdict must not change across runs.
LLM temperature = 0 in all agents. If temperature is not 0, set it now —
non-determinism is a governance failure in a financial system.

```python
async def eval_consistency(n_runs: int = 5) -> dict:
    """Run sample dispute 5 times, check verdict stability."""
    SAMPLE = {
        "transaction_id": "TXN-2024-0318-8821",
        "amount": 1240.00,
        "merchant": "ElectroMart Inc.",
        "reason_code": "10.4",
        "customer_statement": "I did not authorize this charge.",
    }

    verdicts = []
    for i in range(n_runs):
        graph = build_graph()
        state = await graph.ainvoke({"raw_input": SAMPLE})
        verdicts.append(state["auditor"].verdict)

    unique_verdicts = set(verdicts)
    consistent      = len(unique_verdicts) == 1

    print(f"\n── CONSISTENCY TEST ────────────────────────────────")
    print(f"Verdicts across {n_runs} runs: {verdicts}")
    print(f"Consistent: {consistent}")
    if not consistent:
        print(f"  WARNING: non-deterministic verdicts detected: {unique_verdicts}")
        print(f"  Fix: set temperature=0 in all OpenAI calls")

    return {
        "consistent":     consistent,
        "verdicts":       verdicts,
        "unique_verdicts": list(unique_verdicts),
        "n_runs":          n_runs,
    }
```

### Expected results for 25 demo disputes

| Scenario group | Count | Expected verdict accuracy |
|---|---|---|
| `clean_unauthorized` | 6 | 100% |
| `geo_mismatch` | 4 | 100% |
| `high_velocity` | 4 | 100% |
| `legitimate_charge` | 3 | 100% |
| `duplicate_dispute` | 1 | 100% |
| `outside_time_window` | 1 | 100% |
| `policy_violation` | 4 | 100% — Agent 3 must catch all |
| `escalate` (ambiguous) | 4 | ≥ 75% — some variance acceptable |
| `unknown_merchant` | 1 | 100% |
| `outside_time_window` | 1 | 100% |

Overall target: **≥ 22/25 (88%)** — escalate cases are the only acceptable
source of variance given their deliberately ambiguous signals.

---

## Evaluation 4 — Governance layer

```python
# backend/eval/eval_governance.py

import sqlite3, json
from backend.governance.audit_writer import AuditRecord

REQUIRED_FIELDS = [
    "run_id", "transaction_id", "timestamp",
    "triage_entities", "triage_masked_stmt",
    "inv_tool_calls", "inv_dt_fraud_prob", "inv_dt_decision_path",
    "inv_rag_query", "inv_rag_chunks", "inv_recommendation", "inv_confidence",
    "aud_rules_checked", "aud_violations", "aud_verdict",
    "total_latency_ms", "total_tokens", "total_cost_usd",
]

def eval_governance() -> dict:
    conn = sqlite3.connect("data/disputes.db")
    rows = conn.execute(
        "SELECT run_id, full_record_json FROM audit_runs"
    ).fetchall()
    conn.close()

    if not rows:
        print("  WARNING: No audit records found. Run pipeline eval first.")
        return {"error": "no_records"}

    completeness_failures = []
    reasoning_mismatches  = []

    for run_id, record_json in rows:
        record = json.loads(record_json)

        # Check completeness — no required field should be None
        for field in REQUIRED_FIELDS:
            if record.get(field) is None:
                completeness_failures.append({
                    "run_id": run_id,
                    "missing_field": field,
                })

        # Check reasoning consistency:
        # LLM reasoning text should mention the same verdict as structured output
        reasoning = record.get("inv_llm_reasoning", "")
        verdict   = record.get("inv_recommendation", "")
        if reasoning and verdict:
            reasoning_lower = reasoning.lower()
            if verdict == "approve" and "reject" in reasoning_lower and "approve" not in reasoning_lower:
                reasoning_mismatches.append({"run_id": run_id, "verdict": verdict,
                                             "issue": "reasoning says reject, verdict=approve"})
            elif verdict == "reject" and "approve" in reasoning_lower and "reject" not in reasoning_lower:
                reasoning_mismatches.append({"run_id": run_id, "verdict": verdict,
                                             "issue": "reasoning says approve, verdict=reject"})

    total              = len(rows)
    complete           = total - len(set(f["run_id"] for f in completeness_failures))
    completeness_rate  = complete / total if total > 0 else 0
    consistency_rate   = (total - len(reasoning_mismatches)) / total if total > 0 else 0

    print(f"\n── GOVERNANCE EVALUATION ───────────────────────────")
    print(f"Audit records:         {total}")
    print(f"Completeness:          {complete}/{total} ({completeness_rate*100:.1f}%)")
    print(f"Reasoning consistency: {total - len(reasoning_mismatches)}/{total} ({consistency_rate*100:.1f}%)")

    if completeness_failures:
        print(f"\nCompleteness failures:")
        for f in completeness_failures[:5]:
            print(f"  {f['run_id'][:8]}: missing {f['missing_field']}")

    if reasoning_mismatches:
        print(f"\nReasoning mismatches:")
        for m in reasoning_mismatches:
            print(f"  {m['run_id'][:8]}: {m['issue']}")

    return {
        "total_records":       total,
        "completeness_rate":   round(completeness_rate, 4),
        "consistency_rate":    round(consistency_rate, 4),
        "completeness_failures": completeness_failures,
        "reasoning_mismatches":  reasoning_mismatches,
    }
```

---

## Master eval runner

**File:** `backend/eval/run_eval.py`

```python
"""
Run full system evaluation and write results to eval_results.json.
Usage:
  python backend/eval/run_eval.py --mode all
  python backend/eval/run_eval.py --mode rag
  python backend/eval/run_eval.py --mode classifier
  python backend/eval/run_eval.py --mode pipeline
  python backend/eval/run_eval.py --mode governance
"""

import asyncio, argparse, json, time
from datetime import datetime

from backend.eval.eval_classifier import eval_classifier
from backend.eval.eval_rag        import eval_rag
from backend.eval.eval_pipeline   import eval_pipeline, eval_consistency
from backend.eval.eval_governance import eval_governance

BANNER = """
══ DISPUTE AGENT — FULL EVALUATION REPORT ══════════════════════════
Visa Core Rules and Visa Product and Service Rules · 18 April 2026
"""

def print_summary(results: dict):
    print(BANNER)

    if "classifier" in results:
        c = results["classifier"]
        print(f"── 1. ML CLASSIFIER ────────────────────────────────────────")
        print(f"  F1:       {c['f1']:.4f}   Precision: {c['precision']:.4f}   Recall: {c['recall']:.4f}")
        print(f"  ROC-AUC:  {c['roc_auc']:.4f}   Eval rows: {c['n_eval']:,}")

    if "rag" in results:
        r = results["rag"]
        print(f"\n── 2. RAG RETRIEVAL ────────────────────────────────────────")
        print(f"  Hit@1:  {r['hit_at_1']:.3f}   Hit@3:  {r['hit_at_3']:.3f}   MRR: {r['mrr']:.3f}")
        print(f"  Failed queries: {len(r['failed_queries'])}/{r['n_queries']}")
        for fq in r["failed_queries"]:
            print(f"    [{fq['id']}] §{fq['expected_section']} — {fq['query'][:60]}")

    if "pipeline" in results:
        p = results["pipeline"]
        co = results.get("consistency", {})
        print(f"\n── 3. PIPELINE END-TO-END ──────────────────────────────────")
        print(f"  Verdict accuracy:    {p['correct']}/{p['total']} ({p['verdict_accuracy']*100:.1f}%)")
        print(f"  Violation catch rate: {p['violation_catch_rate']*100:.1f}%")
        if co:
            print(f"  Consistency (5 runs): {'PASS' if co['consistent'] else 'FAIL — ' + str(co['unique_verdicts'])}")

    if "governance" in results:
        g = results["governance"]
        print(f"\n── 4. GOVERNANCE ───────────────────────────────────────────")
        print(f"  Audit completeness:   {g['completeness_rate']*100:.1f}%")
        print(f"  Reasoning consistency: {g['consistency_rate']*100:.1f}%")

    print(f"\n══ GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ══════════════════════")

async def run_all():
    results = {}
    results["classifier"]   = eval_classifier()
    results["rag"]          = eval_rag()
    results["pipeline"]     = await eval_pipeline()
    results["consistency"]  = await eval_consistency(n_runs=5)
    results["governance"]   = eval_governance()

    print_summary(results)

    with open("eval_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved → eval_results.json")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "classifier", "rag", "pipeline", "governance"])
    args = parser.parse_args()

    if args.mode == "all":
        asyncio.run(run_all())
    elif args.mode == "classifier":
        print_summary({"classifier": eval_classifier()})
    elif args.mode == "rag":
        print_summary({"rag": eval_rag()})
    elif args.mode == "pipeline":
        results = asyncio.run(eval_pipeline())
        results["consistency"] = asyncio.run(eval_consistency())
        print_summary({"pipeline": results})
    elif args.mode == "governance":
        print_summary({"governance": eval_governance()})
```

---

## Expected output (target numbers for portfolio)

```
══ DISPUTE AGENT — FULL EVALUATION REPORT ══════════════════════════

── 1. ML CLASSIFIER ────────────────────────────────────────────────
  F1:       0.81   Precision: 0.79   Recall: 0.83
  ROC-AUC:  0.91   Eval rows: 118,904

── 2. RAG RETRIEVAL ────────────────────────────────────────────────
  Hit@1:  0.70   Hit@3:  0.85   MRR: 0.76
  Failed queries: 3/20
    [rag_011] §11.7.5.3 — 10.4 invalid dispute cryptocurrency NFT...
    [rag_018] §1.11.2.7 — enforcement appeal member 30 days...
    [rag_020] §11.10.2.5 — 13.1 not received processing requirements...

── 3. PIPELINE END-TO-END ──────────────────────────────────────────
  Verdict accuracy:     22/25 (88.0%)
  Violation catch rate: 100.0%
  Consistency (5 runs): PASS

── 4. GOVERNANCE ───────────────────────────────────────────────────
  Audit completeness:    100.0%
  Reasoning consistency: 92.0%

══ GENERATED: 2024-03-18 14:22:01 ══════════════════════════════════
```

Save this output as `eval_report.png` (screenshot) and add to your
portfolio README alongside the `eval_results.json` file.

---

## Files created by this module

```
backend/eval/
├── rag_golden_set.py      20 query→chunk pairs from real Visa rules
├── eval_classifier.py     DT evaluation on IEEE-CIS test set
├── eval_rag.py            RAG retrieval evaluation
├── eval_pipeline.py       End-to-end verdict accuracy + consistency
├── eval_governance.py     Audit completeness + reasoning consistency
└── run_eval.py            Master runner — produces eval_results.json

eval_results.json          Generated output (add to .gitignore or commit as artefact)
```

## Update CLAUDE.md

Add to the run section:
```bash
# Evaluation (run after at least one full pipeline run)
python backend/eval/run_eval.py --mode all
```

Add to Definition of done:
```
- [ ] run_eval.py --mode all completes without errors
- [ ] RAG Hit@3 >= 0.80
- [ ] Pipeline verdict accuracy >= 22/25
- [ ] Violation catch rate = 100%
- [ ] eval_results.json committed or screenshotted for portfolio
```
---

## Evaluation Report Storage

### What gets stored

Every `run_eval.py` run writes two outputs:

| Output | Format | Purpose |
|---|---|---|
| `eval_results.json` | Machine-readable JSON | Frontend dashboard, CI checks, trend tracking |
| `eval_report.md` | Human-readable Markdown | Portfolio README, stakeholder sharing, audit trail |

Both are written to the project root. Commit `eval_report.md` to git.
Add `eval_results.json` to `.gitignore` or commit it too — your call.

---

### `eval_report.md` — generated output format

The report writer appends a timestamped run to `eval_report.md` rather
than overwriting it, so you accumulate a history of every eval run.
This is useful for showing improvement over time in your portfolio.

**File:** `backend/eval/report_writer.py`

```python
"""
Writes human-readable eval report to eval_report.md.
Appends each run — never overwrites — so history is preserved.
"""

import json
from datetime import datetime, timezone

REPORT_PATH = "eval_report.md"

def write_report(results: dict):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_id    = results.get("run_id", timestamp)

    lines = []
    lines.append(f"\n---\n")
    lines.append(f"## Eval run — {timestamp}\n")

    # ── 1. Classifier ────────────────────────────────────────────────────
    if "classifier" in results:
        c = results["classifier"]
        lines.append(f"### 1. ML Classifier (Decision Tree)\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| F1 | {c['f1']:.4f} |")
        lines.append(f"| Precision | {c['precision']:.4f} |")
        lines.append(f"| Recall | {c['recall']:.4f} |")
        lines.append(f"| ROC-AUC | {c['roc_auc']:.4f} |")
        lines.append(f"| Eval rows | {c['n_eval']:,} |")
        lines.append("")

    # ── 2. RAG ───────────────────────────────────────────────────────────
    if "rag" in results:
        r = results["rag"]
        lines.append(f"### 2. RAG Retrieval\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Hit@1 | {r['hit_at_1']:.3f} |")
        lines.append(f"| Hit@3 (runtime precision) | {r['hit_at_3']:.3f} |")
        lines.append(f"| MRR | {r['mrr']:.3f} |")
        lines.append(f"| Queries evaluated | {r['n_queries']} |")
        lines.append(f"| Failed queries | {len(r['failed_queries'])} |")

        if r["failed_queries"]:
            lines.append(f"\n**Failed queries:**\n")
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
        lines.append(f"### 3. Pipeline End-to-End\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(
            f"| Verdict accuracy | {p['correct']}/{p['total']} "
            f"({p['verdict_accuracy']*100:.1f}%) |"
        )
        lines.append(
            f"| Violation catch rate | "
            f"{p['violation_catch_rate']*100:.1f}% |"
        )
        if co:
            status = "✅ PASS" if co["consistent"] else f"❌ FAIL — {co['unique_verdicts']}"
            lines.append(f"| Consistency ({co['n_runs']} runs) | {status} |")

        # Per-scenario breakdown
        if "results" in p:
            scenario_map = {}
            for r in p["results"]:
                s = r["scenario"]
                if s not in scenario_map:
                    scenario_map[s] = {"correct": 0, "total": 0}
                scenario_map[s]["total"] += 1
                if r["correct"]:
                    scenario_map[s]["correct"] += 1

            lines.append(f"\n**Per-scenario breakdown:**\n")
            lines.append(f"| Scenario | Correct | Total | Accuracy |")
            lines.append(f"|---|---|---|---|")
            for scenario, counts in sorted(scenario_map.items()):
                acc = counts["correct"] / counts["total"]
                flag = "" if acc == 1.0 else " ⚠️"
                lines.append(
                    f"| `{scenario}` | {counts['correct']} | "
                    f"{counts['total']} | {acc*100:.0f}%{flag} |"
                )

        # List any mismatches
        mismatches = [r for r in p.get("results", []) if not r["correct"]]
        if mismatches:
            lines.append(f"\n**Mismatches:**\n")
            for m in mismatches:
                lines.append(
                    f"- `{m['transaction_id']}` — "
                    f"expected `{m['expected']}`, got `{m['predicted']}`"
                )
        lines.append("")

    # ── 4. Governance ────────────────────────────────────────────────────
    if "governance" in results:
        g = results["governance"]
        lines.append(f"### 4. Governance Layer\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Audit records | {g['total_records']} |")
        lines.append(
            f"| Completeness | "
            f"{g['completeness_rate']*100:.1f}% |"
        )
        lines.append(
            f"| Reasoning consistency | "
            f"{g['consistency_rate']*100:.1f}% |"
        )

        if g.get("completeness_failures"):
            lines.append(f"\n**Completeness failures:**\n")
            for f in g["completeness_failures"][:5]:
                lines.append(
                    f"- Run `{f['run_id'][:8]}`: missing field `{f['missing_field']}`"
                )

        if g.get("reasoning_mismatches"):
            lines.append(f"\n**Reasoning mismatches:**\n")
            for m in g["reasoning_mismatches"]:
                lines.append(f"- Run `{m['run_id'][:8]}`: {m['issue']}")
        lines.append("")

    # ── Overall health ───────────────────────────────────────────────────
    warnings = _collect_warnings(results)
    if warnings:
        lines.append(f"### ⚠️ Warnings\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append(f"### ✅ All checks passed\n")

    # Write — append, never overwrite
    with open(REPORT_PATH, "a") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nReport appended → {REPORT_PATH}")


def _collect_warnings(results: dict) -> list[str]:
    """Return list of actionable warnings based on eval results."""
    warnings = []

    if "rag" in results:
        r = results["rag"]
        if r["hit_at_1"] < 0.60:
            warnings.append(
                f"RAG Hit@1 = {r['hit_at_1']:.2f} < 0.60 — "
                f"improve chunking: section headings and time limit tables "
                f"are likely being split across chunks"
            )
        if r["hit_at_3"] < 0.80:
            warnings.append(
                f"RAG Hit@3 = {r['hit_at_3']:.2f} < 0.80 — "
                f"check query formulation in Investigator system prompt; "
                f"queries must include reason code number and 'time limit'"
            )
        if r["mrr"] < 0.65:
            warnings.append(
                f"MRR = {r['mrr']:.2f} < 0.65 — ranking is poor even when "
                f"correct chunk is retrieved; try prefixing chunks with "
                f"section number during ingestion"
            )

    if "classifier" in results:
        c = results["classifier"]
        if c["f1"] < 0.70:
            warnings.append(
                f"Classifier F1 = {c['f1']:.4f} < 0.70 — "
                f"increase fraud_velocity_lambda from 3.4 to 4.5 in "
                f"generate_training_data.py CALIBRATION dict and retrain"
            )

    if "pipeline" in results:
        p = results["pipeline"]
        if p["verdict_accuracy"] < 0.88:
            warnings.append(
                f"Verdict accuracy = {p['verdict_accuracy']*100:.1f}% < 88% — "
                f"check AGENTS.md Investigator system prompt; confidence "
                f"floor (0.85) may be too aggressive for ambiguous scenarios"
            )
        if p["violation_catch_rate"] < 1.0:
            warnings.append(
                f"⚠️ CRITICAL: Violation catch rate = "
                f"{p['violation_catch_rate']*100:.1f}% < 100% — "
                f"Agent 3 missed a policy violation. Review auditor.py "
                f"pre_check_rules() and system prompt immediately."
            )

    if "consistency" in results:
        co = results["consistency"]
        if not co.get("consistent"):
            warnings.append(
                f"Non-deterministic verdicts detected: {co['unique_verdicts']} — "
                f"set temperature=0 in ALL OpenAI client calls"
            )

    if "governance" in results:
        g = results["governance"]
        if g.get("completeness_rate", 1.0) < 1.0:
            warnings.append(
                f"Audit completeness = {g['completeness_rate']*100:.1f}% — "
                f"some AuditRecord fields are None; check build_audit_record() "
                f"in audit_writer.py"
            )

    return warnings
```

### Wire into `run_eval.py`

Add two lines to the `run_all()` function in `run_eval.py`, after
`print_summary(results)`:

```python
from backend.eval.report_writer import write_report

async def run_all():
    results = {}
    results["run_id"]       = datetime.now(timezone.utc).isoformat()
    results["classifier"]   = eval_classifier()
    results["rag"]          = eval_rag()
    results["pipeline"]     = await eval_pipeline()
    results["consistency"]  = await eval_consistency(n_runs=5)
    results["governance"]   = eval_governance()

    print_summary(results)
    write_report(results)                        # ← add this

    with open("eval_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved → eval_results.json")
    return results
```

---

### `eval_report.md` file header

Create this file manually once before the first eval run.
Claude Code should create it as part of project setup:

```markdown
# Dispute Agent — Evaluation Report

Visa Core Rules and Visa Product and Service Rules · 18 April 2026 edition
Three-agent pipeline: Triage (gpt-4o-mini) · Investigator (gpt-4o) · Auditor (gpt-4o-mini)
Decision Tree classifier: dt_v1 · 5,000 synthetic rows · IEEE-CIS calibrated

Runs are appended below in reverse chronological order.
```

---

### Example `eval_report.md` after one run

```markdown
# Dispute Agent — Evaluation Report
...header...

---

## Eval run — 2024-03-18 14:22 UTC

### 1. ML Classifier (Decision Tree)

| Metric | Value |
|---|---|
| F1 | 0.8134 |
| Precision | 0.7921 |
| Recall | 0.8360 |
| ROC-AUC | 0.9108 |
| Eval rows | 118,904 |

### 2. RAG Retrieval

| Metric | Value |
|---|---|
| Hit@1 | 0.700 |
| Hit@3 (runtime precision) | 0.850 |
| MRR | 0.763 |
| Queries evaluated | 20 |
| Failed queries | 3 |

**Failed queries:**
- `[rag_011]` §11.7.5.3 — `10.4 invalid dispute cryptocurrency NFT cardholder deceived`
- `[rag_018]` §1.11.2.7 — `enforcement appeal member 30 days new evidence violation`
- `[rag_020]` §11.10.2.5 — `13.1 not received processing requirements cardholder`

### 3. Pipeline End-to-End

| Metric | Value |
|---|---|
| Verdict accuracy | 22/25 (88.0%) |
| Violation catch rate | 100.0% |
| Consistency (5 runs) | ✅ PASS |

**Per-scenario breakdown:**

| Scenario | Correct | Total | Accuracy |
|---|---|---|---|
| `clean_unauthorized` | 6 | 6 | 100% |
| `duplicate_dispute` | 1 | 1 | 100% |
| `escalate` (ambiguous) | 3 | 4 | 75% ⚠️ |
| `geo_mismatch` | 4 | 4 | 100% |
| `high_velocity` | 4 | 4 | 100% |
| `legitimate_charge` | 3 | 3 | 100% |
| `outside_time_window` | 1 | 1 | 100% |
| `policy_violation` | 4 | 4 | 100% |
| `unknown_merchant` | 1 | 1 | 100% |

**Mismatches:**
- `TXN-2024-0313-6622` — expected `escalate`, got `approve`

### 4. Governance Layer

| Metric | Value |
|---|---|
| Audit records | 25 |
| Completeness | 100.0% |
| Reasoning consistency | 92.0% |

### ⚠️ Warnings

- RAG Hit@1 = 0.70 < threshold — consider prefixing chunks with section number
```

---

### Add to Definition of done (in `CLAUDE.md`)

```
- [ ] eval_report.md exists and contains at least one completed run
- [ ] eval_report.md is committed to git
- [ ] No CRITICAL warnings (violation catch rate < 100%) in latest run
```