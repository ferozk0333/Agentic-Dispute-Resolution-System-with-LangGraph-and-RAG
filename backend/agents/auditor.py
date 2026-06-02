"""
Agent 3 — Auditor
Model: gpt-4o-mini

Receives ONLY Agent 2's structured output (no tool call history, no chain of
thought). All five policy rules are evaluated deterministically in Python; the
LLM receives the pre-computed boolean results and acts as a verdict compiler —
producing the final structured output with natural-language detail strings.

On every run, appends a structured entry to audit_log.jsonl.
"""
import json
import os
import sqlite3
import time
from datetime import date, datetime, timezone
from typing import Optional
from uuid import uuid4

from dateutil import parser as dateutil_parser

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from backend.agents.triage import AgentMetrics, DisputeInput, TriageOutput
from backend.agents.investigator import InvestigatorOutput
from backend.governance.audit_writer import AuditRecord, write_audit_record
from backend.tools.sql_tool import DB_PATH

load_dotenv()

MODEL        = "gpt-4o-mini"
AUDIT_LOG    = os.getenv("AUDIT_LOG_PATH", "./data/audit_log.jsonl")

PRICING = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o":      {"input": 0.002500, "output": 0.010000},
}

# ── Schemas ───────────────────────────────────────────────────────────────────

class AuditRuleResult(BaseModel):
    rule:   str
    passed: bool
    detail: str


class AuditorOutput(BaseModel):
    verdict:       str               # approve | reject | policy_violation | escalate
    rules_checked: list[AuditRuleResult]
    violations:    list[str]
    reasoning:     str               # 2-3 sentence plain-language explanation


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an independent compliance auditor at a payment network.

All policy rules have already been evaluated deterministically. Their boolean results
and detail strings are provided. Your job is to compile the final verdict.

── RULE CLASSIFICATION ──────────────────────────────────────────────────────────

Hard rules — failure here means the case cannot be processed as submitted:
  • transaction_exists   — transaction must exist in the database
  • refund_threshold     — dispute amount must not exceed the auto-processing limit
  • no_duplicate         — transaction must not have been previously resolved
  • reason_code_match    — reason code must align with the dispute category

Adjudication rules — failure here shapes the verdict but is not a policy breach:
  • dispute_window       — 120-day Visa filing window; failure means the claim is
                           time-barred and must be rejected on eligibility grounds
  • confidence_floor     — investigator confidence below threshold; case requires
                           human review before a final approve/reject can be issued

── VERDICT LOGIC (apply in order, stop at first match) ─────────────────────────

1. If ANY hard rule has passed=false
   → verdict = "policy_violation"
   → violations = [names of all failed hard rules]

2. Else if dispute_window has passed=false
   → verdict = "reject"
   → violations = []
   → reasoning must state the claim is time-barred under Visa §11 dispute window rules

3. Else if confidence_floor has passed=false
   → verdict = "escalate"
   → violations = []
   → reasoning must state that automated confidence is insufficient for a final decision

4. Else if all rules passed AND Agent 2 recommendation is "escalate"
   → verdict = "escalate"
   → violations = []

5. Else (all rules passed, Agent 2 is approve or reject)
   → adopt Agent 2's recommendation verbatim as the verdict
   → violations = []

── INSTRUCTIONS ─────────────────────────────────────────────────────────────────

- Accept every pre-computed "passed" value exactly as given — do not override.
- Write a concise, specific "detail" string for each rule (keep the facts from the
  pre-computed detail; add interpretive context where useful).
- "violations" lists only the names of failed hard rules. Leave it empty for
  adjudication rule failures (dispute_window, confidence_floor).
- "reasoning": 2-3 sentences in plain language naming the specific signals or rule
  results that drove the verdict. State the recommended action (e.g. refund issued,
  claim denied as time-barred, escalated to senior analyst for manual review).
- Do not use em dashes (—) anywhere in your output. Use a period or comma instead.

Output ONLY valid JSON:
{
  "verdict": "approve" | "reject" | "policy_violation" | "escalate",
  "rules_checked": [{"rule": string, "passed": bool, "detail": string}],
  "violations": [string],
  "reasoning": string
}\
"""

# ── Deterministic pre-checks ──────────────────────────────────────────────────

def _get_db_transaction_date(transaction_id: str) -> Optional[str]:
    """Read transaction_date directly from demo_disputes as a fallback when triage didn't extract it."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            "SELECT transaction_date FROM demo_disputes WHERE transaction_id = ?",
            (transaction_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    return None


def _check_transaction_exists(transaction_id: str) -> AuditRuleResult:
    """Verify the transaction_id is present in demo_disputes before processing."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            "SELECT 1 FROM demo_disputes WHERE transaction_id = ?",
            (transaction_id,),
        )
        found = cur.fetchone() is not None
        conn.close()
    except Exception as exc:
        return AuditRuleResult(
            rule="transaction_exists",
            passed=False,
            detail=f"database lookup failed: {exc}",
        )
    return AuditRuleResult(
        rule="transaction_exists",
        passed=found,
        detail=(
            f"Record confirmed in database for {transaction_id}"
            if found else
            f"No database record found for {transaction_id}. Cannot dispute a non-existent transaction."
        ),
    )


def _parse_date(s: str) -> Optional[date]:
    """Try ISO first, then fall back to dateutil for natural language dates."""
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    try:
        return dateutil_parser.parse(s, default=datetime(2024, 1, 1)).date()
    except Exception:
        return None


def _check_dispute_window(
    transaction_date_str: Optional[str],
    filing_date: date,
    max_days: int = 120,
) -> AuditRuleResult:
    if not transaction_date_str:
        return AuditRuleResult(
            rule="dispute_window",
            passed=True,
            detail="Transaction date not provided. The 120-day Visa window requires manual confirmation before finalising.",
        )
    txn_date = _parse_date(transaction_date_str)
    if txn_date is None:
        return AuditRuleResult(
            rule="dispute_window",
            passed=True,
            detail=f"Transaction date format unrecognized ({transaction_date_str!r}). Window check skipped, defaulted to pass.",
        )
    delta  = (filing_date - txn_date).days
    passed = delta <= max_days
    return AuditRuleResult(
        rule="dispute_window",
        passed=passed,
        detail=(
            f"Filed {delta} days after the transaction date, within the {max_days}-day Visa dispute window"
            if passed else
            f"Dispute is time-barred: {delta} days have elapsed since {txn_date}, exceeding the {max_days}-day Visa limit"
        ),
    )


def _check_refund_threshold(amount: float, limit: float = 5000.0) -> AuditRuleResult:
    passed = amount <= limit
    return AuditRuleResult(
        rule="refund_threshold",
        passed=passed,
        detail=(
            f"Dispute amount ${amount:,.2f} is within the ${limit:,.2f} automated processing limit"
            if passed else
            f"Dispute amount ${amount:,.2f} exceeds the ${limit:,.2f} automated processing limit. Requires manual approval."
        ),
    )


def _check_no_duplicate(transaction_id: str) -> AuditRuleResult:
    if not os.path.exists(AUDIT_LOG):
        return AuditRuleResult(
            rule="no_duplicate",
            passed=True,
            detail="No prior dispute resolutions found in the audit log",
        )
    seen = False
    try:
        with open(AUDIT_LOG) as f:
            for line in f:
                entry = json.loads(line)
                if (
                    entry.get("transaction_id") == transaction_id
                    and entry.get("final_verdict") in ("approve", "reject")
                ):
                    seen = True
                    break
    except (json.JSONDecodeError, OSError):
        pass
    return AuditRuleResult(
        rule="no_duplicate",
        passed=not seen,
        detail=(
            f"No prior settlement on record for {transaction_id}"
            if not seen else
            f"A final settlement already exists for {transaction_id}. Cannot process the same transaction twice."
        ),
    )


# Visa reason-code → valid dispute categories (prefix match on first two chars)
_REASON_CODE_MAP: dict[str, set[str]] = {
    "10": {"unauthorized"},
    "11": {"unauthorized"},
    "12": {"incorrect_amount", "other"},
    "13": {"not_received", "incorrect_amount", "other"},
}


def _check_reason_code_match(reason_code: str, dispute_category: str) -> AuditRuleResult:
    prefix = reason_code.split(".")[0] if reason_code else ""
    allowed = _REASON_CODE_MAP.get(prefix)
    if allowed is None:
        return AuditRuleResult(
            rule="reason_code_match",
            passed=True,
            detail=f"Reason code {reason_code!r} has no defined category mapping, defaulted to pass.",
        )
    passed = dispute_category in allowed
    return AuditRuleResult(
        rule="reason_code_match",
        passed=passed,
        detail=(
            f"Reason code {reason_code} is valid for a '{dispute_category}' dispute"
            if passed else
            f"Reason code {reason_code} does not support '{dispute_category}' disputes (valid categories: {sorted(allowed)})"
        ),
    )


def _check_confidence_floor(
    recommendation: str,
    confidence: float,
    floor: float = 0.85,
) -> AuditRuleResult:
    # Only applies when Agent 2 recommends approve or reject
    if recommendation == "escalate":
        return AuditRuleResult(
            rule="confidence_floor",
            passed=True,
            detail="Recommendation is escalate. Confidence threshold is not evaluated for escalated cases.",
        )
    passed = confidence >= floor
    return AuditRuleResult(
        rule="confidence_floor",
        passed=passed,
        detail=(
            f"Investigator confidence {confidence:.0%} meets the {floor:.0%} minimum for an automated {recommendation} decision"
            if passed else
            f"Investigator confidence {confidence:.0%} is below the {floor:.0%} minimum. Human review required before issuing a {recommendation} decision."
        ),
    )


def pre_check_rules(
    triage:       TriageOutput,
    investigator: InvestigatorOutput,
    filing_date:  date,
) -> list[AuditRuleResult]:
    # Prefer the date extracted by triage; fall back to the DB record if triage missed it
    txn_date = triage.entities.transaction_date or _get_db_transaction_date(triage.transaction_id)
    return [
        _check_transaction_exists(triage.transaction_id),
        _check_dispute_window(txn_date, filing_date),
        _check_refund_threshold(triage.amount),
        _check_reason_code_match(triage.reason_code, triage.entities.dispute_category),
        _check_no_duplicate(triage.transaction_id),
        _check_confidence_floor(investigator.recommendation, investigator.confidence),
    ]


# ── Cost helper ───────────────────────────────────────────────────────────────

def _track_cost(prompt_tokens: int, completion_tokens: int, model: str = MODEL) -> float:
    p = PRICING[model]
    return (prompt_tokens / 1000 * p["input"]) + (completion_tokens / 1000 * p["output"])


# ── Audit record builder ───────────────────────────────────────────────────────

def _build_and_write_audit(
    run_id:              str,
    dispute:             DisputeInput,
    triage:              TriageOutput,
    investigator:        InvestigatorOutput,
    auditor:             AuditorOutput,
    triage_metrics:      AgentMetrics,
    investigator_metrics: AgentMetrics,
    auditor_metrics:     AgentMetrics,
) -> None:
    clf = investigator.classifier_output or {}
    record = AuditRecord(
        run_id           = run_id,
        transaction_id   = dispute.transaction_id,
        timestamp        = datetime.now(timezone.utc).isoformat(),
        raw_input        = dispute.model_dump(),

        triage_entities    = triage.entities.model_dump(),
        triage_masked_stmt = triage.masked_statement,
        triage_latency_ms  = triage_metrics.latency_ms,
        triage_tokens_in   = triage_metrics.tokens_in,
        triage_tokens_out  = triage_metrics.tokens_out,
        triage_cost_usd    = triage_metrics.cost_usd,

        inv_tool_calls_log  = investigator.tool_calls_log,
        inv_dt_fraud_prob   = clf.get("fraud_probability", 0.0),
        inv_dt_prediction   = clf.get("prediction", "unknown"),
        inv_dt_confidence   = clf.get("confidence", 0.0),
        inv_dt_decision_path = clf.get("decision_path", []),
        inv_dt_top_features = clf.get("top_features", []),
        inv_rag_query       = investigator.rag_query,
        inv_rag_chunks      = investigator.rag_chunks,
        inv_rag_precision   = investigator.rag_precision,
        inv_recommendation  = investigator.recommendation,
        inv_confidence      = investigator.confidence,
        inv_fraud_signals   = investigator.fraud_signals,
        inv_llm_reasoning   = investigator.llm_reasoning,
        inv_latency_ms      = investigator_metrics.latency_ms,
        inv_tokens_in       = investigator_metrics.tokens_in,
        inv_tokens_out      = investigator_metrics.tokens_out,
        inv_cost_usd        = investigator_metrics.cost_usd,

        aud_rules_checked  = [r.model_dump() for r in auditor.rules_checked],
        aud_violations     = auditor.violations,
        aud_verdict        = auditor.verdict,
        aud_latency_ms     = auditor_metrics.latency_ms,
        aud_tokens_in      = auditor_metrics.tokens_in,
        aud_tokens_out     = auditor_metrics.tokens_out,
        aud_cost_usd       = auditor_metrics.cost_usd,

        total_latency_ms = round(
            triage_metrics.latency_ms + investigator_metrics.latency_ms + auditor_metrics.latency_ms, 1
        ),
        total_tokens = (
            triage_metrics.tokens_in + triage_metrics.tokens_out
            + investigator_metrics.tokens_in + investigator_metrics.tokens_out
            + auditor_metrics.tokens_in + auditor_metrics.tokens_out
        ),
        total_cost_usd = round(
            triage_metrics.cost_usd + investigator_metrics.cost_usd + auditor_metrics.cost_usd, 6
        ),

        model_triage      = "gpt-4o-mini",
        model_investigator = "gpt-4o",
        model_auditor     = "gpt-4o-mini",
        dt_model_version  = clf.get("model_version", "dt_v1"),
        pipeline_version  = os.getenv("PIPELINE_VERSION", "v1.0.0"),
    )
    write_audit_record(record)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_auditor(
    dispute:              DisputeInput,
    triage:               TriageOutput,
    investigator:         InvestigatorOutput,
    triage_metrics:       AgentMetrics,
    investigator_metrics: AgentMetrics,
    filing_date:          Optional[date] = None,
) -> tuple[AuditorOutput, AgentMetrics]:
    """
    Run deterministic rule checks, compile verdict via LLM, write full audit record.
    """
    run_id = str(uuid4())
    if filing_date is None:
        filing_date = date.today()

    # ── 1. Deterministic pre-checks ───────────────────────────────────────────
    pre_results = pre_check_rules(triage, investigator, filing_date)

    # ── 2. LLM verdict compilation ────────────────────────────────────────────
    user_payload = {
        "agent2_recommendation": investigator.recommendation,
        "agent2_confidence":     investigator.confidence,
        "fraud_signals":         investigator.fraud_signals,
        "rag_source_clause":     investigator.rag_source_clause,
        "all_rules": [r.model_dump() for r in pre_results],
    }

    client = OpenAI()
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(user_payload)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    output = AuditorOutput.model_validate(json.loads(response.choices[0].message.content))

    auditor_metrics = AgentMetrics(
        model=MODEL,
        latency_ms=round(latency_ms, 1),
        tokens_in=response.usage.prompt_tokens,
        tokens_out=response.usage.completion_tokens,
        cost_usd=round(_track_cost(response.usage.prompt_tokens, response.usage.completion_tokens), 6),
    )

    # ── 3. Write full audit record ────────────────────────────────────────────
    _build_and_write_audit(
        run_id, dispute, triage, investigator, output,
        triage_metrics, investigator_metrics, auditor_metrics,
    )

    return output, auditor_metrics


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from backend.agents.triage import run_triage
    from backend.agents.investigator import run_investigator

    # filing_date matches demo data era (transactions now use 2026 dates)
    DEMO_FILING_DATE = date(2026, 5, 31)

    samples = [
        # 0 — clean unauthorized, should escalate (confidence < 0.85)
        DisputeInput(
            transaction_id="TXN-2024-0318-8821",
            amount=1240.00,
            merchant="ElectroMart Inc.",
            reason_code="10.4",
            customer_statement=(
                "I never made this $1,240 purchase at ElectroMart on March 18th. "
                "My card was in my wallet."
            ),
        ),
        # 1 — exceeds $5,000 threshold → policy_violation
        DisputeInput(
            transaction_id="TXN-2024-0316-9934",
            amount=7800.00,
            merchant="LuxuryGoods Direct",
            reason_code="10.4",
            customer_statement=(
                "I did not authorise a $7,800 purchase from LuxuryGoods Direct "
                "to a UAE address. This is fraudulent."
            ),
        ),
        # 2 — outside 120-day window (Sep 2023)
        DisputeInput(
            transaction_id="TXN-2023-0901-4421",
            amount=650.00,
            merchant="FurnitureWorld",
            reason_code="13.1",
            customer_statement=(
                "I want to dispute a charge from FurnitureWorld dated September 1st 2023 "
                "for furniture I never received."
            ),
        ),
    ]

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dispute = samples[idx]

    print(f"\n=== STEP 1: Triage ===")
    triage_out, triage_m = run_triage(dispute)
    print(triage_out.model_dump_json(indent=2))
    print(f"  → ${triage_m.cost_usd:.6f}  {triage_m.latency_ms:.0f}ms")

    print(f"\n=== STEP 2: Investigator ===")
    inv_out, inv_m = run_investigator(triage_out)
    print(inv_out.model_dump_json(indent=2))
    print(f"  → ${inv_m.cost_usd:.6f}  {inv_m.latency_ms:.0f}ms")

    print(f"\n=== STEP 3: Auditor ===")
    aud_out, aud_m = run_auditor(
        dispute, triage_out, inv_out,
        triage_m, inv_m,
        filing_date=DEMO_FILING_DATE,
    )
    print(aud_out.model_dump_json(indent=2))
    print(f"\n--- Auditor metrics ---")
    print(aud_m.model_dump_json(indent=2))

    total = round(triage_m.cost_usd + inv_m.cost_usd + aud_m.cost_usd, 6)
    print(f"\n--- Total pipeline cost: ${total:.6f} ---")
    print(f"Audit log written → {AUDIT_LOG}")
