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
import time
from datetime import date, datetime, timezone
from typing import Optional

from dateutil import parser as dateutil_parser

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from backend.agents.triage import AgentMetrics, DisputeInput, TriageOutput
from backend.agents.investigator import InvestigatorOutput

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


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an independent compliance auditor at a payment network.

All five policy rules have already been evaluated deterministically and their
boolean results are provided to you. Your job is to compile the final output:
- Accept the pre-computed "passed" values exactly as given — do not override them.
- Write a concise, specific "detail" string for each rule.
- Determine the verdict:
    • If ANY rule has passed=false  → "policy_violation"
    • If all rules passed AND Agent 2 recommendation is "escalate" → "escalate"
    • If all rules passed → adopt Agent 2's recommendation verbatim
- List every rule name whose passed=false in "violations" (empty array if none).

Output ONLY valid JSON:
{
  "verdict": "approve" | "reject" | "policy_violation" | "escalate",
  "rules_checked": [
    {"rule": string, "passed": bool, "detail": string}
  ],
  "violations": [string]
}\
"""

# ── Deterministic pre-checks ──────────────────────────────────────────────────

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
            detail="transaction_date not provided — cannot verify window, defaulting pass",
        )
    txn_date = _parse_date(transaction_date_str)
    if txn_date is None:
        return AuditRuleResult(
            rule="dispute_window",
            passed=True,
            detail=f"transaction_date unparseable ({transaction_date_str!r}) — defaulting pass",
        )
    delta  = (filing_date - txn_date).days
    passed = delta <= max_days
    return AuditRuleResult(
        rule="dispute_window",
        passed=passed,
        detail=f"transaction_date={txn_date}, filing_date={filing_date}, gap={delta} days (limit {max_days})",
    )


def _check_refund_threshold(amount: float, limit: float = 5000.0) -> AuditRuleResult:
    passed = amount <= limit
    return AuditRuleResult(
        rule="refund_threshold",
        passed=passed,
        detail=f"amount=${amount:,.2f} (limit ${limit:,.2f})",
    )


def _check_no_duplicate(transaction_id: str) -> AuditRuleResult:
    if not os.path.exists(AUDIT_LOG):
        return AuditRuleResult(
            rule="no_duplicate",
            passed=True,
            detail="no prior audit log found",
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
        detail=f"transaction_id={transaction_id} {'already resolved' if seen else 'not seen before'}",
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
            detail=f"reason_code={reason_code!r} has no mapping — defaulting pass",
        )
    passed = dispute_category in allowed
    return AuditRuleResult(
        rule="reason_code_match",
        passed=passed,
        detail=(
            f"reason_code={reason_code} (category {prefix}.x) "
            f"{'matches' if passed else 'does not match'} "
            f"dispute_category={dispute_category!r} "
            f"(allowed: {sorted(allowed)})"
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
            detail="recommendation=escalate — floor does not apply",
        )
    passed = confidence >= floor
    return AuditRuleResult(
        rule="confidence_floor",
        passed=passed,
        detail=f"confidence={confidence:.2f} (floor {floor:.2f}), recommendation={recommendation}",
    )


def pre_check_rules(
    triage:       TriageOutput,
    investigator: InvestigatorOutput,
    filing_date:  date,
) -> list[AuditRuleResult]:
    return [
        _check_dispute_window(triage.entities.transaction_date, filing_date),
        _check_refund_threshold(triage.amount),
        _check_reason_code_match(triage.reason_code, triage.entities.dispute_category),
        _check_no_duplicate(triage.transaction_id),
        _check_confidence_floor(investigator.recommendation, investigator.confidence),
    ]


# ── Cost helper ───────────────────────────────────────────────────────────────

def _track_cost(prompt_tokens: int, completion_tokens: int, model: str = MODEL) -> float:
    p = PRICING[model]
    return (prompt_tokens / 1000 * p["input"]) + (completion_tokens / 1000 * p["output"])


# ── Audit log writer ──────────────────────────────────────────────────────────

def _write_audit_log(
    dispute:             DisputeInput,
    triage:              TriageOutput,
    investigator:        InvestigatorOutput,
    auditor:             AuditorOutput,
    triage_metrics:      AgentMetrics,
    investigator_metrics: AgentMetrics,
    auditor_metrics:     AgentMetrics,
) -> None:
    os.makedirs(os.path.dirname(AUDIT_LOG) or ".", exist_ok=True)
    entry = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "transaction_id": dispute.transaction_id,
        "final_verdict":  auditor.verdict,
        "violations":     auditor.violations,
        "triage":         triage.model_dump(),
        "investigator":   investigator.model_dump(),
        "auditor":        auditor.model_dump(),
        "metrics": {
            "triage":       triage_metrics.model_dump(),
            "investigator": investigator_metrics.model_dump(),
            "auditor":      auditor_metrics.model_dump(),
            "total_cost_usd": round(
                triage_metrics.cost_usd
                + investigator_metrics.cost_usd
                + auditor_metrics.cost_usd,
                6,
            ),
        },
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


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
    Run all five deterministic rule checks, then call the LLM to compile the
    final structured output (verdict + detail strings). Writes to audit_log.jsonl.
    """
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
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        cost_usd=round(_track_cost(response.usage.prompt_tokens, response.usage.completion_tokens), 6),
    )

    # ── 3. Write audit log ────────────────────────────────────────────────────
    _write_audit_log(
        dispute, triage, investigator, output,
        triage_metrics, investigator_metrics, auditor_metrics,
    )

    return output, auditor_metrics


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from backend.agents.triage import run_triage
    from backend.agents.investigator import run_investigator

    # filing_date anchored to demo data era so dispute_window checks are realistic
    DEMO_FILING_DATE = date(2024, 3, 20)

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
