"""
Agent 1 — Triage & Router
Model: gpt-4o-mini

Extracts structured entities from a raw dispute, masks PII, and classifies
urgency. Returns a TriageOutput (frozen; passed to Investigator) plus
AgentMetrics (latency, tokens, cost).
"""
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, field_validator

load_dotenv()

# ── Pricing table (per 1K tokens) ────────────────────────────────────────────
PRICING = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o":      {"input": 0.002500, "output": 0.010000},
}

MODEL = "gpt-4o-mini"

# ── Schemas ───────────────────────────────────────────────────────────────────

_VALID_REASON_CODES = {"10.4", "10.5", "11.3", "12.1", "12.4", "12.5", "13.1", "13.2"}


class DisputeInput(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    reason_code: str
    customer_statement: str

    @field_validator("transaction_id")
    @classmethod
    def txn_id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("transaction_id cannot be empty")
        return v

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be greater than 0")
        if v > 1_000_000:
            raise ValueError("amount exceeds the $1,000,000 single-dispute limit")
        return round(v, 2)

    @field_validator("merchant")
    @classmethod
    def merchant_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("merchant cannot be empty")
        return v

    @field_validator("reason_code")
    @classmethod
    def reason_code_valid(cls, v: str) -> str:
        if v not in _VALID_REASON_CODES:
            raise ValueError(
                f"reason_code must be one of {sorted(_VALID_REASON_CODES)}"
            )
        return v

    @field_validator("customer_statement")
    @classmethod
    def statement_min_length(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 20:
            raise ValueError("customer_statement must be at least 20 characters")
        return v


class TriageEntities(BaseModel):
    transaction_date: Optional[str]
    dispute_category: str   # unauthorized | not_received | incorrect_amount | other
    urgency: str            # standard | high_value | time_sensitive


class TriageOutput(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    reason_code: str
    masked_statement: str
    entities: TriageEntities


class AgentMetrics(BaseModel):
    model: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a financial dispute triage agent at a payment network.

Your job:
1. Extract structured entities from the customer dispute input.
2. Mask PERSONAL PII only in the masked_statement field:
   - Personal names (people) → [NAME]
   - Personal home/mailing addresses → [LOCATION]
   - Personal contact details (email, phone, account numbers) → [CONTACT]
   DO NOT mask: business/merchant names, product names, transaction amounts, dates, times,
   cities used as destinations (e.g. "shipped to Dubai"), or any other non-personal information.
   Dates such as "May 30, 2026" or "March 18th" are NEVER PII — leave them as-is.
   Copy transaction_id, amount, merchant, and reason_code verbatim from the input.
3. Output ONLY valid JSON matching the schema below. No prose, no explanation.

Output schema:
{
  "transaction_id": string,
  "amount": float,
  "merchant": string,
  "reason_code": string,
  "masked_statement": string,
  "entities": {
    "transaction_date": string | null,
    "dispute_category": "unauthorized" | "not_received" | "incorrect_amount" | "other",
    "urgency": "standard" | "high_value" | "time_sensitive"
  }
}

transaction_date: Extract the date the disputed charge occurred from the customer statement.
Any mention of a date ("May 30, 2026", "March 18th", "last Friday") MUST be extracted and
normalised to ISO 8601 (YYYY-MM-DD). For relative dates, anchor to 2024-01-01 as a proxy.
If only month/year is mentioned, use the 1st of that month. Set null ONLY if no date at all
appears anywhere in the statement. This field is critical — do not omit it if a date is present.
High-value = amount > $1000. Time-sensitive = customer mentions travel, emergency, or recurring charge.\
"""


# ── Cost helper ───────────────────────────────────────────────────────────────

def track_cost(usage, model: str = MODEL) -> float:
    p = PRICING[model]
    return (usage.prompt_tokens / 1000 * p["input"]) + \
           (usage.completion_tokens / 1000 * p["output"])


# ── Runner ────────────────────────────────────────────────────────────────────

def run_triage(dispute: DisputeInput) -> tuple[TriageOutput, AgentMetrics]:
    """Call the triage LLM and return a validated TriageOutput + cost metrics."""
    client = OpenAI()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": json.dumps(dispute.model_dump())},
    ]

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = json.loads(response.choices[0].message.content)
    output = TriageOutput.model_validate(raw)

    metrics = AgentMetrics(
        model=MODEL,
        latency_ms=round(latency_ms, 1),
        tokens_in=response.usage.prompt_tokens,
        tokens_out=response.usage.completion_tokens,
        cost_usd=round(track_cost(response.usage), 6),
    )

    return output, metrics


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    samples = [
        # clean unauthorized — high-value
        DisputeInput(
            transaction_id="TXN-2024-0318-8821",
            amount=1240.00,
            merchant="ElectroMart Inc.",
            reason_code="10.4",
            customer_statement=(
                "Hi, I'm John Smith at john.smith@gmail.com. "
                "I never made this $1,240 purchase at ElectroMart on March 18th. "
                "I was in Seattle at the time and my card was in my wallet."
            ),
        ),
        # subscription fraud — time-sensitive
        DisputeInput(
            transaction_id="TXN-2024-0201-4432",
            amount=89.99,
            merchant="StreamFlix Pro",
            reason_code="13.2",
            customer_statement=(
                "This recurring charge keeps appearing but I cancelled my subscription "
                "three months ago. Please refund immediately — I'm on a fixed income "
                "and this is an emergency for me. My number is 555-867-5309."
            ),
        ),
        # geo mismatch — standard
        DisputeInput(
            transaction_id="TXN-2024-0225-3310",
            amount=215.00,
            merchant="AirTravel Bookings",
            reason_code="10.4",
            customer_statement=(
                "There's a charge from AirTravel Bookings for $215 dated February 25th "
                "that I did not authorise. I did not travel to the UK at that time."
            ),
        ),
    ]

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dispute = samples[idx]

    print(f"\n--- Input ---")
    print(dispute.model_dump_json(indent=2))

    print(f"\n--- Running triage (model={MODEL}) ---")
    output, metrics = run_triage(dispute)

    print(f"\n--- TriageOutput ---")
    print(output.model_dump_json(indent=2))

    print(f"\n--- Metrics ---")
    print(metrics.model_dump_json(indent=2))
