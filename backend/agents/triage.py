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
from pydantic import BaseModel

load_dotenv()

# ── Pricing table (per 1K tokens) ────────────────────────────────────────────
PRICING = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o":      {"input": 0.002500, "output": 0.010000},
}

MODEL = "gpt-4o-mini"

# ── Schemas ───────────────────────────────────────────────────────────────────

class DisputeInput(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    reason_code: str
    customer_statement: str


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
   - Personal locations/addresses → [LOCATION]
   - Personal contact details (email, phone, account numbers) → [CONTACT]
   DO NOT mask business/merchant names, product names, or transaction amounts.
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

transaction_date: extract and normalise to ISO 8601 (YYYY-MM-DD). If only month/year is mentioned, use the 1st of that month. If no date is mentioned, set null.
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
