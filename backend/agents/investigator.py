"""
Agent 2 — Investigator
Model: gpt-4o

Calls four tools in sequence (query_transactions → get_velocity_history →
check_merchant_risk → search_compliance_docs), then synthesises a structured
recommendation with confidence score and cited Visa rule clause.

Receives: TriageOutput from Agent 1
Returns:  InvestigatorOutput + AgentMetrics (tokens accumulated across all turns)
"""
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from backend.agents.triage import AgentMetrics, TriageOutput
from backend.tools.rag_tool import compute_rag_precision, search_compliance_docs
from backend.tools.risk_tool import check_merchant_risk
from backend.tools.sql_tool import query_transactions
from backend.tools.velocity_tool import get_velocity_history

load_dotenv()

MODEL = "gpt-4o"

PRICING = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o":      {"input": 0.002500, "output": 0.010000},
}

# ── Output schema ─────────────────────────────────────────────────────────────

class InvestigatorOutput(BaseModel):
    recommendation:   str           # approve | reject | escalate
    confidence:       float
    fraud_signals:    list[str]
    rag_source_clause: str
    rag_precision:    float


# ── Tool definitions (OpenAI function calling format) ─────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_transactions",
            "description": "Fetch full transaction record from the database by transaction ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_velocity_history",
            "description": (
                "Get recent transaction velocity for a card: count, amounts, "
                "and countries over a lookback window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {"type": "string"},
                    "hours":   {"type": "integer", "default": 24},
                },
                "required": ["card_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_merchant_risk",
            "description": (
                "Return the risk score (0.0–1.0) and risk category for a merchant. "
                "Higher score = riskier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_name": {"type": "string"},
                },
                "required": ["merchant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_compliance_docs",
            "description": (
                "Search the Visa compliance rulebook for relevant dispute rules. "
                "Returns top matching chunks with source section and page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a fraud investigator at a payment network. You have access to four tools.
Use them in this order:
1. query_transactions — get the full transaction record
2. get_velocity_history — check recent card activity for anomalies
3. check_merchant_risk — get the merchant's risk profile
4. search_compliance_docs — retrieve the applicable Visa dispute rule

After calling all tools, output ONLY valid JSON:
{
  "recommendation": "approve" | "reject" | "escalate",
  "confidence": float (0.0–1.0),
  "fraud_signals": [list of specific signals found, max 5],
  "rag_source_clause": "exact section text retrieved from compliance docs",
  "rag_precision": float (your estimate of retrieval relevance, 0.0–1.0)
}

Rules:
- confidence > 0.85 required to recommend approve or reject. Otherwise: escalate.
- Always cite the exact Visa rule clause. If not found, set rag_source_clause to "NOT FOUND" and escalate.
- fraud_signals must be specific: "3 transactions in 2 hours" not "unusual activity".\
"""

# ── Tool dispatcher ───────────────────────────────────────────────────────────

_TOOL_MAP = {
    "query_transactions":   query_transactions,
    "get_velocity_history": get_velocity_history,
    "check_merchant_risk":  check_merchant_risk,
    "search_compliance_docs": search_compliance_docs,
}


def _dispatch(name: str, args: dict):
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn(**args)


# ── Cost helper ───────────────────────────────────────────────────────────────

def _track_cost(prompt_tokens: int, completion_tokens: int, model: str = MODEL) -> float:
    p = PRICING[model]
    return (prompt_tokens / 1000 * p["input"]) + (completion_tokens / 1000 * p["output"])


# ── Runner ────────────────────────────────────────────────────────────────────

def run_investigator(triage: TriageOutput) -> tuple[InvestigatorOutput, AgentMetrics]:
    """Run the investigator tool-call loop and return structured output + metrics."""
    client = OpenAI()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": triage.model_dump_json()},
    ]

    total_prompt     = 0
    total_completion = 0
    t0 = time.perf_counter()

    # Track the last RAG call so we can compute objective precision after the loop
    last_rag_query:  Optional[str]       = None
    last_rag_chunks: Optional[list[dict]] = None

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        total_prompt     += response.usage.prompt_tokens
        total_completion += response.usage.completion_tokens

        msg = response.choices[0].message

        if msg.tool_calls:
            # Append the assistant turn (with tool_calls)
            messages.append(msg)
            # Dispatch each tool call and append results
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                if name == "search_compliance_docs":
                    last_rag_query = args.get("query", "")
                result = _dispatch(name, args)
                if name == "search_compliance_docs":
                    last_rag_chunks = result if isinstance(result, list) else None
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result),
                })
        else:
            # No more tool calls — parse final JSON response
            latency_ms = (time.perf_counter() - t0) * 1000
            # Strip markdown code fences that gpt-4o sometimes wraps around JSON
            raw = (msg.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0].strip()
            output = InvestigatorOutput.model_validate(json.loads(raw))

            # Replace self-reported rag_precision with objective LLM judge score
            if last_rag_query and last_rag_chunks:
                output = output.model_copy(
                    update={"rag_precision": compute_rag_precision(last_rag_query, last_rag_chunks)}
                )

            metrics = AgentMetrics(
                model=MODEL,
                latency_ms=round(latency_ms, 1),
                tokens_in=total_prompt,
                tokens_out=total_completion,
                cost_usd=round(_track_cost(total_prompt, total_completion), 6),
            )
            return output, metrics


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from backend.agents.triage import DisputeInput, run_triage

    samples = [
        # clean unauthorized — high-value ($1,240, low-risk merchant)
        DisputeInput(
            transaction_id="TXN-2024-0318-8821",
            amount=1240.00,
            merchant="ElectroMart Inc.",
            reason_code="10.4",
            customer_statement=(
                "I never made this $1,240 purchase at ElectroMart on March 18th. "
                "I was in Seattle at the time and my card was in my wallet."
            ),
        ),
        # high velocity — three hits same day on card_****5511
        DisputeInput(
            transaction_id="TXN-2024-0318-5511",
            amount=399.00,
            merchant="GameStore Digital",
            reason_code="10.4",
            customer_statement=(
                "There are three identical $399 charges from GameStore Digital "
                "on March 18th that I did not authorise."
            ),
        ),
        # geo mismatch — US card used in France
        DisputeInput(
            transaction_id="TXN-2024-0305-7743",
            amount=499.00,
            merchant="LuxuryWatch Co.",
            reason_code="10.4",
            customer_statement=(
                "A $499 charge appeared from LuxuryWatch Co. in France. "
                "I have never been to France and my card never left the US."
            ),
        ),
    ]

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dispute = samples[idx]

    print(f"\n=== STEP 1: Triage ===")
    triage_out, triage_metrics = run_triage(dispute)
    print(triage_out.model_dump_json(indent=2))
    print(f"  → cost ${triage_metrics.cost_usd:.6f}, {triage_metrics.latency_ms:.0f}ms")

    print(f"\n=== STEP 2: Investigator ===")
    inv_out, inv_metrics = run_investigator(triage_out)
    print(inv_out.model_dump_json(indent=2))
    print(f"\n--- Metrics ---")
    print(inv_metrics.model_dump_json(indent=2))
