# ARCHITECTURE.md — Agent Pipeline Design

## Overview

Three agents run sequentially inside a LangGraph `StateGraph`. Each agent reads from shared state, writes its structured output back, and passes control to the next node. Agent 3 receives only Agent 2's decision struct — never its reasoning trace.

## LangGraph state schema

```python
from typing import TypedDict, Optional, List
from dataclasses import dataclass

@dataclass
class AgentMetrics:
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float

@dataclass
class TriageOutput:
    transaction_id: str
    amount: float
    merchant: str
    reason_code: str
    masked_statement: str
    entities: dict
    metrics: AgentMetrics

@dataclass
class InvestigatorOutput:
    recommendation: str          # "approve" | "reject" | "escalate"
    confidence: float            # 0.0–1.0
    fraud_signals: List[str]
    rag_source_clause: str       # exact Visa rule clause retrieved
    rag_precision: float
    tool_calls_made: List[str]
    metrics: AgentMetrics

@dataclass
class AuditorOutput:
    verdict: str                 # "approve" | "reject" | "policy_violation" | "escalate"
    rules_checked: List[dict]    # [{rule, result, passed}]
    violations: List[str]        # empty if all passed
    audit_log_entry: dict
    metrics: AgentMetrics

class DisputeState(TypedDict):
    raw_input: dict
    triage: Optional[TriageOutput]
    investigator: Optional[InvestigatorOutput]
    auditor: Optional[AuditorOutput]
    pipeline_complete: bool
    halted: bool
    halt_reason: Optional[str]
```

## LangGraph graph definition

```python
from langgraph.graph import StateGraph, END

def build_graph():
    graph = StateGraph(DisputeState)

    graph.add_node("triage", run_triage)
    graph.add_node("investigator", run_investigator)
    graph.add_node("auditor", run_auditor)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "investigator")
    graph.add_edge("investigator", "auditor")

    graph.add_conditional_edges(
        "auditor",
        route_after_audit,
        {
            "approve": END,
            "reject": END,
            "escalate": END,
            "policy_violation": END,
        }
    )

    return graph.compile()

def route_after_audit(state: DisputeState) -> str:
    return state["auditor"].verdict
```

## Agent isolation rule

Agent 3 must only receive `InvestigatorOutput.recommendation` and `InvestigatorOutput.confidence` — not the tool call history, not the RAG text, not the fraud signals list. This enforces independent adjudication.

```python
def run_auditor(state: DisputeState) -> DisputeState:
    inv = state["investigator"]
    # Pass ONLY the decision struct — not the full investigator output
    auditor_input = {
        "recommendation": inv.recommendation,
        "confidence": inv.confidence,
        "transaction": state["triage"].entities,
    }
    # ... call auditor agent with auditor_input only
```

## FastAPI endpoint

```python
# POST /resolve
# Accepts: DisputeInput (see AGENTS.md)
# Returns: Server-Sent Events stream, one event per agent step

@app.post("/resolve")
async def resolve_dispute(dispute: DisputeInput):
    async def event_stream():
        graph = build_graph()
        async for step in graph.astream({"raw_input": dispute.dict()}):
            agent_name = list(step.keys())[0]
            payload = step[agent_name]
            yield f"data: {json.dumps({'agent': agent_name, 'output': payload})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

## Pipeline halt on violation

When Auditor finds a policy violation, it sets `halted=True` and `halt_reason` in state. The graph routes to END but the frontend reads the `policy_violation` verdict and renders the violation panel instead of the approval panel.

```python
def run_auditor(state):
    # ... run checks
    if violations:
        state["halted"] = True
        state["halt_reason"] = violations[0]
        state["auditor"].verdict = "policy_violation"
    return state
```

## Audit log

Every completed pipeline run writes one line to `audit_log.jsonl`:

```json
{
  "timestamp": "2024-03-18T14:22:01Z",
  "transaction_id": "TXN-20240318-8821",
  "amount": 1240.00,
  "merchant": "ElectroMart Inc.",
  "reason_code": "10.4",
  "agent2_recommendation": "approve",
  "agent2_confidence": 0.91,
  "violations": [],
  "final_verdict": "approve",
  "total_latency_ms": 2442,
  "total_tokens": 3692,
  "total_cost_usd": 0.0083
}
```

Write this in `run_auditor` after verdict is determined, before returning state.

## CORS configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["POST"],
    allow_headers=["*"],
)
```
