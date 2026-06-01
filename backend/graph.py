"""
LangGraph StateGraph — wires Triage → Investigator → Auditor.

Isolation guarantee: the auditor node passes only the InvestigatorOutput
struct (recommendation + confidence) to run_auditor; the LLM inside
run_auditor receives only those two scalar fields via user_payload.
The full investigator tool-call history never touches Agent 3.
"""
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from backend.agents.auditor import AuditorOutput, run_auditor
from backend.agents.investigator import InvestigatorOutput, run_investigator
from backend.agents.triage import AgentMetrics, DisputeInput, TriageOutput, run_triage


# ── Shared pipeline state ─────────────────────────────────────────────────────

class DisputeState(TypedDict):
    raw_input:             dict
    triage_output:         Optional[TriageOutput]
    triage_metrics:        Optional[AgentMetrics]
    investigator_output:   Optional[InvestigatorOutput]
    investigator_metrics:  Optional[AgentMetrics]
    auditor_output:        Optional[AuditorOutput]
    auditor_metrics:       Optional[AgentMetrics]
    pipeline_complete:     bool
    halted:                bool
    halt_reason:           Optional[str]


def _initial_state(dispute: DisputeInput) -> DisputeState:
    return DisputeState(
        raw_input=dispute.model_dump(),
        triage_output=None,
        triage_metrics=None,
        investigator_output=None,
        investigator_metrics=None,
        auditor_output=None,
        auditor_metrics=None,
        pipeline_complete=False,
        halted=False,
        halt_reason=None,
    )


# ── Node functions ────────────────────────────────────────────────────────────

def triage_node(state: DisputeState) -> dict:
    dispute = DisputeInput.model_validate(state["raw_input"])
    out, metrics = run_triage(dispute)
    return {"triage_output": out, "triage_metrics": metrics}


def investigator_node(state: DisputeState) -> dict:
    out, metrics = run_investigator(state["triage_output"])
    return {"investigator_output": out, "investigator_metrics": metrics}


def auditor_node(state: DisputeState) -> dict:
    dispute = DisputeInput.model_validate(state["raw_input"])
    aud_out, aud_metrics = run_auditor(
        dispute=dispute,
        triage=state["triage_output"],
        investigator=state["investigator_output"],
        triage_metrics=state["triage_metrics"],
        investigator_metrics=state["investigator_metrics"],
    )
    halted = aud_out.verdict == "policy_violation"
    return {
        "auditor_output":  aud_out,
        "auditor_metrics": aud_metrics,
        "pipeline_complete": True,
        "halted":          halted,
        "halt_reason":     aud_out.violations[0] if halted and aud_out.violations else None,
    }


def _route_after_audit(state: DisputeState) -> str:
    return state["auditor_output"].verdict


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(DisputeState)

    graph.add_node("triage",       triage_node)
    graph.add_node("investigator", investigator_node)
    graph.add_node("auditor",      auditor_node)

    graph.set_entry_point("triage")
    graph.add_edge("triage",       "investigator")
    graph.add_edge("investigator", "auditor")
    graph.add_conditional_edges(
        "auditor",
        _route_after_audit,
        {"approve": END, "reject": END, "escalate": END, "policy_violation": END},
    )

    return graph.compile()


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from datetime import date

    samples = [
        DisputeInput(
            transaction_id="TXN-2024-0318-8821",
            amount=1240.00,
            merchant="ElectroMart Inc.",
            reason_code="10.4",
            customer_statement=(
                "I never made this $1,240 purchase at ElectroMart on March 18th. "
                "My card was in my wallet the whole time."
            ),
        ),
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
    ]

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dispute = samples[idx]

    print(f"\n=== Running pipeline for {dispute.transaction_id} ===\n")
    graph = build_graph()

    final_state = None
    for step in graph.stream(_initial_state(dispute)):
        for agent_name, updates in step.items():
            print(f"--- {agent_name} ---")
            if agent_name == "triage" and updates.get("triage_output"):
                out = updates["triage_output"]
                m   = updates["triage_metrics"]
                print(out.model_dump_json(indent=2))
                print(f"  → ${m.cost_usd:.6f}  {m.latency_ms:.0f}ms")
            elif agent_name == "investigator" and updates.get("investigator_output"):
                out = updates["investigator_output"]
                m   = updates["investigator_metrics"]
                print(out.model_dump_json(indent=2))
                print(f"  → ${m.cost_usd:.6f}  {m.latency_ms:.0f}ms")
            elif agent_name == "auditor" and updates.get("auditor_output"):
                out = updates["auditor_output"]
                m   = updates["auditor_metrics"]
                print(out.model_dump_json(indent=2))
                print(f"  → ${m.cost_usd:.6f}  {m.latency_ms:.0f}ms")
                print(f"\nFinal verdict: {out.verdict}")
                if updates.get("halted"):
                    print(f"Pipeline halted — violation: {updates['halt_reason']}")
