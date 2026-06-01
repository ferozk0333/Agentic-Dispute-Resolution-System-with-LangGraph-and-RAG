"""
FastAPI entry point.

POST /resolve  — accepts DisputeInput JSON, streams SSE events per agent step.
GET  /health   — liveness check.

Each SSE event is a JSON object:
  { "agent": "triage"|"investigator"|"auditor",
    "output": {...},
    "metrics": {latency_ms, prompt_tokens, completion_tokens, cost_usd},
    "halted": bool,          # auditor only
    "halt_reason": str|null  # auditor only, first violation if halted
  }
A final `event: done` signals the stream is closed.
"""
import asyncio
import json
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.agents.triage import DisputeInput
from backend.graph import DisputeState, _initial_state, build_graph

load_dotenv()

app = FastAPI(title="Dispute Resolution API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _serialise(obj) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


async def _stream_pipeline(dispute: DisputeInput) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph pipeline in a thread pool and yield SSE-formatted strings
    as each agent completes. The sync graph.stream() call blocks its thread;
    events are passed back to the async generator via an asyncio.Queue.
    """
    graph = build_graph()
    initial = _initial_state(dispute)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            for step in graph.stream(initial):
                for agent_name, updates in step.items():
                    payload: dict = {"agent": agent_name}

                    if agent_name == "triage" and updates.get("triage_output"):
                        payload["output"]  = _serialise(updates["triage_output"])
                        payload["metrics"] = _serialise(updates["triage_metrics"])

                    elif agent_name == "investigator" and updates.get("investigator_output"):
                        payload["output"]  = _serialise(updates["investigator_output"])
                        payload["metrics"] = _serialise(updates["investigator_metrics"])

                    elif agent_name == "auditor" and updates.get("auditor_output"):
                        payload["output"]      = _serialise(updates["auditor_output"])
                        payload["metrics"]     = _serialise(updates["auditor_metrics"])
                        payload["halted"]      = updates.get("halted", False)
                        payload["halt_reason"] = updates.get("halt_reason")

                    loop.call_soon_threadsafe(queue.put_nowait, payload)

        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"error": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # end sentinel

    future = loop.run_in_executor(None, _run)

    while True:
        item = await queue.get()
        if item is None:
            yield "event: done\ndata: {}\n\n"
            break
        yield f"data: {json.dumps(item)}\n\n"

    await future  # surface any thread exception


@app.post("/resolve")
async def resolve_dispute(dispute: DisputeInput):
    return StreamingResponse(
        _stream_pipeline(dispute),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
