# METRICS.md — Performance Instrumentation

## Metrics tracked

| Metric | Scope | Source | Displayed in UI |
|---|---|---|---|
| Latency (ms) | Per agent + total | `time.perf_counter()` | Yes — each step card |
| Tokens in | Per agent + total | `response.usage.prompt_tokens` | Yes — each step card |
| Tokens out | Per agent + total | `response.usage.completion_tokens` | Yes — each step card |
| Cost (USD) | Per agent + total | Tokens × model pricing | Yes — each step card + summary |
| F1 score | Batch eval only | IEEE-CIS test set | Yes — results panel |
| RAG precision | Per Investigator run | Relevance score from ChromaDB | Yes — results panel |
| Policy catch rate | Batch eval only | Violations caught / total violations seeded | Yes — results panel |

---

## Latency instrumentation

Wrap every OpenAI API call:

```python
import time

def timed_completion(client, **kwargs) -> tuple:
    """Returns (response, latency_ms)"""
    start = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
    return response, round(latency_ms, 1)
```

For multi-turn agents (Investigator has a tool call loop), accumulate latency across all turns:

```python
total_latency = 0.0
while True:
    response, turn_latency = timed_completion(client, ...)
    total_latency += turn_latency
    if not response.choices[0].message.tool_calls:
        break
```

---

## Token & cost instrumentation

### Usage accumulation for multi-turn agents

```python
from dataclasses import dataclass, field

@dataclass
class UsageAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage):
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens
```

### Cost computation

```python
MODEL_PRICING = {
    "gpt-4o": {
        "input":  0.002500,   # USD per 1K tokens
        "output": 0.010000,
    },
    "gpt-4o-mini": {
        "input":  0.000150,
        "output": 0.000600,
    },
}

def compute_cost(usage: UsageAccumulator, model: str) -> float:
    p = MODEL_PRICING[model]
    cost = (usage.prompt_tokens / 1000 * p["input"]) + \
           (usage.completion_tokens / 1000 * p["output"])
    return round(cost, 6)
```

### AgentMetrics dataclass

Populate this for every agent and attach to its output:

```python
@dataclass
class AgentMetrics:
    model: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float

    def to_dict(self):
        return {
            "model": self.model,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
        }
```

### Pipeline totals

Compute in the FastAPI response after all agents complete:

```python
def compute_pipeline_totals(triage, investigator, auditor) -> dict:
    agents = [triage, investigator, auditor]
    return {
        "total_latency_ms": round(sum(a.metrics.latency_ms for a in agents), 1),
        "total_tokens_in":  sum(a.metrics.tokens_in for a in agents),
        "total_tokens_out": sum(a.metrics.tokens_out for a in agents),
        "total_tokens":     sum(a.metrics.tokens_in + a.metrics.tokens_out for a in agents),
        "total_cost_usd":   round(sum(a.metrics.cost_usd for a in agents), 6),
    }
```

---

## F1 score — batch evaluation

**File:** `backend/eval/run_eval.py`

Run this separately from the live demo. It runs 100–500 disputes from `test_set.csv` through the full pipeline and computes aggregate metrics.

```python
import pandas as pd
from sklearn.metrics import f1_score, classification_report

def run_batch_eval(n_samples: int = 200):
    df = pd.read_csv("./data/test_set.csv").sample(n_samples, random_state=42)

    y_true = []
    y_pred = []
    costs  = []
    latencies = []

    for _, row in df.iterrows():
        dispute_input = row_to_dispute_input(row)
        result = run_pipeline_sync(dispute_input)

        # Map pipeline verdict to binary fraud prediction
        predicted_fraud = 1 if result["verdict"] in ["reject", "policy_violation"] else 0
        y_true.append(int(row["isFraud"]))
        y_pred.append(predicted_fraud)
        costs.append(result["total_cost_usd"])
        latencies.append(result["total_latency_ms"])

    f1 = f1_score(y_true, y_pred, average="binary")
    print(classification_report(y_true, y_pred))
    print(f"F1:               {f1:.4f}")
    print(f"Avg cost/dispute: ${sum(costs)/len(costs):.5f}")
    print(f"Avg latency:      {sum(latencies)/len(latencies):.0f}ms")
    print(f"Total eval cost:  ${sum(costs):.4f}")

    return {
        "f1": round(f1, 4),
        "avg_cost_usd": round(sum(costs)/len(costs), 5),
        "avg_latency_ms": round(sum(latencies)/len(latencies), 1),
        "n_samples": n_samples,
    }
```

---

## RAG precision instrumentation

RAG precision = fraction of retrieved chunks that are genuinely relevant.

Use a lightweight judge call to score each retrieved chunk:

```python
def score_rag_chunk(query: str, chunk_text: str) -> float:
    """Returns 1.0 if chunk is relevant to query, 0.0 if not."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # cheap judge
        messages=[{
            "role": "user",
            "content": f"Is this text relevant to the query: '{query}'?\n\nText: {chunk_text}\n\nReply with only 1 (relevant) or 0 (not relevant)."
        }],
        max_tokens=1,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0

def compute_rag_precision(query: str, chunks: list[dict]) -> float:
    scores = [score_rag_chunk(query, c["text"]) for c in chunks]
    return round(sum(scores) / len(scores), 3) if scores else 0.0
```

Note: this adds a small cost per Investigator run (~$0.0001). Only run in demo mode, not batch eval.

---

## SSE payload schema

The FastAPI `/resolve` endpoint streams one SSE event per agent. Each event carries the agent output plus its metrics. The frontend reads this stream and populates each step panel.

```json
{
  "agent": "triage",
  "output": {
    "transaction_id": "TXN-20240318-8821",
    "masked_statement": "...",
    "entities": { ... }
  },
  "metrics": {
    "model": "gpt-4o-mini",
    "latency_ms": 312,
    "tokens_in": 410,
    "tokens_out": 88,
    "cost_usd": 0.0001
  }
}
```

Final event (after auditor) also includes `pipeline_totals` and optionally `eval_metrics` if a batch eval has been run.

---

## Metric display rules for frontend

- Latency: display as integer ms (e.g. `312ms`). Never show decimal.
- Tokens: display as integer with comma separator for 4+ digits (e.g. `2,210`).
- Cost: display with 4 significant figures (e.g. `$0.0081`). Never round to 0.
- F1: display as 2 decimal places (e.g. `0.91`).
- RAG precision: display as 2 decimal places (e.g. `0.87`).
- Total cost: sum across all agents. If < $0.001, display as `< $0.001`.

All rounding must happen in JS before rendering — never display raw floats.
