# AGENTS.md — Agent Prompts, Tools & Output Contracts

## Agent 1 — Triage & Router

**Model:** `gpt-4o-mini`  
**File:** `backend/agents/triage.py`

### Responsibilities
- Extract structured entities from the raw dispute input
- Mask PII (cardholder name, location, contact details) before passing downstream
- Classify dispute urgency (standard / high-value / time-sensitive)

### System prompt
```
You are a financial dispute triage agent at a payment network.

Your job:
1. Extract structured entities from the customer dispute input.
2. Mask any PII: replace names with [NAME], locations with [LOCATION], contact details with [CONTACT].
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

High-value = amount > $1000. Time-sensitive = customer mentions travel, emergency, or recurring charge.
```

### Input schema
```python
class DisputeInput(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    reason_code: str
    customer_statement: str
```

### Cost tracking
```python
def track_cost(usage, model="gpt-4o-mini"):
    # Pricing as of 2024 — update if changed
    pricing = {
        "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},  # per 1K tokens
        "gpt-4o":      {"input": 0.002500, "output": 0.010000},
    }
    p = pricing[model]
    return (usage.prompt_tokens / 1000 * p["input"]) + \
           (usage.completion_tokens / 1000 * p["output"])
```

---

## Agent 2 — Investigator

**Model:** `gpt-4o`  
**File:** `backend/agents/investigator.py`

### Responsibilities
- Call all four tools in appropriate order
- Synthesize tool results with retrieved compliance rule
- Output a structured recommendation with confidence score and fraud signals
- The RAG check should be sophisticated. It should use MMR retrieval. 

### System prompt
```
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
- fraud_signals must be specific: "3 transactions in 2 hours" not "unusual activity".
```

### Tool schemas (OpenAI function calling format)

```python
INVESTIGATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_transactions",
            "description": "Fetch full transaction record from the database by transaction ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"}
                },
                "required": ["transaction_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_velocity_history",
            "description": "Get recent transaction velocity for a card: count, amounts, and geo data over the past 24 hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {"type": "string"},
                    "hours": {"type": "integer", "default": 24}
                },
                "required": ["card_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_merchant_risk",
            "description": "Return the risk score (0.0–1.0) and risk category for a merchant. Higher = riskier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_name": {"type": "string"}
                },
                "required": ["merchant_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_compliance_docs",
            "description": "Search the Visa compliance rulebook for relevant dispute rules. Returns top matching chunks with source section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]
```

### Tool dispatch loop
```python
import json
from openai import OpenAI

client = OpenAI()

def run_investigator(state):
    messages = [
        {"role": "system", "content": INVESTIGATOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(state["triage"].entities)}
    ]

    while True:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=INVESTIGATOR_TOOLS,
            tool_choice="auto"
        )
        latency = (time.perf_counter() - start) * 1000

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                result = dispatch_tool(tc.function.name, json.loads(tc.function.arguments))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })
        else:
            output = json.loads(msg.content)
            # accumulate usage across all turns
            return output, response.usage, latency
```

---

## Agent 3 — Auditor

**Model:** `gpt-4o-mini`  
**File:** `backend/agents/auditor.py`

### Responsibilities
- Evaluate Agent 2's recommendation against deterministic policy rules
- Never access Agent 2's tool call history or reasoning — only its structured output
- Halt pipeline and log violation if any rule fails

### System prompt
```
You are an independent compliance auditor at a payment network.

You will receive a dispute summary and Agent 2's recommendation.
Check the recommendation against each rule below. Output ONLY valid JSON.

Rules to check (in order):
1. dispute_window: Dispute must be within 120 days of transaction date.
2. refund_threshold: Refund amount must not exceed $5,000 for automated approval.
3. reason_code_match: Reason code must match the dispute category.
4. no_duplicate: Transaction ID must not appear in recent approved disputes.
5. confidence_floor: Agent 2 confidence must be >= 0.85 for approve/reject.

Output schema:
{
  "verdict": "approve" | "reject" | "policy_violation" | "escalate",
  "rules_checked": [
    {"rule": string, "passed": bool, "detail": string}
  ],
  "violations": [string]   // empty array if all passed
}

If ANY rule fails: verdict = "policy_violation" and list the violation.
```

### Deterministic pre-checks (run before LLM call)

Some rules can be checked in code before calling the LLM — do this to save tokens:

```python
def pre_check_rules(triage_output, investigator_output) -> list[dict]:
    """Run deterministic checks in Python. Return list of rule results."""
    results = []

    # Rule: confidence floor
    results.append({
        "rule": "confidence_floor",
        "passed": investigator_output.confidence >= 0.85,
        "detail": f"confidence={investigator_output.confidence:.2f}"
    })

    # Rule: refund threshold
    results.append({
        "rule": "refund_threshold",
        "passed": triage_output.amount <= 5000,
        "detail": f"amount=${triage_output.amount:.2f}"
    })

    return results
```

Pass these results into the auditor prompt so the LLM only needs to check the remaining context-dependent rules (dispute window, reason code match, duplicate check).

---

## Tool implementations

### `sql_tool.py`
```python
import sqlite3, os

DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/ieee_fraud.db")

def query_transactions(transaction_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM transactions WHERE TransactionID = ?", (transaction_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"error": "Transaction not found"}
    return dict(row)

def get_velocity_history(card_id: str, hours: int = 24) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as count, SUM(TransactionAmt) as total,
               COUNT(DISTINCT addr1) as unique_locations
        FROM transactions
        WHERE card1 = ? AND TransactionDT > (
            SELECT MAX(TransactionDT) - ? * 3600 FROM transactions WHERE card1 = ?
        )
    """, (card_id, hours, card_id))
    row = cur.fetchone()
    conn.close()
    return {"count": row[0], "total_amount": row[1], "unique_locations": row[2]}
```

### `rag_tool.py`
```python
import chromadb, os
from openai import OpenAI

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
client = OpenAI()
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection("visa_rules")

def search_compliance_docs(query: str, top_k: int = 3) -> list[dict]:
    embedding = client.embeddings.create(
        input=query, model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "section": meta.get("section", "unknown"),
            "page": meta.get("page", "unknown"),
            "relevance_score": round(1 - dist, 3)
        })
    return chunks

### `risk_tool.py`
```python
# Merchant risk scores derived from IEEE-CIS merchant frequency + fraud rate
# Precomputed at ingest time and stored in SQLite merchant_risk table

def check_merchant_risk(merchant_name: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT risk_score, risk_category FROM merchant_risk WHERE merchant_name LIKE ?",
        (f"%{merchant_name}%",)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"risk_score": 0.5, "risk_category": "unknown", "note": "merchant not in database"}
    return {"risk_score": row[0], "risk_category": row[1]}
```
