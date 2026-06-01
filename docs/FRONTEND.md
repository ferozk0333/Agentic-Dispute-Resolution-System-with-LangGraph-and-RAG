# FRONTEND.md — Demo UI Specification

## Overview

A single-page HTML/CSS/JS demo that walks a user through a dispute resolution in five steps. Each step reveals one agent's work in real time via SSE streaming from the FastAPI backend. No frameworks. No build step. One `index.html`, one `style.css`, one `app.js`.

---

## File structure

```
frontend/
├── index.html
├── style.css
└── app.js
```

Open `index.html` directly in a browser (or serve with `python -m http.server 5500`).

---

## Design principles

- Dark background (`#0f0f11`) with light card surfaces — makes metrics pop
- Monospace font for transaction IDs, JSON payloads, code values
- Color coding matches agent identity throughout:
  - Agent 1 (Triage): purple (`#7c3aed`)
  - Agent 2 (Investigator): teal (`#0d9488`)
  - Agent 3 (Auditor): orange (`#d97706`)
  - Results: green (approve) / red (reject/violation)
- No animations on load. Animate only when data arrives via SSE.
- All numbers rounded before display — never show raw floats.

---

## Step structure

Five steps, rendered as panels. Only the active panel is visible. A step bar at the top shows progress.

```
[Step 1: Input] → [Step 2: Triage] → [Step 3: Investigate] → [Step 4: Audit] → [Step 5: Results]
```

Step bar behavior:
- Incomplete steps: gray circle with step number
- Active step: colored circle (agent color) with pulse animation
- Complete steps: filled circle with checkmark

---

## Step 1 — Dispute input

Form fields:
- Transaction ID (text input, monospace, pre-filled with sample)
- Amount (text input, pre-filled with "$1,240.00")
- Merchant name (text input)
- Reason code (dropdown: 10.4 Unauthorized / 13.1 Not received / 12.5 Incorrect amount)
- Customer statement (textarea, 4 rows)
- "Run dispute pipeline" button — triggers SSE stream on click

On button click:
1. Disable the form
2. Advance step bar to Step 2
3. Open SSE connection to `POST /resolve`
4. As each SSE event arrives, populate the corresponding step panel and advance the step bar

---

## Step 2 — Triage panel

Populated when SSE event `agent: "triage"` arrives.

Sections:
1. Header: "Agent 1 — triage & router" + model badge (`gpt-4o-mini`)
2. Extracted entities: rendered as pill badges (transaction ID, amount, merchant, reason code, urgency)
3. PII-masked statement: rendered in a code block with masked tokens highlighted in amber
4. Metric row (4 cards side by side):
   - Latency: `{latency_ms}ms`
   - Tokens in / out: `{tokens_in} / {tokens_out}`
   - Cost: `${cost_usd}`
   - Model: `gpt-4o-mini`

Animation: cards fade in sequentially (50ms stagger) when data arrives.

---

## Step 3 — Investigator panel

Populated when SSE event `agent: "investigator"` arrives.

Sections:
1. Header: "Agent 2 — investigator" + model badge (`gpt-4o`)
2. Tool call log: animated list. Each tool call appears as a row with:
   - Green pulse dot (animated)
   - Tool name + arguments in monospace
   - Result badge (e.g. "hit", "3 txn / 2h", "score 0.21", "2 chunks")
3. RAG retrieval block: gray code-style box showing the retrieved Visa rule clause + section + page
4. Fraud signals: bullet list of specific signals found
5. Recommendation badge: large colored badge (green = approve, red = reject, yellow = escalate)
6. Confidence bar: horizontal progress bar, 0–100%
7. Metric row (4 cards):
   - Latency
   - Tokens in / out
   - Cost
   - RAG precision score

Tool call animation: each tool row appears with a 300ms delay between rows, simulating live execution.

---

## Step 4 — Auditor panel

Populated when SSE event `agent: "auditor"` arrives.

Sections:
1. Header: "Agent 3 — auditor (independent)" + model badge (`gpt-4o-mini`)
2. Policy rules checklist: table with columns Rule / Status / Detail
   - Pass rows: green checkmark icon
   - Fail rows: red X icon, row background tinted red
3. Audit log entry: collapsible JSON block (collapsed by default, "View audit log" toggle)
4. Verdict: large badge — Approve (green) / Reject (red) / Policy violation (amber) / Escalate (gray)

If `verdict === "policy_violation"`:
- Show a red alert banner: "Pipeline halted — policy violation detected"
- List all violations in the banner
- Skip Step 5 and show violation summary instead

5. Metric row (4 cards):
   - Latency
   - Tokens in / out
   - Cost
   - Violations caught (integer)

---

## Step 5 — Results panel

Populated after auditor SSE event, using `pipeline_totals`.

Sections:
1. Final verdict — large centered badge
2. Pipeline summary metric grid (2×2 or 4 across):
   - Total latency
   - Total tokens
   - Total cost
   - Policy catch rate
3. Per-agent breakdown table:
   | Agent | Model | Latency | Tokens | Cost |
   |---|---|---|---|---|
   | Triage | gpt-4o-mini | ... | ... | ... |
   | Investigator | gpt-4o | ... | ... | ... |
   | Auditor | gpt-4o-mini | ... | ... | ... |
4. Evaluation metrics row:
   - F1 score (from batch eval, static display — loaded from `eval_results.json` if present)
   - RAG precision (from this run)
5. "Run another dispute" button — resets to Step 1

---

## SSE connection in JavaScript

```javascript
async function runPipeline(formData) {
  const response = await fetch("http://localhost:8000/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formData),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split("\n").filter(l => l.startsWith("data: "));

    for (const line of lines) {
      const event = JSON.parse(line.slice(6));
      handleAgentEvent(event);
    }
  }
}

function handleAgentEvent(event) {
  switch (event.agent) {
    case "triage":       populateTriage(event);       advanceStep(2); break;
    case "investigator": populateInvestigator(event); advanceStep(3); break;
    case "auditor":      populateAuditor(event);      advanceStep(4); break;
    case "complete":     populateResults(event);      advanceStep(5); break;
  }
}
```

---

## Number formatting rules

All numbers must be formatted before injecting into the DOM:

```javascript
const fmt = {
  ms:      (v) => `${Math.round(v)}ms`,
  tokens:  (v) => v.toLocaleString(),
  cost:    (v) => v < 0.001 ? "< $0.001" : `$${v.toFixed(4)}`,
  f1:      (v) => v.toFixed(2),
  pct:     (v) => `${Math.round(v * 100)}%`,
  score:   (v) => v.toFixed(2),
};
```

Never render a raw float. Every value through `fmt.*` before display.

---

## Error handling

- If SSE connection fails: show error banner "Pipeline error — check backend is running on :8000"
- If an agent returns an error: show error state for that step panel, allow user to retry
- If policy violation: halt step progression at Step 4, render violation panel, offer "Review manually" and "Run new dispute" buttons
- Network timeout after 30 seconds: show timeout error with retry button

---

## Sample data for demo

Pre-fill the form with this dispute to demonstrate a clean approve flow:

```javascript
const SAMPLE_DISPUTE = {
  transaction_id: "TXN-20240318-8821",
  amount: 1240.00,
  merchant: "ElectroMart Inc.",
  reason_code: "10.4",
  customer_statement: "I did not authorize this charge. My card was used while I was travelling internationally and I have never shopped at this merchant."
};
```

Include a second sample that triggers a policy violation (amount > $5000):

```javascript
const SAMPLE_VIOLATION = {
  transaction_id: "TXN-20240318-9934",
  amount: 7800.00,
  merchant: "LuxuryGoods Direct",
  reason_code: "10.4",
  customer_statement: "Unauthorized purchase on my account."
};
```

Add a "Load sample" dropdown above the form so the user can switch between samples without typing.
