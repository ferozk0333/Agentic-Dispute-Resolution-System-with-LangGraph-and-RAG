'use strict';

const API = 'http://localhost:8000';

// ── Sample disputes ───────────────────────────────────────────────────────────

const SAMPLES = {
  approve: {
    transaction_id: 'TXN-2024-0318-8821',
    amount: 1240.00,
    merchant: 'ElectroMart Inc.',
    reason_code: '10.4',
    customer_statement:
      'I did not authorize this charge. My card was in my wallet the entire time and I have never shopped at this merchant.',
  },
  violation: {
    transaction_id: 'TXN-2024-0316-9934',
    amount: 7800.00,
    merchant: 'LuxuryGoods Direct',
    reason_code: '10.4',
    customer_statement:
      'I did not authorize a $7,800 purchase from LuxuryGoods Direct shipped to a UAE address. This is fraudulent.',
  },
};

// ── Number formatting ─────────────────────────────────────────────────────────

const fmt = {
  ms:     (v) => `${Math.round(v)}ms`,
  tokens: (v) => Number(v).toLocaleString(),
  cost:   (v) => (v < 0.001 ? '< $0.001' : `$${v.toFixed(4)}`),
  pct:    (v) => `${Math.round(v * 100)}%`,
  score:  (v) => parseFloat(v).toFixed(2),
  f1:     (v) => parseFloat(v).toFixed(2),
};

// ── Pipeline state ────────────────────────────────────────────────────────────

const STATE = {
  triage: null,
  investigator: null,
  auditor: null,
};

// ── Step bar ──────────────────────────────────────────────────────────────────

function setStepState(stepNum, state) {
  // state: 'pending' | 'active' | 'complete'
  document.querySelector(`.step[data-step="${stepNum}"]`).dataset.state = state;
}

function advanceStep(completedStep) {
  setStepState(completedStep, 'complete');
  if (completedStep < 5) setStepState(completedStep + 1, 'active');
}

// ── Panel switching ───────────────────────────────────────────────────────────

function showPanel(n) {
  document.querySelectorAll('.panel').forEach((p) => p.classList.remove('panel-active'));
  const panel = document.getElementById(`panel-${n}`);
  panel.classList.add('panel-active');
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }

function metricCard(label, value, sub) {
  return `<div class="metric-card">
    <div class="metric-label">${label}</div>
    <div class="metric-value">${value}</div>
    ${sub ? `<div class="metric-sub">${sub}</div>` : ''}
  </div>`;
}

function verdictClass(verdict) {
  return `verdict-${verdict.toLowerCase().replace(/ /g, '_')}`;
}

function verdictLabel(verdict) {
  const map = {
    approve:          '✓ Approve',
    reject:           '✗ Reject',
    escalate:         '⇡ Escalate',
    policy_violation: '⚠ Policy Violation',
  };
  return map[verdict] || verdict;
}

// ── Panel 2: Triage ───────────────────────────────────────────────────────────

function populateTriage(event) {
  STATE.triage = event;
  const out  = event.output;
  const m    = event.metrics;
  const ents = out.entities;

  // Entities pills
  const pills = [
    { label: 'txn', value: out.transaction_id },
    { label: 'amount', value: `$${Number(out.amount).toLocaleString()}` },
    { label: 'merchant', value: out.merchant },
    { label: 'reason', value: out.reason_code },
    { label: 'category', value: ents.dispute_category },
    { label: 'urgency', value: ents.urgency.replace(/_/g, ' ') },
    ents.transaction_date ? { label: 'date', value: ents.transaction_date } : null,
  ].filter(Boolean);

  el('triageEntities').innerHTML = pills
    .map(p => `<span class="entity-pill urgency-${ents.urgency}">
      <span class="pill-label">${p.label}</span>
      <span class="pill-value">${p.label === 'urgency' ? p.value : p.value}</span>
    </span>`)
    .join('');

  // Masked statement — highlight PII tokens in amber
  const highlighted = out.masked_statement.replace(
    /\[(NAME|LOCATION|CONTACT|PII)\]/g,
    '<mark class="pii-mask">[$1]</mark>'
  );
  el('maskedStatement').innerHTML = highlighted;

  // Metrics
  el('triageMetrics').innerHTML = [
    metricCard('Latency',      fmt.ms(m.latency_ms)),
    metricCard('Tokens in/out', `${fmt.tokens(m.tokens_in)} / ${fmt.tokens(m.tokens_out)}`),
    metricCard('Cost',         fmt.cost(m.cost_usd)),
    metricCard('Model',        m.model),
  ].join('');

  el('triageLoading').classList.add('hidden');
  el('triageContent').classList.remove('hidden');
}

// ── Panel 3: Investigator ─────────────────────────────────────────────────────

function populateInvestigator(event) {
  STATE.investigator = event;
  const out = event.output;
  const m   = event.metrics;
  const tri = STATE.triage ? STATE.triage.output : {};

  // Tool log — 4 rows, staggered via CSS animation-delay
  const tools = buildToolRows(tri, out);
  el('toolLog').innerHTML = tools.map(t =>
    `<div class="tool-row">
      <span class="tool-pulse"></span>
      <span class="tool-name">${t.name}</span>
      <span class="tool-args">${t.args}</span>
      <span class="tool-badge ${t.badgeClass}">${t.badge}</span>
    </div>`
  ).join('');

  // RAG block
  const ragText = out.rag_source_clause || 'NOT FOUND';
  const ragMeta = event.output.rag_precision !== undefined
    ? `<span>precision: ${fmt.score(out.rag_precision)}</span>`
    : '';
  el('ragBlock').innerHTML =
    `${escHtml(ragText)}<div class="rag-meta">${ragMeta}</div>`;

  // Fraud signals
  const signals = out.fraud_signals || [];
  el('signalList').innerHTML = signals.length
    ? signals.map(s => `<li>${escHtml(s)}</li>`).join('')
    : '<li>No specific signals detected</li>';

  // Recommendation badge
  const rec = out.recommendation;
  el('recBadge').className = `verdict-badge ${verdictClass(rec)}`;
  el('recBadge').textContent = verdictLabel(rec);

  // Confidence bar
  const conf = out.confidence || 0;
  const confClass = conf >= 0.85 ? 'conf-high' : conf >= 0.70 ? 'conf-medium' : 'conf-low';
  el('confidenceFill').style.width = fmt.pct(conf);
  el('confidenceFill').className = `confidence-fill ${confClass}`;
  el('confidencePct').textContent = fmt.pct(conf);

  // Metrics
  el('invMetrics').innerHTML = [
    metricCard('Latency',      fmt.ms(m.latency_ms)),
    metricCard('Tokens in/out', `${fmt.tokens(m.tokens_in)} / ${fmt.tokens(m.tokens_out)}`),
    metricCard('Cost',         fmt.cost(m.cost_usd)),
    metricCard('RAG precision', fmt.score(out.rag_precision ?? 0)),
  ].join('');

  el('invLoading').classList.add('hidden');
  el('invContent').classList.remove('hidden');
}

function buildToolRows(tri, inv) {
  const txnId   = tri.transaction_id || '…';
  const merchant = tri.merchant || '…';
  const signals = inv.fraud_signals || [];
  const rag     = inv.rag_source_clause || '';

  const hasVelocity = signals.some(s => /transaction|velocity|hour/i.test(s));
  const ragNotFound = rag.startsWith('NOT FOUND');

  return [
    {
      name: 'query_transactions',
      args: `("${txnId}")`,
      badge: 'fetched',
      badgeClass: 'badge-ok',
    },
    {
      name: 'get_velocity_history',
      args: '(card, 24h)',
      badge: hasVelocity ? 'velocity spike' : 'checked',
      badgeClass: hasVelocity ? 'badge-warn' : 'badge-ok',
    },
    {
      name: 'check_merchant_risk',
      args: `("${merchant.slice(0, 20)}")`,
      badge: 'risk assessed',
      badgeClass: 'badge-ok',
    },
    {
      name: 'search_compliance_docs',
      args: '(dispute condition)',
      badge: ragNotFound ? 'not found' : (rag.split(':')[0] || 'retrieved').slice(0, 24),
      badgeClass: ragNotFound ? 'badge-err' : 'badge-ok',
    },
  ];
}

// ── Panel 4: Auditor ──────────────────────────────────────────────────────────

function populateAuditor(event) {
  STATE.auditor = event;
  const out        = event.output;
  const m          = event.metrics;
  const violations = out.violations || [];
  const isViolation = out.verdict === 'policy_violation';

  // Violation banner
  if (isViolation && violations.length) {
    const banner = el('violationBanner');
    banner.innerHTML =
      `<strong>⚠ Pipeline halted — policy violation detected</strong>
       <ul>${violations.map(v => `<li>${escHtml(v.replace(/_/g, ' '))}</li>`).join('')}</ul>`;
    banner.classList.remove('hidden');
  }

  // Rules table
  el('rulesBody').innerHTML = (out.rules_checked || []).map(r => {
    const pass = r.passed;
    return `<tr class="${pass ? '' : 'row-fail'}">
      <td class="rule-name">${escHtml(r.rule)}</td>
      <td><span class="rule-status ${pass ? 'pass' : 'fail'}">${pass ? '✓ Pass' : '✗ Fail'}</span></td>
      <td class="rule-detail">${escHtml(r.detail)}</td>
    </tr>`;
  }).join('');

  // Verdict
  el('auditorVerdict').className = `verdict-badge large ${verdictClass(out.verdict)}`;
  el('auditorVerdict').textContent = verdictLabel(out.verdict);

  // Audit log JSON (collapsed by default)
  el('auditPre').textContent = JSON.stringify(out, null, 2);

  // Metrics
  const violationCount = violations.length;
  el('auditorMetrics').innerHTML = [
    metricCard('Latency',      fmt.ms(m.latency_ms)),
    metricCard('Tokens in/out', `${fmt.tokens(m.tokens_in)} / ${fmt.tokens(m.tokens_out)}`),
    metricCard('Cost',         fmt.cost(m.cost_usd)),
    metricCard('Violations',   violationCount, violationCount ? 'policy halted' : 'all clear'),
  ].join('');

  el('auditorLoading').classList.add('hidden');
  el('auditorContent').classList.remove('hidden');

  // Show "Review / Reset" buttons for policy violations
  if (isViolation) {
    el('violationActions').classList.remove('hidden');
  }

  // Populate panel 5 data now so it's ready if we advance
  populateResults(event);
}

// ── Panel 5: Results ──────────────────────────────────────────────────────────

function populateResults(auditorEvent) {
  const verdict = auditorEvent.output.verdict;
  const totals  = auditorEvent.pipeline_totals || {};

  // Final verdict badge
  el('finalVerdict').className = `verdict-badge xlarge ${verdictClass(verdict)}`;
  el('finalVerdict').textContent = verdictLabel(verdict);

  // Totals grid
  el('totalsGrid').innerHTML = [
    metricCard('Total latency',   fmt.ms(totals.total_latency_ms  ?? 0)),
    metricCard('Total tokens',    fmt.tokens(totals.total_tokens  ?? 0)),
    metricCard('Total cost',      fmt.cost(totals.total_cost_usd  ?? 0)),
    metricCard('Violations',      (auditorEvent.output.violations || []).length,
      (auditorEvent.output.violations || []).length ? 'policy halt' : 'all clear'),
  ].join('');

  // Per-agent breakdown
  const rows = [
    { agent: 'Triage',       data: STATE.triage       },
    { agent: 'Investigator', data: STATE.investigator },
    { agent: 'Auditor',      data: STATE.auditor      },
  ];
  el('breakdownBody').innerHTML = rows.map(r => {
    if (!r.data) return '';
    const m = r.data.metrics;
    return `<tr>
      <td class="agent-cell">${r.agent}</td>
      <td class="model-cell">${m.model}</td>
      <td>${fmt.ms(m.latency_ms)}</td>
      <td>${fmt.tokens(m.tokens_in)}</td>
      <td>${fmt.tokens(m.tokens_out)}</td>
      <td>${fmt.cost(m.cost_usd)}</td>
    </tr>`;
  }).join('');

  // Eval metrics — RAG precision from this run
  const ragPrecision = STATE.investigator?.output?.rag_precision;
  const evalCards = [];

  if (ragPrecision !== undefined) {
    evalCards.push(`<div class="eval-card">
      <div class="metric-label">RAG precision (this run)</div>
      <div class="metric-value">${fmt.score(ragPrecision)}</div>
      <div class="metric-sub">LLM judge over retrieved chunks</div>
    </div>`);
  }

  // Try to show F1 score if eval has been run
  if (window._f1Score !== undefined) {
    evalCards.push(`<div class="eval-card">
      <div class="metric-label">F1 score (batch eval)</div>
      <div class="metric-value">${fmt.f1(window._f1Score)}</div>
      <div class="metric-sub">IEEE-CIS eval set · ${window._f1Samples ?? '?'} samples</div>
    </div>`);
  }

  el('evalRow').innerHTML = evalCards.join('');
}

// ── SSE pipeline runner ───────────────────────────────────────────────────────

async function runPipeline(formData) {
  let response;
  try {
    response = await fetch(`${API}/resolve`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(formData),
      signal:  AbortSignal.timeout(120_000),
    });
  } catch (err) {
    showError('Pipeline error — check backend is running on :8000');
    resetForm();
    return;
  }

  if (!response.ok) {
    showError(`Backend error: ${response.status} ${response.statusText}`);
    resetForm();
    return;
  }

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const raw = line.slice(5).trim();
          if (!raw || raw === '{}') continue;
          try {
            const event = JSON.parse(raw);
            handleAgentEvent(event);
          } catch (_) { /* malformed — skip */ }
        }
        if (line.startsWith('event: done')) {
          // Final event — advance to results if not a violation
          const verdict = STATE.auditor?.output?.verdict;
          if (verdict && verdict !== 'policy_violation') {
            setTimeout(() => {
              advanceStep(4);
              showPanel(5);
            }, 600);
          }
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      showError('Stream interrupted — ' + err.message);
    }
  }
}

function handleAgentEvent(event) {
  if (event.error) {
    showError('Agent error: ' + event.error);
    return;
  }

  switch (event.agent) {
    case 'triage':
      populateTriage(event);
      advanceStep(2);
      showPanel(2);
      setStepState(3, 'active');
      showPanel(3);
      break;

    case 'investigator':
      populateInvestigator(event);
      advanceStep(3);
      showPanel(3);
      setStepState(4, 'active');
      showPanel(4);
      break;

    case 'auditor':
      populateAuditor(event);
      advanceStep(4);
      showPanel(4);
      break;
  }
}

// ── Form handling ─────────────────────────────────────────────────────────────

function fillForm(sample) {
  el('f_txn_id').value     = sample.transaction_id;
  el('f_amount').value     = sample.amount;
  el('f_merchant').value   = sample.merchant;
  el('f_reason').value     = sample.reason_code;
  el('f_statement').value  = sample.customer_statement;
}

function readForm() {
  return {
    transaction_id:     el('f_txn_id').value.trim(),
    amount:             parseFloat(el('f_amount').value),
    merchant:           el('f_merchant').value.trim(),
    reason_code:        el('f_reason').value,
    customer_statement: el('f_statement').value.trim(),
  };
}

function disableForm() {
  el('submitBtn').disabled = true;
  el('disputeForm').querySelectorAll('input,select,textarea').forEach(e => e.disabled = true);
}

function resetForm() {
  el('submitBtn').disabled = false;
  el('disputeForm').querySelectorAll('input,select,textarea').forEach(e => e.disabled = false);
}

function resetPipeline() {
  // Clear state
  STATE.triage = STATE.investigator = STATE.auditor = null;

  // Reset step bar
  [1,2,3,4,5].forEach(n => setStepState(n, n === 1 ? 'active' : 'pending'));

  // Clear panels
  ['triageEntities','maskedStatement','triageMetrics',
   'toolLog','ragBlock','signalList','recBadge','invMetrics',
   'rulesBody','auditorMetrics','auditPre',
   'totalsGrid','breakdownBody','evalRow'].forEach(id => {
    const node = el(id);
    if (node) node.innerHTML = '';
  });

  // Reset loading states
  ['triageLoading','invLoading','auditorLoading'].forEach(id => {
    el(id).classList.remove('hidden');
  });
  ['triageContent','invContent','auditorContent'].forEach(id => {
    el(id).classList.add('hidden');
  });
  el('violationBanner').classList.add('hidden');
  el('auditPre').classList.add('hidden');
  el('violationActions').classList.add('hidden');
  el('errorBanner').classList.add('hidden');
  el('auditorVerdict').className = 'verdict-badge large';
  el('auditorVerdict').textContent = '';
  el('confidenceFill').style.width = '0%';

  resetForm();
  showPanel(1);
}

// ── Error display ─────────────────────────────────────────────────────────────

function showError(msg) {
  const banner = el('errorBanner');
  banner.textContent = msg;
  banner.classList.remove('hidden');
  setTimeout(() => banner.classList.add('hidden'), 6000);
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Eval results (F1 score) ───────────────────────────────────────────────────

async function loadEvalResults() {
  try {
    const res = await fetch(`${API}/eval-results`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.f1 !== undefined) {
      window._f1Score   = data.f1;
      window._f1Samples = data.n_samples;
    }
  } catch (_) { /* backend not running or no results yet */ }
}

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Load sample F1 results if available
  loadEvalResults();

  // Sample buttons
  el('loadApprove').addEventListener('click', () => fillForm(SAMPLES.approve));
  el('loadViolation').addEventListener('click', () => fillForm(SAMPLES.violation));

  // Reset buttons
  el('resetBtn').addEventListener('click', resetPipeline);
  el('resetFromViolation').addEventListener('click', resetPipeline);
  el('reviewBtn').addEventListener('click', () =>
    alert('Manual review queue: feature not implemented in this demo.')
  );

  // Audit log toggle
  el('auditToggle').addEventListener('click', () => {
    const pre = el('auditPre');
    const hidden = pre.classList.toggle('hidden');
    el('auditToggle').textContent = (hidden ? '▶' : '▼') + ' View audit log entry';
  });

  // Form submit
  el('disputeForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = readForm();

    if (!data.transaction_id || !data.merchant || !data.customer_statement) {
      showError('Please fill in all fields.');
      return;
    }
    if (isNaN(data.amount) || data.amount <= 0) {
      showError('Amount must be a positive number.');
      return;
    }

    disableForm();
    el('errorBanner').classList.add('hidden');

    // Advance to triage panel immediately to show "running" state
    setStepState(1, 'complete');
    setStepState(2, 'active');
    showPanel(2);

    await runPipeline(data);
  });

  // Pre-fill with clean dispute sample on load
  fillForm(SAMPLES.approve);
});
