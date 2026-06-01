'use strict';

const API = 'http://localhost:8000';

// ── Samples ───────────────────────────────────────────────────────────────────
const SAMPLES = {
  approve: {
    transaction_id: 'TXN-2024-0318-8821',
    amount: 1240.00,
    merchant: 'ElectroMart Inc.',
    reason_code: '10.4',
    customer_statement:
      'I did not authorize this $1,240 charge. My card was in my wallet the entire time and I have never shopped at this merchant before.',
  },
  violation: {
    transaction_id: 'TXN-2024-0316-9934',
    amount: 7800.00,
    merchant: 'LuxuryGoods Direct',
    reason_code: '10.4',
    customer_statement:
      'I did not authorize a $7,800 purchase from LuxuryGoods Direct shipped to a UAE address. This is clearly fraudulent activity on my account.',
  },
};

// ── Formatting ────────────────────────────────────────────────────────────────
const fmt = {
  ms:     v => `${Math.round(v)}ms`,
  tokens: v => Number(v).toLocaleString(),
  cost:   v => v < 0.001 ? '< $0.001' : `$${v.toFixed(4)}`,
  pct:    v => `${Math.round(v * 100)}%`,
  score:  v => parseFloat(v).toFixed(2),
  f1:     v => parseFloat(v).toFixed(2),
};

// ── Tool definitions for the investigator animation ───────────────────────────
const TOOL_DEFS = [
  { id: 'sql',        icon: 'DB',  iconClass: 'db',    name: 'query_transactions',    label: 'Transaction DB lookup' },
  { id: 'velocity',   icon: 'V',   iconClass: 'speed', name: 'get_velocity_history',   label: 'Card velocity check (24h)' },
  { id: 'risk',       icon: 'MR',  iconClass: 'risk',  name: 'check_merchant_risk',    label: 'Merchant risk profile' },
  { id: 'classifier', icon: 'DT',  iconClass: 'doc',   name: 'run_fraud_classifier',   label: 'Decision Tree fraud classifier' },
  { id: 'rag',        icon: 'RAG', iconClass: 'doc',   name: 'search_compliance_docs', label: 'Visa rulebook retrieval' },
];

// ── State ─────────────────────────────────────────────────────────────────────
const STATE = { triage: null, investigator: null, auditor: null };
let invAnimDone = Promise.resolve(); // resolved when investigator animation finishes

// ── Helpers ───────────────────────────────────────────────────────────────────
const el   = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function metricTile(label, value, sub) {
  return `<div class="metric-tile">
    <div class="metric-tile-label">${label}</div>
    <div class="metric-tile-value">${value}</div>
    ${sub ? `<div class="metric-tile-sub">${sub}</div>` : ''}
  </div>`;
}

function verdictClass(v) {
  return 'v-' + (v || '').toLowerCase().replace(/ /g, '_');
}
function verdictLabel(v) {
  return { approve: '✓ Approve', reject: '✗ Reject', escalate: 'Escalate',
           policy_violation: '⚠ Policy Violation' }[v] || v;
}

// ── Field validation ──────────────────────────────────────────────────────────
function setFieldError(inputId, errId, msg) {
  const input = el(inputId);
  const errEl = el(errId);
  if (msg) {
    input.classList.add('invalid');
    if (errEl) errEl.textContent = msg;
  } else {
    input.classList.remove('invalid');
    if (errEl) errEl.textContent = '';
  }
  return !msg;
}

function validateForm() {
  const data = readForm();
  let ok = true;

  ok = setFieldError('f_txn', 'err_txn',
    !data.transaction_id ? 'Transaction ID is required' : null) && ok;

  ok = setFieldError('f_amt', 'err_amt',
    isNaN(data.amount) || data.amount <= 0 ? 'Enter a positive amount' :
    data.amount > 1_000_000 ? 'Amount exceeds $1,000,000 limit' : null) && ok;

  ok = setFieldError('f_merch', 'err_merch',
    !data.merchant ? 'Merchant name is required' : null) && ok;

  ok = setFieldError('f_stmt', 'err_stmt',
    !data.customer_statement ? 'Statement is required' :
    data.customer_statement.length < 20 ? 'Please provide at least 20 characters' : null) && ok;

  return ok;
}

// ── Step bar ──────────────────────────────────────────────────────────────────
function setStep(n, state) {
  document.querySelector(`.step-item[data-step="${n}"]`).dataset.state = state;
}
function completeStep(n) {
  setStep(n, 'complete');
  if (n < 5) setStep(n + 1, 'active');
}

// ── Panel display ─────────────────────────────────────────────────────────────
function showPanel(n) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('panel-active'));
  el(`panel-${n}`).classList.add('panel-active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Panel 2: Triage ───────────────────────────────────────────────────────────
function populateTriage(event) {
  STATE.triage = event;
  const { output: o, metrics: m } = event;

  // Entity pills
  const pills = [
    { k: 'txn',      v: o.transaction_id },
    { k: 'amount',   v: `$${Number(o.amount).toLocaleString()}` },
    { k: 'merchant', v: o.merchant },
    { k: 'reason',   v: o.reason_code },
    { k: 'category', v: o.entities.dispute_category },
    { k: 'urgency',  v: o.entities.urgency.replace(/_/g, ' ') },
    o.entities.transaction_date ? { k: 'date', v: o.entities.transaction_date } : null,
  ].filter(Boolean);

  el('triagePills').innerHTML = pills.map(p =>
    `<span class="pill urgency-${o.entities.urgency}">
       <span class="pill-key">${p.k}</span>
       <span class="pill-val">${esc(p.v)}</span>
     </span>`
  ).join('');

  // Masked statement — highlight PII tokens
  el('maskedStmt').innerHTML = o.masked_statement.replace(
    /\[(NAME|LOCATION|CONTACT|PII)\]/g,
    '<span class="pii-mask">[$1]</span>'
  );

  el('triageMetrics').innerHTML = [
    metricTile('Latency',       fmt.ms(m.latency_ms)),
    metricTile('Tokens in',     fmt.tokens(m.tokens_in)),
    metricTile('Tokens out',    fmt.tokens(m.tokens_out)),
    metricTile('Cost',          fmt.cost(m.cost_usd)),
  ].join('');

  el('triageStatus').innerHTML = '<span style="color:var(--green)">✓ Complete</span>';
  el('triageStatus').className = 'agent-status done';
  el('triageBody').classList.remove('hidden');
}

// ── Classifier section renderer ───────────────────────────────────────────────
function renderClassifier(clf) {
  if (!clf) return;

  const prob = clf.fraud_probability ?? 0;
  const pred = clf.prediction ?? 'unknown';

  // Animated probability bar (defer to next frame so transition fires)
  const bar = el('probBar');
  bar.style.width = '0%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { bar.style.width = `${Math.round(prob * 100)}%`; });
  });

  el('probMeta').textContent =
    `${(prob * 100).toFixed(1)}% · confidence: ${clf.confidence ?? '—'} · ${clf.model_version ?? 'dt_v1'}`;

  const predEl = el('classifierPred');
  predEl.textContent = pred === 'fraud' ? 'FRAUD' : 'LEGITIMATE';
  predEl.className   = `classifier-pred ${pred === 'fraud' ? 'fraud' : 'legitimate'}`;

  // Decision path
  const pathEl = el('decisionPath');
  pathEl.innerHTML = (clf.decision_path || []).map(step => {
    const isLeaf = step === 'FRAUD' || step === 'LEGITIMATE';
    if (isLeaf) {
      const cls = step === 'FRAUD' ? 'path-leaf-fraud' : 'path-leaf-legit';
      return `<div class="${cls}">→ ${step}</div>`;
    }
    return `<div>${esc(step)}</div>`;
  }).join('');

  // Feature importance bars — normalise so max bar = 100%
  const features = clf.top_features || [];
  const maxImp   = Math.max(...features.map(f => f.importance), 0.01);
  el('featureList').innerHTML = features.map(f => {
    const barPct = Math.round((f.importance / maxImp) * 100);
    return `<div class="feat-bar-row">
      <div class="feat-name" title="${esc(f.feature)}">${esc(f.feature)}</div>
      <div class="feat-bar-track"><div class="feat-bar-fill" style="width:${barPct}%"></div></div>
      <div class="feat-value">${parseFloat(f.importance).toFixed(2)}</div>
    </div>`;
  }).join('');
}

// ── Panel 3: Investigator — animated tool log ─────────────────────────────────
async function populateInvestigator(event) {
  STATE.investigator = event;
  const { output: o, metrics: m } = event;
  const tri = STATE.triage?.output || {};

  // Build result strings for each tool from the investigator output
  const hasVelocity  = (o.fraud_signals || []).some(s => /transaction|velocity|hour/i.test(s));
  const ragNotFound  = (o.rag_source_clause || '').startsWith('NOT FOUND');
  const ragLabel     = ragNotFound ? 'Not found' : (o.rag_source_clause || '').split(':')[0].slice(0, 28);
  const clf          = o.classifier_output || null;
  const clfProb      = clf ? clf.fraud_probability : null;
  const clfPred      = clf ? clf.prediction : null;

  const toolResults = [
    {
      status:  'ok',
      result:  `Transaction ${tri.transaction_id || '…'} · $${Number(tri.amount || 0).toLocaleString()}`,
    },
    {
      status:  hasVelocity ? 'warn' : 'ok',
      result:  hasVelocity
        ? (o.fraud_signals || []).find(s => /transaction|velocity/i.test(s)) || 'Elevated velocity'
        : 'Velocity within normal range',
    },
    {
      status:  'ok',
      result:  `${esc(tri.merchant || '…')}: risk profiled`,
    },
    {
      status:  clfPred === 'fraud' ? 'warn' : 'ok',
      result:  clf
        ? `p(fraud)=${clfProb?.toFixed(2)} · ${clfPred}`
        : 'Classifier not available',
    },
    {
      status:  ragNotFound ? 'error' : 'ok',
      result:  ragNotFound ? 'Rule not found — will escalate' : ragLabel,
    },
  ];

  const timeline = el('toolTimeline');
  timeline.innerHTML = '';

  // Animate rows one by one
  let resolveAnim;
  invAnimDone = new Promise(r => { resolveAnim = r; });

  for (let i = 0; i < TOOL_DEFS.length; i++) {
    const def = TOOL_DEFS[i];
    const res = toolResults[i];

    // Step A: row appears in "running" state
    const row = document.createElement('div');
    row.className = 'tool-row';
    row.style.animationDelay = '0s';
    row.innerHTML = `
      <div class="tool-row-left">
        <div class="tool-icon ${def.iconClass}">${def.icon}</div>
        <div class="tool-text">
          <div class="tool-name">${def.name}()</div>
          <div class="tool-args">${esc(def.label)}</div>
          <div class="tool-result" id="tool-result-${i}"></div>
        </div>
      </div>
      <div class="tool-row-right">
        <span class="status-chip running" id="tool-chip-${i}">
          <span class="tool-spinner"></span> Running
        </span>
      </div>`;
    timeline.appendChild(row);

    await sleep(700);   // simulate execution time

    // Step B: update to done state with result
    el(`tool-chip-${i}`).outerHTML =
      `<span class="status-chip ${res.status}" id="tool-chip-${i}">
         ${res.status === 'ok' ? '✓ Done' : res.status === 'warn' ? '⚠ Warning' : '✗ Not found'}
       </span>`;
    const resultEl = el(`tool-result-${i}`);
    if (resultEl) resultEl.textContent = res.result;

    await sleep(250);
  }

  resolveAnim();

  // After animation: render classifier section
  renderClassifier(clf);

  // After animation: show RAG, signals, verdict, metrics
  // RAG block
  const ragText = o.rag_source_clause || 'NOT FOUND';
  el('ragCard').innerHTML =
    `${esc(ragText)}<div class="rag-meta"><span>section: ${esc(o.rag_source_clause?.split(':')[0] || '—')}</span><span>precision: ${fmt.score(o.rag_precision ?? 0)}</span></div>`;

  // Signals
  const signals = o.fraud_signals || [];
  el('signalList').innerHTML = signals.length
    ? signals.map(s => `<li>${esc(s)}</li>`).join('')
    : '<li>No specific signals detected</li>';

  // Recommendation badge
  el('recBadge').className = `verdict-badge ${verdictClass(o.recommendation)}`;
  el('recBadge').textContent = verdictLabel(o.recommendation);

  // Confidence bar
  const conf = o.confidence ?? 0;
  const confClass = conf >= 0.85 ? 'conf-high' : conf >= 0.70 ? 'conf-medium' : 'conf-low';
  el('confFill').style.width = fmt.pct(conf);
  el('confFill').className = `conf-fill ${confClass}`;
  el('confPct').textContent = fmt.pct(conf);

  // Metrics
  el('invMetrics').innerHTML = [
    metricTile('Latency',       fmt.ms(m.latency_ms)),
    metricTile('Tokens in',     fmt.tokens(m.tokens_in)),
    metricTile('Tokens out',    fmt.tokens(m.tokens_out)),
    metricTile('RAG precision', fmt.score(o.rag_precision ?? 0)),
  ].join('');

  el('invStatus').innerHTML = '<span style="color:var(--green)">✓ Complete</span>';
  el('invStatus').className = 'agent-status done';
  el('invBodyExtra').classList.remove('hidden');
}

// ── Panel 4: Auditor ──────────────────────────────────────────────────────────
async function populateAuditor(event) {
  // Wait for investigator animation to finish before rendering auditor
  await invAnimDone;

  STATE.auditor = event;
  const { output: o, metrics: m } = event;
  const violations = o.violations || [];
  const isViol = o.verdict === 'policy_violation';

  // Reasoning box
  const reasoning = o.reasoning || '';
  if (reasoning) {
    el('reasoningBox').textContent = reasoning;
    el('reasoningBox').classList.remove('hidden');
  }

  // Violation banner
  if (isViol && violations.length) {
    const b = el('violationBanner');
    b.innerHTML = `<div class="alert-banner-title">⚠ Pipeline halted — policy violation detected</div>
      <ul>${violations.map(v => `<li>${esc(v.replace(/_/g, ' '))}</li>`).join('')}</ul>`;
    b.classList.remove('hidden');
  }

  // Rules list
  el('rulesList').innerHTML = (o.rules_checked || []).map(r =>
    `<div class="rule-row ${r.passed ? '' : 'fail'}">
       <div class="rule-body">
         <div class="rule-name-text">${esc(r.rule)}</div>
         <div class="rule-detail-text">${esc(r.detail)}</div>
       </div>
       <span class="rule-status-tag ${r.passed ? 'pass' : 'fail'}">${r.passed ? 'Pass' : 'Fail'}</span>
     </div>`
  ).join('');

  // Verdict
  el('auditVerdict').className = `verdict-badge large ${verdictClass(o.verdict)}`;
  el('auditVerdict').textContent = verdictLabel(o.verdict);

  // Audit log
  el('auditPre').textContent = JSON.stringify(o, null, 2);

  // Metrics
  el('auditorMetrics').innerHTML = [
    metricTile('Latency',     fmt.ms(m.latency_ms)),
    metricTile('Tokens in',   fmt.tokens(m.tokens_in)),
    metricTile('Tokens out',  fmt.tokens(m.tokens_out)),
    metricTile('Violations',  violations.length, violations.length ? 'policy halt' : 'all clear'),
  ].join('');

  el('auditorStatus').innerHTML = '<span style="color:var(--green)">✓ Complete</span>';
  el('auditorStatus').className = 'agent-status done';
  el('auditorBody').classList.remove('hidden');

  if (isViol) el('violationActions').classList.remove('hidden');

  // Pre-populate results panel
  populateResults(event);

  // Auto-advance to results for non-violation verdicts
  if (!isViol) {
    await sleep(1000);
    completeStep(4);
    showPanel(5);
  }
}

// ── Panel 5: Results ──────────────────────────────────────────────────────────
function populateResults(audEvent) {
  setStep(5, 'complete');   // makes step 5 clickable in the stepper
  const verdict = audEvent.output.verdict;
  const totals  = audEvent.pipeline_totals || {};

  el('finalVerdict').className = `verdict-badge xlarge ${verdictClass(verdict)}`;
  el('finalVerdict').textContent = verdictLabel(verdict);

  el('summaryGrid').innerHTML = [
    metricTile('Total latency', fmt.ms(totals.total_latency_ms ?? 0)),
    metricTile('Total tokens',  fmt.tokens(totals.total_tokens ?? 0)),
    metricTile('Total cost',    fmt.cost(totals.total_cost_usd ?? 0)),
    metricTile('Violations',    (audEvent.output.violations || []).length,
      (audEvent.output.violations || []).length ? 'policy halt' : 'all clear'),
  ].join('');

  const rows = [
    { name: 'Triage',       data: STATE.triage },
    { name: 'Investigator', data: STATE.investigator },
    { name: 'Auditor',      data: STATE.auditor },
  ];
  el('breakdownBody').innerHTML = rows.map(r => {
    if (!r.data) return '';
    const m = r.data.metrics;
    return `<tr>
      <td class="agent-name-cell">${r.name}</td>
      <td>${fmt.ms(m.latency_ms)}</td>
      <td>${fmt.tokens(m.tokens_in)}</td>
      <td>${fmt.tokens(m.tokens_out)}</td>
      <td>${fmt.cost(m.cost_usd)}</td>
    </tr>`;
  }).join('');

  const reasoning = STATE.auditor?.output?.reasoning || '';
  el('reasoningSummary').innerHTML = reasoning
    ? `<div class="section-title" style="margin-top:28px">Decision rationale</div>
       <div class="reasoning-box">${esc(reasoning)}</div>`
    : '';

  const evalCards = [];
  const ragP = STATE.investigator?.output?.rag_precision;
  if (ragP !== undefined) {
    evalCards.push(`<div class="eval-tile">
      <div class="metric-tile-label">RAG precision (this run)</div>
      <div class="metric-tile-value">${fmt.score(ragP)}</div>
      <div class="metric-tile-sub">LLM judge · retrieved chunks</div>
    </div>`);
  }
  if (window._f1Score !== undefined) {
    evalCards.push(`<div class="eval-tile">
      <div class="metric-tile-label">F1 score (batch eval)</div>
      <div class="metric-tile-value">${fmt.f1(window._f1Score)}</div>
      <div class="metric-tile-sub">IEEE-CIS · ${window._f1Samples ?? '?'} samples</div>
    </div>`);
  }
  el('evalRow').innerHTML = evalCards.join('');
}

// ── SSE pipeline runner ───────────────────────────────────────────────────────
async function runPipeline(formData) {
  let response;
  try {
    response = await fetch(`${API}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
      signal: AbortSignal.timeout(120_000),
    });
  } catch {
    showErr('Cannot connect to backend — is uvicorn running on :8000?');
    resetForm(); return;
  }
  if (!response.ok) {
    if (response.status === 422) {
      try {
        const body = await response.json();
        const details = body.detail;
        const msg = Array.isArray(details)
          ? details.map(d => d.msg.replace('Value error, ', '')).join('; ')
          : JSON.stringify(details);
        showErr(`Validation: ${msg}`);
      } catch { showErr(`Validation error — check your inputs.`); }
    } else {
      showErr(`Backend error: ${response.status}`);
    }
    resetForm(); return;
  }

  const reader = response.body.getReader();
  const dec    = new TextDecoder();
  let   buf    = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const raw = line.slice(5).trim();
          if (!raw || raw === '{}') continue;
          try { dispatch(JSON.parse(raw)); } catch { /* skip malformed */ }
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') showErr('Stream interrupted: ' + err.message);
  }
}

async function dispatch(event) {
  if (event.error) { showErr('Agent error: ' + event.error); return; }

  switch (event.agent) {
    case 'triage':
      populateTriage(event);
      completeStep(2);
      showPanel(2);
      setStep(3, 'active');
      // Show panel 3 immediately so the tool animation is visible
      showPanel(3);
      break;

    case 'investigator':
      // Start animation (async); panel 3 is already visible
      populateInvestigator(event);
      completeStep(3);
      setStep(4, 'active');
      // Show panel 4 after animation completes
      invAnimDone.then(() => showPanel(4));
      break;

    case 'auditor':
      // populateAuditor awaits invAnimDone internally
      populateAuditor(event);
      completeStep(4);
      break;
  }
}

// ── Form ──────────────────────────────────────────────────────────────────────
function fillForm(s) {
  el('f_txn').value   = s.transaction_id;
  el('f_amt').value   = s.amount;
  el('f_merch').value = s.merchant;
  el('f_reason').value= s.reason_code;
  el('f_stmt').value  = s.customer_statement;
}

function readForm() {
  return {
    transaction_id:     el('f_txn').value.trim(),
    amount:             parseFloat(el('f_amt').value),
    merchant:           el('f_merch').value.trim(),
    reason_code:        el('f_reason').value,
    customer_statement: el('f_stmt').value.trim(),
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

function resetAll() {
  STATE.triage = STATE.investigator = STATE.auditor = null;
  invAnimDone = Promise.resolve();

  [1,2,3,4,5].forEach(n => setStep(n, n === 1 ? 'active' : 'pending'));

  // Clear dynamic content
  ['triagePills','maskedStmt','triageMetrics',
   'toolTimeline','ragCard','signalList','recBadge','invMetrics',
   'probMeta','decisionPath','featureList',
   'rulesList','auditorMetrics','auditPre',
   'summaryGrid','breakdownBody','evalRow'].forEach(id => {
    const node = el(id); if (node) node.innerHTML = '';
  });
  el('probBar').style.width = '0%';
  el('classifierPred').className = 'classifier-pred';
  el('classifierPred').textContent = '';

  el('triageStatus').innerHTML  = '<span class="spinner-sm"></span> Running…';
  el('triageStatus').className  = 'agent-status';
  el('invStatus').innerHTML     = '<span class="spinner-sm"></span> Running…';
  el('invStatus').className     = 'agent-status';
  el('auditorStatus').innerHTML = '<span class="spinner-sm"></span> Running…';
  el('auditorStatus').className = 'agent-status';

  el('triageBody').classList.add('hidden');
  el('invBodyExtra').classList.add('hidden');
  el('auditorBody').classList.add('hidden');
  el('violationBanner').classList.add('hidden');
  el('violationActions').classList.add('hidden');
  el('auditPre').classList.add('hidden');
  el('liveIndicator').classList.add('hidden');
  el('errBanner').classList.add('hidden');

  // Reset reasoning boxes
  el('reasoningBox').classList.add('hidden');
  el('reasoningBox').textContent = '';
  el('reasoningSummary').innerHTML = '';

  // Reset flag button
  const reviewBtn = el('reviewBtn');
  reviewBtn.textContent = 'Flag for manual review';
  reviewBtn.disabled = false;
  reviewBtn.classList.remove('flag-success');

  // Clear field validation errors
  [['f_txn','err_txn'],['f_amt','err_amt'],['f_merch','err_merch'],['f_stmt','err_stmt']].forEach(
    ([fid, eid]) => setFieldError(fid, eid, null)
  );

  resetForm();
  showPanel(1);
}

// ── Misc ──────────────────────────────────────────────────────────────────────
function showErr(msg) {
  const b = el('errBanner');
  b.textContent = msg;
  b.classList.remove('hidden');
  setTimeout(() => b.classList.add('hidden'), 7000);
}

async function loadEvalResults() {
  try {
    const r = await fetch(`${API}/eval-results`);
    if (!r.ok) return;
    const d = await r.json();
    if (d.f1 !== undefined) { window._f1Score = d.f1; window._f1Samples = d.n_samples; }
  } catch { /* backend not running yet */ }
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadEvalResults();
  fillForm(SAMPLES.approve);

  // Stepper: clicking any non-pending step navigates to that panel
  el('stepper').addEventListener('click', e => {
    const item = e.target.closest('.step-item');
    if (!item || item.dataset.state === 'pending') return;
    showPanel(parseInt(item.dataset.step));
  });

  // Clear per-field validation error on user input
  [['f_txn','err_txn'],['f_amt','err_amt'],['f_merch','err_merch'],['f_stmt','err_stmt']].forEach(
    ([fid, eid]) => el(fid).addEventListener('input', () => setFieldError(fid, eid, null))
  );

  el('loadApprove').addEventListener('click',    () => fillForm(SAMPLES.approve));
  el('loadViolation').addEventListener('click',  () => fillForm(SAMPLES.violation));
  el('resetBtn').addEventListener('click',       resetAll);
  el('resetFromViolation').addEventListener('click', resetAll);
  el('reviewBtn').addEventListener('click', async () => {
    const btn = el('reviewBtn');
    if (btn.classList.contains('flag-success')) return;
    const txnId = STATE.triage?.output?.transaction_id || 'unknown';
    btn.textContent = 'Flagging…';
    btn.disabled = true;
    try {
      await fetch(`${API}/flag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: txnId }),
      });
      btn.textContent = '✓ Flagged for review';
      btn.classList.add('flag-success');
    } catch {
      btn.textContent = 'Flag for manual review';
      btn.disabled = false;
    }
  });

  el('auditToggle').addEventListener('click', () => {
    const pre = el('auditPre');
    const hidden = pre.classList.toggle('hidden');
    el('auditToggle').textContent = (hidden ? 'Show' : 'Hide') + ' audit log entry ' + (hidden ? '↓' : '↑');
  });

  el('disputeForm').addEventListener('submit', async e => {
    e.preventDefault();
    if (!validateForm()) return;
    const data = readForm();

    disableForm();
    el('liveIndicator').classList.remove('hidden');
    el('errBanner').classList.add('hidden');

    setStep(1, 'complete');
    setStep(2, 'active');
    showPanel(2);

    await runPipeline(data);

    el('liveIndicator').classList.add('hidden');
  });
});
