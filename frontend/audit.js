'use strict';

const API      = 'http://localhost:8000';
const PAGE_SZ  = 20;

// ── State ─────────────────────────────────────────────────────────────────────
let _page       = 0;
let _totalRuns  = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────
const el    = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

function esc(str) {
  return String(str ?? '')
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

const fmt = {
  ms:     v => v != null ? `${Math.round(v)}ms` : '—',
  cost:   v => v == null ? '—' : (v < 0.001 ? '< $0.001' : `$${parseFloat(v).toFixed(4)}`),
  pct:    v => v != null ? `${Math.round(v * 100)}%` : '—',
  score:  v => v != null ? parseFloat(v).toFixed(2) : '—',
  tokens: v => v != null ? Number(v).toLocaleString() : '—',
  prob:   v => v != null ? `${(v * 100).toFixed(1)}%` : '—',
  ts:     v => v ? new Date(v).toLocaleString() : '—',
};

function verdictClass(v) { return 'v-' + (v || '').toLowerCase().replace(/ /g, '_'); }
function verdictLabel(v) {
  return { approve: '✓ Approve', reject: '✗ Reject', escalate: 'Escalate',
           policy_violation: '⚠ Violation' }[v] || v;
}

function verdictBadge(v) {
  return `<span class="verdict-badge ${verdictClass(v)}" style="font-size:11.5px;padding:3px 10px">${verdictLabel(v)}</span>`;
}

function showErr(msg) {
  const b = el('errBanner');
  b.textContent = msg;
  b.classList.remove('hidden');
  setTimeout(() => b.classList.add('hidden'), 8000);
}

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res   = await fetch(`${API}/audit/stats`);
    const stats = await res.json();
    el('total-runs').textContent    = stats.total_runs   ?? 0;
    el('stat-approved').textContent = stats.approved     ?? 0;
    el('stat-rejected').textContent = stats.rejected     ?? 0;
    el('stat-escalated').textContent= stats.escalated    ?? 0;
    el('stat-violations').textContent = stats.violations ?? 0;
    el('stat-cost').textContent     = fmt.cost(stats.total_cost_usd);
    el('stat-latency').textContent  = fmt.ms(stats.avg_latency_ms);
    el('stat-rag').textContent      = fmt.score(stats.avg_rag_precision);
  } catch {
    /* backend not running */
  }
}

// ── Runs table ────────────────────────────────────────────────────────────────
async function loadRuns(page = 0) {
  _page = page;
  const verdict  = el('verdictFilter').value;
  const txnSearch = (el('txnSearch').value || '').trim();

  const params = new URLSearchParams({
    limit:  PAGE_SZ,
    offset: page * PAGE_SZ,
    ...(verdict   && { verdict }),
    ...(txnSearch && { txn_id: txnSearch }),
  });

  try {
    const res  = await fetch(`${API}/audit/runs?${params}`);
    const data = await res.json();
    _totalRuns = data.total;
    renderTable(data.runs);
    renderPagination(data.total, page);
  } catch {
    el('runsBody').innerHTML = '<tr><td colspan="8" class="table-empty">Could not connect to backend.</td></tr>';
  }
}

function renderTable(runs) {
  if (!runs.length) {
    el('runsBody').innerHTML = '<tr><td colspan="8" class="table-empty">No runs found.</td></tr>';
    return;
  }
  el('runsBody').innerHTML = runs.map(r => {
    const prob      = r.inv_dt_fraud_prob;
    const probColor = prob >= 0.7 ? 'var(--red)' : prob >= 0.4 ? 'var(--amber)' : 'var(--green)';
    const conf      = r.inv_confidence;
    const confLabel = conf >= 0.85 ? 'high' : conf >= 0.70 ? 'medium' : 'low';
    const viols     = r.aud_violations_count || 0;
    return `<tr class="run-row" data-run-id="${esc(r.run_id)}">
      <td class="mono-cell">${esc(r.transaction_id)}</td>
      <td class="ts-cell">${fmt.ts(r.timestamp)}</td>
      <td>${verdictBadge(r.final_verdict)}</td>
      <td style="color:${probColor};font-family:var(--mono);font-size:12.5px">${fmt.prob(prob)}</td>
      <td style="font-family:var(--mono);font-size:12.5px">${conf != null ? conf.toFixed(2) : '—'} <span style="color:var(--text-muted);font-size:10.5px">${confLabel}</span></td>
      <td style="font-family:var(--mono);font-size:12.5px">${viols > 0 ? `<span style="color:var(--red)">${viols}</span>` : '0'}</td>
      <td style="font-family:var(--mono);font-size:12.5px">${fmt.cost(r.total_cost_usd)}</td>
      <td style="font-family:var(--mono);font-size:12.5px">${fmt.ms(r.total_latency_ms)}</td>
    </tr>`;
  }).join('');

  el('runsBody').querySelectorAll('.run-row').forEach(row => {
    row.addEventListener('click', () => replayRun(row.dataset.runId));
  });
}

function renderPagination(total, page) {
  const totalPages = Math.ceil(total / PAGE_SZ);
  if (totalPages <= 1) { el('pagination').innerHTML = ''; return; }
  el('pagination').innerHTML = `
    <button class="btn-ghost" id="prevPage" ${page === 0 ? 'disabled' : ''}>← Prev</button>
    <span class="page-label">Page ${page + 1} of ${totalPages}</span>
    <button class="btn-ghost" id="nextPage" ${page >= totalPages - 1 ? 'disabled' : ''}>Next →</button>
  `;
  el('prevPage')?.addEventListener('click', () => loadRuns(page - 1));
  el('nextPage')?.addEventListener('click', () => loadRuns(page + 1));
}

// ── Replay ────────────────────────────────────────────────────────────────────
async function replayRun(runId) {
  try {
    const res    = await fetch(`${API}/audit/runs/${runId}`);
    if (!res.ok) { showErr('Run not found.'); return; }
    const record = await res.json();
    renderReplayPanel(record);
  } catch {
    showErr('Could not load replay — is the backend running?');
  }
}

function renderReplayPanel(record) {
  // Banner
  const ts = fmt.ts(record.timestamp);
  el('replayBanner').innerHTML =
    `<span class="replay-run-label">REPLAY</span> &nbsp;Run <code>${(record.run_id || '').slice(0, 8)}</code> · ${esc(ts)}`;

  populateTriageSection(record);
  populateInvestigatorSection(record);
  populateAuditorSection(record);
  populateTotalsSection(record);

  el('replayPanel').classList.remove('hidden');
  el('replayPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Triage section ────────────────────────────────────────────────────────────
function populateTriageSection(r) {
  const entities = r.triage_entities || {};
  const input    = r.raw_input || {};

  const pills = [
    { k: 'txn',      v: input.transaction_id },
    { k: 'amount',   v: `$${Number(input.amount || 0).toLocaleString()}` },
    { k: 'merchant', v: input.merchant },
    { k: 'reason',   v: input.reason_code },
    { k: 'category', v: entities.dispute_category },
    { k: 'urgency',  v: (entities.urgency || '').replace(/_/g, ' ') },
    entities.transaction_date ? { k: 'date', v: entities.transaction_date } : null,
  ].filter(Boolean);

  el('rp-triagePills').innerHTML = pills.map(p =>
    `<span class="pill urgency-${entities.urgency}">
       <span class="pill-key">${p.k}</span>
       <span class="pill-val">${esc(p.v)}</span>
     </span>`
  ).join('');

  el('rp-maskedStmt').innerHTML = esc(r.triage_masked_stmt || '').replace(
    /\[(NAME|LOCATION|CONTACT|PII)\]/g,
    '<span class="pii-mask">[$1]</span>'
  );

  el('rp-triageMetrics').innerHTML = [
    metricTile('Latency',    fmt.ms(r.triage_latency_ms)),
    metricTile('Tokens in',  fmt.tokens(r.triage_tokens_in)),
    metricTile('Tokens out', fmt.tokens(r.triage_tokens_out)),
    metricTile('Cost',       fmt.cost(r.triage_cost_usd)),
  ].join('');
}

// ── Investigator section ──────────────────────────────────────────────────────
function populateInvestigatorSection(r) {
  // Tool call log
  const log = r.inv_tool_calls_log || [];
  if (log.length) {
    el('rp-toolLog').innerHTML = log.map(tc => {
      const statusClass = tc.status === 'error' ? 'error' : 'ok';
      const statusLabel = tc.status === 'error' ? '✗ Error' : '✓ Done';
      const resultStr   = _summariseResult(tc.tool, tc.result);
      return `<div class="tool-row">
        <div class="tool-row-left">
          <div class="tool-icon ${_toolIconClass(tc.tool)}">${_toolIcon(tc.tool)}</div>
          <div class="tool-text">
            <div class="tool-name">${esc(tc.tool)}()</div>
            <div class="tool-result">${esc(resultStr)}</div>
          </div>
        </div>
        <div class="tool-row-right">
          <span class="status-chip ${statusClass}">${statusLabel}</span>
        </div>
      </div>`;
    }).join('');
  } else {
    el('rp-toolLog').innerHTML = '<div style="color:var(--text-muted);font-size:13px;padding:12px">No tool calls logged.</div>';
  }

  // Classifier
  const clf = {
    fraud_probability: r.inv_dt_fraud_prob,
    prediction:        r.inv_dt_prediction,
    confidence:        r.inv_dt_confidence,
    decision_path:     r.inv_dt_decision_path || [],
    top_features:      r.inv_dt_top_features  || [],
    model_version:     r.dt_model_version || 'dt_v1',
  };
  renderReplayClassifier(clf);

  // RAG chunks
  const chunks = r.inv_rag_chunks || [];
  if (chunks.length) {
    el('rp-ragChunks').innerHTML = chunks.map((c, i) =>
      `<div class="rag-card" style="margin-bottom:10px">
         <div style="font-size:10px;color:var(--text-muted);font-weight:600;margin-bottom:4px">
           Chunk ${i + 1} · ${esc(c.section || '—')} · p.${esc(String(c.page || '?'))}
         </div>
         ${esc(c.text || JSON.stringify(c))}
         ${c.relevance != null ? `<div class="rag-meta"><span>relevance: ${parseFloat(c.relevance).toFixed(3)}</span></div>` : ''}
       </div>`
    ).join('');
  } else {
    const clause = r.inv_rag_query
      ? `<div class="rag-card">Query: <em>${esc(r.inv_rag_query)}</em><div class="rag-meta"><span>No chunks retrieved</span></div></div>`
      : '<div style="color:var(--text-muted);font-size:13px">No RAG chunks logged.</div>';
    el('rp-ragChunks').innerHTML = clause;
  }

  // Fraud signals
  const signals = r.inv_fraud_signals || [];
  el('rp-signalList').innerHTML = signals.length
    ? signals.map(s => `<li>${esc(s)}</li>`).join('')
    : '<li>No signals logged</li>';

  // LLM reasoning (collapsible)
  const reasoning = (r.inv_llm_reasoning || '').trim();
  if (reasoning) {
    el('rp-reasoning').textContent = reasoning;
    el('rp-reasoningToggle').classList.remove('hidden');
  } else {
    el('rp-reasoningToggle').classList.add('hidden');
  }

  // Recommendation + confidence
  el('rp-recBadge').className = `verdict-badge ${verdictClass(r.inv_recommendation)}`;
  el('rp-recBadge').textContent = verdictLabel(r.inv_recommendation);

  const conf = r.inv_confidence ?? 0;
  const confClass = conf >= 0.85 ? 'conf-high' : conf >= 0.70 ? 'conf-medium' : 'conf-low';
  el('rp-confFill').style.width  = fmt.pct(conf);
  el('rp-confFill').className    = `conf-fill ${confClass}`;
  el('rp-confPct').textContent   = fmt.pct(conf);

  el('rp-invMetrics').innerHTML = [
    metricTile('Latency',       fmt.ms(r.inv_latency_ms)),
    metricTile('Tokens in',     fmt.tokens(r.inv_tokens_in)),
    metricTile('Tokens out',    fmt.tokens(r.inv_tokens_out)),
    metricTile('RAG precision', fmt.score(r.inv_rag_precision)),
  ].join('');
}

function renderReplayClassifier(clf) {
  const prob = clf.fraud_probability ?? 0;
  const pred = clf.prediction ?? 'unknown';

  el('rp-probBar').style.width = `${Math.round(prob * 100)}%`;

  el('rp-probMeta').textContent =
    `${(prob * 100).toFixed(1)}% · confidence: ${clf.confidence ?? '—'} · ${clf.model_version ?? 'dt_v1'}`;

  const predEl = el('rp-classifierPred');
  predEl.textContent = pred === 'fraud' ? 'FRAUD' : 'LEGITIMATE';
  predEl.className   = `classifier-pred ${pred === 'fraud' ? 'fraud' : 'legitimate'}`;

  el('rp-decisionPath').innerHTML = (clf.decision_path || []).map(step => {
    const isLeaf = step === 'FRAUD' || step === 'LEGITIMATE';
    if (isLeaf) {
      const cls = step === 'FRAUD' ? 'path-leaf-fraud' : 'path-leaf-legit';
      return `<div class="${cls}">→ ${step}</div>`;
    }
    return `<div>${esc(step)}</div>`;
  }).join('');

  const features = clf.top_features || [];
  const maxImp   = Math.max(...features.map(f => f.importance), 0.01);
  el('rp-featureList').innerHTML = features.map(f => {
    const barPct = Math.round((f.importance / maxImp) * 100);
    return `<div class="feat-bar-row">
      <div class="feat-name" title="${esc(f.feature)}">${esc(f.feature)}</div>
      <div class="feat-bar-track"><div class="feat-bar-fill" style="width:${barPct}%"></div></div>
      <div class="feat-value">${parseFloat(f.importance).toFixed(2)}</div>
    </div>`;
  }).join('');
}

// ── Auditor section ───────────────────────────────────────────────────────────
function populateAuditorSection(r) {
  const violations = r.aud_violations || [];
  const isViol     = r.aud_verdict === 'policy_violation';

  if (isViol && violations.length) {
    const b = el('rp-violationBanner');
    b.innerHTML = `<div class="alert-banner-title">⚠ Pipeline halted — policy violation</div>
      <ul>${violations.map(v => `<li>${esc(v.replace(/_/g, ' '))}</li>`).join('')}</ul>`;
    b.classList.remove('hidden');
  }

  el('rp-rulesList').innerHTML = (r.aud_rules_checked || []).map(rule =>
    `<div class="rule-row ${rule.passed ? '' : 'fail'}">
       <div class="rule-body">
         <div class="rule-name-text">${esc(rule.rule)}</div>
         <div class="rule-detail-text">${esc(rule.detail)}</div>
       </div>
       <span class="rule-status-tag ${rule.passed ? 'pass' : 'fail'}">${rule.passed ? 'Pass' : 'Fail'}</span>
     </div>`
  ).join('');

  const verdictEl = el('rp-verdict');
  verdictEl.className   = `verdict-badge large ${verdictClass(r.aud_verdict)}`;
  verdictEl.textContent = verdictLabel(r.aud_verdict);

  const audReasoning = el('rp-audReasoning');
  if (r.aud_reasoning) {
    audReasoning.textContent = r.aud_reasoning;
    audReasoning.classList.remove('hidden');
  } else {
    audReasoning.classList.add('hidden');
  }

  el('rp-audMetrics').innerHTML = [
    metricTile('Latency',    fmt.ms(r.aud_latency_ms)),
    metricTile('Tokens in',  fmt.tokens(r.aud_tokens_in)),
    metricTile('Tokens out', fmt.tokens(r.aud_tokens_out)),
    metricTile('Violations', violations.length, violations.length ? 'policy halt' : 'all clear'),
  ].join('');
}

// ── Pipeline totals ───────────────────────────────────────────────────────────
function populateTotalsSection(r) {
  el('rp-totals').innerHTML = [
    metricTile('Total latency', fmt.ms(r.total_latency_ms)),
    metricTile('Total tokens',  fmt.tokens(r.total_tokens)),
    metricTile('Total cost',    fmt.cost(r.total_cost_usd)),
    metricTile('Models', `${r.model_triage ?? '—'} / ${r.model_investigator ?? '—'}`, r.pipeline_version ?? ''),
  ].join('');
}

// ── Tool display helpers ──────────────────────────────────────────────────────
function _toolIconClass(name) {
  if (name === 'query_transactions')    return 'db';
  if (name === 'get_velocity_history')  return 'speed';
  if (name === 'check_merchant_risk')   return 'risk';
  return 'doc';
}
function _toolIcon(name) {
  if (name === 'query_transactions')    return 'DB';
  if (name === 'get_velocity_history')  return 'V';
  if (name === 'check_merchant_risk')   return 'MR';
  if (name === 'run_fraud_classifier')  return 'DT';
  return 'RAG';
}
function _summariseResult(name, result) {
  if (!result || typeof result !== 'object') return String(result ?? '');
  if (result.error) return `Error: ${result.error}`;
  if (name === 'query_transactions')
    return `${result.transaction_id ?? '—'} · $${result.transaction_amt ?? '?'}`;
  if (name === 'get_velocity_history')
    return `${result.transaction_count ?? 0} txns · $${result.total_amount ?? 0} · ${(result.unique_countries || []).length} countries`;
  if (name === 'check_merchant_risk')
    return `risk ${result.risk_score ?? '?'} (${result.risk_category ?? '?'})`;
  if (name === 'run_fraud_classifier')
    return `p(fraud)=${result.fraud_probability?.toFixed(2) ?? '?'} · ${result.prediction ?? '?'}`;
  if (name === 'search_compliance_docs')
    return Array.isArray(result) ? `${result.length} chunks retrieved` : 'Not found';
  return JSON.stringify(result).slice(0, 80);
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadRuns(0);

  el('refreshBtn').addEventListener('click', () => {
    loadStats();
    loadRuns(0);
  });

  el('verdictFilter').addEventListener('change', () => loadRuns(0));

  let searchTimer;
  el('txnSearch').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadRuns(0), 350);
  });

  el('closeReplay').addEventListener('click', () => {
    el('replayPanel').classList.add('hidden');
    // reset replay state
    el('rp-violationBanner').classList.add('hidden');
    el('rp-reasoning').classList.add('hidden');
    el('rp-reasoning').textContent = '';
    el('rp-reasoningToggle').textContent = 'Show ↓';
    el('rp-audReasoning').classList.add('hidden');
  });

  el('rp-reasoningToggle').addEventListener('click', () => {
    const pre    = el('rp-reasoning');
    const hidden = pre.classList.toggle('hidden');
    el('rp-reasoningToggle').textContent = hidden ? 'Show ↓' : 'Hide ↑';
  });
});
