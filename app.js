// ============================================
// CONFIGURATION
// ============================================
const API_BASE = 'http://localhost:8000';

const CLASS_META = {
  colon_aca: { label: 'Colon Adenocarcinoma', organ: 'Colon', status: 'CANCER', color: '#e25454' },
  colon_n:   { label: 'Normal Colon Tissue', organ: 'Colon', status: 'NORMAL', color: '#3dba78' },
  lung_aca:  { label: 'Lung Adenocarcinoma', organ: 'Lung', status: 'CANCER', color: '#e07a40' },
  lung_n:    { label: 'Normal Lung Tissue', organ: 'Lung', status: 'NORMAL', color: '#3dba78' },
  lung_scc:  { label: 'Lung Squamous Cell Carcinoma', organ: 'Lung', status: 'CANCER', color: '#c94f4f' },
};

// ============================================
// STATE
// ============================================
let state = { 
  mode: 'image', 
  imageFile: null, 
  csvFile: null, 
  running: false, 
  history: [] 
};
let lastResult = null;

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  setupNav();
  setupModeTabs();
  applyMode('image');
});

// ============================================
// NAVIGATION
// ============================================
function setupNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const view = item.dataset.view;
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      item.classList.add('active');
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      document.getElementById('view-' + view).classList.add('active');
      if (view === 'history') renderFullHistory();
    });
  });
}

// ============================================
// MODE TABS
// ============================================
function setupModeTabs() {
  document.querySelectorAll('.mode-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.mode = tab.dataset.mode;
      applyMode(state.mode);
      updateRunHint();
    });
  });
}

function applyMode(mode) {
  const imgCard = document.getElementById('img-upload-card');
  const csvCard = document.getElementById('csv-upload-card');

  if (mode === 'image') {
    imgCard.style.display = '';
    csvCard.style.display = 'none';
  } else if (mode === 'mirna') {
    imgCard.style.display = 'none';
    csvCard.style.display = '';
  } else {
    imgCard.style.display = '';
    csvCard.style.display = '';
  }

  if (mode === 'image') { state.csvFile = null; resetCard('csv'); }
  if (mode === 'mirna') { state.imageFile = null; resetCard('img'); }

  updateRunHint();
}

function resetCard(type) {
  if (type === 'img') {
    document.getElementById('img-upload-card').classList.remove('has-file');
    document.getElementById('img-preview').style.display = 'none';
    const zone = document.getElementById('img-zone');
    if (zone) {
      const text = zone.querySelector('.upload-text');
      const subText = zone.querySelector('.upload-sub-text');
      const icon = zone.querySelector('.upload-icon-wrap');
      if (text) text.style.display = '';
      if (subText) subText.style.display = '';
      if (icon) icon.style.display = '';
    }
    state.imageFile = null;
  } else {
    document.getElementById('csv-upload-card').classList.remove('has-file');
    document.getElementById('csv-uploaded-info').style.display = 'none';
    const zone = document.getElementById('csv-zone');
    if (zone) {
      const text = zone.querySelector('.upload-text');
      const subText = zone.querySelector('.upload-sub-text');
      const icon = zone.querySelector('.upload-icon-wrap');
      if (text) text.style.display = '';
      if (subText) subText.style.display = '';
      if (icon) icon.style.display = '';
    }
    state.csvFile = null;
  }
}

function updateRunHint() {
  const hint = document.getElementById('run-hint');
  const btn = document.getElementById('run-btn');
  const hasImage = !!state.imageFile;
  const hasCsv = !!state.csvFile;

  let ready = false;
  if (state.mode === 'image' && hasImage) ready = true;
  if (state.mode === 'mirna' && hasCsv) ready = true;
  if (state.mode === 'combined' && hasImage && hasCsv) ready = true;

  if (btn) btn.disabled = !ready || state.running;

  if (!ready && hint) {
    const needs = state.mode === 'combined' ? 'both image and miRNA CSV'
                : state.mode === 'mirna' ? 'a miRNA CSV file'
                : 'a histopathology image';
    hint.textContent = `Upload ${needs} to begin`;
  } else if (hint) {
    hint.textContent = 'Ready to analyse';
  }
}

// ============================================
// FILE UPLOAD HANDLERS
// ============================================
function triggerUpload(type) {
  const inputId = type === 'image' ? 'img-file-input' : 'csv-file-input';
  const input = document.getElementById(inputId);
  if (input) input.click();
}

function onDrag(e) {
  e.preventDefault();
  e.currentTarget.classList.add('drag-over');
}

function onDrop(e, type) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) processFile(file, type);
}

function handleFile(e, type) {
  const file = e.target.files[0];
  if (file) processFile(file, type);
}

function processFile(file, type) {
  if (type === 'image') {
    state.imageFile = file;
    const card = document.getElementById('img-upload-card');
    if (card) card.classList.add('has-file');
    
    const reader = new FileReader();
    reader.onload = ev => {
      const previewImg = document.getElementById('img-preview-src');
      const filenameSpan = document.getElementById('img-filename');
      const previewDiv = document.getElementById('img-preview');
      const zone = document.getElementById('img-zone');
      
      if (previewImg) previewImg.src = ev.target.result;
      if (filenameSpan) filenameSpan.textContent = file.name;
      if (previewDiv) previewDiv.style.display = 'flex';
      
      if (zone) {
        const text = zone.querySelector('.upload-text');
        const subText = zone.querySelector('.upload-sub-text');
        const icon = zone.querySelector('.upload-icon-wrap');
        if (text) text.style.display = 'none';
        if (subText) subText.style.display = 'none';
        if (icon) icon.style.display = 'none';
      }
    };
    reader.readAsDataURL(file);
  } else {
    state.csvFile = file;
    const card = document.getElementById('csv-upload-card');
    if (card) card.classList.add('has-file');
    
    const filenameSpan = document.getElementById('csv-filename');
    const infoDiv = document.getElementById('csv-uploaded-info');
    const zone = document.getElementById('csv-zone');
    
    if (filenameSpan) filenameSpan.textContent = file.name;
    if (infoDiv) infoDiv.style.display = 'flex';
    
    if (zone) {
      const text = zone.querySelector('.upload-text');
      const subText = zone.querySelector('.upload-sub-text');
      const icon = zone.querySelector('.upload-icon-wrap');
      if (text) text.style.display = 'none';
      if (subText) subText.style.display = 'none';
      if (icon) icon.style.display = 'none';
    }
  }
  updateRunHint();
}

// ============================================
// RUN ANALYSIS
// ============================================
async function runAnalysis() {
  if (state.running) return;
  state.running = true;

  const runBtn = document.getElementById('run-btn');
  const runInner = document.getElementById('run-btn-inner');
  const runSpinner = document.getElementById('run-spinner');
  const resultsSection = document.getElementById('results-section');
  
  if (runBtn) runBtn.disabled = true;
  if (runInner) runInner.style.display = 'none';
  if (runSpinner) runSpinner.style.display = 'flex';
  if (resultsSection) resultsSection.style.display = 'none';

  // Remove any existing dual results or warnings
  const existingDual = document.querySelector('.dual-results');
  if (existingDual) existingDual.remove();
  const existingWarning = document.querySelector('.conflict-warning');
  if (existingWarning) existingWarning.remove();

  try {
    let result;
    if (state.mode === 'image') {
      result = await predictImage(state.imageFile);
    } else if (state.mode === 'mirna') {
      result = await predictTabular(state.csvFile);
    } else {
      result = await predictCombined(state.imageFile, state.csvFile);
    }
    showResult(result);
  } catch (err) {
    console.error('Prediction error:', err);
    showError(err.message || 'Prediction failed. Is the FastAPI server running?');
  }

  state.running = false;
  if (runBtn) runBtn.disabled = false;
  if (runInner) runInner.style.display = 'flex';
  if (runSpinner) runSpinner.style.display = 'none';
}

// ============================================
// API CALLS
// ============================================
async function predictImage(file) {
  const form = new FormData();
  form.append('image_file', file);
  const res = await fetch(`${API_BASE}/predict-image`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return { ...data, mode: 'Image (EfficientNet-B3)' };
}

async function predictTabular(file) {
  const form = new FormData();
  form.append('csv_file', file);
  const res = await fetch(`${API_BASE}/predict-tabular`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return {
    predicted_class: data.subtype_prediction || 'unknown',
    label: data.subtype_prediction || 'Unknown',
    status: (data.binary_prediction || '').toLowerCase().includes('normal') ? 'NORMAL' : 'CANCER',
    organ: (data.subtype_prediction || '').toLowerCase().includes('colon') ? 'Colon' : 'Lung',
    confidence: data.confidence || 0,
    probabilities: null,
    mode: 'miRNA Ensemble',
  };
}

async function predictCombined(imgFile, csvFile) {
  const form = new FormData();
  form.append('image_file', imgFile);
  form.append('csv_file', csvFile);
  const res = await fetch(`${API_BASE}/predict-combined`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// ============================================
// RENDER RESULT - MAIN ENTRY POINT
// ============================================
function showResult(data) {
  if (data.image && data.mirna) {
    showCombinedResult(data);
  } else {
    showSingleResult(data);
  }
}

// ============================================
// SHOW COMBINED RESULT (Image + miRNA side by side)
// ============================================
function showCombinedResult(data) {
  const meta = CLASS_META[data.predicted_class] || {};
  
  // ── HIDE the overall verdict banner completely ──────────
  const banner = document.getElementById('verdict-banner');
  if (banner) {
    banner.style.display = 'none';
  }
  
  // ── Hide the Stage 2 classification card ────────────────
  const resultsGrid = document.querySelector('.results-grid');
  if (resultsGrid) {
    resultsGrid.style.display = 'none';
  }
  
  // ── Hide the section divider ────────────────────────────
  const divider = document.querySelector('.section-divider');
  if (divider) {
    divider.style.display = 'none';
  }
  
  // ── Get or create container ─────────────────────────────
  let container = document.getElementById('dual-results-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'dual-results-container';
    container.style.cssText = 'margin: 20px 0;';
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
      resultsSection.appendChild(container);
    }
  }
  
  // Clear container
  container.innerHTML = '';
  
  // ── Build Side-by-Side Results ──────────────────────────
  const imgStatusColor = data.image?.status === 'CANCER' ? '#f87171' : '#34d399';
  const imgStatusBg = data.image?.status === 'CANCER' ? 'rgba(248,113,113,0.08)' : 'rgba(52,211,153,0.08)';
  const imgConfColor = (data.image?.confidence || 0) >= 75 ? '#34d399' : '#f87171';
  
  // miRNA confidence ALWAYS in RED (since it shows cancer)
  const mirnaConfColor = '#f87171'; // Always red
  
  const mirnaStatus = data.mirna?.binary && data.mirna.binary !== 'N/A' ? data.mirna.binary : 'Unknown';
  const mirnaIsCancer = mirnaStatus.includes('Cancer') || mirnaStatus.includes('Carcinoma');
  const mirnaStatusColor = mirnaIsCancer ? '#f87171' : '#34d399';
  const mirnaStatusBg = mirnaIsCancer ? 'rgba(248,113,113,0.08)' : 'rgba(52,211,153,0.08)';
  
  container.innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0;">
      <!-- Image Result -->
      <div style="background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.12);border-radius:18px;padding:18px 20px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
          <span style="font-size:16px;">🖼️</span>
          <span style="font-weight:600;color:#e2e8f8;font-size:13px;">Histopathology Image</span>
          <span style="margin-left:auto;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;background:${imgStatusBg};color:${imgStatusColor}">
            ${data.image?.status || 'UNKNOWN'}
          </span>
        </div>
        <div style="font-size:22px;font-weight:700;color:#e2e8f8;margin-bottom:4px;">
          ${data.image?.label || 'N/A'}
        </div>
        <div style="display:flex;align-items:center;gap:12px;margin-top:8px;">
          <span style="color:#7a8aaa;font-size:12px;">Confidence:</span>
          <span style="font-size:20px;font-weight:700;color:${imgConfColor}">
            ${Math.round(data.image?.confidence || 0)}%
          </span>
        </div>
        <div style="margin-top:8px;font-size:11px;color:#3d4d6a;">
          Class: ${data.image?.prediction || 'N/A'}
        </div>
        <!-- Stage 2 Classification details moved here -->
        <div style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06);">
          <div style="font-size:10.5px;color:#3d4d6a;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Classification Details</div>
          <div style="display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:12px;">
            <span style="color:#7a8aaa;">Type:</span>
            <span style="color:#e2e8f8;">${data.label || '—'}</span>
            <span style="color:#7a8aaa;">Organ:</span>
            <span style="color:#e2e8f8;">${data.organ || '—'}</span>
            <span style="color:#7a8aaa;">Class ID:</span>
            <code style="color:#93c5fd;background:rgba(59,130,246,0.1);padding:1px 8px;border-radius:4px;font-size:11px;font-family:'JetBrains Mono',monospace;">${data.predicted_class || '—'}</code>
            <span style="color:#7a8aaa;">Model:</span>
            <span style="color:#e2e8f8;">Combined (Image + miRNA)</span>
          </div>
          <div style="margin-top:8px;">
            <span style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500;background:rgba(255,255,255,0.06);color:#7a8aaa;border:1px solid rgba(255,255,255,0.06);">
              ${data.confidence >= 90 ? '🔒 Very high confidence' :
                data.confidence >= 75 ? '✅ High confidence' :
                data.confidence >= 55 ? '🟡 Moderate confidence' :
                '⚠️ Low confidence — verify manually'}
            </span>
          </div>
        </div>
      </div>
      
      <!-- miRNA Result -->
      <div style="background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.12);border-radius:18px;padding:18px 20px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
          <span style="font-size:16px;">🧬</span>
          <span style="font-weight:600;color:#e2e8f8;font-size:13px;">miRNA Expression</span>
          <span style="margin-left:auto;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;background:${mirnaStatusBg};color:${mirnaStatusColor}">
            ${mirnaStatus}
          </span>
        </div>
        <div style="font-size:22px;font-weight:700;color:#e2e8f8;margin-bottom:4px;">
          ${data.mirna?.subtype && data.mirna.subtype !== 'N/A' ? data.mirna.subtype : 'Not Available'}
        </div>
        <div style="display:flex;align-items:center;gap:12px;margin-top:8px;">
          <span style="color:#7a8aaa;font-size:12px;">Confidence:</span>
          <span style="font-size:20px;font-weight:700;color:${mirnaConfColor}">
            ${Math.round(data.mirna?.confidence || 0)}%
          </span>
        </div>
        ${data.mirna?.error ? `<div style="margin-top:8px;font-size:11px;color:#f87171;">⚠️ ${data.mirna.error}</div>` : ''}
        <div style="margin-top:8px;font-size:11px;color:#3d4d6a;">
          ${data.mirna?.subtype && data.mirna.subtype !== 'N/A' ? `Subtype: ${data.mirna.subtype}` : 'No miRNA data available'}
        </div>
      </div>
    </div>
  `;
  
  // ── Show results section ─────────────────────────────────
  const resultsSection = document.getElementById('results-section');
  const downloadRow = document.getElementById('download-row');
  if (resultsSection) resultsSection.style.display = 'block';
  if (downloadRow) downloadRow.style.display = 'flex';
  
  // ── Store for PDF ────────────────────────────────────────
  lastResult = data;
  
  // ── History ──────────────────────────────────────────────
  state.history.unshift({
    time: new Date().toLocaleTimeString(),
    label: data.label || meta.label || data.predicted_class || '—',
    status: data.status || 'UNKNOWN',
    organ: data.organ || meta.organ || '—',
    conf: Math.round(data.confidence || 0),
    mode: 'Combined (Image + miRNA)',
  });
}

// ============================================
// SHOW SINGLE RESULT (Image or miRNA only)
// ============================================
function showSingleResult(data) {
  const meta = CLASS_META[data.predicted_class] || {};
  const conf = Math.round(data.confidence || 0);
  const isCanc = (data.status || '').toUpperCase() === 'CANCER';

  const banner = document.getElementById('verdict-banner');
  if (banner) {
    banner.style.display = 'flex';
    banner.className = 'verdict-banner ' + (isCanc ? 'cancer' : 'normal');
    document.getElementById('verdict-icon').textContent = isCanc ? '🔴' : '🟢';
    document.getElementById('verdict-title').textContent = isCanc ? 'Cancer Detected' : 'No Cancer Detected';
    document.getElementById('verdict-conf-num').textContent = conf + '%';
  }

  document.getElementById('r-type').textContent = data.label || meta.label || '—';
  document.getElementById('r-organ').textContent = data.organ || meta.organ || '—';
  document.getElementById('r-class').textContent = data.predicted_class || '—';
  document.getElementById('r-model').textContent = data.mode || '—';

  const tier = conf >= 90 ? '🔒 Very high confidence'
             : conf >= 75 ? '✅ High confidence'
             : conf >= 55 ? '🟡 Moderate confidence'
             : '⚠️ Low confidence — verify manually';
  document.getElementById('tier-badge').textContent = tier;

  const barsEl = document.getElementById('prob-bars');
  if (barsEl && data.probabilities && Object.keys(data.probabilities).length) {
    const sorted = Object.entries(data.probabilities).sort(([,a],[,b]) => b - a);
    barsEl.innerHTML = sorted.map(([cls, prob]) => {
      const m = CLASS_META[cls] || { label: cls, color: '#666' };
      const pct = Math.round(prob * 100);
      const lbl = m.label.replace('Lung ', '').replace('Colon ', '')
                         .replace(' Carcinoma', '').replace(' Cell', '');
      return `<div class="prob-bar-row">
        <div class="prob-bar-label">${lbl}</div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width:${pct}%;background:${m.color}"></div>
        </div>
        <div class="prob-bar-pct">${pct}%</div>
      </div>`;
    }).join('');
  } else if (barsEl) {
    barsEl.innerHTML = `<p style="font-size:12px;color:var(--text3);padding:8px 0">
      Probability breakdown not available for this endpoint.</p>`;
  }

  const resultsGrid = document.querySelector('.results-grid');
  if (resultsGrid) resultsGrid.style.display = 'grid';
  
  const divider = document.querySelector('.section-divider');
  if (divider) divider.style.display = 'flex';

  const resultsSection = document.getElementById('results-section');
  const downloadRow = document.getElementById('download-row');
  if (resultsSection) resultsSection.style.display = 'block';
  if (downloadRow) downloadRow.style.display = 'flex';

  lastResult = data;
  state.history.unshift({
    time: new Date().toLocaleTimeString(),
    label: data.label || meta.label || data.predicted_class || '—',
    status: data.status || 'UNKNOWN',
    organ: data.organ || meta.organ || '—',
    conf: conf,
    mode: data.mode || '—',
  });
}

// ============================================
// SHOW ERROR
// ============================================
function showError(msg) {
  const resultsSection = document.getElementById('results-section');
  if (resultsSection) {
    resultsSection.style.display = 'block';
    resultsSection.innerHTML = `<div style="padding:24px;background:rgba(248,113,113,0.06);
      border:1px solid rgba(248,113,113,0.25);border-radius:var(--radius-lg);
      color:#f87171;font-size:13px;display:flex;align-items:center;gap:10px">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span><strong>Error:</strong> ${msg}</span></div>`;
  }
}

// ============================================
// DOWNLOAD PDF
// ============================================
async function downloadReport() {
  if (!lastResult) return;
  
  const btn = document.getElementById('download-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spin-ring" style="border-top-color:#fff;width:14px;height:14px"></span> Generating…`;
  }

  try {
    const form = new FormData();
    
    if (lastResult.image && lastResult.mirna) {
      form.append('label', lastResult.image.label || lastResult.label || '');
      form.append('confidence', String(lastResult.confidence || 0));
      form.append('status', lastResult.status || '');
      form.append('organ', lastResult.organ || '');
      form.append('class_id', lastResult.predicted_class || '');
      form.append('model_used', 'Combined (Image + miRNA)');
      if (lastResult.image?.probabilities)
        form.append('probabilities', JSON.stringify(lastResult.image.probabilities));
      
      const mirna_info = `miRNA: ${lastResult.mirna.subtype || 'N/A'} (${Math.round(lastResult.mirna.confidence || 0)}%)`;
      form.append('additional_info', mirna_info);
    } else {
      form.append('label', lastResult.label || '');
      form.append('confidence', String(lastResult.confidence || 0));
      form.append('status', lastResult.status || '');
      form.append('organ', lastResult.organ || '');
      form.append('class_id', lastResult.predicted_class || '');
      form.append('model_used', lastResult.mode || '');
      if (lastResult.probabilities)
        form.append('probabilities', JSON.stringify(lastResult.probabilities));
    }

    const res = await fetch(`${API_BASE}/generate-report`, { method: 'POST', body: form });
    if (!res.ok) throw new Error('PDF generation failed');

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'CancerAI_Report.pdf';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert('PDF error: ' + err.message);
  }

  if (btn) {
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      Download PDF Report`;
  }
}

// ============================================
// HISTORY
// ============================================
function renderFullHistory() {
  const empty = document.getElementById('empty-history');
  const table = document.getElementById('history-table');
  const tbody = document.getElementById('history-tbody');

  if (!state.history.length) {
    if (empty) empty.style.display = 'flex';
    if (table) table.style.display = 'none';
    return;
  }
  
  if (empty) empty.style.display = 'none';
  if (table) table.style.display = 'table';
  
  if (tbody) {
    tbody.innerHTML = state.history.map(h => `
      <tr>
        <td style="color:var(--text3);font-family:var(--font-mono);font-size:12px">${h.time}</td>
        <td style="color:var(--text)">${h.label}</td>
        <td><span class="status-pill ${h.status.toLowerCase()}">${h.status}</span></td>
        <td>${h.organ}</td>
        <td style="font-family:var(--font-mono)">${h.conf}%</td>
        <td style="color:var(--text3)">${h.mode}</td>
      </tr>
    `).join('');
  }
}

function clearHistory() {
  state.history = [];
  renderFullHistory();
}