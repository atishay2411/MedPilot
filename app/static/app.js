// ── DOM refs ───────────────────────────────────────────────────────────────
const transcriptEl       = document.getElementById("transcript");
const promptEl           = document.getElementById("prompt");
const patientUuidEl      = document.getElementById("patient-uuid");
const fileInputEl        = document.getElementById("file-input");
const uploadLabelEl      = document.getElementById("upload-label");
const patientContextEl   = document.getElementById("patient-context");
const llmBadgeTextEl     = document.getElementById("llm-badge-text");
const newChatBtnEl       = document.getElementById("new-chat-btn");
const patientSearchEl    = document.getElementById("patient-search-input");
const browseAllBtnEl     = document.getElementById("browse-all-btn");
const patientListEl      = document.getElementById("patient-browser-list");
const paginationEl       = document.getElementById("browser-pagination");
const loadMoreBtnEl      = document.getElementById("load-more-btn");

const SESSION_STORAGE_KEY = "medpilot-session-id";
const PAGE_SIZE = 20;

// ── App state ──────────────────────────────────────────────────────────────
const state = {
  sessionId:            window.localStorage.getItem(SESSION_STORAGE_KEY) || "",
  currentPatientUuid:   "",
  currentPatientDisplay:"",
  browser: {
    query:      "",
    startIndex: 0,
    hasMore:    false,
    loading:    false,
  },
};

// ── Helpers ────────────────────────────────────────────────────────────────
function escapeHtml(value) {
  return String(value)
    .replaceAll("&",  "&amp;")
    .replaceAll("<",  "&lt;")
    .replaceAll(">",  "&gt;")
    .replaceAll('"',  "&quot;");
}

function stringify(value) {
  return JSON.stringify(value, null, 2);
}

// ── Patient Context ────────────────────────────────────────────────────────
function renderPatientContext() {
  if (!state.currentPatientUuid) {
    patientContextEl.innerHTML = `<p class="no-patient">No patient selected yet.</p>`;
    patientContextEl.classList.remove("active");
    return;
  }
  patientContextEl.classList.add("active");
  patientContextEl.innerHTML = `
    <span class="status-dot"></span>
    <span class="patient-name">${escapeHtml(state.currentPatientDisplay || "Resolved patient")}</span>
    <p class="patient-uuid">${escapeHtml(state.currentPatientUuid)}</p>
  `;
  patientUuidEl.value = state.currentPatientUuid;
}

function selectPatient(uuid, display) {
  state.currentPatientUuid   = uuid;
  state.currentPatientDisplay = display;
  patientUuidEl.value = uuid;
  renderPatientContext();
  // Highlight selected row in browser
  document.querySelectorAll(".patient-row").forEach((row) => {
    row.classList.toggle("selected", row.dataset.uuid === uuid);
  });
}

function applySessionState(sessionState) {
  if (!sessionState) return;
  if (sessionState.id) {
    state.sessionId = sessionState.id;
    window.localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);
  }
  if (sessionState.current_patient_uuid) {
    state.currentPatientUuid   = sessionState.current_patient_uuid;
    state.currentPatientDisplay = sessionState.current_patient_display || "";
  }
  renderPatientContext();
}

// ── Session ────────────────────────────────────────────────────────────────
async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const res = await fetch("/api/chat/session", { method: "POST" });
  const body = await res.json();
  if (!res.ok || body.ok === false) throw new Error("Unable to create chat session.");
  applySessionState(body.data);
  return state.sessionId;
}

async function restoreSession() {
  if (!state.sessionId) return;
  const res = await fetch(`/api/chat/session/${encodeURIComponent(state.sessionId)}`);
  const body = await res.json();
  if (!res.ok || body.ok === false) {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    state.sessionId = "";
    return;
  }
  applySessionState(body.data);
}

// ── Patient Browser ────────────────────────────────────────────────────────
function renderPatientRow(patient) {
  const uuid    = patient.uuid || "";
  const display = patient.display || "Unknown patient";
  const identifiers = (patient.identifiers || []).map((id) => id.identifier).filter(Boolean).join(", ");

  const row = document.createElement("button");
  row.className = "patient-row";
  row.dataset.uuid = uuid;
  if (uuid === state.currentPatientUuid) row.classList.add("selected");
  row.innerHTML = `
    <span class="pr-name">${escapeHtml(display)}</span>
    ${identifiers ? `<span class="pr-id">${escapeHtml(identifiers)}</span>` : ""}
  `;
  row.addEventListener("click", () => selectPatient(uuid, display));
  return row;
}

function setPatientListLoading() {
  patientListEl.innerHTML = `
    <div class="browser-loading">
      <span class="loading-dot"></span>
      <span class="loading-dot"></span>
      <span class="loading-dot"></span>
    </div>`;
}

function setPatientListEmpty(query) {
  patientListEl.innerHTML = `<p class="browser-hint">No patients found${query ? ` for "<strong>${escapeHtml(query)}</strong>"` : ""}.</p>`;
}

async function loadPatients(query = "", startIndex = 0, append = false) {
  if (state.browser.loading) return;
  state.browser.loading = true;
  browseAllBtnEl.disabled = true;

  if (!append) setPatientListLoading();

  try {
    const params = new URLSearchParams({ limit: PAGE_SIZE, startIndex });
    if (query.trim()) params.set("q", query.trim());
    const res  = await fetch(`/api/patients?${params}`);
    const body = await res.json();
    if (!res.ok || body.ok === false) throw new Error(body.error || "Failed to fetch patients.");

    const patients  = body.data?.results || [];
    const hasMore   = body.data?.has_more || false;

    state.browser.query      = query;
    state.browser.startIndex = startIndex;
    state.browser.hasMore    = hasMore;

    if (!append) patientListEl.innerHTML = "";

    if (patients.length === 0 && !append) {
      setPatientListEmpty(query);
    } else {
      patients.forEach((p) => patientListEl.appendChild(renderPatientRow(p)));
    }

    paginationEl.style.display = hasMore ? "block" : "none";
  } catch (err) {
    patientListEl.innerHTML = `<p class="browser-hint error">⚠ ${escapeHtml(err.message)}</p>`;
    paginationEl.style.display = "none";
  } finally {
    state.browser.loading   = false;
    browseAllBtnEl.disabled = false;
  }
}

// Debounced live search
let searchDebounceTimer = null;
patientSearchEl.addEventListener("input", () => {
  clearTimeout(searchDebounceTimer);
  const q = patientSearchEl.value.trim();
  if (!q) {
    patientListEl.innerHTML = `<p class="browser-hint">Search above or click "All" to browse existing patients.</p>`;
    paginationEl.style.display = "none";
    return;
  }
  searchDebounceTimer = setTimeout(() => loadPatients(q, 0, false), 320);
});

browseAllBtnEl.addEventListener("click", () => {
  patientSearchEl.value = "";
  loadPatients("", 0, false);
});

loadMoreBtnEl.addEventListener("click", () => {
  loadPatients(state.browser.query, state.browser.startIndex + PAGE_SIZE, true);
});

// ── Chat rendering ─────────────────────────────────────────────────────────
function appendMessage(role, html) {
  const card = document.createElement("article");
  card.className = `message ${role}`;
  card.innerHTML = html;
  transcriptEl.appendChild(card);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return card;
}

function removeTypingIndicator() {
  document.getElementById("typing-indicator")?.remove();
}

function showTypingIndicator() {
  removeTypingIndicator();
  const el = document.createElement("div");
  el.className = "typing-indicator";
  el.id = "typing-indicator";
  el.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
  transcriptEl.appendChild(el);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function renderWorkflow(workflow = []) {
  if (!workflow.length) return "";
  return `
    <ul class="workflow">
      ${workflow.map((s) => `<li data-status="${escapeHtml(s.status)}"><strong>${escapeHtml(s.title)}</strong>: ${escapeHtml(s.detail)}</li>`).join("")}
    </ul>`;
}

function renderEvidence(evidence = []) {
  if (!evidence.length) return "";
  return `
    <details>
      <summary>Evidence (${evidence.length})</summary>
      <ul class="evidence-list">
        ${evidence.map((item) =>
          `<li><strong>${escapeHtml(item.label)}</strong> [${escapeHtml(item.resource_type)}${item.resource_uuid ? ` ${escapeHtml(item.resource_uuid)}` : ""}] ${escapeHtml(item.note)}</li>`
        ).join("")}
      </ul>
    </details>`;
}

function renderData(data) {
  if (data === undefined || data === null) return "";
  return `
    <details>
      <summary>Structured data</summary>
      <pre class="json-block">${escapeHtml(stringify(data))}</pre>
    </details>`;
}

function renderPendingAction(pendingAction) {
  if (!pendingAction) return "";
  const destructive = pendingAction.destructive ? "destructive" : "";
  const confirmInput = pendingAction.destructive
    ? `<input type="text" placeholder='Type DELETE to confirm' data-confirm-text="${escapeHtml(pendingAction.id)}" />`
    : "";
  return `
    <div class="pending-card ${destructive}">
      <strong>${escapeHtml(pendingAction.action)}</strong>
      <p class="muted" style="font-size:0.75rem;margin-top:4px;">${escapeHtml(pendingAction.endpoint)}</p>
      <pre class="json-block">${escapeHtml(stringify(pendingAction.payload || pendingAction.metadata || {}))}</pre>
      <div class="pending-actions">
        ${confirmInput}
        <button class="${pendingAction.destructive ? "danger" : "secondary"}"
          data-confirm-action="${escapeHtml(pendingAction.id)}"
          data-destructive="${String(pendingAction.destructive)}">
          ${pendingAction.destructive ? "⚠ Confirm Delete" : "✓ Confirm Action"}
        </button>
      </div>
    </div>`;
}

function renderAssistantResponse(payload) {
  removeTypingIndicator();

  if (payload.session_id) {
    state.sessionId = payload.session_id;
    window.localStorage.setItem(SESSION_STORAGE_KEY, payload.session_id);
  }
  applySessionState(payload.session_state);

  // Also update browser patient context if resolved
  const pc = payload.patient_context;
  if (pc?.uuid && !payload.session_state) {
    state.currentPatientUuid   = pc.uuid;
    state.currentPatientDisplay = pc.display || pc.uuid;
    renderPatientContext();
    // Reflect selection in the browser list
    document.querySelectorAll(".patient-row").forEach((row) => {
      row.classList.toggle("selected", row.dataset.uuid === pc.uuid);
    });
  }

  appendMessage(
    "assistant",
    `<div class="message-head">
      <span class="sender">MedPilot</span>
      <span class="intent-badge">${escapeHtml(payload.intent || "")}</span>
    </div>
    <div class="message-body">${escapeHtml(payload.message || "")}</div>
    ${payload.summary ? `<div class="message-summary">${escapeHtml(payload.summary)}</div>` : ""}
    ${renderWorkflow(payload.workflow)}
    ${renderEvidence(payload.evidence)}
    ${renderData(payload.data)}
    ${renderPendingAction(payload.pending_action)}`,
  );
}

// ── Chat API ───────────────────────────────────────────────────────────────
async function sendChat(prompt, file = null) {
  await ensureSession();
  const fd = new FormData();
  fd.append("prompt",     prompt);
  fd.append("session_id", state.sessionId);
  if (patientUuidEl.value.trim()) fd.append("patient_uuid", patientUuidEl.value.trim());
  if (file) fd.append("file", file);

  const res  = await fetch("/api/chat", { method: "POST", body: fd });
  const body = await res.json();
  if (!res.ok || body.ok === false) throw new Error(typeof body.error === "string" ? body.error : stringify(body.error));
  return body.data;
}

async function confirmAction(actionId, destructiveConfirmText = "") {
  const fd = new FormData();
  fd.append("action_id", actionId);
  if (destructiveConfirmText) fd.append("destructive_confirm_text", destructiveConfirmText);
  const res  = await fetch("/api/chat/confirm", { method: "POST", body: fd });
  const body = await res.json();
  if (!res.ok || body.ok === false) throw new Error(typeof body.error === "string" ? body.error : stringify(body.error));
  return body.data;
}

async function refreshLLMStatus() {
  try {
    const res  = await fetch("/api/llm/status");
    const body = await res.json();
    if (!res.ok || body.ok === false) return;
    const s = body.data;
    llmBadgeTextEl.textContent = (!s.enabled || s.provider === "none")
      ? "Deterministic reasoning"
      : `${s.provider}: ${s.model || "configured"}`;
  } catch {
    llmBadgeTextEl.textContent = "LLM status unavailable";
  }
}

// ── Event listeners ────────────────────────────────────────────────────────
document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  appendMessage(
    "user",
    `<div class="message-head"><span class="sender">Clinician</span></div>
     <div class="message-body">${escapeHtml(prompt)}</div>`,
  );
  promptEl.value = "";
  promptEl.style.height = "auto";
  showTypingIndicator();

  try {
    const payload = await sendChat(prompt, fileInputEl.files[0] || null);
    fileInputEl.value = "";
    uploadLabelEl.classList.remove("has-file");
    renderAssistantResponse(payload);
  } catch (err) {
    removeTypingIndicator();
    appendMessage("assistant", `<div class="message-head"><span class="sender">MedPilot</span></div><div class="message-body">⚠ ${escapeHtml(err.message)}</div>`);
  }
});

// Textarea auto-resize
promptEl.addEventListener("input", () => {
  promptEl.style.height = "auto";
  promptEl.style.height = Math.min(promptEl.scrollHeight, 160) + "px";
});

// Shift+Enter = new line, Enter = submit
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
});

// File badge
fileInputEl.addEventListener("change", () => {
  uploadLabelEl.classList.toggle("has-file", fileInputEl.files.length > 0);
});

// Confirm action delegation
transcriptEl.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-confirm-action]");
  if (!btn) return;
  const actionId     = btn.dataset.confirmAction;
  const destructive  = btn.dataset.destructive === "true";
  const container    = btn.closest(".pending-card");
  const input        = container?.querySelector(`[data-confirm-text="${actionId}"]`);
  const confirmText  = destructive ? (input?.value?.trim() || "") : "";

  showTypingIndicator();
  try {
    const payload = await confirmAction(actionId, confirmText);
    renderAssistantResponse(payload);
    btn.disabled = true;
    if (input) input.disabled = true;
  } catch (err) {
    removeTypingIndicator();
    appendMessage("assistant", `<div class="message-head"><span class="sender">MedPilot</span></div><div class="message-body">⚠ ${escapeHtml(err.message)}</div>`);
  }
});

// Quick action buttons
document.querySelectorAll(".quick-action").forEach((btn) => {
  btn.addEventListener("click", () => {
    promptEl.value = btn.dataset.prompt || "";
    promptEl.dispatchEvent(new Event("input"));
    promptEl.focus();
  });
});

// New chat
newChatBtnEl.addEventListener("click", () => {
  window.localStorage.removeItem(SESSION_STORAGE_KEY);
  state.sessionId            = "";
  state.currentPatientUuid   = "";
  state.currentPatientDisplay = "";
  patientUuidEl.value        = "";
  transcriptEl.innerHTML     = "";
  renderPatientContext();
  appendWelcome();
  ensureSession().catch(() => {});
});

// ── Welcome message ────────────────────────────────────────────────────────
function appendWelcome() {
  appendMessage(
    "assistant",
    `<div class="message-head"><span class="sender">MedPilot</span><span class="intent-badge">ready</span></div>
     <div class="message-body">
       Ask me to search, summarize, analyze, or update a patient chart. You can also pick a patient from the <strong>Patient Browser</strong> in the sidebar to instantly set context.
     </div>`,
  );
}

// ── Boot ───────────────────────────────────────────────────────────────────
appendWelcome();
renderPatientContext();
refreshLLMStatus();
(async () => {
  await restoreSession().catch(() => {});
  await ensureSession().catch(() => {});
})();
