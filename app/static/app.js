const transcriptEl = document.getElementById("transcript");
const promptEl = document.getElementById("prompt");
const patientUuidEl = document.getElementById("patient-uuid");
const fileInputEl = document.getElementById("file-input");
const patientContextEl = document.getElementById("patient-context");
const llmBadgeEl = document.getElementById("llm-badge");
const SESSION_STORAGE_KEY = "medpilot-session-id";

const state = {
  sessionId: window.localStorage.getItem(SESSION_STORAGE_KEY) || "",
  currentPatientUuid: "",
  currentPatientDisplay: "",
};

function renderPatientContext() {
  if (!state.currentPatientUuid) {
    patientContextEl.innerHTML = "<p>No patient selected yet.</p>";
    return;
  }
  patientContextEl.innerHTML = `
    <strong>${escapeHtml(state.currentPatientDisplay || "Resolved patient")}</strong>
    <p class="muted">UUID: ${escapeHtml(state.currentPatientUuid)}</p>
  `;
  patientUuidEl.value = state.currentPatientUuid;
}

function applySessionState(sessionState) {
  if (!sessionState) return;
  if (sessionState.id) {
    state.sessionId = sessionState.id;
    window.localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);
  }
  state.currentPatientUuid = sessionState.current_patient_uuid || "";
  state.currentPatientDisplay = sessionState.current_patient_display || "";
  renderPatientContext();
}

async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const response = await fetch("/api/chat/session", { method: "POST" });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error("Unable to create chat session.");
  }
  applySessionState(payload.data);
  return state.sessionId;
}

async function restoreSession() {
  if (!state.sessionId) return;
  const response = await fetch(`/api/chat/session/${encodeURIComponent(state.sessionId)}`);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    state.sessionId = "";
    return;
  }
  applySessionState(payload.data);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function stringify(value) {
  return JSON.stringify(value, null, 2);
}

function appendMessage(role, content) {
  const card = document.createElement("article");
  card.className = `message ${role}`;
  card.innerHTML = content;
  transcriptEl.appendChild(card);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function renderWorkflow(workflow = []) {
  if (!workflow.length) return "";
  return `
    <ul class="workflow">
      ${workflow.map((step) => `<li><strong>${escapeHtml(step.title)}</strong>: ${escapeHtml(step.detail)} <span class="muted">(${escapeHtml(step.status)})</span></li>`).join("")}
    </ul>
  `;
}

function renderEvidence(evidence = []) {
  if (!evidence.length) return "";
  return `
    <details>
      <summary>Evidence</summary>
      <ul class="evidence-list">
        ${evidence
          .map(
            (item) =>
              `<li><strong>${escapeHtml(item.label)}</strong> [${escapeHtml(item.resource_type)}${item.resource_uuid ? ` ${escapeHtml(item.resource_uuid)}` : ""}] ${escapeHtml(item.note)}</li>`,
          )
          .join("")}
      </ul>
    </details>
  `;
}

function renderData(data) {
  if (data === undefined || data === null) return "";
  return `
    <details>
      <summary>Structured data</summary>
      <pre class="json-block">${escapeHtml(stringify(data))}</pre>
    </details>
  `;
}

function renderPendingAction(pendingAction) {
  if (!pendingAction) return "";
  const destructive = pendingAction.destructive ? "destructive" : "";
  const confirmInput = pendingAction.destructive
    ? `<input type="text" placeholder="Type DELETE" data-confirm-text="${escapeHtml(pendingAction.id)}" />`
    : "";
  return `
    <div class="pending-card ${destructive}">
      <strong>${escapeHtml(pendingAction.action)}</strong>
      <p class="muted">${escapeHtml(pendingAction.endpoint)}</p>
      <pre class="json-block">${escapeHtml(stringify(pendingAction.payload || pendingAction.metadata || {}))}</pre>
      <div class="pending-actions">
        ${confirmInput}
        <button class="${pendingAction.destructive ? "danger" : "secondary"}" data-confirm-action="${escapeHtml(pendingAction.id)}" data-destructive="${String(pendingAction.destructive)}">
          ${pendingAction.destructive ? "Confirm Delete" : "Confirm Action"}
        </button>
      </div>
    </div>
  `;
}

function renderAssistantResponse(payload) {
  if (payload.session_id) {
    state.sessionId = payload.session_id;
    window.localStorage.setItem(SESSION_STORAGE_KEY, payload.session_id);
  }
  applySessionState(payload.session_state);
  const patientContext = payload.patient_context;
  if (patientContext?.uuid && !payload.session_state) {
    state.currentPatientUuid = patientContext.uuid;
    state.currentPatientDisplay = patientContext.display || patientContext.uuid;
    renderPatientContext();
  }

  appendMessage(
    "assistant",
    `
      <div class="message-head">
        <strong>MedPilot</strong>
        <span>${escapeHtml(payload.intent)}</span>
      </div>
      <div>${escapeHtml(payload.message)}</div>
      ${payload.summary ? `<div class="message-summary">${escapeHtml(payload.summary)}</div>` : ""}
      ${renderWorkflow(payload.workflow)}
      ${renderEvidence(payload.evidence)}
      ${renderData(payload.data)}
      ${renderPendingAction(payload.pending_action)}
    `,
  );
}

async function sendChat(prompt, file = null) {
  await ensureSession();
  const formData = new FormData();
  formData.append("prompt", prompt);
  formData.append("session_id", state.sessionId);
  if (patientUuidEl.value.trim()) {
    formData.append("patient_uuid", patientUuidEl.value.trim());
  }
  if (file) {
    formData.append("file", file);
  }

  const response = await fetch("/api/chat", { method: "POST", body: formData });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(typeof payload.error === "string" ? payload.error : stringify(payload.error));
  }
  return payload.data;
}

async function confirmAction(actionId, destructiveConfirmText = "") {
  const formData = new FormData();
  formData.append("action_id", actionId);
  if (destructiveConfirmText) {
    formData.append("destructive_confirm_text", destructiveConfirmText);
  }
  const response = await fetch("/api/chat/confirm", { method: "POST", body: formData });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(typeof payload.error === "string" ? payload.error : stringify(payload.error));
  }
  return payload.data;
}

async function refreshLLMStatus() {
  try {
    const response = await fetch("/api/llm/status");
    const payload = await response.json();
    if (!response.ok || payload.ok === false) return;
    const status = payload.data;
    if (!status.enabled || status.provider === "none") {
      llmBadgeEl.textContent = "Deterministic reasoning active";
      return;
    }
    llmBadgeEl.textContent = `${status.provider}: ${status.model || "configured"}`;
  } catch (_error) {
    llmBadgeEl.textContent = "LLM status unavailable";
  }
}

document.getElementById("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  appendMessage(
    "user",
    `
      <div class="message-head">
        <strong>Clinician</strong>
      </div>
      <div>${escapeHtml(prompt)}</div>
    `,
  );

  try {
    const payload = await sendChat(prompt, fileInputEl.files[0] || null);
    renderAssistantResponse(payload);
    promptEl.value = "";
    fileInputEl.value = "";
  } catch (error) {
    appendMessage("assistant", `<div class="message-head"><strong>MedPilot</strong></div><div>${escapeHtml(error.message)}</div>`);
  }
});

transcriptEl.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-confirm-action]");
  if (!button) return;
  const actionId = button.dataset.confirmAction;
  const destructive = button.dataset.destructive === "true";
  const container = button.closest(".pending-card");
  const input = container?.querySelector(`[data-confirm-text="${actionId}"]`);
  const destructiveText = destructive ? input?.value?.trim() || "" : "";

  try {
    const payload = await confirmAction(actionId, destructiveText);
    renderAssistantResponse(payload);
    button.disabled = true;
  } catch (error) {
    appendMessage("assistant", `<div class="message-head"><strong>MedPilot</strong></div><div>${escapeHtml(error.message)}</div>`);
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    promptEl.value = chip.dataset.prompt || "";
    promptEl.focus();
  });
});

appendMessage(
  "assistant",
  `
    <div class="message-head">
      <strong>MedPilot</strong>
      <span>ready</span>
    </div>
    <div>Ask for patient search, chart summarization, condition updates, patient intake, or patient switching like “change patient to Maria Santos.” I will keep session memory, reuse the active patient when you say “this patient,” and prepare confirmation-gated workflows for writes.</div>
  `,
);

renderPatientContext();
refreshLLMStatus();
(async () => {
  await restoreSession().catch(() => {});
  await ensureSession().catch(() => {});
})();
