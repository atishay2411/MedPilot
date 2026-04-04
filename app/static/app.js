// ── DOM Elements ──────────────────────────────────────────────────────
const transcriptEl = document.getElementById("transcript");
const promptEl = document.getElementById("prompt");
const fileInputEl = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const llmBadgeEl = document.getElementById("llm-badge");
const welcomeScreen = document.getElementById("welcome-screen");

// ── State ─────────────────────────────────────────────────────────────
const state = {
  hasMessages: false,
  conversationHistory: [], // [{role: "user"|"assistant", content: "..."}]
  currentPatientUuid: "",
};

function newChat() {
  state.hasMessages = false;
  state.conversationHistory = [];
  state.currentPatientUuid = "";
  transcriptEl.innerHTML = "";
  showWelcome();
}

// ── Utils ─────────────────────────────────────────────────────────────
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

/** Simple markdown-like rendering: **bold**, *italic*, line breaks */
function renderMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
}

// ── Welcome screen ────────────────────────────────────────────────────
function showWelcome() {
  if (welcomeScreen) {
    welcomeScreen.style.display = "flex";
    return;
  }
  // Rebuild if needed
  const welcome = document.createElement("div");
  welcome.id = "welcome-screen";
  welcome.className = "welcome-screen";
  welcome.innerHTML = `
    <div class="welcome-icon">🏥</div>
    <h2 class="welcome-title">How can I help you today?</h2>
    <p class="welcome-subtitle">I'm your AI clinical copilot. Ask me anything in plain language.</p>
  `;
  transcriptEl.prepend(welcome);
}

function hideWelcome() {
  const welcome = document.getElementById("welcome-screen");
  if (welcome) welcome.style.display = "none";
}

// ── Typing indicator ──────────────────────────────────────────────────
function showTypingIndicator() {
  const existing = document.getElementById("typing-indicator");
  if (existing) return;
  const indicator = document.createElement("div");
  indicator.id = "typing-indicator";
  indicator.className = "typing-indicator";
  indicator.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  transcriptEl.appendChild(indicator);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function removeTypingIndicator() {
  const indicator = document.getElementById("typing-indicator");
  if (indicator) indicator.remove();
}

// ── Message rendering ─────────────────────────────────────────────────
function appendMessage(role, content) {
  hideWelcome();
  state.hasMessages = true;
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
      ${workflow.map((step) => `<li data-status="${escapeHtml(step.status)}"><strong>${escapeHtml(step.title)}</strong>: ${escapeHtml(step.detail)}</li>`).join("")}
    </ul>
  `;
}

function renderEvidence(evidence = []) {
  if (!evidence.length) return "";
  return `
    <details>
      <summary>Evidence (${evidence.length})</summary>
      <ul class="evidence-list">
        ${evidence.map((item) => `<li><strong>${escapeHtml(item.label)}</strong> [${escapeHtml(item.resource_type)}] ${escapeHtml(item.note)}</li>`).join("")}
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
    ? `<input type="text" placeholder="Type DELETE to confirm" data-confirm-text="${escapeHtml(pendingAction.id)}" />`
    : "";
  return `
    <div class="pending-card ${destructive}">
      <strong>${escapeHtml(pendingAction.action)}</strong>
      <p class="muted" style="font-size: 0.72rem; margin-top: 4px;">${escapeHtml(pendingAction.endpoint)}</p>
      <pre class="json-block">${escapeHtml(stringify(pendingAction.payload || pendingAction.metadata || {}))}</pre>
      <div class="pending-actions">
        ${confirmInput}
        <button class="${pendingAction.destructive ? "danger" : "secondary"}" data-confirm-action="${escapeHtml(pendingAction.id)}" data-destructive="${String(pendingAction.destructive)}">
          ${pendingAction.destructive ? "⚠ Confirm Delete" : "✓ Confirm Action"}
        </button>
      </div>
    </div>
  `;
}

function renderAssistantResponse(payload) {
  const patientContext = payload.patient_context;
  if (patientContext?.uuid) {
    state.currentPatientUuid = patientContext.uuid;
  }

  const intentBadge = payload.intent && !["inform", "clarify"].includes(payload.intent)
    ? `<span class="intent-badge">${escapeHtml(payload.intent)}</span>`
    : "";

  appendMessage(
    "assistant",
    `
      <div class="message-head">
        <span class="sender">MedPilot</span>
        ${intentBadge}
      </div>
      <div class="message-body">${renderMarkdown(payload.message)}</div>
      ${payload.summary ? `<div class="message-summary">${renderMarkdown(payload.summary)}</div>` : ""}
      ${renderWorkflow(payload.workflow)}
      ${renderEvidence(payload.evidence)}
      ${renderData(payload.data)}
      ${renderPendingAction(payload.pending_action)}
    `,
  );
}

// ── API calls ─────────────────────────────────────────────────────────
async function sendChat(prompt, file = null) {
  const formData = new FormData();
  formData.append("prompt", prompt);
  if (state.conversationHistory.length > 0) {
    formData.append("history", JSON.stringify(state.conversationHistory.slice(-14)));
  }
  if (file) formData.append("file", file);
  if (state.currentPatientUuid) {
    formData.append("patient_uuid", state.currentPatientUuid);
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
  if (destructiveConfirmText) formData.append("destructive_confirm_text", destructiveConfirmText);
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
      llmBadgeEl.innerHTML = '<span class="dot" style="background: var(--accent-amber)"></span> LLM Not Configured';
      llmBadgeEl.style.color = "var(--accent-amber)";
      llmBadgeEl.style.borderColor = "rgba(245, 158, 11, 0.2)";
      llmBadgeEl.style.background = "rgba(245, 158, 11, 0.12)";
      return;
    }
    llmBadgeEl.innerHTML = `<span class="dot"></span> ${escapeHtml(status.provider)}: ${escapeHtml(status.model || "ready")}`;
  } catch (_error) {
    llmBadgeEl.innerHTML = '<span class="dot" style="background: var(--accent-red)"></span> Offline';
  }
}

// ── Auto-resize textarea ──────────────────────────────────────────────
function autoResize() {
  promptEl.style.height = "auto";
  promptEl.style.height = Math.min(promptEl.scrollHeight, 160) + "px";
}

promptEl.addEventListener("input", autoResize);

// ── File input indicator ──────────────────────────────────────────────
fileInputEl.addEventListener("change", () => {
  uploadBtn.classList.toggle("has-file", fileInputEl.files.length > 0);
});

// ── Chat form submit ──────────────────────────────────────────────────
document.getElementById("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  appendMessage(
    "user",
    `
      <div class="message-head">
        <span class="sender">You</span>
      </div>
      <div class="message-body">${renderMarkdown(prompt)}</div>
    `,
  );

  promptEl.value = "";
  autoResize();
  showTypingIndicator();

  try {
    const payload = await sendChat(prompt, fileInputEl.files[0] || null);
    removeTypingIndicator();
    renderAssistantResponse(payload);
    // Update conversation history for context
    state.conversationHistory.push({ role: "user", content: prompt });
    const assistantContent = payload.summary || payload.message || "";
    if (assistantContent) {
      state.conversationHistory.push({ role: "assistant", content: assistantContent });
    }
    fileInputEl.value = "";
    uploadBtn.classList.remove("has-file");
  } catch (error) {
    removeTypingIndicator();
    appendMessage(
      "assistant",
      `
        <div class="message-head"><span class="sender">MedPilot</span></div>
        <div class="message-body" style="color: var(--accent-red);">${escapeHtml(error.message)}</div>
      `,
    );
  }
});

// ── Enter to send, Shift+Enter for newline ────────────────────────────
promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
});

// ── Confirm action clicks ─────────────────────────────────────────────
transcriptEl.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-confirm-action]");
  if (!button) return;
  const actionId = button.dataset.confirmAction;
  const destructive = button.dataset.destructive === "true";
  const container = button.closest(".pending-card");
  const input = container?.querySelector(`[data-confirm-text="${actionId}"]`);
  const destructiveText = destructive ? input?.value?.trim() || "" : "";

  button.disabled = true;
  button.textContent = "Processing...";

  try {
    const payload = await confirmAction(actionId, destructiveText);
    renderAssistantResponse(payload);
  } catch (error) {
    appendMessage(
      "assistant",
      `
        <div class="message-head"><span class="sender">MedPilot</span></div>
        <div class="message-body" style="color: var(--accent-red);">${escapeHtml(error.message)}</div>
      `,
    );
    button.disabled = false;
    button.textContent = destructive ? "⚠ Confirm Delete" : "✓ Confirm Action";
  }
});

// ── Quick action & suggestion clicks ──────────────────────────────────
document.querySelectorAll(".quick-action, .welcome-suggestion").forEach((el) => {
  el.addEventListener("click", () => {
    const prompt = el.dataset.prompt;
    if (prompt) {
      promptEl.value = prompt;
      autoResize();
      promptEl.focus();
    }
  });
});

// ── New chat button ───────────────────────────────────────────────────
document.getElementById("new-chat-btn").addEventListener("click", newChat);

// ── Init ──────────────────────────────────────────────────────────────
refreshLLMStatus();
