const logEl = document.getElementById("event-log");
const registrationPreviewEl = document.getElementById("registration-preview");
const confirmRegistrationBtn = document.getElementById("confirm-registration");
let pendingRegistration = null;

function logEvent(title, body) {
  const article = document.createElement("article");
  article.innerHTML = `<strong>${title}</strong><div>${body}</div>`;
  logEl.prepend(article);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(typeof data.error === "string" ? data.error : JSON.stringify(data.error));
  }
  return data.data;
}

document.getElementById("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = document.getElementById("search-query").value;
  const results = await api("/api/patients/search", { method: "POST", body: JSON.stringify({ query }) });
  document.getElementById("search-results").textContent = JSON.stringify(results, null, 2);
  logEvent("Patient search", `Returned ${results.length} candidate records.`);
});

document.getElementById("register-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = Object.fromEntries(form.entries());
  pendingRegistration = await api("/api/patients/preview", { method: "POST", body: JSON.stringify(payload) });
  registrationPreviewEl.textContent = JSON.stringify(pendingRegistration, null, 2);
  confirmRegistrationBtn.disabled = false;
  logEvent("Registration preview", "Write payload prepared and waiting for confirmation.");
});

confirmRegistrationBtn.addEventListener("click", async () => {
  if (!pendingRegistration) return;
  const result = await api("/api/writes/execute", {
    method: "POST",
    body: JSON.stringify({
      ...pendingRegistration,
      confirmed: true,
      prompt: "UI registration confirmation",
    }),
  });
  logEvent("Patient created", JSON.stringify(result, null, 2));
  confirmRegistrationBtn.disabled = true;
  pendingRegistration = null;
});

document.getElementById("preview-pdf").addEventListener("click", async () => {
  const fileInput = document.getElementById("pdf-file");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const response = await fetch("/api/ingestion/pdf/preview", { method: "POST", body: formData });
  const data = await response.json();
  document.getElementById("pdf-preview").textContent = JSON.stringify(data.data, null, 2);
  logEvent("PDF preview", "PDF parsed successfully.");
});

document.getElementById("pdf-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData();
  formData.append("patient_uuid", document.getElementById("pdf-patient-uuid").value);
  formData.append("confirmed", "true");
  formData.append("file", document.getElementById("pdf-file").files[0]);
  const response = await fetch("/api/ingestion/pdf/execute", { method: "POST", body: formData });
  const data = await response.json();
  document.getElementById("pdf-preview").textContent = JSON.stringify(data.data, null, 2);
  logEvent("PDF ingestion executed", `Processed ${data.data.length} entities.`);
});

document.getElementById("hg-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = Object.fromEntries(form.entries());
  const preview = await api("/api/health-gorilla/preview", { method: "POST", body: JSON.stringify(payload) });
  document.getElementById("hg-preview").textContent = JSON.stringify(preview, null, 2);
  logEvent("Health Gorilla preview", `Prepared ${preview.conditions.length} conditions for potential import.`);
});

document.getElementById("delete-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const conditionUuid = document.getElementById("delete-condition-uuid").value;
  const patientUuid = document.getElementById("delete-patient-uuid").value;
  const confirmText = document.getElementById("delete-confirm").value;
  const result = await api("/api/writes/execute", {
    method: "POST",
    body: JSON.stringify({
      intent: "delete_condition",
      action: "Delete Condition",
      permission: "delete:condition",
      endpoint: "DELETE /ws/fhir2/R4/Condition/{uuid}",
      payload: { condition_uuid: conditionUuid },
      confirmed: true,
      destructive: true,
      destructive_confirm_text: confirmText,
      patient_uuid: patientUuid,
      prompt: "UI destructive confirmation",
    }),
  });
  logEvent("Condition deleted", JSON.stringify(result, null, 2));
});
