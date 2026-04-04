"""
MedPilot — Streamlit Frontend
Clinical workflow copilot dashboard
"""
import streamlit as st
import requests
import json
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="MedPilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; color: #e0e0e0; }

    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    [data-testid="stChatInput"] input {
        background: #1e2535 !important;
        border: 1px solid #3d4a6b !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }

    [data-testid="stChatMessage"] {
        background: #1e2535;
        border-radius: 12px;
        border: 1px solid #2d3748;
        margin-bottom: 8px;
    }

    [data-testid="stMetric"] {
        background: #1e2535;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f8ef7, #7c5cbf);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(79, 142, 247, 0.4);
    }

    .success-banner {
        background: linear-gradient(135deg, #1a4731, #276749);
        border: 1px solid #38a169;
        border-radius: 10px;
        padding: 12px 16px;
        color: #9ae6b4;
        margin: 8px 0;
    }
    .error-banner {
        background: linear-gradient(135deg, #4a1a1a, #742020);
        border: 1px solid #e53e3e;
        border-radius: 10px;
        padding: 12px 16px;
        color: #fed7d7;
        margin: 8px 0;
    }
    .info-card {
        background: #1e2535;
        border: 1px solid #3d4a6b;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .medpilot-header {
        background: linear-gradient(135deg, #1a1f2e, #243047);
        border: 1px solid #3d4a6b;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_confirmation" not in st.session_state:
    st.session_state.pending_confirmation = None
if "current_patient" not in st.session_state:
    st.session_state.current_patient = None


# ── Helper Functions — defined BEFORE any page logic ─────────────────────────

def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def render_patient_card(patient: dict):
    name = patient.get("person", {}).get("display", "Unknown")
    pid = patient.get("display", "")
    gender = patient.get("person", {}).get("gender", "—")
    age = patient.get("person", {}).get("age", "—")
    st.markdown(f"""
    <div class="info-card">
        <h4 style="color:#7ec8f7;margin:0">👤 {name}</h4>
        <p style="color:#a0aec0;margin:4px 0 0">ID: {pid} &nbsp;|&nbsp; Gender: {gender} &nbsp;|&nbsp; Age: {age}</p>
    </div>
    """, unsafe_allow_html=True)


def render_data_table(items: list, label: str):
    if not items:
        st.info(f"No {label} found.")
        return
    rows = []
    for item in items:
        rows.append({"Display": item.get("display", str(item))})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_chat_data(data):
    """Render structured data returned from the chat API."""
    if not data:
        return
    if "patients" in data:
        patients = data["patients"]
        if patients:
            for p in patients[:5]:
                render_patient_card(p)
        else:
            st.info("No patients found.")
    elif "vitals" in data:
        render_data_table(data["vitals"], "vitals")
    elif "conditions" in data:
        render_data_table(data["conditions"], "conditions")
    elif "allergies" in data:
        render_data_table(data["allergies"], "allergies")
    elif "medications" in data:
        render_data_table(data["medications"], "medications")
    elif data.get("success") is True:
        st.markdown('<div class="success-banner">✅ Action completed successfully</div>', unsafe_allow_html=True)
    elif data.get("success") is False:
        st.markdown(f'<div class="error-banner">❌ {data.get("error", "Unknown error")}</div>', unsafe_allow_html=True)


def _handle_chat_response(resp: dict):
    """Append assistant response to session message history."""
    msg_text = resp.get("message", "Done.")
    st.session_state.messages.append({
        "role": "assistant",
        "content": msg_text,
        "data": resp.get("data")
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedPilot")
    st.markdown("*Clinical Workflow Copilot*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["💬 Chat", "🔍 Patient Search", "📄 PDF Ingest"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**OpenMRS Status**")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3)
        if health.status_code == 200:
            st.success("API Connected ✅")
        else:
            st.error("API Error ❌")
    except Exception:
        st.error("API Offline ❌")

    st.divider()
    if st.session_state.current_patient:
        p = st.session_state.current_patient
        name = p.get("person", {}).get("display", "Unknown") if isinstance(p, dict) else str(p)
        st.markdown(f"**Active Patient**\n\n`{name}`")
        if st.button("Clear Patient", use_container_width=True):
            st.session_state.current_patient = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":
    st.markdown("""
    <div class="medpilot-header">
        <span style="font-size:2rem">🤖</span>
        <div>
            <h2 style="margin:0;color:#7ec8f7">MedPilot Chat</h2>
            <p style="margin:0;color:#a0aec0;font-size:0.85rem">Natural language clinical workflow assistant</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 Example commands"):
        st.markdown("""
        - *"Search for patient John Smith"*
        - *"Show me vitals for patient 10003A6"*
        - *"Add allergy Penicillin with severe hives reaction for patient 10003A6"*
        - *"Add condition Hypertension for patient 10003A6"*
        - *"What medications does patient 10003A6 have?"*
        """)

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("data"):
                _render_chat_data(msg["data"])

    # Pending confirmation UI
    if st.session_state.pending_confirmation:
        pending = st.session_state.pending_confirmation
        st.warning(f"⚠️ **Confirmation Required:** {pending.get('confirmation_message', 'Confirm this action?')}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Proceed", use_container_width=True, key="confirm_yes"):
                confirmed_messages = st.session_state.messages.copy()
                confirmed_messages.append({
                    "role": "user",
                    "content": f"Yes, confirmed. Execute with params: {json.dumps({**pending.get('params', {}), 'confirmed': True})}"
                })
                resp = api_post("/api/chat", {"messages": [{"role": m["role"], "content": m["content"]} for m in confirmed_messages]})
                _handle_chat_response(resp)
                st.session_state.pending_confirmation = None
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True, key="confirm_no"):
                st.session_state.pending_confirmation = None
                st.session_state.messages.append({"role": "assistant", "content": "Action cancelled."})
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask MedPilot anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {"messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]}
                resp = api_post("/api/chat", payload)

            if "error" in resp:
                st.error(f"API Error: {resp['error']}")
            else:
                msg_text = resp.get("message", "")
                st.markdown(msg_text)

                if resp.get("requires_confirmation"):
                    st.session_state.pending_confirmation = resp
                else:
                    data = resp.get("data")
                    _render_chat_data(data)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": msg_text,
                        "data": data
                    })


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Patient Search":
    st.markdown("""
    <div class="medpilot-header">
        <span style="font-size:2rem">🔍</span>
        <div>
            <h2 style="margin:0;color:#7ec8f7">Patient Search</h2>
            <p style="margin:0;color:#a0aec0;font-size:0.85rem">Search, view, and manage patient records</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("Search by name or patient ID", placeholder="e.g. 10003A6 or John Smith")

    if query:
        with st.spinner("Searching..."):
            res = api_get(f"/api/patients/search?q={query}")
        patients = res.get("patients", [])

        if not patients:
            st.warning("No patients found.")
        else:
            st.success(f"Found {len(patients)} patient(s)")
            for p in patients:
                with st.expander(f"👤 {p.get('person', {}).get('display', 'Unknown')} — {p.get('display', '')}"):
                    uuid = p.get("uuid", "")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("📊 Vitals", key=f"v_{uuid}"):
                            vit = api_get(f"/api/vitals/{uuid}")
                            render_data_table(vit.get("vitals", []), "vitals")
                    with col2:
                        if st.button("🩺 Conditions", key=f"c_{uuid}"):
                            cond = api_get(f"/api/conditions/{uuid}")
                            render_data_table(cond.get("conditions", []), "conditions")
                    with col3:
                        if st.button("💊 Medications", key=f"m_{uuid}"):
                            meds = api_get(f"/api/medications/{uuid}")
                            render_data_table(meds.get("medications", []), "medications")

                    st.subheader("Allergies")
                    al = api_get(f"/api/allergies/{uuid}")
                    render_data_table(al.get("allergies", []), "allergies")

                    if st.button("Set as Active Patient", key=f"set_{uuid}"):
                        st.session_state.current_patient = p
                        st.success(f"Active patient set to {p.get('person', {}).get('display')}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PDF INGEST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 PDF Ingest":
    st.markdown("""
    <div class="medpilot-header">
        <span style="font-size:2rem">📄</span>
        <div>
            <h2 style="margin:0;color:#7ec8f7">PDF / EHR Ingestion</h2>
            <p style="margin:0;color:#a0aec0;font-size:0.85rem">Upload a patient record PDF and reconcile before committing</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # FIX: Declare patient_id at page scope so it's available to the commit section
    patient_id = st.text_input("Target Patient ID", placeholder="e.g. 10003A6", key="ingest_patient_id")

    col_upload, col_preview = st.columns([1, 2])

    with col_upload:
        st.subheader("1. Upload PDF")
        uploaded = st.file_uploader("Patient Record PDF", type=["pdf"], label_visibility="collapsed")

        if uploaded and st.button("📥 Parse PDF", use_container_width=True):
            with st.spinner("Parsing PDF with NLP..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                try:
                    r = requests.post(f"{API_BASE}/api/ingest/parse", files=files, timeout=30)
                    result = r.json()
                except Exception as e:
                    result = {"error": str(e)}

            if result.get("success"):
                st.session_state["parsed_pdf"] = result["parsed"]
                st.success("PDF parsed successfully!")
            else:
                st.error(f"Parse failed: {result.get('error', result.get('detail', 'Unknown error'))}")

    with col_preview:
        st.subheader("2. Review & Reconcile")

        if "parsed_pdf" in st.session_state:
            parsed = st.session_state["parsed_pdf"]

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["👤 Demographics", "🩺 Conditions", "💉 Vitals", "⚠️ Allergies", "💊 Medications"])

            with tab1:
                st.json({k: v for k, v in parsed.items() if k not in ["allergies", "conditions", "observations", "medications"]})

            with tab2:
                if parsed.get("conditions"):
                    st.dataframe(pd.DataFrame(parsed["conditions"]), use_container_width=True)
                else:
                    st.info("No conditions extracted.")

            with tab3:
                if parsed.get("observations"):
                    df = pd.DataFrame([{"Vital": k, "Value": v} for k, v in parsed["observations"].items()])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No vitals extracted.")

            with tab4:
                if parsed.get("allergies"):
                    st.dataframe(pd.DataFrame(parsed["allergies"]), use_container_width=True)
                else:
                    st.info("No allergies extracted.")

            with tab5:
                if parsed.get("medications"):
                    st.dataframe(pd.DataFrame(parsed["medications"]), use_container_width=True)
                else:
                    st.info("No medications extracted.")

            st.divider()
            st.subheader("3. Commit to OpenMRS")

            if not patient_id:
                st.warning("⚠️ Please enter the Target Patient ID (above) before committing.")
            else:
                st.warning(f"This will write the extracted data to patient **{patient_id}** in OpenMRS.")
                if st.button("✅ Confirm & Commit to OpenMRS", use_container_width=True):
                    with st.spinner("Writing to OpenMRS..."):
                        try:
                            r = requests.post(
                                f"{API_BASE}/api/ingest/commit",
                                params={"patient_id": patient_id},
                                json=parsed,
                                timeout=60
                            )
                            res = r.json()
                        except Exception as e:
                            res = {"error": str(e)}

                    if res.get("success"):
                        results = res.get("results", {})
                        st.success("✅ Data committed to OpenMRS!")
                        with st.expander("View commit results"):
                            st.json(results)
                        del st.session_state["parsed_pdf"]
                    else:
                        st.error(f"Commit failed: {res.get('error', res.get('detail', 'Unknown error'))}")
        else:
            st.markdown("""
            <div class="info-card" style="text-align:center;padding:40px">
                <p style="color:#a0aec0;font-size:1.1rem">Upload a PDF on the left to see the extracted data here</p>
            </div>
            """, unsafe_allow_html=True)
