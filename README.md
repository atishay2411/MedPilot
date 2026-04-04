# MedPilot 🏥

A clinical workflow copilot that lets clinicians manage OpenMRS patient records using **natural language**.

Built with: **Python FastAPI** · **Streamlit** · **OpenAI GPT-4o** · **OpenMRS REST API**

---

## Quick Start

### 1. Start OpenMRS

```bash
cd D:\CS595\LOF-CS595\labs\openmrs
docker-compose up -d
```

Wait ~60s for OpenMRS to be ready at `http://localhost:8080/openmrs`

---

### 2. Set Up Backend

```bash
cd backend
pip install -r requirements.txt
```

Create your `.env` file:

```bash
cp ../.env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Start the API:

```bash
cd ..
uvicorn backend.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

---

### 3. Set Up Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Dashboard available at: `http://localhost:8501`

---

## Features

| Feature | Description |
|---------|-------------|
| 💬 Natural Language Chat | Ask anything — search patients, add vitals, manage allergies |
| 🔍 Patient Search | Find patients by name or ID, view all clinical data |
| 📄 PDF Ingestion | Upload patient record PDFs, review extracted data, commit to OpenMRS |
| ✅ Confirmation Dialogs | All write/delete actions require explicit confirmation |
| 🔒 Deduplication | Prevents redundant allergies and medication orders |

## Project Structure

```
medpilot/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt
│   ├── services/
│   │   ├── openmrs.py       # OpenMRS REST client
│   │   ├── nlp.py           # OpenAI GPT intent resolver
│   │   └── pdf_parser.py    # PDF → structured data
│   └── routers/
│       ├── chat.py          # /api/chat
│       ├── patients.py      # /api/patients
│       ├── vitals.py        # /api/vitals
│       ├── conditions.py    # /api/conditions
│       ├── allergies.py     # /api/allergies
│       ├── medications.py   # /api/medications
│       └── ingest.py        # /api/ingest
└── frontend/
    ├── app.py               # Streamlit dashboard
    └── requirements.txt
```

## OpenMRS Connection

- **URL:** `http://localhost:8080/openmrs`
- **Username:** `admin`
- **Password:** `Admin123`
