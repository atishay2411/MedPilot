# MedPilot

MedPilot is a chat-first clinical copilot for OpenMRS. It combines a FastAPI backend, a lightweight browser UI, deterministic intent routing, optional LLM reasoning, and direct OpenMRS REST/FHIR integrations so clinicians can search patients, review charts, record data, and prepare write actions from natural-language prompts.

This repository is more than a chatbot demo. It includes:

- a full backend service layer for patient, observation, condition, allergy, medication, encounter, visit, appointment, relationship, and lab-order workflows
- evidence-grounded patient summaries
- PDF ingestion for structured patient record import
- Health Gorilla preview/import support
- confirmation-gated writes, destructive-action safeguards, RBAC checks, and audit logging
- configured filesystem-backed paths for audit and chat/session artifacts

## Why this project exists

OpenMRS is powerful, but many clinical workflows still require multiple clicks, context switching, and careful navigation between modules. MedPilot provides a single conversational workspace that can:

- find the right patient quickly
- keep chart context across follow-up prompts
- summarize the chart using current data
- prepare write operations safely before anything is submitted
- expose OpenMRS capabilities through a cleaner, more discoverable interface

## Core capabilities

MedPilot currently supports these workflow groups:

- Patient search, patient counts, identifier lookup, patient switching, registration, update, and deletion
- Chart review for vitals/observations, conditions, allergies, medications, encounters, visits, and program-related lookups
- Clinical writes for vitals, conditions, allergies, medications, encounters, notes, relationships, appointments, and lab orders
- Patient intake workflows that bundle registration plus chart data creation
- PDF preview and ingestion for patient records
- Health Gorilla patient/condition preview and import workflows
- FHIR metadata inspection

## How the app works

1. A user sends a free-form message from the UI or `POST /api/chat`.
2. MedPilot first tries a deterministic classifier for common workflows.
3. If needed, it falls back to an LLM provider for richer intent extraction, clarification handling, and summary generation.
4. Domain services translate the request into OpenMRS REST/FHIR operations.
5. Write actions are returned as pending confirmations before execution.
6. Audit events and runtime filesystem artifacts are stored under `data/`.

## Architecture

### Backend

- `FastAPI` serves both the API and the static frontend
- `app/services/` contains the domain logic and chat orchestration
- `app/clients/` contains OpenMRS and Health Gorilla HTTP clients
- `app/llm/` abstracts provider-specific model calls

### Frontend

- `app/static/index.html` is the single-page chat shell
- `app/static/app.js` handles chat state, rendering, uploads, and confirmations
- `app/static/styles.css` provides the UI styling

### External systems

- `OpenMRS` is the primary system of record
- `Health Gorilla` is optional and used for patient/condition preview/import
- `OpenAI`, `Ollama`, or `Anthropic` can be configured for reasoning

## Repository structure

```text
.
├── app/
│   ├── api/              # FastAPI routes
│   ├── clients/          # OpenMRS and Health Gorilla API clients
│   ├── core/             # audit logging, security, confirmations, exceptions
│   ├── llm/              # provider abstraction, factory, and provider implementations
│   ├── models/           # Pydantic request/response/domain models
│   ├── parsers/          # PDF parsing logic
│   ├── services/         # chat agent and clinical workflow services
│   ├── static/           # browser UI served directly by FastAPI
│   └── main.py           # application entrypoint
├── data/
│   ├── audit/            # append-only audit log output
│   └── chat/             # configured chat/session storage path
├── Documentation/        # supporting product/spec documentation
├── openmrs/              # local OpenMRS compose file and integration assets
├── tests/                # unit tests
├── requirements.txt      # Python dependencies
└── README.md
```

## Prerequisites

Before you run MedPilot, make sure you have:

- Python 3.10 or newer
- `pip`
- Docker and Docker Compose if you want to use the provided OpenMRS stack
- an accessible OpenMRS instance
- optional LLM credentials or a local Ollama server if you want richer conversational behavior
- an optional Health Gorilla sandbox token if you want to test that integration

## Setup

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file and adjust it for your environment:

```bash
cp .env.example .env
```

Important settings:

- `OPENMRS_BASE_URL`, `OPENMRS_USERNAME`, `OPENMRS_PASSWORD`
- `MEDPILOT_USER_ROLE` for the default app actor
- `MEDPILOT_LLM_PROVIDER` and the corresponding provider credentials if you want LLM-backed reasoning
- `HEALTH_GORILLA_TOKEN` if you want external lab-condition preview/import

### 3. Start OpenMRS

You have two options:

#### Option A: Use an existing OpenMRS instance

Point `OPENMRS_BASE_URL` and credentials in `.env` to your running environment.

#### Option B: Use the included OpenMRS Docker stack

```bash
cd openmrs
docker compose up -d
```

Important: the provided `openmrs/docker-compose.yml` does not provision MySQL. It expects a MySQL database named `openmrs` to already be reachable from the containers at `host.docker.internal` with:

- username: `root`
- password: `password`

If you do not already have that database running locally, either:

- adapt the compose file to include MySQL, or
- point MedPilot at another OpenMRS instance

Default ports in the provided stack:

- OpenMRS gateway: `http://localhost`
- OpenMRS backend: `http://localhost:8080/openmrs`

### 4. Run MedPilot

From the repo root:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- App UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- FastAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## LLM configuration

MedPilot has a deterministic classifier for many common actions, but the best conversational experience comes from configuring an LLM provider. Without one, common patterned prompts can still work, but broader free-form requests and richer summaries may fall back to a "provider not configured" response.

### OpenAI

```env
MEDPILOT_LLM_PROVIDER=openai
MEDPILOT_LLM_MODEL=gpt-5.4-mini
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Ollama

```env
MEDPILOT_LLM_PROVIDER=ollama
MEDPILOT_LLM_MODEL=qwen2.5:14b
OLLAMA_BASE_URL=http://localhost:11434/api
```

### Anthropic

```env
MEDPILOT_LLM_PROVIDER=anthropic
MEDPILOT_LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=your_api_key_here
```

Useful optional toggles:

```env
MEDPILOT_LLM_ENABLE_INTENT_REASONING=true
MEDPILOT_LLM_ENABLE_SUMMARY_REASONING=true
MEDPILOT_LLM_TIMEOUT_SECONDS=30
MEDPILOT_LLM_MAX_OUTPUT_TOKENS=2000
MEDPILOT_LLM_REASONING_EFFORT=medium
```

## End-to-end run guide

This is the fastest way to validate the whole project flow.

### 1. Confirm the app is healthy

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/llm/status
```

### 2. Open the UI

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) and use the chat interface.

### 3. Try a patient search

Example prompts:

- `Find patient Maria Santos`
- `Find patient with id 10001D`
- `List all patients`
- `How many patients are there?`

### 4. Switch or confirm patient context

Example prompts:

- `Change patient to Maria Santos`
- `Open chart for John Doe`
- `Summarize this patient`

### 5. Read chart data

Example prompts:

- `Show their vitals`
- `Show their conditions`
- `Show their allergies`
- `Show their medications`
- `Show encounters for this patient`
- `Show visits for this patient`
- `Show appointments for this patient`
- `Show lab orders for this patient`
- `Show family members for this patient`

### 6. Prepare a write action

Example prompts:

- `Record blood pressure 140/90 for this patient`
- `Add condition diabetes for this patient`
- `Add penicillin allergy for this patient`
- `Prescribe metformin 500 mg orally twice daily for 30 days for this patient`
- `Book a General Medicine appointment for this patient tomorrow at 9am`
- `Order a CBC for this patient`
- `Add note for this patient: Follow-up visit for chest pain`

Writes are preview-gated before execution:

- standard writes require explicit confirmation
- destructive actions require typing `DELETE`

### 7. Test PDF ingestion

You can attach a PDF in the chat UI or use the sample asset under `openmrs/EHR Lab/patient_record.pdf`.

Typical flow:

1. open a patient chart
2. attach the PDF
3. ask MedPilot to ingest it for the active patient
4. review the preview/confirmation
5. confirm execution

### 8. Inspect runtime artifacts

Runtime artifacts are stored locally:

- audit log: `data/audit/audit.log`
- configured chat/session storage path: `data/chat/sessions/`

## Useful shell commands

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app locally

```bash
uvicorn app.main:app --reload
```

### Run the test suite

```bash
PYTHONPATH=. pytest -q
```

### Run a deterministic-classifier smoke test

```bash
python test_classifier.py
```

### Start the included OpenMRS services

```bash
cd openmrs
docker compose up -d
```

## API overview

FastAPI automatically exposes interactive docs at `/docs`. The most important endpoints are:

- `GET /api/health` - app health check
- `GET /api/llm/status` - active LLM/provider status
- `POST /api/chat` - primary chat endpoint with optional file upload
- `POST /api/chat/confirm` - confirm pending write actions
- `POST /api/intent` - direct intent classification
- `POST /api/patients/search` - patient search
- `GET /api/patients/{patient_uuid}/summary` - structured patient summary
- `POST /api/observations`, `PUT /api/observations`, `DELETE /api/observations/{id}` - observation CRUD
- `POST /api/conditions`, `PATCH /api/conditions/{id}`, `DELETE /api/conditions/{id}` - condition CRUD
- `POST /api/allergies`, `PATCH /api/allergies/{id}`, `DELETE /api/allergies/{id}` - allergy CRUD
- `POST /api/medications`, `PATCH /api/medications/{id}`, `POST /api/medications/dispense` - medication workflows
- `POST /api/encounters` - encounter creation
- `POST /api/ingestion/pdf/preview` and `POST /api/ingestion/pdf/execute` - PDF parsing and ingestion
- `POST /api/health-gorilla/preview` - Health Gorilla preview

Not every workflow is exposed as a dedicated standalone route yet. Some capabilities, especially the richer orchestration flows, are currently driven through the chat agent and service layer.

### Example chat request

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -F "prompt=Find patient Maria Santos"
```

### Example PDF preview request

```bash
curl -X POST http://127.0.0.1:8000/api/ingestion/pdf/preview \
  -F "file=@openmrs/EHR Lab/patient_record.pdf"
```

## Safety model

MedPilot includes several safeguards by design:

- role-based permission enforcement in `app/core/security.py`
- confirmation gates before writes
- destructive confirmation text for delete operations
- audit logging with redaction in `app/core/audit.py`
- active patient context during a conversation to reduce context-loss mistakes

## Testing

The repository includes unit tests for:

- deterministic classification
- PDF parsing
- chat-agent handlers
- summary generation
- service-layer behaviors
- scope isolation rules

Run all tests with:

```bash
PYTHONPATH=. pytest -q
```

If you see `ModuleNotFoundError: No module named 'app'`, it means the repo root is not on `PYTHONPATH`.

## Troubleshooting

### The UI shows "LLM Not Configured"

That means the app started successfully, but the selected provider is disabled or missing credentials. Check:

- `MEDPILOT_LLM_PROVIDER`
- provider API key or local endpoint
- `MEDPILOT_LLM_MODEL`

### OpenMRS requests fail

Verify:

- `OPENMRS_BASE_URL`
- OpenMRS username/password
- the OpenMRS server is reachable
- the OpenMRS instance supports the REST/FHIR endpoints MedPilot uses

### Docker OpenMRS starts but backend cannot connect

The bundled compose file assumes a host-side MySQL database. If that database is missing, the OpenMRS backend container will not initialize correctly.

### PDF ingestion is not working

Check:

- the uploaded file is a real PDF
- the PDF contains extractable text
- the active patient is set before executing ingestion

## Development notes

- The frontend is static and does not require a separate build step.
- Session memory is file-backed, not database-backed.
- Audit logs are append-only text records with basic field redaction.
- The repo includes sample OpenMRS-related assets under `openmrs/` that are useful for local experimentation and parser tests.

## License

This project is distributed under the GNU General Public License v3. See `LICENSE` for details.
