# MedPilot

MedPilot is a production-structured FastAPI application with a chat-first clinician copilot UI implementing the MedPilot SRS v2.0 requirements for OpenMRS-centric patient workflows.

## What is included

- Full-stack application scaffold with modular backend services and a browser UI
- Chatbot-style clinician workspace with deterministic natural-language intent parsing and agentic workflow orchestration
- Provider-agnostic LLM adapter with OpenAI and Ollama support for intent reasoning and clinician-grade chart summarization
- OpenMRS integrations for patient search, demographics, summary, conditions, observations, allergies, medications, encounters, and population counts
- Evidence-grounded patient analysis and summarization
- Preview-gated write execution and destructive delete confirmation
- PDF patient record parsing and ingestion pipeline
- Health Gorilla preview and import integration with configurable condition caps
- RBAC enforcement, append-only audit logging, retry logic, and testable service boundaries

## Structure

- `app/main.py`: FastAPI entrypoint
- `app/api/routes.py`: API endpoints
- `app/clients/`: OpenMRS and Health Gorilla clients
- `app/llm/`: provider adapters and factory for OpenAI and Ollama
- `app/services/`: domain workflows
- `app/services/chat_agent.py`: chat orchestration and pending-action execution
- `app/services/llm_reasoning.py`: hybrid reasoning layer that upgrades intent extraction and patient summaries
- `app/services/prompt_parser.py`: natural-language intent and entity extraction
- `app/parsers/`: PDF ingestion parser
- `app/static/`: browser UI
- `tests/`: parser and identifier tests

## Run

1. Create a virtualenv and install dependencies with `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your OpenMRS and Health Gorilla credentials
3. Start the app with `uvicorn app.main:app --reload`
4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

## LLM configuration

MedPilot always works with deterministic parsing and summary logic. To enable model-backed reasoning, configure one provider:

### OpenAI

```env
MEDPILOT_LLM_PROVIDER=openai
MEDPILOT_LLM_MODEL=gpt-5.4-mini
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
MEDPILOT_LLM_REASONING_EFFORT=medium
```

### Ollama

```env
MEDPILOT_LLM_PROVIDER=ollama
MEDPILOT_LLM_MODEL=qwen2.5:14b
OLLAMA_BASE_URL=http://localhost:11434/api
MEDPILOT_LLM_REASONING_EFFORT=medium
```

Optional flags:

- `MEDPILOT_LLM_ENABLE_INTENT_REASONING=true`
- `MEDPILOT_LLM_ENABLE_SUMMARY_REASONING=true`
- `MEDPILOT_LLM_MAX_OUTPUT_TOKENS=2000`
- `MEDPILOT_LLM_TIMEOUT_SECONDS=30`

## Notes

- The app is designed to talk to a live OpenMRS instance like the one described in [`/Users/atishayjain/Desktop/Hackathon/scarlethacks26/openmrs/docker-compose.yml`](/Users/atishayjain/Desktop/Hackathon/scarlethacks26/openmrs/docker-compose.yml).
- Audit logs are written to `data/audit/audit.log`.
- The chat copilot supports natural-language search, chart summary/analysis, encounter creation, patient registration, condition/allergy workflows, vital recording, medication stop/order/dispense workflows, compound patient intake workflows, PDF ingestion, and Health Gorilla sync previews with confirmation-gated execution.
- The UI shows the active reasoning mode via `/api/llm/status`.
