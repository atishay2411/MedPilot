# MedPilot

MedPilot is a production-structured FastAPI application and static clinical operations UI implementing the MedPilot SRS v2.0 requirements for OpenMRS-centric patient workflows.

## What is included

- Full-stack application scaffold with modular backend services and a browser UI
- OpenMRS integrations for patient search, demographics, summary, conditions, observations, allergies, medications, encounters, and population counts
- Preview-gated patient creation and destructive delete confirmation
- PDF patient record parsing and ingestion pipeline
- Health Gorilla preview integration with configurable condition caps
- RBAC enforcement, append-only audit logging, retry logic, and testable service boundaries

## Structure

- `app/main.py`: FastAPI entrypoint
- `app/api/routes.py`: API endpoints
- `app/clients/`: OpenMRS and Health Gorilla clients
- `app/services/`: domain workflows
- `app/parsers/`: PDF ingestion parser
- `app/static/`: browser UI
- `tests/`: parser and identifier tests

## Run

1. Create a virtualenv and install dependencies with `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your OpenMRS and Health Gorilla credentials
3. Start the app with `uvicorn app.main:app --reload`
4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Notes

- The app is designed to talk to a live OpenMRS instance like the one described in [`/Users/atishayjain/Desktop/Hackathon/scarlethacks26/openmrs/docker-compose.yml`](/Users/atishayjain/Desktop/Hackathon/scarlethacks26/openmrs/docker-compose.yml).
- Audit logs are written to `data/audit/audit.log`.
- The generic write executor currently wires patient creation, condition creation, and condition deletion directly, while the ingestion flows use dedicated endpoints. The service modules already provide the rest of the CRUD primitives for straightforward route expansion.
