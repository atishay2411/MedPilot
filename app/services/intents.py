from __future__ import annotations

from app.models.common import PendingWrite
from app.models.domain import PatientRegistration
from app.services.patients import PatientService


class IntentService:
    def __init__(self, patient_service: PatientService):
        self.patient_service = patient_service

    def classify(self, prompt: str) -> dict:
        lowered = prompt.lower()
        if "add patient" in lowered or "register" in lowered:
            return {"intent": "create_patient", "write": True}
        if "summary" in lowered:
            return {"intent": "patient_summary", "write": False}
        if "allergy" in lowered and "remove" in lowered:
            return {"intent": "delete_allergy", "write": True}
        if "condition" in lowered and "inactive" in lowered:
            return {"intent": "update_condition", "write": True}
        if "sync" in lowered and "health gorilla" in lowered:
            return {"intent": "sync_health_gorilla", "write": True}
        if "pdf" in lowered and "ingest" in lowered:
            return {"intent": "ingest_pdf", "write": True}
        return {"intent": "search_patient", "write": False}

    def preview_create_patient(self, registration: PatientRegistration) -> PendingWrite:
        payload = self.patient_service.build_create_payload(registration)
        duplicates = self.patient_service.find_duplicate_candidates(registration)
        return PendingWrite(
            intent="create_patient",
            action="Create Patient",
            permission="write:patient",
            endpoint="POST /ws/rest/v1/patient",
            payload=payload,
            duplicate_warnings=[f"Possible duplicate: {match}" for match in duplicates],
        )
