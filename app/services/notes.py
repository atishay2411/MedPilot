from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.services.utils import now_iso


_CLINICAL_NOTE_CONCEPT = "162169AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # FHIR obs text concept
_CHIEF_COMPLAINT_CONCEPT = "5219AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"   # Chief complaint
_CLINICAL_IMPRESSION_CONCEPT = "159395AAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # Clinical impression/note

_NOTE_CONCEPT_MAP = {
    "note": _CLINICAL_NOTE_CONCEPT,
    "clinical note": _CLINICAL_NOTE_CONCEPT,
    "progress note": _CLINICAL_NOTE_CONCEPT,
    "chief complaint": _CHIEF_COMPLAINT_CONCEPT,
    "complaint": _CHIEF_COMPLAINT_CONCEPT,
    "impression": _CLINICAL_IMPRESSION_CONCEPT,
    "clinical impression": _CLINICAL_IMPRESSION_CONCEPT,
    "assessment": _CLINICAL_IMPRESSION_CONCEPT,
}


class NotesService:
    def __init__(self, client: OpenMRSClient):
        self.client = client

    def _concept_uuid(self, note_type: str) -> str:
        return _NOTE_CONCEPT_MAP.get(note_type.lower().strip(), _CLINICAL_NOTE_CONCEPT)

    def build_fhir_payload(self, patient_uuid: str, note_text: str, note_type: str = "note") -> dict[str, Any]:
        concept_uuid = self._concept_uuid(note_type)
        return {
            "resourceType": "Observation",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "exam",
                            "display": "Exam",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {"code": concept_uuid, "display": note_type.title()},
                ],
                "text": note_type.title(),
            },
            "subject": {"reference": f"Patient/{patient_uuid}"},
            "effectiveDateTime": now_iso(),
            "issued": now_iso(),
            "valueString": note_text,
        }

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/ws/fhir2/R4/Observation", payload)

    def list_text_obs_for_patient(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get("/ws/fhir2/R4/Observation", params={
            "patient": patient_uuid,
            "code": _CLINICAL_NOTE_CONCEPT,
        })
