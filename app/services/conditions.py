from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.core.exceptions import ValidationError
from app.services.lookups import LookupService


class ConditionService:
    def __init__(self, client: OpenMRSClient, lookups: LookupService):
        self.client = client
        self.lookups = lookups

    def list_for_patient(self, patient_uuid: str) -> dict:
        return self.client.get("/ws/fhir2/R4/Condition", params={"patient": patient_uuid})

    def build_create_payload(self, patient_uuid: str, condition_name: str, clinical_status: str, verification_status: str, onset_date: str | None) -> dict:
        return {
            "patient": patient_uuid,
            "condition": {"coded": self.lookups.resolve_uuid("concept", condition_name)},
            "clinicalStatus": clinical_status,
            "verificationStatus": verification_status,
            "onsetDate": onset_date,
        }

    def create(self, payload: dict) -> dict:
        return self.client.post("/ws/rest/v1/condition", payload)

    def patch_status(self, condition_uuid: str, status: str) -> dict:
        return self.client.patch(
            f"/ws/fhir2/R4/Condition/{condition_uuid}",
            [{"op": "replace", "path": "/clinicalStatus/coding/0/code", "value": status}],
        )

    def delete(self, condition_uuid: str) -> dict:
        return self.client.delete(f"/ws/fhir2/R4/Condition/{condition_uuid}")

    def create_concept(self, condition_name: str) -> dict:
        return self.client.post(
            "/ws/rest/v1/concept",
            {
                "names": [{"name": condition_name, "locale": "en", "conceptNameType": "FULLY_SPECIFIED", "localePreferred": True}],
                "datatype": "N/A",
                "conceptClass": "Diagnosis",
                "descriptions": [{"description": "Imported from Health Gorilla", "locale": "en"}],
            },
        )

    def find_by_name(self, patient_uuid: str, condition_name: str) -> dict[str, Any] | None:
        entries = self.list_for_patient(patient_uuid).get("entry", [])
        normalized = condition_name.lower()
        for entry in entries:
            resource = entry.get("resource", entry)
            display = ((resource.get("code") or {}).get("coding") or [{}])[0].get("display") or (resource.get("code") or {}).get("text", "")
            if display and normalized in display.lower():
                return resource
        return None

    def resolve_or_create_concept_uuid(self, condition_name: str) -> str:
        try:
            return self.lookups.resolve_uuid("concept", condition_name)
        except ValidationError:
            return self.create_concept(condition_name)["uuid"]
