from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.services.lookups import LookupService


class AllergyService:
    def __init__(self, client: OpenMRSClient, lookups: LookupService):
        self.client = client
        self.lookups = lookups

    def list_for_patient(self, patient_uuid: str) -> dict:
        return self.client.get("/ws/fhir2/R4/AllergyIntolerance", params={"patient": patient_uuid})

    def list_rest_for_patient(self, patient_uuid: str) -> dict:
        return self.client.get(f"/ws/rest/v1/patient/{patient_uuid}/allergy")

    def allergy_exists(self, patient_uuid: str, allergen_name: str) -> bool:
        allergen_uuid = self.lookups.resolve_uuid("concept", allergen_name)
        results = self.list_rest_for_patient(patient_uuid).get("results", [])
        return any(item.get("allergen", {}).get("codedAllergen", {}).get("uuid") == allergen_uuid for item in results)

    def build_rest_payload(self, allergen_name: str, severity: str, reaction: str, comment: str | None) -> dict:
        return {
            "allergen": {
                "allergenType": "DRUG",
                "codedAllergen": {"uuid": self.lookups.resolve_uuid("concept", allergen_name)},
            },
            "severity": {"uuid": self.lookups.resolve_uuid("concept", severity)},
            "reactions": [{"reaction": {"uuid": self.lookups.resolve_uuid("concept", reaction)}}],
            "comment": comment or f"Patient reports {reaction} to {allergen_name}.",
        }

    def create(self, patient_uuid: str, payload: dict) -> dict:
        return self.client.post(f"/ws/rest/v1/patient/{patient_uuid}/allergy", payload)

    def patch_severity(self, allergy_uuid: str, severity: str) -> dict:
        return self.client.patch(
            f"/ws/fhir2/R4/AllergyIntolerance/{allergy_uuid}",
            [{"op": "replace", "path": "/reaction/0/severity", "value": severity}],
        )

    def delete(self, allergy_uuid: str) -> dict:
        return self.client.delete(f"/ws/fhir2/R4/AllergyIntolerance/{allergy_uuid}")

    def find_by_allergen(self, patient_uuid: str, allergen_name: str) -> dict[str, Any] | None:
        entries = self.list_for_patient(patient_uuid).get("entry", [])
        normalized = allergen_name.lower()
        for entry in entries:
            resource = entry.get("resource", entry)
            display = ((resource.get("code") or {}).get("coding") or [{}])[0].get("display") or (resource.get("code") or {}).get("text", "")
            if display and normalized in display.lower():
                return resource
        return None
