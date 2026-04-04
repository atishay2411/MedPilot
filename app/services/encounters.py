from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.models.domain import EncounterInput
from app.services.lookups import LookupService


class EncounterService:
    def __init__(self, client: OpenMRSClient, lookups: LookupService):
        self.client = client
        self.lookups = lookups

    def list_for_patient(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get("/ws/fhir2/R4/Encounter", params={"patient": patient_uuid})

    def build_rest_payload(self, payload: EncounterInput) -> dict[str, Any]:
        return {
            "encounterDatetime": payload.encounter_datetime,
            "patient": payload.patient_uuid,
            "encounterType": self.lookups.resolve_uuid("encountertype", payload.encounter_type_name),
            "location": self.lookups.resolve_uuid("location", payload.location_name),
            "encounterProviders": [
                {
                    "provider": self.lookups.resolve_uuid("provider", payload.provider_name),
                    "encounterRole": self.lookups.resolve_uuid("encounterrole", payload.encounter_role_name),
                }
            ],
        }

    def create_rest(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/ws/rest/v1/encounter", payload)
