from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient


class PopulationService:
    def __init__(self, client: OpenMRSClient):
        self.client = client

    def count_patients(self, query: str = "") -> dict[str, Any]:
        if query.strip():
            bundle = self.client.get("/ws/fhir2/R4/Patient", params={"name": query.strip(), "_count": 1})
        else:
            bundle = self.client.get("/ws/fhir2/R4/Patient", params={"_count": 1})
        return {"label": "patients", "count": int(bundle.get("total", 0))}

    def count_encounters(self, patient_uuid: str) -> dict[str, Any]:
        entries = self.client.get("/ws/fhir2/R4/Encounter", params={"patient": patient_uuid}).get("entry", [])
        return {"label": "encounters", "count": len(entries)}

    def count_by_condition(self, patient_uuid: str) -> dict[str, Any]:
        entries = self.client.get("/ws/fhir2/R4/Condition", params={"patient": patient_uuid}).get("entry", [])
        return {"label": "conditions", "count": len(entries)}
