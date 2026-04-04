from __future__ import annotations

from app.clients.openmrs import OpenMRSClient


class PopulationService:
    def __init__(self, client: OpenMRSClient):
        self.client = client

    def count_patients(self, query: str = "") -> dict:
        results = self.client.get("/ws/rest/v1/patient", params={"q": query or ""}).get("results", [])
        return {"label": "patients", "count": len(results)}

    def count_encounters(self, patient_uuid: str) -> dict:
        entries = self.client.get("/ws/fhir2/R4/Encounter", params={"patient": patient_uuid}).get("entry", [])
        return {"label": "encounters", "count": len(entries)}

    def count_by_condition(self, patient_uuid: str) -> dict:
        entries = self.client.get("/ws/fhir2/R4/Condition", params={"patient": patient_uuid}).get("entry", [])
        return {"label": "conditions", "count": len(entries)}
