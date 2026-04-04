from __future__ import annotations

from app.clients.openmrs import OpenMRSClient
from app.models.domain import ObservationInput, ObservationUpdateInput


VITALS_CODE_MAP = {
    "Height": "5090",
    "Weight (kg)": "5089",
    "Temperature": "5088",
    "Respiratory Rate": "5242",
    "Oxygen Saturation (SpO2)": "5092",
    "Systolic blood pressure": "5085",
    "Diastolic blood pressure": "5086",
}


class ObservationService:
    def __init__(self, client: OpenMRSClient):
        self.client = client

    def list_for_patient(self, patient_uuid: str) -> dict:
        return self.client.get("/ws/fhir2/R4/Observation", params={"patient": patient_uuid})

    def get(self, observation_uuid: str) -> dict:
        return self.client.get(f"/ws/fhir2/R4/Observation/{observation_uuid}")

    def build_fhir_payload(self, payload: ObservationInput) -> dict:
        return {
            "resourceType": "Observation",
            "status": "final",
            "code": {"coding": [{"system": "http://openmrs.org/fhir/codesystem/observation-codes", "code": payload.code, "display": payload.display}]},
            "subject": {"reference": f"Patient/{payload.patient_uuid}"},
            "effectiveDateTime": payload.effective_datetime,
            "valueQuantity": {"value": payload.value, "unit": payload.unit},
        }

    def create(self, payload: dict) -> dict:
        return self.client.post("/ws/fhir2/R4/Observation", payload)

    def update(self, payload: ObservationUpdateInput) -> dict:
        resource = self.build_fhir_payload(payload)
        resource["id"] = payload.observation_uuid
        return self.client.put(f"/ws/fhir2/R4/Observation/{payload.observation_uuid}", resource)

    def delete(self, observation_uuid: str) -> dict:
        return self.client.delete(f"/ws/fhir2/R4/Observation/{observation_uuid}")
