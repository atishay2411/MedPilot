from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.models.domain import ObservationInput, ObservationUpdateInput


# Short concept codes → full CIEL concept UUID (padded to 36 chars with 'A')
# Format: shortCode.ljust(36, 'A')
VITALS_CODE_MAP: dict[str, str] = {
    "Height (cm)": "5090",
    "Height": "5090",
    "Weight (kg)": "5089",
    "Weight": "5089",
    "Temperature (°C)": "5088",
    "Temperature": "5088",
    "Respiratory Rate": "5242",
    "Respiratory rate": "5242",
    "Oxygen Saturation (SpO2)": "5092",
    "Oxygen saturation": "5092",
    "SpO2": "5092",
    "Systolic blood pressure": "5085",
    "Systolic Blood Pressure": "5085",
    "Diastolic blood pressure": "5086",
    "Diastolic Blood Pressure": "5086",
}

# Observation display → FHIR category code
_CATEGORY_MAP: dict[str, str] = {
    "Height (cm)": "exam",
    "Height": "exam",
    "Weight (kg)": "exam",
    "Weight": "exam",
}
_DEFAULT_CATEGORY = "vital-signs"

# Unit → UCUM code
_UNIT_CODE_MAP: dict[str, str] = {
    "cm": "cm",
    "kg": "kg",
    "°C": "Cel",
    "C": "Cel",
    "Celsius": "Cel",
    "breaths/min": "/min",
    "breaths per minute": "/min",
    "%": "%",
    "mmHg": "mm[Hg]",
}


def _long_code(short_code: str) -> str:
    """Pad a short CIEL concept code to 36 characters with trailing A's."""
    if len(short_code) >= 36:
        return short_code
    return short_code + "A" * (36 - len(short_code))


class ObservationService:
    def __init__(self, client: OpenMRSClient):
        self.client = client

    def list_for_patient(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get("/ws/fhir2/R4/Observation", params={"patient": patient_uuid})

    def get(self, observation_uuid: str) -> dict[str, Any]:
        return self.client.get(f"/ws/fhir2/R4/Observation/{observation_uuid}")

    def build_fhir_payload(self, payload: ObservationInput) -> dict[str, Any]:
        """Build a FHIR Observation resource that OpenMRS FHIR2 accepts.

        Uses the dual-coding pattern required by OpenMRS:
        - First coding: full CIEL UUID (shortCode padded to 36 chars with A's)
        - Second coding: short CIEL code with cielterminology.org system
        """
        display = payload.display or ""
        short_code = payload.code

        # Ensure we have the correct short code for known vitals
        if display in VITALS_CODE_MAP:
            short_code = VITALS_CODE_MAP[display]
        elif short_code not in VITALS_CODE_MAP.values():
            # Keep whatever code was passed (custom observations)
            pass

        full_code = _long_code(short_code)
        category_code = _CATEGORY_MAP.get(display, _DEFAULT_CATEGORY)
        category_display = "Exam" if category_code == "exam" else "Vital Signs"
        unit = payload.unit or ""
        ucum_code = _UNIT_CODE_MAP.get(unit, unit)

        return {
            "resourceType": "Observation",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": category_code,
                            "display": category_display,
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "code": full_code,
                        "display": display,
                    },
                    {
                        "system": "https://cielterminology.org",
                        "code": short_code,
                    },
                ],
                "text": display,
            },
            "subject": {"reference": f"Patient/{payload.patient_uuid}"},
            "effectiveDateTime": payload.effective_datetime,
            "issued": payload.effective_datetime,
            "valueQuantity": {
                "value": payload.value,
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": ucum_code,
            },
        }

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/ws/fhir2/R4/Observation", payload)

    def update(self, payload: ObservationUpdateInput) -> dict[str, Any]:
        resource = self.build_fhir_payload(payload)
        resource["id"] = payload.observation_uuid
        return self.client.put(
            f"/ws/fhir2/R4/Observation/{payload.observation_uuid}", resource
        )

    def delete(self, observation_uuid: str) -> dict[str, Any]:
        return self.client.delete(f"/ws/fhir2/R4/Observation/{observation_uuid}")

    def find_latest_by_display(
        self, patient_uuid: str, display: str
    ) -> dict[str, Any] | None:
        entries = self.list_for_patient(patient_uuid).get("entry", [])
        matches = []
        for entry in entries:
            resource = entry.get("resource", entry)
            code_obj = resource.get("code") or {}
            codings = code_obj.get("coding") or [{}]
            resource_display = codings[0].get("display") or code_obj.get("text", "")
            if resource_display.lower() == display.lower():
                matches.append(resource)
        if not matches:
            return None
        matches.sort(
            key=lambda item: item.get("effectiveDateTime", ""), reverse=True
        )
        return matches[0]

    @staticmethod
    def extract_observation_snapshot(resource: dict[str, Any]) -> dict[str, Any]:
        code = (resource.get("code") or {}).get("coding", [{}])[0]
        value = resource.get("valueQuantity", {})
        return {
            "uuid": resource.get("id"),
            "display": code.get("display") or (resource.get("code") or {}).get("text"),
            "value": value.get("value"),
            "unit": value.get("unit"),
            "effectiveDateTime": resource.get("effectiveDateTime"),
        }
