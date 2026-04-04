from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.core.exceptions import ValidationError
from app.services.lookups import LookupService


class MedicationService:
    def __init__(self, client: OpenMRSClient, lookups: LookupService):
        self.client = client
        self.lookups = lookups

    def list_for_patient(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get("/ws/fhir2/R4/MedicationRequest", params={"patient": patient_uuid})

    def list_orders_for_patient(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get("/ws/rest/v1/order", params={"patient": patient_uuid})

    def medication_exists(self, patient_uuid: str, drug_name: str) -> bool:
        results = self.list_orders_for_patient(patient_uuid).get("results", [])
        return any(drug_name.lower() in item.get("display", "").lower() and item.get("action") == "NEW" for item in results)

    def build_create_payload(self, patient_uuid: str, encounter_uuid: str, medication: dict[str, Any]) -> dict[str, Any]:
        drug_uuid = self.lookups.resolve_uuid("drug", medication["drug_name"])
        drug_detail = self.client.get(f"/ws/rest/v1/drug/{drug_uuid}")
        return {
            "type": "drugorder",
            "patient": patient_uuid,
            "drug": drug_uuid,
            "concept": drug_detail["concept"]["uuid"],
            "dose": medication["dose"],
            "doseUnits": self.lookups.resolve_uuid("concept", medication["dose_units_name"]),
            "route": self.lookups.resolve_uuid("concept", medication["route_name"]),
            "frequency": self.lookups.resolve_uuid("concept", medication["frequency_name"]),
            "duration": medication["duration"],
            "durationUnits": self.lookups.resolve_uuid("concept", medication["duration_units_name"]),
            "quantity": medication["quantity"],
            "quantityUnits": self.lookups.resolve_uuid("concept", medication["quantity_units_name"]),
            "careSetting": self.lookups.resolve_uuid("caresetting", medication["care_setting_name"]),
            "encounter": encounter_uuid,
            "orderer": self.lookups.resolve_uuid("provider", medication["orderer_name"]),
        }

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/ws/rest/v1/order", payload)

    def patch_status(self, medication_request_uuid: str, status: str) -> dict[str, Any]:
        return self.client.patch(
            f"/ws/fhir2/R4/MedicationRequest/{medication_request_uuid}",
            [{"op": "replace", "path": "/status", "value": status}],
        )

    def medication_dispense(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get("/ws/fhir2/R4/MedicationDispense", params={"patient": patient_uuid})

    def create_dispense(self, patient_uuid: str, medication_reference: str, quantity: float, unit: str, when_handed_over: str, dosage_text: str) -> dict[str, Any]:
        return self.client.post(
            "/ws/fhir2/R4/MedicationDispense",
            {
                "resourceType": "MedicationDispense",
                "status": "completed",
                "subject": {"reference": f"Patient/{patient_uuid}"},
                "medicationReference": {"reference": medication_reference},
                "quantity": {"value": quantity, "unit": unit},
                "whenHandedOver": when_handed_over,
                "dosageInstruction": [{"text": dosage_text}],
            },
        )

    def find_by_name(self, patient_uuid: str, drug_name: str) -> dict[str, Any] | None:
        entries = self.list_for_patient(patient_uuid).get("entry", [])
        normalized = drug_name.lower()
        for entry in entries:
            resource = entry.get("resource", entry)
            codeable = resource.get("medicationCodeableConcept", {})
            display = (
                ((codeable.get("coding") or [{}])[0].get("display"))
                or codeable.get("text")
                or (resource.get("medicationReference") or {}).get("display", "")
            )
            if display and normalized in display.lower():
                return resource
        return None

    def resolve_medication_reference(self, patient_uuid: str, drug_name: str) -> str:
        resource = self.find_by_name(patient_uuid, drug_name)
        if resource:
            medication_reference = (resource.get("medicationReference") or {}).get("reference")
            if medication_reference:
                return medication_reference
        try:
            drug_uuid = self.lookups.resolve_uuid("drug", drug_name)
        except ValidationError as exc:
            raise ValidationError(f"Could not resolve a medication reference for '{drug_name}'.") from exc
        return f"Medication/{drug_uuid}"
