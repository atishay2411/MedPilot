from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.core.exceptions import ValidationError
from app.services.lookups import LookupService

# Normalise free-text frequency expressions to standard OpenMRS concept names.
# Keys are lowercase; values are tried in order against the OpenMRS concept search.
_FREQUENCY_ALIASES: dict[str, list[str]] = {
    "once daily": ["Once daily", "OD", "Daily"],
    "once a day": ["Once daily", "OD", "Daily"],
    "twice daily": ["Twice daily", "BD", "BID", "Twice a day"],
    "twice a day": ["Twice daily", "BD", "BID"],
    "2 times a day": ["Twice daily", "BD", "BID"],
    "three times daily": ["Three times a day", "TDS", "TID"],
    "three times a day": ["Three times a day", "TDS", "TID"],
    "3 times a day": ["Three times a day", "TDS", "TID"],
    "four times daily": ["Four times a day", "QID"],
    "four times a day": ["Four times a day", "QID"],
    "4 times a day": ["Four times a day", "QID"],
    "once weekly": ["Once weekly", "Weekly", "Once a week"],
    "once a week": ["Once weekly", "Weekly"],
    "twice weekly": ["Twice weekly", "Twice a week"],
    "twice a week": ["Twice weekly", "Twice a week"],
    "2 times a week": ["Twice weekly", "Twice a week"],
    "every other day": ["Every other day", "Alternate days"],
    "every morning": ["Once daily", "Every morning"],
    "at night": ["Once daily at night", "At night", "Once nightly"],
    "at bedtime": ["Once daily at bedtime", "At bedtime"],
    "as needed": ["As needed", "PRN"],
    "prn": ["As needed", "PRN"],
}


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

    def _resolve_frequency(self, frequency_name: str) -> str:
        """Resolve a frequency name to an OpenMRS concept UUID.

        Tries the raw name first, then walks a normalisation alias table so that
        common natural-language expressions like '2 times a week' resolve to a
        valid OpenMRS concept.
        """
        candidates = [frequency_name]
        candidates += _FREQUENCY_ALIASES.get(frequency_name.lower(), [])
        for candidate in candidates:
            try:
                return self.lookups.resolve_uuid("concept", candidate)
            except ValidationError:
                continue
        raise ValidationError(f"No concept found for frequency '{frequency_name}'. Try a standard name like 'Once daily', 'Twice daily', or 'Twice weekly'.")

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
            "frequency": self._resolve_frequency(medication["frequency_name"]),
            "duration": medication["duration"],
            "durationUnits": self.lookups.resolve_uuid("concept", medication["duration_units_name"]),
            "quantity": medication["quantity"],
            "quantityUnits": self.lookups.resolve_uuid("concept", medication["quantity_units_name"]),
            "careSetting": self.lookups.resolve_uuid("caresetting", medication.get("care_setting_name", "Outpatient")),
            "encounter": encounter_uuid,
            "orderer": self.lookups.resolve_uuid("provider", medication.get("orderer_name", "Super User")),
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
