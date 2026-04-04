from __future__ import annotations

from app.clients.health_gorilla import HealthGorillaClient
from app.clients.openmrs import OpenMRSClient
from app.config import Settings
from app.models.common import EntityResult
from app.core.exceptions import ValidationError
from app.parsers.patient_pdf import parse_patient_pdf
from app.services.allergies import AllergyService
from app.services.conditions import ConditionService
from app.services.encounters import EncounterService
from app.services.medications import MedicationService
from app.services.observations import ObservationService, VITALS_CODE_MAP
from app.services.patients import PatientService
from app.services.utils import now_iso


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        openmrs: OpenMRSClient,
        hg_client: HealthGorillaClient,
        patients: PatientService,
        encounters: EncounterService,
        observations: ObservationService,
        conditions: ConditionService,
        allergies: AllergyService,
        medications: MedicationService,
    ):
        self.settings = settings
        self.openmrs = openmrs
        self.hg_client = hg_client
        self.patients = patients
        self.encounters = encounters
        self.observations = observations
        self.conditions = conditions
        self.allergies = allergies
        self.medications = medications

    def parse_pdf(self, path: str):
        return parse_patient_pdf(path)

    def ingest_pdf(self, patient_uuid: str, path: str) -> list[EntityResult]:
        parsed = self.parse_pdf(path)
        results: list[EntityResult] = []

        encounter_payload = self.encounters.build_rest_payload(
            type("EncounterPayload", (), {
                "patient_uuid": patient_uuid,
                "encounter_type_name": parsed.encounter_type_name or "Vitals",
                "location_name": parsed.location_name or "Outpatient Clinic",
                "provider_name": parsed.provider_name or "Super User",
                "encounter_role_name": parsed.encounter_role_name or "Clinician",
                "encounter_datetime": now_iso(),
            })()
        )
        encounter = self.encounters.create_rest(encounter_payload)
        encounter_uuid = encounter["uuid"]
        results.append(EntityResult(entity_type="encounter", name="Encounter", outcome="success", detail=encounter_uuid))

        for name, value in parsed.observations.items():
            payload = self.observations.build_fhir_payload(
                type("ObsInput", (), {
                    "patient_uuid": patient_uuid,
                    "code": VITALS_CODE_MAP.get(name, name),
                    "display": name,
                    "value": value,
                    "unit": "%" if "Saturation" in name else "mmHg" if "pressure" in name else "cm" if name == "Height" else "kg",
                    "effective_datetime": now_iso(),
                })()
            )
            self.observations.create(payload)
            results.append(EntityResult(entity_type="observation", name=name, outcome="success", detail="Created"))

        for condition in parsed.conditions:
            payload = self.conditions.build_create_payload(patient_uuid, condition["condition_name"], condition["clinical_status"], condition["verification_status"], condition.get("onset_date"))
            self.conditions.create(payload)
            results.append(EntityResult(entity_type="condition", name=condition["condition_name"], outcome="success", detail="Created"))

        for allergy in parsed.allergies:
            if self.allergies.allergy_exists(patient_uuid, allergy["allergen_name"]):
                results.append(EntityResult(entity_type="allergy", name=allergy["allergen_name"], outcome="skipped", detail="Duplicate"))
                continue
            payload = self.allergies.build_rest_payload(allergy["allergen_name"], allergy["severity_name"], allergy["reaction_name"], allergy.get("comment"))
            self.allergies.create(patient_uuid, payload)
            results.append(EntityResult(entity_type="allergy", name=allergy["allergen_name"], outcome="success", detail="Created"))

        for medication in parsed.medications:
            if self.medications.medication_exists(patient_uuid, medication["drug_name"]):
                results.append(EntityResult(entity_type="medication", name=medication["drug_name"], outcome="skipped", detail="Duplicate"))
                continue
            payload = self.medications.build_create_payload(patient_uuid, encounter_uuid, medication)
            self.medications.create(payload)
            results.append(EntityResult(entity_type="medication", name=medication["drug_name"], outcome="success", detail="Created"))

        return results

    def health_gorilla_preview(self, given_name: str, family_name: str, birthdate: str) -> dict:
        patient_bundle = self.hg_client.search_patient(given_name, family_name, birthdate)
        entries = patient_bundle.get("entry", [])
        if not entries:
            return {"matches": [], "conditions": []}
        patient_entry = entries[0]
        patient_id = patient_entry["resource"]["id"]
        conditions_bundle = self.hg_client.get_conditions(patient_id)
        prepared = []
        for entry in conditions_bundle.get("entry", [])[: self.settings.health_gorilla_max_conditions]:
            resource = entry.get("resource", {})
            coding = resource.get("code", {}).get("coding", [])
            display = coding[0].get("display") if coding else resource.get("code", {}).get("text")
            if not display:
                continue
            prepared.append(
                {
                    "condition_name": display,
                    "clinical_status": "active",
                    "verification_status": "confirmed",
                    "onset_date": resource.get("onsetDateTime") or resource.get("recordedDate"),
                }
            )
        return {"matches": entries, "conditions": prepared}

    def sync_health_gorilla(self, match_resource: dict, conditions: list[dict]) -> list[EntityResult]:
        patient_payload = self.patients.build_create_payload_from_fhir(match_resource)
        created_patient = self.patients.create(patient_payload)
        patient_uuid = created_patient["uuid"]
        results = [EntityResult(entity_type="patient", name=self.patients.format_patient_display(match_resource), outcome="success", detail=patient_uuid)]

        for condition in conditions[: self.settings.health_gorilla_max_conditions]:
            concept_uuid = self.conditions.resolve_or_create_concept_uuid(condition["condition_name"])
            payload = {
                "patient": patient_uuid,
                "condition": {"coded": concept_uuid},
                "clinicalStatus": condition.get("clinical_status", "active"),
                "verificationStatus": condition.get("verification_status", "confirmed"),
                "onsetDate": condition.get("onset_date"),
            }
            self.conditions.create(payload)
            results.append(EntityResult(entity_type="condition", name=condition["condition_name"], outcome="success", detail="Imported from Health Gorilla"))

        return results
