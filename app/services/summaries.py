from __future__ import annotations

from app.services.allergies import AllergyService
from app.services.conditions import ConditionService
from app.services.encounters import EncounterService
from app.services.medications import MedicationService
from app.services.observations import ObservationService
from app.services.patients import PatientService


class SummaryService:
    def __init__(
        self,
        patients: PatientService,
        observations: ObservationService,
        conditions: ConditionService,
        allergies: AllergyService,
        medications: MedicationService,
        encounters: EncounterService,
    ):
        self.patients = patients
        self.observations = observations
        self.conditions = conditions
        self.allergies = allergies
        self.medications = medications
        self.encounters = encounters

    def patient_summary(self, patient_uuid: str) -> dict:
        return {
            "patient": self.patients.get_demographics(patient_uuid),
            "observations": self.observations.list_for_patient(patient_uuid),
            "conditions": self.conditions.list_for_patient(patient_uuid),
            "allergies": self.allergies.list_for_patient(patient_uuid),
            "medications": self.medications.list_for_patient(patient_uuid),
            "encounters": self.encounters.list_for_patient(patient_uuid),
        }
