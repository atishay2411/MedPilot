from __future__ import annotations

from typing import Any

from app.models.common import EvidenceItem
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

    def build_clinical_brief(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        patient = snapshot["patient"]
        observations = [entry.get("resource", entry) for entry in snapshot["observations"].get("entry", [])]
        conditions = [entry.get("resource", entry) for entry in snapshot["conditions"].get("entry", [])]
        allergies = [entry.get("resource", entry) for entry in snapshot["allergies"].get("entry", [])]
        medications = [entry.get("resource", entry) for entry in snapshot["medications"].get("entry", [])]
        encounters = [entry.get("resource", entry) for entry in snapshot["encounters"].get("entry", [])]

        patient_name = self.patients.format_patient_display(patient)
        patient_uuid = patient.get("id") or patient.get("uuid")
        active_conditions = [self._condition_display(item) for item in conditions if self._status(item, "clinicalStatus") != "inactive"]
        allergy_names = [self._allergy_display(item) for item in allergies]
        medication_names = [self._medication_display(item) for item in medications]
        latest_vitals = self._latest_vitals(observations)
        evidence: list[EvidenceItem] = [
            EvidenceItem(label="Patient", resource_type="Patient", resource_uuid=patient_uuid, note=f"Resolved patient context for {patient_name}.")
        ]

        for item in conditions[:5]:
            evidence.append(EvidenceItem(label=self._condition_display(item), resource_type="Condition", resource_uuid=item.get("id"), note=f"Clinical status: {self._status(item, 'clinicalStatus') or 'unknown'}"))
        for item in allergies[:5]:
            evidence.append(EvidenceItem(label=self._allergy_display(item), resource_type="AllergyIntolerance", resource_uuid=item.get("id"), note="Recorded allergy on patient chart."))
        for item in medications[:5]:
            evidence.append(EvidenceItem(label=self._medication_display(item), resource_type="MedicationRequest", resource_uuid=item.get("id"), note=f"Medication status: {item.get('status', 'unknown')}"))
        for vital in latest_vitals.values():
            evidence.append(EvidenceItem(label=vital["display"], resource_type="Observation", resource_uuid=vital["uuid"], note=f"Latest reading: {vital['value']} {vital['unit']}"))

        narrative = (
            f"{patient_name} is a {patient.get('gender', 'unknown')} patient"
            f" born {patient.get('birthDate', 'unknown')}. "
            f"Active problems: {', '.join(active_conditions[:5]) if active_conditions else 'none documented'}. "
            f"Allergies: {', '.join(allergy_names[:5]) if allergy_names else 'none documented'}. "
            f"Active medications: {', '.join(medication_names[:5]) if medication_names else 'none documented'}. "
            f"Encounters on file: {len(encounters)}."
        )

        analysis = self._analysis_points(latest_vitals, conditions)
        return {
            "narrative": narrative,
            "analysis": analysis,
            "evidence": [item.model_dump() for item in evidence],
            "highlights": {
                "active_conditions": active_conditions,
                "allergies": allergy_names,
                "medications": medication_names,
                "latest_vitals": latest_vitals,
                "encounter_count": len(encounters),
            },
        }

    def summarize_patient(self, patient_uuid: str) -> dict[str, Any]:
        snapshot = self.patient_summary(patient_uuid)
        return self.build_clinical_brief(snapshot)

    @staticmethod
    def _condition_display(resource: dict[str, Any]) -> str:
        return ((resource.get("code") or {}).get("coding") or [{}])[0].get("display") or (resource.get("code") or {}).get("text", "Unknown condition")

    @staticmethod
    def _allergy_display(resource: dict[str, Any]) -> str:
        return ((resource.get("code") or {}).get("coding") or [{}])[0].get("display") or (resource.get("code") or {}).get("text", "Unknown allergy")

    @staticmethod
    def _medication_display(resource: dict[str, Any]) -> str:
        codeable = resource.get("medicationCodeableConcept", {})
        return ((codeable.get("coding") or [{}])[0].get("display")) or codeable.get("text") or (resource.get("medicationReference") or {}).get("display", "Unknown medication")

    @staticmethod
    def _status(resource: dict[str, Any], field: str) -> str:
        return (((resource.get(field) or {}).get("coding") or [{}])[0].get("code") or resource.get(field, "") or "").lower()

    def _latest_vitals(self, observations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for resource in sorted(observations, key=lambda item: item.get("effectiveDateTime", ""), reverse=True):
            snapshot = self.observations.extract_observation_snapshot(resource)
            display = snapshot["display"]
            if display and display not in latest:
                latest[display] = snapshot
        return latest

    @staticmethod
    def _analysis_points(latest_vitals: dict[str, dict[str, Any]], conditions: list[dict[str, Any]]) -> list[str]:
        points: list[str] = []
        systolic = latest_vitals.get("Systolic blood pressure")
        diastolic = latest_vitals.get("Diastolic blood pressure")
        if systolic and systolic["value"] is not None and systolic["value"] >= 140:
            points.append(f"Elevated systolic blood pressure: {systolic['value']} {systolic['unit']} [Observation {systolic['uuid']}].")
        if diastolic and diastolic["value"] is not None and diastolic["value"] >= 90:
            points.append(f"Elevated diastolic blood pressure: {diastolic['value']} {diastolic['unit']} [Observation {diastolic['uuid']}].")
        temperature = latest_vitals.get("Temperature")
        if temperature and temperature["value"] is not None and temperature["value"] >= 38:
            points.append(f"Fever-range temperature documented: {temperature['value']} {temperature['unit']} [Observation {temperature['uuid']}].")
        spo2 = latest_vitals.get("Oxygen Saturation (SpO2)")
        if spo2 and spo2["value"] is not None and spo2["value"] < 92:
            points.append(f"Low oxygen saturation documented: {spo2['value']} {spo2['unit']} [Observation {spo2['uuid']}].")
        active_conditions = [
            ((item.get("code") or {}).get("coding") or [{}])[0].get("display") or (item.get("code") or {}).get("text", "Unknown condition")
            for item in conditions
            if (((item.get("clinicalStatus") or {}).get("coding") or [{}])[0].get("code") or "").lower() not in {"inactive", "resolved"}
        ]
        if active_conditions:
            points.append(f"Active problem list includes {', '.join(active_conditions[:3])}.")
        if not points:
            points.append("No high-priority abnormal trends were detected from the currently available chart data.")
        return points
