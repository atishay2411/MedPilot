from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PatientRegistration(BaseModel):
    given_name: str
    family_name: str
    gender: Literal["M", "F", "O", "U"]
    birthdate: str
    address1: str | None = None
    city_village: str | None = None
    country: str | None = None


class PatientUpdateInput(BaseModel):
    """Partial update to patient demographics. All fields optional except patient_uuid."""

    patient_uuid: str
    given_name: str | None = None
    family_name: str | None = None
    gender: Literal["M", "F", "O", "U"] | None = None
    birthdate: str | None = None
    address1: str | None = None
    city_village: str | None = None
    country: str | None = None


class PatientSearchQuery(BaseModel):
    query: str


class ObservationInput(BaseModel):
    patient_uuid: str
    code: str
    display: str
    value: float
    unit: str
    effective_datetime: str


class ObservationUpdateInput(ObservationInput):
    observation_uuid: str


class ConditionInput(BaseModel):
    patient_uuid: str
    condition_name: str
    clinical_status: str
    verification_status: str
    onset_date: str | None = None


class AllergyInput(BaseModel):
    patient_uuid: str
    allergen_name: str
    severity: str
    reaction: str
    comment: str | None = None


class MedicationInput(BaseModel):
    patient_uuid: str
    encounter_uuid: str
    drug_name: str
    concept_name: str
    dose: float
    dose_units_name: str
    route_name: str
    frequency_name: str
    duration: float
    duration_units_name: str
    quantity: float
    quantity_units_name: str
    care_setting_name: str
    orderer_name: str


class MedicationPatchInput(BaseModel):
    medication_request_uuid: str
    status: str


class MedicationDispenseInput(BaseModel):
    patient_uuid: str
    medication_reference: str
    quantity: float
    unit: str
    when_handed_over: str
    dosage_text: str


class EncounterInput(BaseModel):
    patient_uuid: str
    encounter_type_name: str
    location_name: str
    provider_name: str
    encounter_role_name: str
    encounter_datetime: str


class PdfParseResult(BaseModel):
    encounter_type_name: str | None = None
    location_name: str | None = None
    provider_name: str | None = None
    encounter_role_name: str | None = None
    allergies: list[dict[str, Any]] = Field(default_factory=list)
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    observations: dict[str, Any] = Field(default_factory=dict)
    medications: list[dict[str, Any]] = Field(default_factory=list)
    name: str | None = None
    age: str | None = None
    gender: str | None = None


class HealthGorillaSearchInput(BaseModel):
    given_name: str
    family_name: str
    birthdate: str


class IntentRequest(BaseModel):
    prompt: str
    patient_uuid: str | None = None


class WriteExecutionRequest(BaseModel):
    intent: str
    action: str
    permission: str
    endpoint: str
    payload: dict[str, Any]
    confirmed: bool = False
    destructive_confirm_text: str | None = None
    destructive: bool = False
    patient_uuid: str | None = None
    prompt: str | None = None


class ParsedIntent(BaseModel):
    intent: str
    write: bool
    confidence: float
    entities: dict[str, Any] = Field(default_factory=dict)
