from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class CapabilityField:
    name: str
    description: str
    required: bool = False


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    intent: str
    handler_name: str
    summary: str
    scope: Literal["global", "patient"] = "patient"
    """'global' for population/system queries that do not require an active patient.
    'patient' for chart-level actions that act on a specific patient."""
    fields: tuple[CapabilityField, ...] = ()
    write: bool = False
    destructive: bool = False
    examples: tuple[str, ...] = ()
    guidance: tuple[str, ...] = ()


CAPABILITIES: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        intent="search_patient",
        handler_name="_handle_search_patient",
        summary="Search patients by name, identifier, UUID, or broad list request.",
        scope="global",
        fields=(
            CapabilityField("patient_query", "Search term or identifier. Null for pure follow-up on active patient.", required=False),
            CapabilityField("search_mode", "One of default, starts_with, or contains.", required=False),
        ),
        examples=("Find patient Maria Santos", "Find any patient whose name starts with N", "Search all patients"),
        guidance=(
            "Strip conversational wrappers and keep only the real search term.",
            "Use search_mode='starts_with' for prompts like 'starts with N'.",
        ),
    ),
    CapabilityDefinition(
        intent="search_by_identifier",
        handler_name="_handle_search_by_identifier",
        summary="Look up a patient directly by their identifier number or UUID.",
        scope="global",
        fields=(
            CapabilityField("identifier", "Patient identifier string or UUID to look up directly.", required=True),
        ),
        examples=("Find patient with ID 10001D", "Look up patient UUID abc-123"),
        guidance=("Use this intent when the user provides a specific ID/number, not a name.",),
    ),
    CapabilityDefinition(
        intent="switch_patient",
        handler_name="_handle_switch_patient",
        summary="Switch the active chart context to a different patient.",
        scope="global",
        fields=(CapabilityField("patient_query", "Patient name, identifier, or UUID.", required=True),),
        examples=("Change patient to Maria Santos",),
    ),
    CapabilityDefinition(
        intent="count_patients",
        handler_name="_handle_count_patients",
        summary="Count patients globally or within a filtered search scope.",
        scope="global",
        fields=(
            CapabilityField("patient_query", "Optional search term for filtered counts.", required=False),
            CapabilityField("search_mode", "One of default, starts_with, or contains.", required=False),
        ),
        examples=("How many patients are present?", "How many patients start with N?"),
    ),
    CapabilityDefinition(
        intent="get_metadata",
        handler_name="_handle_get_metadata",
        summary="Fetch the connected OpenMRS FHIR capability statement.",
        scope="global",
        examples=("Show metadata", "What FHIR capabilities are available?"),
    ),
    CapabilityDefinition(
        intent="patient_analysis",
        handler_name="_handle_patient_analysis",
        summary="Summarize and analyze the active or named patient chart.",
        fields=(CapabilityField("patient_query", "Optional patient search term.", required=False),),
        examples=("Summarize this patient", "Analyze Maria Santos"),
    ),
    CapabilityDefinition(
        intent="get_observations",
        handler_name="_handle_get_observations",
        summary="Read chart observations or a specific latest vital sign.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("observation_display", "Optional vital/observation display name.", required=False),
        ),
        examples=("Show their vitals", "Show Maria Santos's blood pressure"),
    ),
    CapabilityDefinition(
        intent="create_observation",
        handler_name="_handle_create_observation",
        summary="Create one or more observations for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("observations", "List of observations with display, code, value, and unit.", required=True),
        ),
        write=True,
        examples=("Record blood pressure 140/90 for Maria Santos",),
        guidance=("Split blood pressure into systolic and diastolic observations.",),
    ),
    CapabilityDefinition(
        intent="update_observation",
        handler_name="_handle_update_observation",
        summary="Update a single existing observation for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("observations", "Single observation item with display, code, value, and unit.", required=True),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="delete_observation",
        handler_name="_handle_delete_observation",
        summary="Delete an observation for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("observation_display", "Observation display name to delete.", required=True),
        ),
        write=True,
        destructive=True,
    ),
    CapabilityDefinition(
        intent="get_conditions",
        handler_name="_handle_get_conditions",
        summary="Read patient conditions/problem list.",
        fields=(CapabilityField("patient_query", "Optional patient search term.", required=False),),
    ),
    CapabilityDefinition(
        intent="create_condition",
        handler_name="_handle_create_condition",
        summary="Create a condition/problem list entry for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("condition_name", "Condition or diagnosis name.", required=True),
            CapabilityField("clinical_status", "Condition clinical status.", required=False),
            CapabilityField("verification_status", "Condition verification status.", required=False),
            CapabilityField("onset_date", "Optional onset date in YYYY-MM-DD.", required=False),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="update_condition",
        handler_name="_handle_update_condition",
        summary="Update an existing condition status for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("condition_name", "Condition name.", required=True),
            CapabilityField("status", "Updated clinical status.", required=True),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="delete_condition",
        handler_name="_handle_delete_condition",
        summary="Delete a condition from the patient chart.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("name", "Condition name to delete.", required=True),
        ),
        write=True,
        destructive=True,
    ),
    CapabilityDefinition(
        intent="get_allergies",
        handler_name="_handle_get_allergies",
        summary="Read patient allergies.",
        fields=(CapabilityField("patient_query", "Optional patient search term.", required=False),),
    ),
    CapabilityDefinition(
        intent="create_allergy",
        handler_name="_handle_create_allergy",
        summary="Create an allergy for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("allergen_name", "Allergen name.", required=True),
            CapabilityField("severity", "Severity such as mild, moderate, or severe.", required=False),
            CapabilityField("reaction", "Reaction or manifestation.", required=False),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="update_allergy",
        handler_name="_handle_update_allergy",
        summary="Update allergy severity for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("allergen_name", "Allergen name.", required=True),
            CapabilityField("severity", "Updated severity.", required=True),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="delete_allergy",
        handler_name="_handle_delete_allergy",
        summary="Delete an allergy from a patient chart.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("name", "Allergy/allergen name to delete.", required=True),
        ),
        write=True,
        destructive=True,
    ),
    CapabilityDefinition(
        intent="get_medications",
        handler_name="_handle_get_medications",
        summary="Read active medications/medication requests for a patient.",
        fields=(CapabilityField("patient_query", "Optional patient search term.", required=False),),
    ),
    CapabilityDefinition(
        intent="get_medication_dispense",
        handler_name="_handle_get_medication_dispense",
        summary="Read dispense history for a patient.",
        fields=(CapabilityField("patient_query", "Optional patient search term.", required=False),),
    ),
    CapabilityDefinition(
        intent="create_medication",
        handler_name="_handle_create_medication",
        summary="Create a medication order for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("drug_name", "Drug name.", required=True),
            CapabilityField("concept_name", "Concept name; defaults to drug_name when omitted.", required=False),
            CapabilityField("dose", "Dose value.", required=True),
            CapabilityField("dose_units_name", "Dose units.", required=True),
            CapabilityField("route_name", "Route.", required=True),
            CapabilityField("frequency_name", "Frequency.", required=True),
            CapabilityField("duration", "Duration value.", required=True),
            CapabilityField("duration_units_name", "Duration units.", required=True),
            CapabilityField("quantity", "Quantity.", required=True),
            CapabilityField("quantity_units_name", "Quantity units.", required=True),
        ),
        write=True,
        examples=("Prescribe metformin 500 mg orally twice daily for 30 days to Maria Santos",),
    ),
    CapabilityDefinition(
        intent="create_medication_dispense",
        handler_name="_handle_create_medication_dispense",
        summary="Create a medication dispense record for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("drug_name", "Drug name.", required=True),
            CapabilityField("quantity", "Quantity dispensed.", required=True),
            CapabilityField("unit", "Dispense unit.", required=False),
            CapabilityField("when_handed_over", "ISO datetime.", required=False),
            CapabilityField("dosage_text", "Free-text dosage instructions.", required=False),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="update_medication",
        handler_name="_handle_update_medication",
        summary="Update medication request status for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("drug_name", "Drug name.", required=True),
            CapabilityField("status", "FHIR medication request status.", required=True),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="create_patient",
        handler_name="_handle_create_patient",
        summary="Register a new patient in OpenMRS.",
        scope="global",
        fields=(
            CapabilityField("given_name", "Patient first/given name.", required=True),
            CapabilityField("family_name", "Patient last/family name.", required=True),
            CapabilityField("birthdate", "Birthdate in YYYY-MM-DD.", required=True),
            CapabilityField("gender", "One of M, F, O, U.", required=False),
            CapabilityField("city_village", "Optional city.", required=False),
        ),
        write=True,
        examples=("Add a patient named Sahil Rochwani born 2000-04-24",),
    ),
    CapabilityDefinition(
        intent="update_patient",
        handler_name="_handle_update_patient",
        summary="Update a patient's demographics (name, gender, birthdate, or address).",
        fields=(
            CapabilityField("patient_query", "Patient name, identifier, or UUID to update.", required=True),
            CapabilityField("given_name", "New given name.", required=False),
            CapabilityField("family_name", "New family name.", required=False),
            CapabilityField("gender", "New gender: M, F, O, or U.", required=False),
            CapabilityField("birthdate", "New birthdate in YYYY-MM-DD.", required=False),
            CapabilityField("city_village", "New city or village.", required=False),
        ),
        write=True,
        examples=("Update Maria Santos gender to F", "Change John Doe's birthdate to 1990-01-15"),
    ),
    CapabilityDefinition(
        intent="delete_patient",
        handler_name="_handle_delete_patient",
        summary="Delete or void a patient record from OpenMRS.",
        scope="global",
        fields=(
            CapabilityField("patient_query", "Patient name, identifier, or UUID.", required=True),
            CapabilityField("purge", "Boolean true only when a hard purge is explicitly requested.", required=False),
        ),
        write=True,
        destructive=True,
        examples=("Delete patient John Doe", "Purge patient test user 123"),
    ),
    CapabilityDefinition(
        intent="patient_intake",
        handler_name="_handle_patient_intake",
        summary="Create a patient with conditions, allergies, observations, medications, and dispenses in one workflow.",
        scope="global",
        fields=(
            CapabilityField("given_name", "Patient first name.", required=True),
            CapabilityField("family_name", "Patient last name.", required=True),
            CapabilityField("birthdate", "Birthdate in YYYY-MM-DD.", required=True),
            CapabilityField("gender", "Patient gender.", required=False),
            CapabilityField("city_village", "Optional city.", required=False),
            CapabilityField("conditions", "List of conditions.", required=False),
            CapabilityField("allergies", "List of allergies.", required=False),
            CapabilityField("observations", "List of observations.", required=False),
            CapabilityField("medications", "List of medications.", required=False),
            CapabilityField("dispenses", "List of dispenses.", required=False),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="create_encounter",
        handler_name="_handle_create_encounter",
        summary="Create an encounter for a patient.",
        fields=(
            CapabilityField("patient_query", "Optional patient search term.", required=False),
            CapabilityField("encounter_type_name", "Encounter type name.", required=False),
            CapabilityField("location_name", "Location name.", required=False),
            CapabilityField("provider_name", "Provider name.", required=False),
            CapabilityField("encounter_role_name", "Encounter role name.", required=False),
        ),
        write=True,
    ),
    CapabilityDefinition(
        intent="ingest_pdf",
        handler_name="_handle_ingest_pdf",
        summary="Parse and ingest a patient PDF attachment.",
        fields=(CapabilityField("patient_query", "Optional patient search term.", required=False),),
        write=True,
    ),
    CapabilityDefinition(
        intent="sync_health_gorilla",
        handler_name="_handle_sync_health_gorilla",
        summary="Match a patient from Health Gorilla and prepare a sync workflow.",
        scope="global",
        fields=(
            CapabilityField("given_name", "Patient first name.", required=True),
            CapabilityField("family_name", "Patient last name.", required=True),
            CapabilityField("birthdate", "Birthdate in YYYY-MM-DD.", required=True),
        ),
        write=True,
    ),
)


CAPABILITY_INDEX: dict[str, CapabilityDefinition] = {
    capability.intent: capability for capability in CAPABILITIES
}


def get_capability(intent: str) -> CapabilityDefinition | None:
    return CAPABILITY_INDEX.get(intent)


def supported_intents() -> tuple[str, ...]:
    return tuple(capability.intent for capability in CAPABILITIES)


def handler_map() -> dict[str, str]:
    return {capability.intent: capability.handler_name for capability in CAPABILITIES}


def is_global_intent(intent: str) -> bool:
    """Return True when the capability is population/system-scoped, not patient-scoped."""
    cap = CAPABILITY_INDEX.get(intent)
    return cap is not None and cap.scope == "global"


def extract_entities(entities: dict[str, Any], capability: CapabilityDefinition) -> dict[str, Any]:
    """Validate and return only the recognised fields for a capability.

    - Required fields missing from *entities* are returned with a ``None`` value
      (the handler is responsible for raising ValidationError if needed).
    - Extra keys in *entities* that are not in the capability's field list are
      silently dropped to prevent pollution downstream.
    - Returns a plain ``dict`` — not a Pydantic model — as each capability has
      a unique field shape.
    """
    field_names: set[str] = {f.name for f in capability.fields}
    # Always pass through patient_query even if not explicitly in field list,
    # because handlers rely on it for patient resolution.
    field_names.add("patient_query")
    return {k: v for k, v in entities.items() if k in field_names}


def render_capability_prompt() -> str:
    lines: list[str] = []
    for index, capability in enumerate(CAPABILITIES, start=1):
        scope_tag = " [global]" if capability.scope == "global" else ""
        lines.append(f"{index}. {capability.intent}{scope_tag}")
        lines.append(f"   summary: {capability.summary}")
        if capability.fields:
            field_parts = []
            for f in capability.fields:
                suffix = "required" if f.required else "optional"
                field_parts.append(f"{f.name} ({suffix}) — {f.description}")
            lines.append("   entities: " + "; ".join(field_parts))
        if capability.guidance:
            lines.append("   guidance: " + " ".join(capability.guidance))
        if capability.examples:
            lines.append("   examples: " + " | ".join(capability.examples))
    return "\n".join(lines)
