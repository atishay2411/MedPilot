from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.config import Settings, get_settings
from app.core.audit import AuditEvent, AuditLogger
from app.core.confirmation import ConfirmationRequest, ensure_confirmation
from app.core.security import Actor, ensure_permission
from app.dependencies import (
    get_allergy_service,
    get_audit_logger,
    get_chat_agent_service,
    get_condition_service,
    get_encounter_service,
    get_ingestion_service,
    get_intent_service,
    get_llm_provider,
    get_medication_service,
    get_observation_service,
    get_patient_service,
    get_population_service,
    get_summary_service,
)
from app.llm.base import LLMProvider
from app.models.common import ApiResponse
from app.models.domain import (
    AllergyInput,
    ConditionInput,
    EncounterInput,
    HealthGorillaSearchInput,
    IntentRequest,
    MedicationInput,
    MedicationDispenseInput,
    MedicationPatchInput,
    ObservationInput,
    ObservationUpdateInput,
    PatientRegistration,
    WriteExecutionRequest,
)
from app.services.allergies import AllergyService
from app.services.chat_agent import ChatAgentService
from app.services.conditions import ConditionService
from app.services.encounters import EncounterService
from app.services.ingestion import IngestionService
from app.services.intents import IntentService
from app.services.medications import MedicationService
from app.services.observations import ObservationService
from app.services.patients import PatientService
from app.services.population import PopulationService
from app.services.summaries import SummaryService


router = APIRouter(prefix="/api")


def get_actor(settings: Settings = Depends(get_settings)) -> Actor:
    return Actor(user_id=settings.medpilot_user_id, role=settings.medpilot_user_role)


@router.get("/health")
def healthcheck() -> ApiResponse:
    return ApiResponse(data={"status": "ok"})


@router.get("/llm/status")
def llm_status(settings: Settings = Depends(get_settings), provider: LLMProvider = Depends(get_llm_provider)) -> ApiResponse:
    return ApiResponse(
        data={
            "provider": settings.medpilot_llm_provider,
            "model": settings.medpilot_llm_model,
            "enabled": provider.enabled,
            "intent_reasoning": settings.medpilot_llm_enable_intent_reasoning,
            "summary_reasoning": settings.medpilot_llm_enable_summary_reasoning,
        }
    )


@router.post("/chat")
async def chat(
    prompt: str = Form(...),
    patient_uuid: str | None = Form(None),
    history: str | None = Form(None),
    file: UploadFile | None = File(None),
    actor: Actor = Depends(get_actor),
    service: ChatAgentService = Depends(get_chat_agent_service),
) -> ApiResponse:
    conversation_history: list[dict] | None = None
    if history:
        try:
            conversation_history = json.loads(history)
        except Exception:
            conversation_history = None
    temp_path: Path | None = None
    try:
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload.bin").suffix or ".bin") as temp_file:
                temp_file.write(await file.read())
                temp_path = Path(temp_file.name)
        response = service.handle_message(prompt, actor, patient_uuid=patient_uuid, attachment_path=str(temp_path) if temp_path else None, conversation_history=conversation_history)
        return ApiResponse(data=response.model_dump())
    except Exception:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


@router.post("/chat/confirm")
def confirm_chat_action(
    action_id: str = Form(...),
    destructive_confirm_text: str | None = Form(None),
    actor: Actor = Depends(get_actor),
    service: ChatAgentService = Depends(get_chat_agent_service),
) -> ApiResponse:
    response = service.confirm_action(action_id, actor, destructive_confirm_text=destructive_confirm_text)
    return ApiResponse(data=response.model_dump())


@router.post("/intent")
def classify_intent(payload: IntentRequest, service: IntentService = Depends(get_intent_service)) -> ApiResponse:
    return ApiResponse(data=service.classify(payload.prompt))


@router.post("/patients/search")
def search_patients(payload: dict, actor: Actor = Depends(get_actor), service: PatientService = Depends(get_patient_service)) -> ApiResponse:
    ensure_permission(actor, "read:patient")
    return ApiResponse(data=service.search(payload["query"]))


@router.get("/patients/{patient_uuid}/summary")
def patient_summary(patient_uuid: str, actor: Actor = Depends(get_actor), service: SummaryService = Depends(get_summary_service)) -> ApiResponse:
    ensure_permission(actor, "read:clinical")
    return ApiResponse(data=service.patient_summary(patient_uuid))


@router.post("/patients/preview")
def preview_create_patient(payload: PatientRegistration, actor: Actor = Depends(get_actor), intents: IntentService = Depends(get_intent_service)) -> ApiResponse:
    ensure_permission(actor, "write:patient")
    return ApiResponse(data=intents.preview_create_patient(payload))


@router.post("/writes/execute")
def execute_write(
    payload: WriteExecutionRequest,
    actor: Actor = Depends(get_actor),
    patients: PatientService = Depends(get_patient_service),
    conditions: ConditionService = Depends(get_condition_service),
    audit: AuditLogger = Depends(get_audit_logger),
) -> ApiResponse:
    ensure_permission(actor, payload.permission)
    ensure_confirmation(ConfirmationRequest(confirmed=payload.confirmed, destructive_confirm_text=payload.destructive_confirm_text), destructive=payload.destructive)

    if payload.intent == "create_patient":
        response = patients.create(payload.payload)
    elif payload.intent == "delete_patient":
        response = patients.delete(payload.payload["patient_uuid"], purge=bool(payload.payload.get("purge")))
    elif payload.intent == "create_condition":
        response = conditions.create(payload.payload)
    elif payload.intent == "delete_condition":
        response = conditions.delete(payload.payload["condition_uuid"])
    else:
        response = {"accepted": True, "note": "Action not yet wired through generic executor; use dedicated routes."}

    audit.log(
        AuditEvent(
            user_id=actor.user_id,
            role=actor.role,
            intent=payload.intent,
            action=payload.action,
            patient_uuid=payload.patient_uuid,
            prompt=payload.prompt,
            endpoint=payload.endpoint,
            request_payload=payload.payload,
            response_status=200,
            outcome="success",
            metadata={"destructive": payload.destructive},
        )
    )
    return ApiResponse(data=response)


@router.delete("/patients/{patient_uuid}")
def delete_patient(
    patient_uuid: str,
    confirm_text: str,
    purge: bool = False,
    actor: Actor = Depends(get_actor),
    service: PatientService = Depends(get_patient_service),
) -> ApiResponse:
    ensure_permission(actor, "delete:patient")
    ensure_confirmation(ConfirmationRequest(confirmed=True, destructive_confirm_text=confirm_text), destructive=True)
    return ApiResponse(data=service.delete(patient_uuid, purge=purge))


@router.post("/encounters")
def create_encounter(
    payload: EncounterInput,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: EncounterService = Depends(get_encounter_service),
) -> ApiResponse:
    ensure_permission(actor, "write:encounter")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    return ApiResponse(data=service.create_rest(service.build_rest_payload(payload)))


@router.get("/observations/{patient_uuid}")
def get_observations(patient_uuid: str, actor: Actor = Depends(get_actor), service: ObservationService = Depends(get_observation_service)) -> ApiResponse:
    ensure_permission(actor, "read:clinical")
    return ApiResponse(data=service.list_for_patient(patient_uuid))


@router.post("/observations")
def create_observation(
    payload: ObservationInput,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: ObservationService = Depends(get_observation_service),
) -> ApiResponse:
    ensure_permission(actor, "write:observation")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    return ApiResponse(data=service.create(service.build_fhir_payload(payload)))


@router.put("/observations")
def update_observation(
    payload: ObservationUpdateInput,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: ObservationService = Depends(get_observation_service),
) -> ApiResponse:
    ensure_permission(actor, "write:observation")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    return ApiResponse(data=service.update(payload))


@router.delete("/observations/{observation_uuid}")
def delete_observation(
    observation_uuid: str,
    confirm_text: str,
    actor: Actor = Depends(get_actor),
    service: ObservationService = Depends(get_observation_service),
) -> ApiResponse:
    ensure_permission(actor, "delete:observation")
    ensure_confirmation(ConfirmationRequest(confirmed=True, destructive_confirm_text=confirm_text), destructive=True)
    return ApiResponse(data=service.delete(observation_uuid))


@router.get("/conditions/{patient_uuid}")
def get_conditions(patient_uuid: str, actor: Actor = Depends(get_actor), service: ConditionService = Depends(get_condition_service)) -> ApiResponse:
    ensure_permission(actor, "read:clinical")
    return ApiResponse(data=service.list_for_patient(patient_uuid))


@router.post("/conditions")
def create_condition(
    payload: ConditionInput,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: ConditionService = Depends(get_condition_service),
) -> ApiResponse:
    ensure_permission(actor, "write:condition")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    body = service.build_create_payload(payload.patient_uuid, payload.condition_name, payload.clinical_status, payload.verification_status, payload.onset_date)
    return ApiResponse(data=service.create(body))


@router.patch("/conditions/{condition_uuid}")
def update_condition(
    condition_uuid: str,
    status: str,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: ConditionService = Depends(get_condition_service),
) -> ApiResponse:
    ensure_permission(actor, "write:condition")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    return ApiResponse(data=service.patch_status(condition_uuid, status))


@router.delete("/conditions/{condition_uuid}")
def delete_condition(
    condition_uuid: str,
    confirm_text: str,
    actor: Actor = Depends(get_actor),
    service: ConditionService = Depends(get_condition_service),
) -> ApiResponse:
    ensure_permission(actor, "delete:condition")
    ensure_confirmation(ConfirmationRequest(confirmed=True, destructive_confirm_text=confirm_text), destructive=True)
    return ApiResponse(data=service.delete(condition_uuid))


@router.get("/allergies/{patient_uuid}")
def get_allergies(patient_uuid: str, actor: Actor = Depends(get_actor), service: AllergyService = Depends(get_allergy_service)) -> ApiResponse:
    ensure_permission(actor, "read:clinical")
    return ApiResponse(data=service.list_for_patient(patient_uuid))


@router.post("/allergies")
def create_allergy(
    payload: AllergyInput,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: AllergyService = Depends(get_allergy_service),
) -> ApiResponse:
    ensure_permission(actor, "write:allergy")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    body = service.build_rest_payload(payload.allergen_name, payload.severity, payload.reaction, payload.comment)
    return ApiResponse(data=service.create(payload.patient_uuid, body))


@router.patch("/allergies/{allergy_uuid}")
def update_allergy(
    allergy_uuid: str,
    severity: str,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: AllergyService = Depends(get_allergy_service),
) -> ApiResponse:
    ensure_permission(actor, "write:allergy")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    return ApiResponse(data=service.patch_severity(allergy_uuid, severity))


@router.delete("/allergies/{allergy_uuid}")
def delete_allergy(
    allergy_uuid: str,
    confirm_text: str,
    actor: Actor = Depends(get_actor),
    service: AllergyService = Depends(get_allergy_service),
) -> ApiResponse:
    ensure_permission(actor, "delete:allergy")
    ensure_confirmation(ConfirmationRequest(confirmed=True, destructive_confirm_text=confirm_text), destructive=True)
    return ApiResponse(data=service.delete(allergy_uuid))


@router.get("/medications/{patient_uuid}")
def get_medications(patient_uuid: str, actor: Actor = Depends(get_actor), service: MedicationService = Depends(get_medication_service)) -> ApiResponse:
    ensure_permission(actor, "read:clinical")
    return ApiResponse(data=service.list_for_patient(patient_uuid))


@router.post("/medications")
def create_medication(
    payload: MedicationInput,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: MedicationService = Depends(get_medication_service),
) -> ApiResponse:
    ensure_permission(actor, "write:medication")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    return ApiResponse(
        data=service.create(
            service.build_create_payload(
                payload.patient_uuid,
                payload.encounter_uuid,
                payload.model_dump(exclude={"patient_uuid", "encounter_uuid"}),
            )
        )
    )


@router.patch("/medications/{medication_request_uuid}")
def update_medication(
    medication_request_uuid: str,
    payload: MedicationPatchInput,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: MedicationService = Depends(get_medication_service),
) -> ApiResponse:
    ensure_permission(actor, "write:medication")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    return ApiResponse(data=service.patch_status(medication_request_uuid, payload.status))


@router.get("/medications/{patient_uuid}/dispense")
def get_medication_dispense(patient_uuid: str, actor: Actor = Depends(get_actor), service: MedicationService = Depends(get_medication_service)) -> ApiResponse:
    ensure_permission(actor, "read:clinical")
    return ApiResponse(data=service.medication_dispense(patient_uuid))


@router.post("/medications/dispense")
def create_medication_dispense(
    payload: MedicationDispenseInput,
    confirmed: bool,
    actor: Actor = Depends(get_actor),
    service: MedicationService = Depends(get_medication_service),
) -> ApiResponse:
    ensure_permission(actor, "write:medication")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    return ApiResponse(
        data=service.create_dispense(
            payload.patient_uuid,
            payload.medication_reference,
            payload.quantity,
            payload.unit,
            payload.when_handed_over,
            payload.dosage_text,
        )
    )


@router.get("/population/patients")
def count_patients(query: str = "", actor: Actor = Depends(get_actor), service: PopulationService = Depends(get_population_service)) -> ApiResponse:
    ensure_permission(actor, "read:population")
    return ApiResponse(data=service.count_patients(query))


@router.get("/population/patients/{patient_uuid}/encounters")
def count_encounters(patient_uuid: str, actor: Actor = Depends(get_actor), service: PopulationService = Depends(get_population_service)) -> ApiResponse:
    ensure_permission(actor, "read:population")
    return ApiResponse(data=service.count_encounters(patient_uuid))


@router.get("/population/patients/{patient_uuid}/conditions")
def count_conditions(patient_uuid: str, actor: Actor = Depends(get_actor), service: PopulationService = Depends(get_population_service)) -> ApiResponse:
    ensure_permission(actor, "read:population")
    return ApiResponse(data=service.count_by_condition(patient_uuid))


@router.post("/ingestion/pdf/preview")
async def preview_pdf_ingestion(file: UploadFile = File(...), service: IngestionService = Depends(get_ingestion_service)) -> ApiResponse:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)
    parsed = service.parse_pdf(temp_path)
    temp_path.unlink(missing_ok=True)
    return ApiResponse(data=parsed.model_dump())


@router.post("/ingestion/pdf/execute")
async def execute_pdf_ingestion(
    patient_uuid: str = Form(...),
    confirmed: bool = Form(False),
    file: UploadFile = File(...),
    actor: Actor = Depends(get_actor),
    service: IngestionService = Depends(get_ingestion_service),
    audit: AuditLogger = Depends(get_audit_logger),
) -> ApiResponse:
    ensure_permission(actor, "write:ingestion")
    ensure_confirmation(ConfirmationRequest(confirmed=confirmed))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)
    results = service.ingest_pdf(patient_uuid, str(temp_path))
    temp_path.unlink(missing_ok=True)
    audit.log(
        AuditEvent(
            user_id=actor.user_id,
            role=actor.role,
            intent="ingest_pdf",
            action="Ingest Patient PDF",
            patient_uuid=patient_uuid,
            prompt=None,
            endpoint="MULTI-STEP PDF INGESTION",
            request_payload={"patient_uuid": patient_uuid},
            response_status=200,
            outcome="success",
            metadata={"entities": len(results)},
        )
    )
    return ApiResponse(data=[result.model_dump() for result in results])


@router.post("/health-gorilla/preview")
def preview_health_gorilla(payload: HealthGorillaSearchInput, actor: Actor = Depends(get_actor), service: IngestionService = Depends(get_ingestion_service)) -> ApiResponse:
    ensure_permission(actor, "write:ingestion")
    return ApiResponse(data=service.health_gorilla_preview(payload.given_name, payload.family_name, payload.birthdate))
