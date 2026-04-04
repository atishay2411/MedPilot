from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from backend.services.nlp import resolve_intent
from backend.services import openmrs

router = APIRouter(prefix="/api/chat", tags=["chat"])


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    intent: str
    message: str
    params: dict
    requires_confirmation: bool
    data: dict | None = None


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Accepts conversation history, resolves intent,
    and executes the action (or returns confirmation prompt).
    """
    history = [{"role": m.role, "content": m.content} for m in request.messages]

    # Resolve intent via OpenAI
    result = resolve_intent(history)

    intent = result.get("intent", "general")
    params = result.get("params", {})
    requires_confirmation = result.get("requires_confirmation", False)
    message = result.get("message", "")
    data = None

    # If it's a read-only intent, execute immediately
    if intent == "search_patient":
        query = params.get("query", params.get("patient_id", ""))
        data = {"patients": openmrs.search_patients(query)}

    elif intent == "get_patient":
        pid = params.get("patient_uuid") or params.get("patient_id")
        if pid:
            uuid = openmrs.get_patient_uuid(pid) if len(pid) < 36 else pid
            data = {"patient": openmrs.get_patient(uuid)}

    elif intent == "get_vitals":
        pid = params.get("patient_uuid") or openmrs.get_patient_uuid(params.get("patient_id", ""))
        if pid:
            data = {"vitals": openmrs.get_vitals(pid)}

    elif intent == "get_conditions":
        pid = params.get("patient_uuid") or openmrs.get_patient_uuid(params.get("patient_id", ""))
        if pid:
            data = {"conditions": openmrs.get_conditions(pid)}

    elif intent == "get_allergies":
        pid = params.get("patient_uuid") or openmrs.get_patient_uuid(params.get("patient_id", ""))
        if pid:
            data = {"allergies": openmrs.get_allergies(pid)}

    elif intent == "get_medications":
        pid = params.get("patient_uuid") or openmrs.get_patient_uuid(params.get("patient_id", ""))
        if pid:
            data = {"medications": openmrs.get_medications(pid)}

    # Write intents only execute if confirmed (confirmed=true in params)
    elif intent in ["add_vital", "add_condition", "add_allergy", "add_medication",
                    "delete_condition", "delete_allergy", "update_condition", "create_patient"]:
        if params.get("confirmed"):
            data = _execute_write(intent, params)
        # else: just return the confirmation prompt

    return ChatResponse(
        intent=intent,
        message=message,
        params=params,
        requires_confirmation=requires_confirmation and not params.get("confirmed"),
        data=data
    )


def _execute_write(intent: str, params: dict) -> dict:
    """Execute confirmed write operations."""
    pid_raw = params.get("patient_id", "")
    patient_uuid = params.get("patient_uuid") or openmrs.get_patient_uuid(pid_raw)

    if intent == "add_vital":
        return openmrs.add_observation(
            patient_uuid,
            params.get("concept_name"),
            params.get("value")
        )

    elif intent == "add_condition":
        return openmrs.add_condition(
            patient_uuid,
            params.get("condition_name"),
            params.get("clinical_status", "ACTIVE"),
            params.get("verification_status", "CONFIRMED"),
            params.get("onset_date")
        )

    elif intent == "update_condition":
        return openmrs.update_condition(
            params.get("condition_uuid"),
            params.get("clinical_status")
        )

    elif intent == "delete_condition":
        return openmrs.delete_condition(params.get("condition_uuid"))

    elif intent == "add_allergy":
        return openmrs.add_allergy(
            patient_uuid,
            params.get("allergen_name"),
            params.get("severity_name"),
            params.get("reaction_name"),
            params.get("comment", "")
        )

    elif intent == "delete_allergy":
        return openmrs.delete_allergy(patient_uuid, params.get("allergy_uuid"))

    elif intent == "create_patient":
        return openmrs.create_patient(params.get("patient_data", {}))

    return {"success": False, "error": "Unknown write intent"}
