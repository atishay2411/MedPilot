from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import Settings
from app.core.exceptions import LLMProviderError
from app.llm.base import LLMProvider
from app.models.common import PendingClarificationSlot
from app.services.capabilities import render_capability_prompt, supported_intents


class ClinicalNarrative(BaseModel):
    summary: str
    analysis_points: list[str] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)


class ConversationalDecision(BaseModel):
    mode: Literal["action", "clarify", "inform"]
    intent: str | None = None
    write: bool = False
    confidence: float = 0.9
    scope: Literal["global", "patient"] = "patient"
    """'global' for population-level queries that should not inherit active patient context.
    'patient' for chart-level actions that apply to a specific patient."""
    entities: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    response_message: str


class OperationalDecision(BaseModel):
    mode: Literal["action", "clarify"]
    intent: str
    write: bool = False
    confidence: float = 0.9
    scope: Literal["global", "patient"] = "patient"
    entities: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    response_message: str


_SUPPORTED_INTENTS = ", ".join(supported_intents())
_CAPABILITY_PROMPT = render_capability_prompt()

_SYSTEM_PROMPT = f"""\
You are MedPilot, an AI clinical copilot for OpenMRS EHR. You understand natural language and help clinicians work the chart.

You are operating inside a connected MedPilot + OpenMRS environment. Never ask which system or facility the user means.
When the user asks to search, count, summarize, create, update, or delete something MedPilot supports, route it directly.

You are embedded in a chat interface where each message is free-form natural language.
Interpret follow-up references like "this patient", "them", "their", or short clarification answers using session context.

RESPONSE FORMAT RULES:
- Executable request → mode="action", correct intent, populate entities.
- Critical data missing → mode="clarify", keep intent, list missing_fields, ask ONE focused question.
- No system action needed → mode="inform", answer naturally.

SCOPE DETECTION:
- scope="global": population/system queries — count_patients, list/search all patients (no specific name), get_metadata.
  Examples: "How many patients?", "List all patients", "Show FHIR metadata"
- scope="patient": any chart action on a specific or active patient — vitals, conditions, allergies, medications, encounters, summaries.
- For scope="global" do NOT inherit the active patient. For scope="patient" with "this patient"/"them"/"their" keep patient_query=null.

PATIENT CONTEXT RULES:
- Named patient → put ONLY the name (not the word "patient") in entities.patient_query. E.g. "Summarize patient John Smith" → patient_query="John Smith", NOT "patient John Smith".
- List-all / count queries → patient_query=null.
- Pending clarification slot → treat collected_entities as authoritative and merge the new answer.
- If a system message says "Active patient context: NAME (UUID: ...)", use that patient for any patient-specific request where no other patient is named.

OBSERVATION RECORDING RULES:
- Map vital signs to the correct display name and CIEL code:
  - Height → display="Height (cm)", code="5090"
  - Weight → display="Weight (kg)", code="5089"
  - Temperature → display="Temperature (°C)", code="5088"
  - Respiratory rate → display="Respiratory Rate", code="5242"
  - Oxygen saturation / SpO2 → display="Oxygen Saturation (SpO2)", code="5092"
  - Systolic BP → display="Systolic blood pressure", code="5085"
  - Diastolic BP → display="Diastolic blood pressure", code="5086"
  - Pulse rate / Heart rate / Pulse → display="Pulse rate", code="5087", unit=/min
  - BMI / Body mass index → display="BMI", code="1342", unit=kg/m2
  - Head circumference → display="Head circumference", code="5314", unit=cm
  - Blood glucose (random) → display="Blood glucose", code="887", unit=mmol/L
  - Fasting blood glucose → display="Fasting blood glucose", code="2339", unit=mmol/L
  - CD4 count → display="CD4 count", code="5497", unit=cells/µL
  - Hemoglobin → display="Hemoglobin", code="21", unit=g/dL
  - Pain scale / Pain score → display="Pain scale", code="160643", unit="" (value 0-10)
- Blood pressure like "120/80" → TWO observations: systolic=120, diastolic=80.
- Each observation must have: display, code, value (numeric), unit.
- observations field must be a LIST even for a single vital sign.

CLINICAL WRITE RULES:
- create_condition: condition_name is required. clinical_status defaults to "active", verification_status defaults to "confirmed".
- create_condition bulk: when user lists multiple conditions (e.g. "Add conditions: Fever, Cough, Cold"), use intent=create_condition with entities.conditions=["Fever","Cough","Cold"]. Do NOT split into separate messages.
- create_allergy: allergen_name required. severity defaults to "moderate". reaction defaults to "Rash". Always use a general English symptom for reaction (e.g. "Rash", "Itching", "Hives", "Sneezing", "Watery eyes") — never use compound phrases like "throat infection".
- create_condition: NEVER use vaccine, immunization, or vaccination names as conditions. If a user asks to record a vaccine, inform them that immunizations must be recorded via the Immunization module, not conditions.
- create_medication: drug_name, dose, dose_units_name, route_name, frequency_name, duration, duration_units_name, quantity, quantity_units_name all required. For frequency_name use standard OpenMRS names: "Once daily", "Twice daily", "Three times a day", "Four times a day", "Once weekly", "Twice weekly", "Every other day", "As needed". Never output free-form frequencies like "2 times a week".
- delete operations: always resolve the patient first via patient_query.
- create_clinical_note: note_text is required. note_type defaults to "note". Other types: "chief complaint", "clinical impression", "assessment".
- search_drugs: drug_query required. Use when user asks about available drugs/medications/formulary.
- search_providers: Use when user asks about providers/clinicians/doctors.
- search_locations: Use when user asks about locations/wards/clinics/facilities.
- get_encounters: Use when user asks about patient encounters/visit history/appointments attended.
- get_visits: Use when user asks about patient visit records from the visit module.
- get_programs: Use when user asks about patient programs/enrollments or lists all programs.
- get_encounter_types: scope=global. Use when user asks what encounter types are available.

NAME PARSING RULES FOR create_patient:
- "Add a patient named John Smith" → given_name="John", family_name="Smith"
- "Full name: Nesh Rochwani, born 24 April 2000" → given_name="Nesh", family_name="Rochwani", birthdate="2000-04-24"
- Always convert date to YYYY-MM-DD.
- First word after "named"/"called" = given_name; remainder before comma/keyword = family_name.

SEARCH RULES:
- "starts with N" or "beginning with S" → search_mode="starts_with"
- "contains Smith" or "with Lopez" → search_mode="contains"
- By ID / identifier / UUID → intent=search_by_identifier, identifier=<value>

SUPPORTED CAPABILITIES:
{_CAPABILITY_PROMPT}

CONVERSATION HISTORY RULES:
- Conversation history is provided as prior messages in the chat.
- Use the history to resolve references like "their", "this patient", "them", "the one you found", "those patients", "list their names".
- If the user says "list their names" after you fetched 58 patients → intent=search_patient with query=null (list all).
- If the user is answering a clarifying question (e.g. "24 nov 2000" after you asked for DOB) → extract the answer and complete the original intent.
- When history shows a clarify turn for create_patient and user provides the missing field → set mode=action, merge all collected entities.

BEHAVIORAL GUIDELINES:
- Prefer action over clarify when enough data is present.
- Populate missing_fields on every clarify response.
- Default gender="U" when unknown; default clinical_status="active"; default verification_status="confirmed".
- Never hallucinate unsupported actions or external systems.
- CRITICAL: When a clarifying answer provides all required fields (combined with already-collected entities from history), set mode=action immediately.
- CRITICAL: mode MUST be exactly one of: "action", "clarify", "inform". Never use any other value.
"""

_FALLBACK_DECISION_PROMPT = f"""\
You are MedPilot's decision fallback layer.

The first-pass decision returned an unsupported or blank intent. Map the user's message to exactly one supported intent.

Rules:
- Choose from: {_SUPPORTED_INTENTS}. Never leave intent blank.
- scope="global" for population queries (count, search all, metadata). scope="patient" for chart actions.
- Do NOT ask what system or facility the user means — it's always the connected OpenMRS.
- If a required field is missing, use mode="clarify" and list missing_fields.
- "Delete patient X" → delete_patient (destructive=true). "Remove allergy X" → delete_allergy. "Remove condition X" → delete_condition.
- "Their vitals / medications / conditions" (pronoun follow-ups) → patient_query=null, scope="patient".
- List-all / count all → intent=search_patient or count_patients, patient_query=null, scope=global.
- "Record/log/measure/add [vital]" → create_observation, provide observations list with display, code, value, unit.
- "Add/create/diagnose condition" → create_condition; "Add/record allergy" → create_allergy.
- "Prescribe/order medication" → create_medication.
- Keep response_message brief and action-oriented.
- CRITICAL: When all required fields are available, always set mode=action.
"""

_CLARIFICATION_RESOLUTION_PROMPT = """\
You are MedPilot's clarification resolution layer.

The user previously answered a clarifying question. You have access to:
- The structured pending clarification slot from session context (intent, already collected entities, remaining missing_fields)
- The conversation history
- The assistant's last clarifying question
- The user's new answer
- An "Already collected" block that shows entities already known — you MUST include all of these in your output entities

Rules:
- Preserve the existing intended action whenever the new message is filling missing details.
- Merge the user's answer into the collected_entities from the slot — do NOT drop entities already known.
- CRITICAL: If all required fields for the intent are now available (from already-collected + new answer), you MUST set mode="action". Do NOT set mode="clarify" when all required fields are present.
- Use mode="clarify" ONLY if a truly required field is still missing after combining all known data.
- If the pending slot has a specific patient_uuid, carry it in scope="patient" and do not reset context.
- Populate entities as the COMPLETE merged set: start with everything in "Already collected", then add/override with what the user just said.
- For create_patient: required fields are given_name, family_name, and birthdate. Once all three are known, use mode=action.
- For dates like "24 April 2000" or "April 24 2000", convert to YYYY-MM-DD (2000-04-24).
- Keep response_message brief and action-oriented.

Example — create_patient:
  Already collected: given_name="Nesh", family_name="Rochwani"
  User answer: "dob is 24 April 2000"
  → mode=action, intent=create_patient, entities={{given_name: "Nesh", family_name: "Rochwani", birthdate: "2000-04-24"}}
"""


class LLMReasoningService:
    def __init__(self, provider: LLMProvider, settings: Settings):
        self.provider = provider
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.provider.enabled

    def generate_conversational_response(
        self,
        prompt: str,
        *,
        session_state: dict[str, Any] | None = None,
        has_file: bool = False,
        conversation_history: list[dict] | None = None,
    ) -> ConversationalDecision:
        if not self.enabled:
            return ConversationalDecision(
                mode="inform",
                response_message=(
                    "I need an LLM provider to be configured to process your request. "
                    "Please set MEDPILOT_LLM_PROVIDER to 'openai' or 'ollama' and provide the required API keys in your .env file."
                ),
            )

        user_prompt = (
            f"User message: {prompt}\n\n"
            f"Attachment present: {'yes' if has_file else 'no'}"
        )
        try:
            return self.provider.generate_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=ConversationalDecision,
                conversation_history=conversation_history,
            )
        except LLMProviderError as exc:
            return ConversationalDecision(
                mode="inform",
                response_message=f"I encountered an error processing your request: {exc}. Please try again.",
            )

    def run_fallback_decision(
        self,
        prompt: str,
        initial_decision: ConversationalDecision,
        *,
        session_state: dict[str, Any] | None = None,
        has_file: bool = False,
        conversation_history: list[dict] | None = None,
    ) -> ConversationalDecision:
        """Single-pass fallback: collapsed repair + operational resolver.

        Only called when the first pass returned an unsupported or blank intent.
        This replaces the previous two-pass repair → operational chain.
        """
        if not self.enabled:
            return initial_decision

        user_prompt = (
            f"User message: {prompt}\n\n"
            f"Attachment present: {'yes' if has_file else 'no'}\n\n"
            f"First-pass decision (needs correction):\n{initial_decision.model_dump_json(indent=2)}"
        )
        try:
            resolved = self.provider.generate_structured(
                system_prompt=_FALLBACK_DECISION_PROMPT,
                user_prompt=user_prompt,
                schema=OperationalDecision,
                conversation_history=conversation_history,
            )
        except LLMProviderError:
            return initial_decision
        return ConversationalDecision.model_validate(resolved.model_dump())

    def resolve_clarification_answer(
        self,
        prompt: str,
        initial_decision: ConversationalDecision,
        *,
        session_state: dict[str, Any] | None = None,
    ) -> ConversationalDecision:
        if not self.enabled:
            return initial_decision

        # Pre-inject collected entities deterministically so small models
        # don't lose already-known fields when producing the JSON response.
        already_collected_block = self._render_already_collected(session_state)

        user_prompt = (
            f"User message: {prompt}\n\n"
            f"{already_collected_block}"
            f"Session context:\n{self._render_session_context(session_state)}\n\n"
            f"Current decision:\n{initial_decision.model_dump_json(indent=2)}"
        )
        try:
            resolved = self.provider.generate_structured(
                system_prompt=_CLARIFICATION_RESOLUTION_PROMPT,
                user_prompt=user_prompt,
                schema=OperationalDecision,
            )
        except LLMProviderError:
            return initial_decision
        return ConversationalDecision.model_validate(resolved.model_dump())

    @staticmethod
    def _render_already_collected(session_state: dict[str, Any] | None) -> str:
        """Render a clear 'Already collected' block from the pending clarification slot.

        This is injected deterministically into the clarification resolution prompt
        so even small LLMs don't lose previously-known entities.
        """
        if not session_state:
            return ""
        pending = session_state.get("pending_clarification")
        if not pending or not isinstance(pending, dict):
            return ""
        collected = pending.get("collected_entities") or {}
        intent = pending.get("intent") or "unknown"
        missing = pending.get("missing_fields") or []
        if not collected:
            return ""
        lines = ["Already collected (you MUST include ALL of these in your output entities):"]
        for k, v in collected.items():
            if v is not None and v != "":
                lines.append(f"  {k}: {json.dumps(v)}")
        lines.append(f"Intent to complete: {intent}")
        if missing:
            lines.append(f"Still missing: {', '.join(missing)}")
        lines.append("")
        return "\n".join(lines) + "\n"

    def render_clinical_summary(
        self,
        patient_display: str,
        brief: dict[str, Any],
        *,
        session_state: dict[str, Any] | None = None,
    ) -> str | None:
        if not (self.enabled and self.settings.medpilot_llm_enable_summary_reasoning):
            return None

        system_prompt = (
            "You are MedPilot's clinical summarization layer. "
            "Use only the supplied chart data and evidence. "
            "Do not hallucinate diagnoses, vitals, or plans. "
            "If data is missing, say it is not documented. "
            "Produce a concise clinician-facing summary and 2-5 analysis points grounded in the evidence."
        )
        user_prompt = (
            f"Patient: {patient_display}\n\n"
            f"Session context:\n{self._render_session_context(session_state)}\n\n"
            f"Deterministic brief:\n{json.dumps(brief, ensure_ascii=True, indent=2)}"
        )
        try:
            response = self.provider.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=ClinicalNarrative,
            )
        except LLMProviderError:
            return None

        parts = [response.summary.strip()]
        if response.analysis_points:
            parts.append("Clinical analysis: " + " ".join(point.strip() for point in response.analysis_points if point.strip()))
        if response.follow_up:
            parts.append("Suggested follow-up: " + " ".join(item.strip() for item in response.follow_up if item.strip()))
        return " ".join(part for part in parts if part)

    @staticmethod
    def _render_slot_context(slot: PendingClarificationSlot) -> str:
        """Serialize a structured clarification slot for LLM prompts."""
        lines = [
            f"PENDING CLARIFICATION SLOT:",
            f"  intent: {slot.intent or 'unknown'}",
            f"  question asked: \"{slot.question}\"",
            f"  missing fields: {', '.join(slot.missing_fields) if slot.missing_fields else 'none'}",
            f"  turn_count: {slot.turn_count}",
        ]
        if slot.collected_entities:
            lines.append(f"  collected entities: {json.dumps(slot.collected_entities, ensure_ascii=True)}")
        if slot.patient_uuid:
            lines.append(f"  patient context: {slot.patient_display or 'Unknown'} (UUID: {slot.patient_uuid})")
        lines.append("The user's current message is likely answering this question. Merge it deterministically.")
        return "\n".join(lines)

    @staticmethod
    def _render_session_context(session_state: dict[str, Any] | None) -> str:
        if not session_state:
            return "No prior session context. This is a new conversation."

        lines: list[str] = []
        if session_state.get("current_patient_display") or session_state.get("current_patient_uuid"):
            lines.append(
                "Active patient: "
                f"{session_state.get('current_patient_display') or 'Unknown'} "
                f"(UUID: {session_state.get('current_patient_uuid') or 'unknown'})"
            )
        else:
            lines.append("No active patient in this session.")

        if session_state.get("last_intent"):
            lines.append(f"Last action performed: {session_state['last_intent']}")

        pending_clarification = session_state.get("pending_clarification")
        if pending_clarification:
            # pending_clarification is now a dict from PendingClarificationSlot.model_dump()
            if isinstance(pending_clarification, dict) and pending_clarification.get("question"):
                slot = PendingClarificationSlot.model_validate(pending_clarification)
                lines.append(LLMReasoningService._render_slot_context(slot))
            elif isinstance(pending_clarification, str):
                # Backward compat: old sessions stored plain text
                lines.append(f'PENDING CLARIFICATION: "{pending_clarification}"')
                lines.append("The user's current message may be answering this question.")

        if session_state.get("pending_workflow"):
            lines.append("Structured pending workflow:")
            lines.append(json.dumps(session_state["pending_workflow"], ensure_ascii=True, indent=2))

        recent_turns = session_state.get("recent_turns") or []
        if recent_turns:
            lines.append(f"\nConversation history ({len(recent_turns)} recent turns):")
            for turn in recent_turns[-12:]:
                role = turn.get("role", "unknown")
                content = str(turn.get("content", "")).strip().replace("\n", " ")
                intent = turn.get("intent")
                if len(content) > 250:
                    content = content[:247] + "..."
                entry = f"  [{role.upper()}]: {content}"
                if intent:
                    entry += f"  [intent: {intent}]"
                lines.append(entry)

        return "\n".join(lines)
