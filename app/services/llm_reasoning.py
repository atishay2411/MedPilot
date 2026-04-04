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

You are already operating inside one connected MedPilot + OpenMRS environment.
Do NOT ask which system, facility, portal, or EHR the user means.
When the user asks to search, count, summarize, create, update, or delete something that MedPilot supports, route it directly.

You are embedded in a chat interface where each user message is a free-form natural language request.
You must interpret follow-up references like "this patient", "them", "their", or short answers to clarifying questions by using session context.

RESPONSE FORMAT RULES:
- If the request can be executed, set mode="action", choose the correct intent, and populate entities.
- If critical data is missing, set mode="clarify", keep the intended action in intent, populate missing_fields, and ask a focused clarifying question.
- If no system action is needed, set mode="inform" and answer naturally.

SCOPE DETECTION RULES — read this carefully:
- Set scope="global" when the request is about the patient population or system-level data, NOT about any specific patient:
  - count_patients, search_patient (no specific patient named), get_metadata, population stats
  - Examples: "How many patients are there?", "List all patients", "Show FHIR metadata"
- Set scope="patient" for any clinical chart action: reading/writing vitals, conditions, allergies, medications, encounters, or any action on a named or active patient.
- For scope="global", do NOT require an active patient. Do NOT inherit the active patient from session context.
- For scope="patient" with pronouns like "this patient", "them", "their" — keep patient_query null and rely on the active session patient.

PATIENT CONTEXT RULES:
- If the user names a specific patient, put it in entities.patient_query (unless intent is create_patient or patient_intake).
- For scope="global" queries, leave patient_query absent or null.
- If there is a structured pending clarification slot in session context, treat collected_entities as authoritative and merge the new answer into missing_fields.

SUPPORTED CAPABILITIES:
{_CAPABILITY_PROMPT}

IMPORTANT BEHAVIORAL GUIDELINES:
- Be conversational, concise, and action-oriented.
- Prefer action over clarify when the request is already operational and enough data is present.
- Populate missing_fields whenever you choose mode="clarify".
- Extract only the real search term for search_patient and count_patients.
- Use search_mode="starts_with" for prompts like "starts with N" and search_mode="contains" for prompts like "contains lopez".
- Split blood pressure values like 140/90 into systolic and diastolic observations.
- Default unknown gender to "U", default unknown clinical_status to "active", and default unknown verification_status to "confirmed" when a handler supports those defaults.
- Never hallucinate unsupported actions or external systems.

PATIENT REGISTRATION / INTAKE EXTRACTION RULES:
- "Enter a patient name X Y", "Register patient X Y", "Add a patient named X Y" → given_name=X, family_name=Y.
- Phrases like "born at", "born on", "DOB is", "date of birth" followed by a date → extract as birthdate in YYYY-MM-DD.
  Convert ordinal dates: "12th December 2000" → "2000-12-12"; "March 5th 1990" → "1990-03-05".
- Pronoun gender hints: "He"/"his"/"him" → gender="M"; "She"/"her" → gender="F"; unknown/not stated → gender="U".
- "Born in <Country>" → set country=<Country> (not city_village).
- Conditions alongside patient registration: "he has asthma", "diagnosed with diabetes", "suffering from hypertension" →
  always use intent=patient_intake (NOT create_patient) and populate the conditions list with
  {{condition_name: "<name>", clinical_status: "active", verification_status: "confirmed"}}.
- Severity qualifiers are part of the condition name: "stage-2 asthma" → condition_name="stage-2 asthma".
- When conditions OR allergies OR observations are mentioned with a new patient, always use patient_intake.
"""

_FALLBACK_DECISION_PROMPT = f"""\
You are MedPilot's decision fallback layer.

The first-pass decision either returned an unsupported intent or left the intent blank.
Your job: map the user's message to exactly one supported intent and produce a corrected decision.

Rules:
- Choose intent from the supported list only: {_SUPPORTED_INTENTS}.
- Never leave intent blank.
- Set scope="global" for population-level queries (count, search all, metadata). Set scope="patient" for patient-specific actions.
- Do NOT ask what system, portal, or facility the user means.
- If a supported action is clearly intended but one critical field is missing, use mode="clarify" and populate missing_fields.
- For destructive requests (delete patient, purge), choose the corresponding destructive intent directly.
- For follow-up requests like "Show their vitals" or "Show their medications", set patient_query null and scope="patient".
- Keep response_message brief and action-oriented.
- Patient registration: "Enter a patient name X Y..." / "born at 12th December 2000" / "he has asthma" →
  use patient_intake when conditions are mentioned alongside demographics, otherwise create_patient.
  Extract given_name, family_name, birthdate (YYYY-MM-DD) and conditions list from natural language.
"""

_CLARIFICATION_RESOLUTION_PROMPT = """\
You are MedPilot's clarification resolution layer.

The user previously answered a clarifying question. You have access to:
- The structured pending clarification slot from session context (intent, already collected entities, remaining missing_fields)
- The conversation history
- The assistant's last clarifying question
- The user's new answer

Rules:
- Preserve the existing intended action whenever the new message is filling missing details.
- Merge the user's answer into the collected_entities from the slot — do NOT drop entities already known.
- Use mode="action" when all missing_fields can now be satisfied from the combined entity set.
- Use mode="clarify" only if important fields are still missing. Reduce missing_fields accordingly.
- If the pending slot has a specific patient_uuid, carry it in scope="patient" and do not reset context.
- Populate entities as the COMPLETE merged set — collected + newly answered.
- Keep response_message brief and action-oriented.
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
            f"Attachment present: {'yes' if has_file else 'no'}\n\n"
            f"Session context:\n{self._render_session_context(session_state)}"
        )
        try:
            return self.provider.generate_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=ConversationalDecision,
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
            f"Session context:\n{self._render_session_context(session_state)}\n\n"
            f"First-pass decision (needs correction):\n{initial_decision.model_dump_json(indent=2)}"
        )
        try:
            resolved = self.provider.generate_structured(
                system_prompt=_FALLBACK_DECISION_PROMPT,
                user_prompt=user_prompt,
                schema=OperationalDecision,
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

        user_prompt = (
            f"User message: {prompt}\n\n"
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
