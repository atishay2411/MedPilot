from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.core.exceptions import LLMProviderError
from app.llm.base import LLMProvider
from app.models.domain import ParsedIntent


class ClinicalNarrative(BaseModel):
    summary: str
    analysis_points: list[str] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)


class LLMIntentOutput(BaseModel):
    intent: str
    write: bool
    confidence: float
    entities: dict[str, Any] = Field(default_factory=dict)


class LLMReasoningService:
    def __init__(self, provider: LLMProvider, settings: Settings):
        self.provider = provider
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.provider.enabled

    def resolve_intent(
        self,
        prompt: str,
        deterministic: ParsedIntent,
        *,
        has_file: bool = False,
        session_state: dict[str, Any] | None = None,
    ) -> ParsedIntent:
        if not (self.enabled and self.settings.medpilot_llm_enable_intent_reasoning):
            return deterministic

        system_prompt = (
            "You are MedPilot's intent extraction layer for a clinical OpenMRS copilot. "
            "Map the user's request into one supported intent and extract only grounded entities. "
            "Supported intents include search_patient, switch_patient, get_metadata, patient_analysis, get_observations, create_observation, "
            "update_observation, delete_observation, get_conditions, create_condition, update_condition, delete_condition, "
            "get_allergies, create_allergy, update_allergy, delete_allergy, get_medications, get_medication_dispense, "
            "create_medication, create_medication_dispense, update_medication, create_patient, patient_intake, create_encounter, ingest_pdf, sync_health_gorilla. "
            "Do not invent entities that are not implied by the prompt. Use patient_intake when a new patient is being created with multiple related clinical items in one request. "
            "Use the session context to resolve follow-up references like 'this patient', 'switch back', 'that chart', or omitted patient names. "
            "For search_patient, extract only the patient identifier or name and strip conversational wrappers such as 'is there a patient called', 'find any related patient whose name is', or 'do we have'. "
            "When the user asks for starts-with or contains matching, include a search_mode entity with values like starts_with, contains, or default."
        )
        user_prompt = (
            f"Prompt:\n{prompt}\n\n"
            f"Attachment present: {'yes' if has_file else 'no'}\n\n"
            f"Session context:\n{self._render_session_context(session_state)}\n\n"
            f"Deterministic candidate:\n{deterministic.model_dump_json(indent=2)}"
        )
        try:
            structured = self.provider.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=LLMIntentOutput,
            )
        except LLMProviderError:
            return deterministic

        llm_parsed = ParsedIntent.model_validate(structured.model_dump())
        return self._merge_intents(deterministic, llm_parsed)

    def render_clinical_summary(self, patient_display: str, brief: dict[str, Any], *, session_state: dict[str, Any] | None = None) -> str | None:
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
    def _render_session_context(session_state: dict[str, Any] | None) -> str:
        if not session_state:
            return "No prior session context."

        lines: list[str] = []
        if session_state.get("current_patient_display") or session_state.get("current_patient_uuid"):
            lines.append(
                "Active patient: "
                f"{session_state.get('current_patient_display') or 'Unknown'} "
                f"({session_state.get('current_patient_uuid') or 'unknown uuid'})"
            )
        if session_state.get("last_intent"):
            lines.append(f"Last intent: {session_state['last_intent']}")

        recent_turns = session_state.get("recent_turns") or []
        if recent_turns:
            lines.append("Recent turns:")
            for turn in recent_turns[-6:]:
                role = turn.get("role", "unknown")
                content = str(turn.get("content", "")).strip().replace("\n", " ")
                if len(content) > 180:
                    content = content[:177] + "..."
                lines.append(f"- {role}: {content}")

        return "\n".join(lines) if lines else "No prior session context."

    @staticmethod
    def _merge_intents(deterministic: ParsedIntent, llm_parsed: ParsedIntent) -> ParsedIntent:
        if deterministic.intent == "create_patient" and llm_parsed.intent == "patient_intake" and not LLMReasoningService._has_clinical_payload(llm_parsed.entities):
            llm_parsed = ParsedIntent(intent="create_patient", write=True, confidence=llm_parsed.confidence, entities=llm_parsed.entities)
        if deterministic.intent == "patient_intake" and llm_parsed.intent == "create_patient" and LLMReasoningService._has_clinical_payload(deterministic.entities):
            llm_parsed = ParsedIntent(intent="patient_intake", write=True, confidence=llm_parsed.confidence, entities=llm_parsed.entities)
        if deterministic.intent == "search_patient" and llm_parsed.intent == "search_patient":
            llm_parsed = ParsedIntent(
                intent="search_patient",
                write=False,
                confidence=llm_parsed.confidence,
                entities=LLMReasoningService._prefer_more_specific_search_query(llm_parsed.entities, deterministic.entities),
            )
        if llm_parsed.confidence < 0.55:
            return deterministic
        if deterministic.confidence < 0.6 and llm_parsed.confidence >= 0.6:
            merged_entities = LLMReasoningService._merge_entity_maps(llm_parsed.entities, deterministic.entities)
            return ParsedIntent(intent=llm_parsed.intent, write=llm_parsed.write, confidence=llm_parsed.confidence, entities=merged_entities)
        chosen = llm_parsed if llm_parsed.confidence >= deterministic.confidence - 0.05 else deterministic
        fallback = deterministic if chosen is llm_parsed else llm_parsed
        merged_entities = LLMReasoningService._merge_entity_maps(chosen.entities, fallback.entities)
        return ParsedIntent(intent=chosen.intent, write=chosen.write, confidence=max(chosen.confidence, fallback.confidence), entities=merged_entities)

    @staticmethod
    def _merge_entity_maps(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
        merged = dict(primary)
        for key, value in secondary.items():
            if key not in merged or merged[key] in (None, "", [], {}):
                merged[key] = value
            elif isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = LLMReasoningService._merge_entity_maps(merged[key], value)
        return merged

    @staticmethod
    def _has_clinical_payload(entities: dict[str, Any]) -> bool:
        return any(entities.get(key) for key in ("conditions", "allergies", "observations", "medications", "dispenses"))

    @staticmethod
    def _prefer_more_specific_search_query(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
        merged = LLMReasoningService._merge_entity_maps(primary, secondary)
        primary_query = str(primary.get("patient_query") or "").strip()
        secondary_query = str(secondary.get("patient_query") or "").strip()
        if primary_query and secondary_query:
            primary_tokens = len(primary_query.split())
            secondary_tokens = len(secondary_query.split())
            if secondary_query.lower() in primary_query.lower() and secondary_tokens < primary_tokens:
                merged["patient_query"] = secondary_query
            elif primary_query.lower() in secondary_query.lower() and primary_tokens < secondary_tokens:
                merged["patient_query"] = primary_query
        return merged
