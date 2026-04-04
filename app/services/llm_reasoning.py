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

    def resolve_intent(self, prompt: str, deterministic: ParsedIntent, *, has_file: bool = False) -> ParsedIntent:
        if not (self.enabled and self.settings.medpilot_llm_enable_intent_reasoning):
            return deterministic

        system_prompt = (
            "You are MedPilot's intent extraction layer for a clinical OpenMRS copilot. "
            "Map the user's request into one supported intent and extract only grounded entities. "
            "Supported intents include search_patient, get_metadata, patient_analysis, get_observations, create_observation, "
            "update_observation, delete_observation, get_conditions, create_condition, update_condition, delete_condition, "
            "get_allergies, create_allergy, update_allergy, delete_allergy, get_medications, get_medication_dispense, "
            "create_medication, create_medication_dispense, update_medication, create_patient, patient_intake, create_encounter, ingest_pdf, sync_health_gorilla. "
            "Do not invent entities that are not implied by the prompt. Use patient_intake when a new patient is being created with multiple related clinical items in one request."
        )
        user_prompt = (
            f"Prompt:\n{prompt}\n\n"
            f"Attachment present: {'yes' if has_file else 'no'}\n\n"
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

    def render_clinical_summary(self, patient_display: str, brief: dict[str, Any]) -> str | None:
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
    def _merge_intents(deterministic: ParsedIntent, llm_parsed: ParsedIntent) -> ParsedIntent:
        if llm_parsed.confidence < 0.55:
            return deterministic
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
