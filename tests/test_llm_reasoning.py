from types import SimpleNamespace

from pydantic import BaseModel

from app.llm.base import LLMProvider
from app.llm.models import LLMGenerationResult
from app.models.domain import ParsedIntent
from app.services.llm_reasoning import ClinicalNarrative, LLMIntentOutput, LLMReasoningService


class DisabledProvider(LLMProvider):
    provider_name = "none"

    @property
    def enabled(self) -> bool:
        return False

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> LLMGenerationResult:  # pragma: no cover
        raise AssertionError("Should not be called")

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema: type[BaseModel]):  # pragma: no cover
        raise AssertionError("Should not be called")


class FakeProvider(LLMProvider):
    provider_name = "fake"

    def __init__(self, payload):
        self.payload = payload

    @property
    def enabled(self) -> bool:
        return True

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        return LLMGenerationResult(provider="fake", model="fake-model", text="ok", raw={})

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema: type[BaseModel]):
        if isinstance(self.payload, BaseModel):
            return schema.model_validate(self.payload.model_dump())
        return schema.model_validate(self.payload)


def make_settings(**overrides):
    base = {
        "medpilot_llm_enable_intent_reasoning": True,
        "medpilot_llm_enable_summary_reasoning": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_llm_reasoning_falls_back_when_disabled():
    service = LLMReasoningService(DisabledProvider(), make_settings())
    deterministic = ParsedIntent(intent="search_patient", write=False, confidence=0.6, entities={"patient_query": "Maria Santos"})

    resolved = service.resolve_intent("Find patient Maria Santos", deterministic)

    assert resolved == deterministic


def test_llm_reasoning_merges_missing_entities_from_deterministic():
    service = LLMReasoningService(
        FakeProvider(LLMIntentOutput(intent="patient_intake", write=True, confidence=0.92, entities={"conditions": [{"condition_name": "diabetes"}]})),
        make_settings(),
    )
    deterministic = ParsedIntent(
        intent="patient_intake",
        write=True,
        confidence=0.8,
        entities={"given_name": "John", "family_name": "Doe", "conditions": [{"condition_name": "diabetes"}]},
    )

    resolved = service.resolve_intent("Add a new patient named John Doe with diabetes", deterministic)

    assert resolved.intent == "patient_intake"
    assert resolved.entities["given_name"] == "John"
    assert resolved.entities["family_name"] == "Doe"


def test_llm_reasoning_renders_summary_when_enabled():
    service = LLMReasoningService(
        FakeProvider(ClinicalNarrative(summary="Maria Santos has active diabetes.", analysis_points=["BP remains elevated [obs-1]."], follow_up=["Review antihypertensive therapy."])),
        make_settings(),
    )

    rendered = service.render_clinical_summary("Maria Santos", {"highlights": {"active_conditions": ["Diabetes"]}, "evidence": []})

    assert "Maria Santos has active diabetes." in rendered
    assert "BP remains elevated" in rendered
