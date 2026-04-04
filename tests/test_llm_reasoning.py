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


class CapturingProvider(FakeProvider):
    def __init__(self, payload):
        super().__init__(payload)
        self.calls: list[dict[str, str]] = []

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema: type[BaseModel]):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return super().generate_structured(system_prompt=system_prompt, user_prompt=user_prompt, schema=schema)


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


def test_llm_reasoning_includes_session_context_in_prompt():
    provider = CapturingProvider(LLMIntentOutput(intent="switch_patient", write=False, confidence=0.91, entities={"patient_query": "Maria Santos"}))
    service = LLMReasoningService(provider, make_settings())
    deterministic = ParsedIntent(intent="search_patient", write=False, confidence=0.5, entities={"patient_query": None})

    resolved = service.resolve_intent(
        "Use Maria Santos instead",
        deterministic,
        session_state={
            "id": "session-1",
            "current_patient_uuid": "patient-123",
            "current_patient_display": "John Doe",
            "last_intent": "patient_analysis",
            "recent_turns": [
                {"role": "user", "content": "Summarize this patient"},
                {"role": "assistant", "content": "Here is the summary for John Doe."},
            ],
        },
    )

    assert resolved.intent == "switch_patient"
    assert "Active patient: John Doe (patient-123)" in provider.calls[0]["user_prompt"]
    assert "Summarize this patient" in provider.calls[0]["user_prompt"]


def test_llm_reasoning_keeps_create_patient_when_llm_overcalls_intake():
    service = LLMReasoningService(
        FakeProvider(LLMIntentOutput(intent="patient_intake", write=True, confidence=0.96, entities={"given_name": "Test", "family_name": "Test", "birthdate": "2000-04-24"})),
        make_settings(),
    )
    deterministic = ParsedIntent(intent="create_patient", write=True, confidence=0.95, entities={"given_name": "Test", "family_name": "Test", "birthdate": "2000-04-24"})

    resolved = service.resolve_intent("Add a patient Test Test who is a male and was born on 24 apr 2000 in Chicago", deterministic)

    assert resolved.intent == "create_patient"


def test_llm_reasoning_prefers_llm_when_deterministic_search_is_low_confidence():
    service = LLMReasoningService(
        FakeProvider(LLMIntentOutput(intent="search_patient", write=False, confidence=0.88, entities={"patient_query": "Nesh test"})),
        make_settings(),
    )
    deterministic = ParsedIntent(intent="search_patient", write=False, confidence=0.45, entities={"patient_query": "any related patient whose name is Nesh test"})

    resolved = service.resolve_intent("find any related patient whose name is Nesh test", deterministic)

    assert resolved.intent == "search_patient"
    assert resolved.entities["patient_query"] == "Nesh test"
