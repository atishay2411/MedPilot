from types import SimpleNamespace

from pydantic import BaseModel

from app.llm.base import LLMProvider
from app.llm.models import LLMGenerationResult
from app.services.llm_reasoning import ClinicalNarrative, ConversationalDecision, LLMReasoningService


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


class SequenceProvider(LLMProvider):
    provider_name = "sequence"

    def __init__(self, payloads):
        self.payloads = list(payloads)

    @property
    def enabled(self) -> bool:
        return True

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        return LLMGenerationResult(provider="sequence", model="fake-model", text="ok", raw={})

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema: type[BaseModel]):
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseModel):
            return schema.model_validate(payload.model_dump())
        return schema.model_validate(payload)


def make_settings(**overrides):
    base = {
        "medpilot_llm_enable_intent_reasoning": True,
        "medpilot_llm_enable_summary_reasoning": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_llm_reasoning_returns_inform_when_disabled():
    service = LLMReasoningService(DisabledProvider(), make_settings())

    decision = service.generate_conversational_response("Find patient Maria Santos")

    assert decision.mode == "inform"
    assert "LLM provider" in decision.response_message


def test_llm_reasoning_generates_action_decision():
    service = LLMReasoningService(
        FakeProvider(ConversationalDecision(
            mode="action", intent="search_patient", confidence=0.95,
            entities={"patient_query": "Maria Santos", "search_mode": "default"},
            response_message="Searching for Maria Santos.",
        )),
        make_settings(),
    )

    decision = service.generate_conversational_response("Find patient Maria Santos")

    assert decision.mode == "action"
    assert decision.intent == "search_patient"
    assert decision.entities["patient_query"] == "Maria Santos"


def test_llm_reasoning_generates_clarify_decision():
    service = LLMReasoningService(
        FakeProvider(ConversationalDecision(
            mode="clarify", intent="create_patient",
            clarifying_question="What is the patient's date of birth?",
            response_message="What is the patient's date of birth?",
        )),
        make_settings(),
    )

    decision = service.generate_conversational_response("Add patient Jane Doe")

    assert decision.mode == "clarify"
    assert "date of birth" in decision.clarifying_question


def test_llm_reasoning_renders_summary_when_enabled():
    service = LLMReasoningService(
        FakeProvider(ClinicalNarrative(
            summary="Maria Santos has active diabetes.",
            analysis_points=["BP remains elevated [obs-1]."],
            follow_up=["Review antihypertensive therapy."],
        )),
        make_settings(),
    )

    rendered = service.render_clinical_summary("Maria Santos", {"highlights": {"active_conditions": ["Diabetes"]}, "evidence": []})

    assert "Maria Santos has active diabetes." in rendered
    assert "BP remains elevated" in rendered


def test_llm_reasoning_includes_session_context_in_prompt():
    provider = CapturingProvider(ConversationalDecision(
        mode="action", intent="switch_patient", confidence=0.91,
        entities={"patient_query": "Maria Santos"},
        response_message="Switching to Maria Santos.",
    ))
    service = LLMReasoningService(provider, make_settings())

    decision = service.generate_conversational_response(
        "Use Maria Santos instead",
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

    assert decision.intent == "switch_patient"
    assert "Active patient: John Doe (UUID: patient-123)" in provider.calls[0]["user_prompt"]
    assert "Summarize this patient" in provider.calls[0]["user_prompt"]


def test_llm_reasoning_includes_pending_clarification_in_context():
    provider = CapturingProvider(ConversationalDecision(
        mode="action", intent="create_patient", write=True, confidence=0.95,
        entities={"given_name": "Jane", "family_name": "Doe", "birthdate": "1990-01-01", "gender": "F"},
        response_message="Creating patient Jane Doe.",
    ))
    service = LLMReasoningService(provider, make_settings())

    service.generate_conversational_response(
        "1990-01-01",
        session_state={
            "id": "session-1",
            "pending_clarification": "What is Jane's date of birth?",
            "recent_turns": [
                {"role": "user", "content": "Add patient Jane Doe"},
                {"role": "assistant", "content": "What is Jane's date of birth?"},
            ],
        },
    )

    prompt_sent = provider.calls[0]["user_prompt"]
    assert "PENDING CLARIFICATION" in prompt_sent
    assert "What is Jane's date of birth?" in prompt_sent


def test_llm_reasoning_includes_structured_pending_workflow_in_context():
    provider = CapturingProvider(ConversationalDecision(
        mode="action",
        intent="create_patient",
        confidence=0.95,
        entities={"given_name": "Jane", "family_name": "Doe", "birthdate": "1990-01-01"},
        response_message="Creating Jane Doe.",
    ))
    service = LLMReasoningService(provider, make_settings())

    service.generate_conversational_response(
        "1990-01-01",
        session_state={
            "id": "session-1",
            "pending_workflow": {
                "intent": "create_patient",
                "original_prompt": "Add patient Jane Doe",
                "collected_entities": {"given_name": "Jane", "family_name": "Doe"},
                "missing_fields": ["birthdate"],
                "clarifying_question": "What is Jane Doe's birthdate?",
            },
        },
    )

    prompt_sent = provider.calls[0]["user_prompt"]
    assert "Structured pending workflow" in prompt_sent
    assert "\"missing_fields\": [" in prompt_sent


def test_llm_reasoning_advertises_delete_patient_capability():
    provider = CapturingProvider(ConversationalDecision(
        mode="action",
        intent="delete_patient",
        confidence=0.94,
        entities={"patient_query": "Jane Doe"},
        response_message="Preparing patient deletion.",
    ))
    service = LLMReasoningService(provider, make_settings())

    service.generate_conversational_response("Delete patient Jane Doe")

    assert "delete_patient" in provider.calls[0]["system_prompt"]


def test_llm_reasoning_repairs_weak_first_pass_decision():
    """Collapsed fallback pass replaces old repair_conversational_response."""
    provider = CapturingProvider(
        ConversationalDecision(
            mode="action",
            intent="search_patient",
            confidence=0.94,
            entities={"patient_query": "Maria Santos", "search_mode": "default"},
            response_message="Searching for Maria Santos.",
        )
    )
    service = LLMReasoningService(provider, make_settings())
    initial = ConversationalDecision(
        mode="clarify",
        confidence=0.9,
        clarifying_question="What system should I search?",
        response_message="What system should I search?",
    )

    repaired = service.run_fallback_decision("Find patient Maria Santos", initial)

    assert repaired.mode == "action"
    assert repaired.intent == "search_patient"
    assert repaired.entities["patient_query"] == "Maria Santos"


def test_llm_reasoning_resolves_missing_operational_intent():
    """Collapsed fallback pass replaces old resolve_operational_decision."""
    service = LLMReasoningService(
        SequenceProvider([
            {
                "mode": "action",
                "intent": "create_patient",
                "write": True,
                "confidence": 0.93,
                "entities": {"given_name": "Sahil", "family_name": "Rochwani"},
                "clarifying_question": "What is Sahil Rochwani's birthdate?",
                "response_message": "I need Sahil Rochwani's birthdate before I can register the patient.",
            }
        ]),
        make_settings(),
    )
    initial = ConversationalDecision(mode="action", intent=None, confidence=0.9, entities={}, response_message="Creating patient.")

    resolved = service.run_fallback_decision("Add a patient named Sahil Rochwani", initial)

    assert resolved.intent == "create_patient"
    assert resolved.entities["given_name"] == "Sahil"


def test_llm_reasoning_merges_clarification_answer_with_prior_context():
    service = LLMReasoningService(
        SequenceProvider([
            {
                "mode": "action",
                "intent": "create_patient",
                "write": True,
                "confidence": 0.97,
                "entities": {
                    "given_name": "Sahil",
                    "family_name": "Rochwani",
                    "birthdate": "2000-04-24",
                    "gender": "U",
                },
                "clarifying_question": None,
                "response_message": "I’m ready to register Sahil Rochwani.",
            }
        ]),
        make_settings(),
    )
    initial = ConversationalDecision(mode="action", intent="create_patient", write=True, confidence=0.9, entities={"birthdate": "2000-04-24"}, response_message="Creating patient.")

    resolved = service.resolve_clarification_answer(
        "24 apr 2000",
        initial,
        session_state={
            "pending_clarification": "Please provide the birthdate for patient Sahil Rochwani.",
            "recent_turns": [
                {"role": "user", "content": "Add a patient called sahil rochwani to openmrs"},
                {"role": "assistant", "content": "Please provide the birthdate for patient Sahil Rochwani."},
            ],
        },
    )

    assert resolved.intent == "create_patient"
    assert resolved.entities["given_name"] == "Sahil"
    assert resolved.entities["family_name"] == "Rochwani"
    assert resolved.entities["birthdate"] == "2000-04-24"
