from pathlib import Path
from types import SimpleNamespace

from app.models.common import ChatHistoryTurn, PendingWorkflowState
from app.services.chat_sessions import ChatSessionStore


def test_chat_session_store_creates_and_persists_session(tmp_path: Path):
    settings = SimpleNamespace(chat_sessions_path=tmp_path)
    store = ChatSessionStore(settings)

    session = store.create()
    assert (tmp_path / f"{session.id}.json").exists()

    session = store.set_current_patient(session, "patient-123", "Maria Santos")
    session = store.append_turn(session, ChatHistoryTurn(role="user", content="Summarize this patient", patient_uuid="patient-123"))
    loaded = store.get(session.id)

    assert loaded.current_patient_uuid == "patient-123"
    assert loaded.current_patient_display == "Maria Santos"
    assert loaded.recent_turns[-1].content == "Summarize this patient"
    assert loaded.snapshot()["current_patient_display"] == "Maria Santos"


def test_chat_session_store_persists_pending_workflow(tmp_path: Path):
    settings = SimpleNamespace(chat_sessions_path=tmp_path)
    store = ChatSessionStore(settings)

    session = store.create()
    workflow = PendingWorkflowState(
        intent="create_patient",
        original_prompt="Add patient Jane Doe",
        collected_entities={"given_name": "Jane", "family_name": "Doe"},
        missing_fields=["birthdate"],
        clarifying_question="What is Jane Doe's birthdate?",
    )

    store.set_pending_workflow(session, workflow)
    loaded = store.get(session.id)

    assert loaded.pending_workflow is not None
    assert loaded.pending_workflow.intent == "create_patient"
    assert loaded.snapshot()["pending_workflow"]["missing_fields"] == ["birthdate"]
