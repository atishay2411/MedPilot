from pathlib import Path
from types import SimpleNamespace

from app.models.common import ChatHistoryTurn
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
