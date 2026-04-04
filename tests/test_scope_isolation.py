"""Tests for scope isolation — global queries must not overwrite the session's active patient.

These tests cover the core 'generalization' bug where a count_patients or
search_patient result was silently switching the session's active patient.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models.common import ChatSessionRecord
from app.services.chat_agent import ChatAgentService
from app.services.llm_reasoning import ConversationalDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(*, patient_uuid: str | None = "patient-abc", patient_display: str | None = "Maria Santos") -> ChatSessionRecord:
    return ChatSessionRecord(
        id="session-test",
        created_at="2026-04-04T00:00:00+00:00",
        updated_at="2026-04-04T00:00:00+00:00",
        current_patient_uuid=patient_uuid,
        current_patient_display=patient_display,
    )


def _make_agent() -> ChatAgentService:
    """Build a minimal ChatAgentService with all dependencies mocked."""
    svc = MagicMock(spec=ChatAgentService)
    # Re-bind the real static/instance methods we actually want to test
    svc._update_session_after_response = ChatAgentService._update_session_after_response.__get__(svc)
    svc._format_search_summary = ChatAgentService._format_search_summary
    return svc


# ---------------------------------------------------------------------------
# Unit tests for _update_session_after_response scope rules
# ---------------------------------------------------------------------------

class FakeSessionStore:
    """Minimal in-memory session store for testing."""

    def __init__(self, session: ChatSessionRecord):
        self._session = session
        self.patient_updates: list[tuple] = []

    def set_current_patient(self, session, uuid, display):
        self.patient_updates.append((uuid, display))
        session.current_patient_uuid = uuid
        session.current_patient_display = display

    def set_last_intent(self, session, intent):
        session.last_intent = intent

    def set_pending_clarification(self, session, slot):
        session.pending_clarification = slot

    def set_pending_workflow(self, session, wf):
        session.pending_workflow = wf

    def append_turn(self, session, turn):
        session.recent_turns.append(turn)
        return session

    def get(self, session_id):
        return self._session


def _make_envelope(intent="get_conditions", patient_uuid="patient-abc", patient_display="Maria Santos"):
    from app.models.common import ChatResponseEnvelope
    return ChatResponseEnvelope(
        intent=intent,
        message="Test response",
        patient_context={"uuid": patient_uuid, "display": patient_display},
    )


def _make_global_envelope(intent="count_patients"):
    from app.models.common import ChatResponseEnvelope
    return ChatResponseEnvelope(
        intent=intent,
        message="There are 42 patients.",
        patient_context=None,  # global responses have no patient context
    )


def test_global_intent_never_overwrites_active_patient():
    """count_patients / search_patient must not clear or change the active session patient."""
    session = _make_session(patient_uuid="patient-abc", patient_display="Maria Santos")
    store = FakeSessionStore(session)

    agent = MagicMock()
    agent.sessions = store
    ChatAgentService._update_session_after_response(
        agent,
        session,
        _make_global_envelope("count_patients"),
        "count_patients",
        is_global=True,
    )

    # No patient update must have been made
    assert store.patient_updates == [], "Global query must NOT update the active patient"
    assert session.current_patient_uuid == "patient-abc"
    assert session.current_patient_display == "Maria Santos"


def test_switch_patient_always_updates_active_patient():
    """switch_patient must always overwrite the active session patient."""
    session = _make_session(patient_uuid="patient-abc", patient_display="Maria Santos")
    store = FakeSessionStore(session)
    envelope = _make_envelope(intent="switch_patient", patient_uuid="patient-xyz", patient_display="John Doe")

    agent = MagicMock()
    agent.sessions = store
    ChatAgentService._update_session_after_response(
        agent,
        session,
        envelope,
        "switch_patient",
        is_global=False,
    )

    assert ("patient-xyz", "John Doe") in store.patient_updates
    assert session.current_patient_uuid == "patient-xyz"


def test_create_patient_updates_active_patient():
    """create_patient confirmation should set the newly created patient as active."""
    session = _make_session(patient_uuid=None, patient_display=None)
    store = FakeSessionStore(session)
    envelope = _make_envelope(intent="create_patient", patient_uuid="new-uuid", patient_display="New Patient")

    agent = MagicMock()
    agent.sessions = store
    ChatAgentService._update_session_after_response(
        agent,
        session,
        envelope,
        "create_patient",
        is_global=False,
    )

    assert ("new-uuid", "New Patient") in store.patient_updates


def test_patient_scoped_read_does_not_overwrite_existing_patient():
    """get_conditions for an already-active patient must not trigger a redundant set_current_patient call."""
    session = _make_session(patient_uuid="patient-abc", patient_display="Maria Santos")
    store = FakeSessionStore(session)
    envelope = _make_envelope(intent="get_conditions", patient_uuid="patient-abc", patient_display="Maria Santos")

    agent = MagicMock()
    agent.sessions = store
    ChatAgentService._update_session_after_response(
        agent,
        session,
        envelope,
        "get_conditions",
        is_global=False,
    )

    # No update needed — the same patient is already active
    assert store.patient_updates == []


def test_first_chart_query_auto_sets_patient():
    """If there is no active patient yet, the first patient-scoped read auto-sets it."""
    session = _make_session(patient_uuid=None, patient_display=None)
    store = FakeSessionStore(session)
    envelope = _make_envelope(intent="get_conditions", patient_uuid="patient-xyz", patient_display="John Doe")

    agent = MagicMock()
    agent.sessions = store
    ChatAgentService._update_session_after_response(
        agent,
        session,
        envelope,
        "get_conditions",
        is_global=False,
    )

    assert ("patient-xyz", "John Doe") in store.patient_updates
    assert session.current_patient_uuid == "patient-xyz"


# ---------------------------------------------------------------------------
# Verify search_patient no longer auto-switches active patient
# ---------------------------------------------------------------------------

def test_handle_search_patient_returns_no_patient_context():
    """_handle_search_patient must return patient_context=None so a search result
    never silently switches the active patient.
    """
    from app.core.security import Actor
    from app.models.common import ChatResponseEnvelope

    session = _make_session(patient_uuid="patient-abc", patient_display="Maria Santos")
    actor = Actor(user_id="test-user", role="clinician")

    agent = MagicMock(spec=ChatAgentService)
    agent.patients = MagicMock()
    agent.patients.search.return_value = [
        {"uuid": "p1", "display": "John Doe"},
        {"uuid": "p2", "display": "Jane Doe"},
    ]
    agent._format_search_summary = ChatAgentService._format_search_summary

    result: ChatResponseEnvelope = ChatAgentService._handle_search_patient(
        agent,
        "find john doe",
        {"patient_query": "john doe", "search_mode": "default"},
        actor,
        [],  # workflow
        session=session,
    )

    assert result.patient_context is None, (
        "search_patient must NOT set patient_context — that would silently switch the active patient"
    )
    assert "John Doe" in result.message or "2 patient" in result.message.lower()


def test_handle_search_patient_single_result_still_no_context():
    """Even a single search result must NOT set patient_context — use switch_patient for that."""
    from app.core.security import Actor
    from app.models.common import ChatResponseEnvelope

    actor = Actor(user_id="test-user", role="clinician")

    agent = MagicMock(spec=ChatAgentService)
    agent.patients = MagicMock()
    agent.patients.search.return_value = [{"uuid": "p1", "display": "Jane Doe"}]
    agent._format_search_summary = ChatAgentService._format_search_summary

    result: ChatResponseEnvelope = ChatAgentService._handle_search_patient(
        agent,
        "find jane doe",
        {"patient_query": "jane doe"},
        actor,
        [],
        session=_make_session(patient_uuid=None, patient_display=None),
    )

    assert result.patient_context is None


# ---------------------------------------------------------------------------
# Verify is_global detection with is_global_intent()
# ---------------------------------------------------------------------------

def test_is_global_intent_used_for_count():
    from app.services.capabilities import is_global_intent
    assert is_global_intent("count_patients") is True


def test_is_global_intent_used_for_patient_analysis():
    from app.services.capabilities import is_global_intent
    assert is_global_intent("patient_analysis") is False
