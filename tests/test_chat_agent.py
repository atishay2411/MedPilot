from app.models.common import ChatSessionRecord
from app.services.llm_reasoning import ConversationalDecision
from app.services.chat_agent import ChatAgentService


def test_search_summary_lists_multiple_prefix_matches():
    summary = ChatAgentService._format_search_summary(
        "N",
        [
            {"display": "Nesh Test"},
            {"display": "Nora Lane"},
            {"display": "Noah Smith"},
        ],
        search_mode="starts_with",
    )

    assert "start with 'N'" in summary
    assert "Nesh Test" in summary
    assert "Nora Lane" in summary
    assert "Noah Smith" in summary


def test_search_summary_handles_no_matches():
    summary = ChatAgentService._format_search_summary("N", [], search_mode="starts_with")

    assert "No patients found" in summary


def test_search_summary_single_match_is_concise():
    summary = ChatAgentService._format_search_summary("Maria Santos", [{"display": "Maria Santos"}])

    assert "Found 1 match" in summary
    assert "Maria Santos" in summary


def test_search_summary_multiple_matches_is_listed():
    summary = ChatAgentService._format_search_summary("all", [{"display": "Betty Williams"}, {"display": "Nora Lane"}])

    assert "Found 2 patients" in summary
    assert "Betty Williams" in summary
    assert "Nora Lane" in summary


def test_build_pending_workflow_state_tracks_missing_fields():
    session = ChatSessionRecord(
        id="session-1",
        created_at="2026-04-04T00:00:00+00:00",
        updated_at="2026-04-04T00:00:00+00:00",
        current_patient_uuid="patient-123",
        current_patient_display="Maria Santos",
    )
    decision = ConversationalDecision(
        mode="clarify",
        intent="create_patient",
        confidence=0.94,
        entities={"given_name": "Jane", "family_name": "Doe"},
        missing_fields=["birthdate"],
        clarifying_question="What is Jane Doe's birthdate?",
        response_message="What is Jane Doe's birthdate?",
    )

    workflow = ChatAgentService._build_pending_workflow_state("Add patient Jane Doe", session, decision)

    assert workflow is not None
    assert workflow.intent == "create_patient"
    assert workflow.missing_fields == ["birthdate"]
    assert workflow.patient_uuid == "patient-123"
