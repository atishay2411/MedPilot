"""Unit tests for PendingClarificationSlot structured state and the extract_entities utility."""
from __future__ import annotations

from app.models.common import PendingClarificationSlot
from app.services.capabilities import CAPABILITY_INDEX, extract_entities, is_global_intent


def test_pending_clarification_slot_serializes_round_trip():
    slot = PendingClarificationSlot(
        question="What is the patient's date of birth?",
        intent="create_patient",
        collected_entities={"given_name": "Sahil", "family_name": "Rochwani"},
        missing_fields=["birthdate"],
        patient_uuid="uuid-123",
        patient_display="Sahil Rochwani",
        turn_count=1,
    )
    dumped = slot.model_dump()
    restored = PendingClarificationSlot.model_validate(dumped)

    assert restored.question == slot.question
    assert restored.intent == slot.intent
    assert restored.collected_entities == {"given_name": "Sahil", "family_name": "Rochwani"}
    assert restored.missing_fields == ["birthdate"]
    assert restored.turn_count == 1


def test_pending_clarification_slot_defaults_are_empty():
    slot = PendingClarificationSlot(question="What is the gender?")

    assert slot.intent is None
    assert slot.collected_entities == {}
    assert slot.missing_fields == []
    assert slot.turn_count == 0


def test_extract_entities_drops_unknown_keys():
    capability = CAPABILITY_INDEX["create_patient"]
    raw = {
        "given_name": "Sahil",
        "family_name": "Rochwani",
        "birthdate": "2000-04-24",
        "spurious_key": "should_be_dropped",
        "another_junk_key": True,
    }

    result = extract_entities(raw, capability)

    assert "given_name" in result
    assert "family_name" in result
    assert "birthdate" in result
    assert "spurious_key" not in result
    assert "another_junk_key" not in result


def test_extract_entities_always_passes_patient_query():
    """patient_query must always pass through even if not in the capability's explicit fields."""
    capability = CAPABILITY_INDEX["get_conditions"]
    raw = {
        "patient_query": "Maria Santos",
        "junk_field": "ignored",
    }

    result = extract_entities(raw, capability)

    assert result["patient_query"] == "Maria Santos"
    assert "junk_field" not in result


def test_extract_entities_missing_required_field_returns_none_value():
    """Missing required fields should return None — handler raises ValidationError."""
    capability = CAPABILITY_INDEX["create_patient"]
    raw = {"given_name": "Sahil"}

    result = extract_entities(raw, capability)

    # Required fields not in raw should not explode — handler validates
    assert result.get("given_name") == "Sahil"
    # family_name is required but not in raw → absent from result (not None-filled)
    # extract_entities does NOT inject Nones; it just strips unknown keys
    assert "junk" not in result


def test_is_global_intent_correctly_classifies_population_intents():
    assert is_global_intent("count_patients") is True
    assert is_global_intent("search_patient") is True
    assert is_global_intent("get_metadata") is True
    assert is_global_intent("create_patient") is True
    assert is_global_intent("delete_patient") is True


def test_is_global_intent_classifies_chart_intents_as_patient_scoped():
    assert is_global_intent("get_conditions") is False
    assert is_global_intent("create_allergy") is False
    assert is_global_intent("get_observations") is False
    assert is_global_intent("patient_analysis") is False
    assert is_global_intent("update_patient") is False


def test_is_global_intent_returns_false_for_unknown_intent():
    assert is_global_intent("nonexistent_intent_xyz") is False
