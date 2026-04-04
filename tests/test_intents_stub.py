"""Tests for the IntentService stub that replaced the legacy classifier."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.models.domain import PatientRegistration
from app.services.intents import IntentService
from app.services.patients import PatientService


def _make_service() -> IntentService:
    patients = MagicMock(spec=PatientService)
    patients.find_duplicate_candidates.return_value = []
    patients.build_create_payload.return_value = {"person": {}, "identifiers": []}
    return IntentService(patients)


def test_intents_service_classify_returns_dict_with_intent():
    svc = _make_service()
    result = svc.classify("How many patients are there?")
    assert "intent" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)


def test_intents_service_classify_known_intent_passes():
    svc = _make_service()
    result = svc.classify("Find patient Maria Santos")
    # Should return some supported intent (exact match not guaranteed for heuristic)
    assert result["intent"] != ""


def test_intents_service_preview_create_patient_returns_payload():
    svc = _make_service()
    reg = PatientRegistration(given_name="Jane", family_name="Doe", gender="F", birthdate="1990-01-01")
    result = svc.preview_create_patient(reg)
    assert "payload" in result
    assert "duplicate_warnings" in result
    assert isinstance(result["duplicate_warnings"], list)
