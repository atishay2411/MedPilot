"""Backward-compatible IntentService stub.

The legacy keyword-matching intent classifier has been superseded by the LLM
routing layer in ``llm_reasoning.py``.  This module keeps the class alive so that
routes that still reference it (/api/intent, /api/patients/preview) do not crash
the application on import.

``classify`` now returns a lightweight response based on the capability registry
instead of keyword heuristics.  ``preview_create_patient`` is retained for the
/api/patients/preview REST route.
"""
from __future__ import annotations

import random
from typing import Any

from app.models.domain import PatientRegistration
from app.services.capabilities import CAPABILITY_INDEX
from app.services.patients import PatientService
from app.services.utils import generate_openmrs_identifier


class IntentService:
    """Thin wrapper used by the legacy REST routes.

    This is intentionally minimal — all intelligent routing goes through
    ``LLMReasoningService`` in the chat pipeline.
    """

    def __init__(self, patients: PatientService) -> None:
        self.patients = patients

    # ------------------------------------------------------------------
    # Public API consumed by routes.py
    # ------------------------------------------------------------------

    def classify(self, prompt: str) -> dict[str, Any]:
        """Return a best-effort intent classification using the capability registry.

        This is a keyword-free heuristic that checks prompt words against
        capability descriptions.  It is only used by the legacy /api/intent
        endpoint — the chat pipeline uses the full LLM reasoning path.
        """
        prompt_lower = prompt.lower()
        best_intent = "inform"
        best_score = 0

        for intent, cap in CAPABILITY_INDEX.items():
            score = sum(1 for word in prompt_lower.split() if word in cap.summary.lower())
            if score > best_score:
                best_score = score
                best_intent = intent

        return {
            "intent": best_intent,
            "confidence": min(0.5 + best_score * 0.1, 0.9),
            "note": "Heuristic classification only. Use /api/chat for full LLM routing.",
        }

    def preview_create_patient(self, registration: PatientRegistration) -> dict[str, Any]:
        """Build a create-patient payload preview without committing.

        Returns the exact payload that would be sent to OpenMRS together with
        any duplicate-candidate warnings.
        """
        payload = self.patients.build_create_payload(registration)
        duplicates = self.patients.find_duplicate_candidates(registration)
        return {
            "payload": payload,
            "duplicate_warnings": duplicates,
        }
