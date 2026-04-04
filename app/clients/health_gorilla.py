from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings
from app.core.exceptions import ExternalServiceError


class HealthGorillaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.health_gorilla_base_url.rstrip("/")
        self.token = settings.health_gorilla_token
        self.timeout = settings.request_timeout_seconds

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise ExternalServiceError("Health Gorilla token is not configured.")
        return {"Authorization": f"Bearer {self.token}"}

    def search_patient(self, given_name: str, family_name: str, birthdate: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}/Patient",
                params={"given": given_name, "family": family_name, "birthdate": birthdate},
                headers=self._headers(),
            )
        response.raise_for_status()
        return response.json()

    def get_conditions(self, patient_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}/Condition",
                params={"patient": patient_id},
                headers=self._headers(),
            )
        response.raise_for_status()
        return response.json()
