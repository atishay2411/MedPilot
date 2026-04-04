from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import Settings
from app.core.exceptions import ExternalServiceError


TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenMRSClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.openmrs_base_url.rstrip("/")
        self.auth = (settings.openmrs_username, settings.openmrs_password)

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, json: Any = None, headers: dict[str, str] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)

        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                with httpx.Client(auth=self.auth, timeout=self.settings.request_timeout_seconds) as client:
                    response = client.request(method, url, params=params, json=json, headers=merged_headers)
                if response.status_code in TRANSIENT_STATUS_CODES and attempt < self.settings.max_retries:
                    time.sleep(0.3 * (2 ** (attempt - 1)))
                    continue
                response.raise_for_status()
                if not response.text:
                    return {}
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    time.sleep(0.3 * (2 ** (attempt - 1)))
                    continue
        raise ExternalServiceError(f"OpenMRS request failed for {method} {path}: {last_error}")

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, json=payload)

    def put(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("PUT", path, json=payload)

    def patch(self, path: str, payload: list[dict[str, Any]]) -> Any:
        return self._request("PATCH", path, json=payload, headers={"Content-Type": "application/json-patch+json"})

    def delete(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("DELETE", path, params=params)

    def search(self, entity: str, query: str) -> list[dict[str, Any]]:
        response = self.get(f"/ws/rest/v1/{entity}", params={"q": query})
        return response.get("results", [])
