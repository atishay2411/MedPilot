from __future__ import annotations

import json

import httpx

from app.config import Settings
from app.core.exceptions import LLMProviderError
from app.llm.base import LLMProvider, StructuredModelT, normalize_structured_schema
from app.llm.models import LLMGenerationResult


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.settings.medpilot_llm_model and self.base_url)

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        payload = self._request(
            {
                "model": self.settings.medpilot_llm_model,
                "messages": self._messages(system_prompt, user_prompt),
                "stream": False,
                "options": {"temperature": 0},
            }
        )
        text = payload.get("message", {}).get("content", "").strip()
        if not text:
            raise LLMProviderError("Ollama returned an empty response.")
        return LLMGenerationResult(provider=self.provider_name, model=self.settings.medpilot_llm_model or "", text=text, raw=payload)

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema: type[StructuredModelT]) -> StructuredModelT:
        payload = self._request(
            {
                "model": self.settings.medpilot_llm_model,
                "messages": self._messages(
                    system_prompt,
                    f"{user_prompt}\n\nReturn valid JSON that matches this schema exactly:\n{json.dumps(normalize_structured_schema(schema), ensure_ascii=True)}",
                ),
                "stream": False,
                "format": normalize_structured_schema(schema),
                "options": {"temperature": 0},
            }
        )
        text = payload.get("message", {}).get("content", "").strip()
        if not text:
            raise LLMProviderError("Ollama returned an empty structured response.")
        try:
            return schema.model_validate_json(text)
        except Exception as exc:
            raise LLMProviderError(f"Ollama structured response could not be validated for {schema.__name__}: {exc}") from exc

    def _request(self, payload: dict) -> dict:
        if not self.enabled:
            raise LLMProviderError("Ollama provider is not configured. Set MEDPILOT_LLM_MODEL and OLLAMA_BASE_URL.")
        try:
            with httpx.Client(timeout=self.settings.medpilot_llm_timeout_seconds) as client:
                response = client.post(f"{self.base_url}/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc

    @staticmethod
    def _messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
