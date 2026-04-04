from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.core.exceptions import LLMProviderError
from app.llm.base import LLMProvider, StructuredModelT, normalize_structured_schema
from app.llm.models import LLMGenerationResult


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = self._build_client()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        self._ensure_enabled()
        try:
            response = self.client.responses.create(
                model=self.settings.medpilot_llm_model,
                instructions=system_prompt,
                input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
                reasoning={"effort": self.settings.medpilot_llm_reasoning_effort},
                max_output_tokens=self.settings.medpilot_llm_max_output_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc
        text = self._extract_text(response.model_dump())
        return LLMGenerationResult(provider=self.provider_name, model=self.settings.medpilot_llm_model or "", text=text, raw=response.model_dump())

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema: type[StructuredModelT]) -> StructuredModelT:
        self._ensure_enabled()
        try:
            response = self.client.responses.create(
                model=self.settings.medpilot_llm_model,
                instructions=system_prompt,
                input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
                reasoning={"effort": self.settings.medpilot_llm_reasoning_effort},
                max_output_tokens=self.settings.medpilot_llm_max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema.__name__,
                        "strict": True,
                        "schema": normalize_structured_schema(schema),
                    }
                },
            )
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise LLMProviderError(f"OpenAI structured request failed: {exc}") from exc
        text = self._extract_text(response.model_dump())
        try:
            return schema.model_validate_json(text)
        except Exception as exc:
            raise LLMProviderError(f"OpenAI structured response could not be validated for {schema.__name__}: {exc}") from exc

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise LLMProviderError("OpenAI provider is not configured. Set OPENAI_API_KEY and MEDPILOT_LLM_MODEL.")

    def _build_client(self):
        if not (self.settings.openai_api_key and self.settings.medpilot_llm_model):
            return None
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise LLMProviderError("OpenAI SDK is not installed. Run `pip install -r requirements.txt`.") from exc
        return OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.medpilot_llm_timeout_seconds,
        )

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        if payload.get("output_text"):
            return payload["output_text"]
        chunks: list[str] = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    chunks.append(content.get("text", ""))
        text = "".join(chunks).strip()
        if not text:
            raise LLMProviderError("OpenAI returned an empty response.")
        return text
