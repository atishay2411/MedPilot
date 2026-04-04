from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.core.exceptions import LLMProviderError
from app.llm.base import LLMProvider, StructuredModelT, normalize_structured_schema
from app.llm.models import LLMGenerationResult


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider using the standard Chat Completions API.

    Works with any OpenAI-compatible endpoint (openai, Azure OpenAI, local
    vLLM / LMStudio, etc.).  Uses ``json_schema`` structured output when
    the model supports it (gpt-4o, gpt-4o-mini, gpt-4.1-*), falling back
    to ``json_object`` mode for older models.
    """

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
            response = self.client.chat.completions.create(
                model=self.settings.medpilot_llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.settings.medpilot_llm_max_output_tokens,
                temperature=0,
            )
        except Exception as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LLMProviderError("OpenAI returned an empty response.")
        return LLMGenerationResult(
            provider=self.provider_name,
            model=self.settings.medpilot_llm_model or "",
            text=text,
            raw={},
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[StructuredModelT],
        conversation_history: list[dict] | None = None,
    ) -> StructuredModelT:
        self._ensure_enabled()
        schema_norm = normalize_structured_schema(schema)

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-12:])
        messages.append({"role": "user", "content": user_prompt})

        # Try json_schema mode first (gpt-4o, gpt-4o-mini, gpt-4.1-*)
        try:
            response = self.client.chat.completions.create(
                model=self.settings.medpilot_llm_model,
                messages=messages,
                max_tokens=self.settings.medpilot_llm_max_output_tokens,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema_norm,
                    },
                },
            )
        except Exception:
            # Fallback: json_object mode with schema hint in the prompt
            schema_hint = json.dumps(schema_norm, ensure_ascii=True)
            fallback_messages = list(messages)
            fallback_messages[-1] = {
                "role": "user",
                "content": (
                    f"{user_prompt}\n\n"
                    f"Return ONLY valid JSON matching this schema "
                    f"(no markdown, no extra text):\n{schema_hint}"
                ),
            }
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.medpilot_llm_model,
                    messages=fallback_messages,
                    max_tokens=self.settings.medpilot_llm_max_output_tokens,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            except Exception as exc2:
                raise LLMProviderError(
                    f"OpenAI structured request failed: {exc2}"
                ) from exc2

        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LLMProviderError("OpenAI returned an empty structured response.")
        try:
            return schema.model_validate_json(text)
        except Exception as exc:
            raise LLMProviderError(
                f"OpenAI structured response could not be validated for "
                f"{schema.__name__}: {exc}"
            ) from exc

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise LLMProviderError(
                "OpenAI provider is not configured. "
                "Set OPENAI_API_KEY and MEDPILOT_LLM_MODEL in your .env file."
            )

    def _build_client(self) -> Any:
        if not (self.settings.openai_api_key and self.settings.medpilot_llm_model):
            return None
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError(
                "OpenAI SDK is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        return OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.medpilot_llm_timeout_seconds,
        )
