from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.core.exceptions import LLMProviderError
from app.llm.base import LLMProvider, StructuredModelT, normalize_structured_schema
from app.llm.models import LLMGenerationResult


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider.

    Uses tool-calling to enforce structured JSON output, which is far more
    reliable than free-text JSON generation for intent classification tasks.
    """

    provider_name = "anthropic"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = self._build_client()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        self._ensure_enabled()
        try:
            response = self.client.messages.create(
                model=self.settings.medpilot_llm_model or "claude-haiku-4-5-20251001",
                max_tokens=self.settings.medpilot_llm_max_output_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
        if not text:
            raise LLMProviderError("Anthropic returned an empty response.")
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
        """Use Claude tool-calling to produce validated structured output.

        Tool-calling is the most reliable way to get schema-conformant JSON
        from Claude — the model fills tool arguments directly, bypassing the
        risk of markdown fences or prose wrapping the JSON.
        """
        self._ensure_enabled()
        schema_dict = normalize_structured_schema(schema)
        tool_def = {
            "name": "respond",
            "description": (
                "Always call this tool to provide your structured response. "
                "Fill in all fields according to the schema."
            ),
            "input_schema": schema_dict,
        }
        msgs: list[dict] = list(conversation_history[-12:]) if conversation_history else []
        msgs.append({"role": "user", "content": user_prompt})
        try:
            response = self.client.messages.create(
                model=self.settings.medpilot_llm_model or "claude-haiku-4-5-20251001",
                max_tokens=self.settings.medpilot_llm_max_output_tokens,
                system=system_prompt,
                tools=[tool_def],
                tool_choice={"type": "tool", "name": "respond"},
                messages=msgs,
            )
        except Exception as exc:
            raise LLMProviderError(f"Anthropic structured request failed: {exc}") from exc

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "respond":
                try:
                    return schema.model_validate(block.input)
                except Exception as exc:
                    raise LLMProviderError(
                        f"Anthropic structured response validation failed for "
                        f"{schema.__name__}: {exc}"
                    ) from exc

        raise LLMProviderError(
            "Anthropic did not call the required tool. Check model and API key."
        )

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise LLMProviderError(
                "Anthropic provider is not configured. "
                "Set ANTHROPIC_API_KEY and MEDPILOT_LLM_MODEL in your .env file."
            )

    def _build_client(self) -> Any:
        api_key = getattr(self.settings, "anthropic_api_key", None)
        if not api_key:
            return None
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise LLMProviderError(
                "Anthropic SDK is not installed. Run `pip install anthropic`."
            ) from exc
        return Anthropic(
            api_key=api_key,
            timeout=self.settings.medpilot_llm_timeout_seconds,
        )
