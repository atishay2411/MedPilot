from __future__ import annotations

from app.core.exceptions import LLMProviderError
from app.llm.base import LLMProvider, StructuredModelT
from app.llm.models import LLMGenerationResult


class NoOpLLMProvider(LLMProvider):
    provider_name = "none"

    @property
    def enabled(self) -> bool:
        return False

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        raise LLMProviderError("No LLM provider is configured.")

    def generate_structured(self, *, system_prompt: str, user_prompt: str, schema: type[StructuredModelT]) -> StructuredModelT:
        raise LLMProviderError("No LLM provider is configured.")
