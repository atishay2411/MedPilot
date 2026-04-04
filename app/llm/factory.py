from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.providers.noop import NoOpLLMProvider
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.providers.openai_provider import OpenAIProvider


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.medpilot_llm_provider.lower().strip()
    if provider == "openai":
        return OpenAIProvider(settings)
    if provider == "ollama":
        return OllamaProvider(settings)
    if provider == "anthropic":
        from app.llm.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(settings)
    return NoOpLLMProvider()
