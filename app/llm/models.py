from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMGenerationResult(BaseModel):
    provider: str
    model: str
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)
