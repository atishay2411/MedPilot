from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TypeVar

from pydantic import BaseModel

from app.llm.models import LLMGenerationResult


StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class LLMProvider(ABC):
    provider_name: str = "base"

    @property
    @abstractmethod
    def enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, *, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[StructuredModelT],
        conversation_history: list[dict] | None = None,
    ) -> StructuredModelT:
        raise NotImplementedError


def normalize_structured_schema(schema: type[StructuredModelT]) -> dict:
    raw = deepcopy(schema.model_json_schema())

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            node.pop("default", None)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(raw)
    return raw
