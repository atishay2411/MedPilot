from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    ok: bool = True
    data: Any | None = None
    error: str | None = None


class PendingWrite(BaseModel):
    intent: str
    action: str
    permission: str
    destructive: bool = False
    endpoint: str
    payload: dict[str, Any]
    patient_uuid: str | None = None
    duplicate_warnings: list[str] = Field(default_factory=list)


class EntityResult(BaseModel):
    entity_type: str
    name: str
    outcome: Literal["success", "skipped", "failed"]
    detail: str


class CountResult(BaseModel):
    label: str
    count: int
