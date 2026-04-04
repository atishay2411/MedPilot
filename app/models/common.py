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


class WorkflowStep(BaseModel):
    status: Literal["planned", "completed", "requires_confirmation", "blocked"]
    title: str
    detail: str


class EvidenceItem(BaseModel):
    label: str
    resource_type: str
    resource_uuid: str | None = None
    note: str


class PendingActionRecord(BaseModel):
    id: str
    action_kind: Literal["write", "workflow"]
    intent: str
    action: str
    permission: str
    endpoint: str
    payload: dict[str, Any] = Field(default_factory=dict)
    patient_uuid: str | None = None
    destructive: bool = False
    prompt: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponseEnvelope(BaseModel):
    session_id: str | None = None
    intent: str
    message: str
    workflow: list[WorkflowStep] = Field(default_factory=list)
    patient_context: dict[str, Any] | None = None
    data: Any | None = None
    summary: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    session_state: dict[str, Any] | None = None


class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    intent: str | None = None
    patient_uuid: str | None = None


class ChatSessionRecord(BaseModel):
    id: str
    created_at: str
    updated_at: str
    current_patient_uuid: str | None = None
    current_patient_display: str | None = None
    last_intent: str | None = None
    recent_turns: list[ChatHistoryTurn] = Field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "current_patient_uuid": self.current_patient_uuid,
            "current_patient_display": self.current_patient_display,
            "last_intent": self.last_intent,
            "recent_turns": [turn.model_dump() for turn in self.recent_turns[-8:]],
        }
