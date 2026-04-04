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


class PendingWorkflowState(BaseModel):
    intent: str | None = None
    original_prompt: str | None = None
    collected_entities: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    patient_uuid: str | None = None
    patient_display: str | None = None


class PendingClarificationSlot(BaseModel):
    """Structured slot state for in-flight clarification — replaces the old plain-text string.

    Stores enough deterministic state for slot-filling so that follow-up
    answers like '24 Apr 2000' can be merged without LLM reconstruction.
    """

    question: str
    """The clarifying question that was posed to the user."""

    intent: str | None = None
    """The intended action that is waiting for the missing information."""

    collected_entities: dict[str, Any] = Field(default_factory=dict)
    """Entities already extracted from the original (and any prior follow-up) message."""

    missing_fields: list[str] = Field(default_factory=list)
    """Fields still needed to complete the action."""

    patient_uuid: str | None = None
    """Active patient UUID at the time the clarification was issued, if any."""

    patient_display: str | None = None
    """Human-readable patient name at the time the clarification was issued."""

    turn_count: int = 0
    """How many clarification turns have been spent on this workflow.
    Prevent infinite clarification loops (cap at 3)."""


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
    scope: Literal["global", "patient"] | None = None
    """Auto-detected query scope: 'global' for population-level queries, 'patient' for chart actions."""


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
    pending_clarification: PendingClarificationSlot | None = None
    """Structured in-flight clarification slot. Replaces old plain-text string."""
    pending_workflow: PendingWorkflowState | None = None
    recent_turns: list[ChatHistoryTurn] = Field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "current_patient_uuid": self.current_patient_uuid,
            "current_patient_display": self.current_patient_display,
            "last_intent": self.last_intent,
            "pending_clarification": self.pending_clarification.model_dump() if self.pending_clarification else None,
            "pending_workflow": self.pending_workflow.model_dump() if self.pending_workflow else None,
            "recent_turns": [turn.model_dump() for turn in self.recent_turns[-12:]],
        }
