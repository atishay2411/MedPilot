from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.core.exceptions import ValidationError
from app.models.common import ChatHistoryTurn, ChatSessionRecord


class ChatSessionStore:
    def __init__(self, settings: Settings):
        self.base_path = settings.chat_sessions_path

    def create(self) -> ChatSessionRecord:
        session = ChatSessionRecord(
            id=str(uuid4()),
            created_at=self._now(),
            updated_at=self._now(),
        )
        self.save(session)
        return session

    def get(self, session_id: str) -> ChatSessionRecord:
        path = self._path(session_id)
        if not path.exists():
            raise ValidationError(f"Chat session '{session_id}' was not found.")
        return ChatSessionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def get_or_create(self, session_id: str | None) -> ChatSessionRecord:
        if session_id:
            return self.get(session_id)
        return self.create()

    def save(self, session: ChatSessionRecord) -> ChatSessionRecord:
        session.updated_at = self._now()
        self._path(session.id).write_text(session.model_dump_json(indent=2), encoding="utf-8")
        return session

    def append_turn(self, session: ChatSessionRecord, turn: ChatHistoryTurn) -> ChatSessionRecord:
        session.recent_turns.append(turn)
        session.recent_turns = session.recent_turns[-20:]
        return self.save(session)

    def set_current_patient(self, session: ChatSessionRecord, patient_uuid: str | None, patient_display: str | None) -> ChatSessionRecord:
        session.current_patient_uuid = patient_uuid
        session.current_patient_display = patient_display
        return self.save(session)

    def set_last_intent(self, session: ChatSessionRecord, intent: str | None) -> ChatSessionRecord:
        session.last_intent = intent
        return self.save(session)

    def _path(self, session_id: str) -> Path:
        return self.base_path / f"{session_id}.json"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
