from __future__ import annotations

from uuid import uuid4

from app.core.exceptions import ValidationError
from app.models.common import PendingActionRecord


class PendingActionStore:
    def __init__(self):
        self._actions: dict[str, PendingActionRecord] = {}

    def create(self, **kwargs) -> PendingActionRecord:
        record = PendingActionRecord(id=str(uuid4()), **kwargs)
        self._actions[record.id] = record
        return record

    def get(self, action_id: str) -> PendingActionRecord:
        record = self._actions.get(action_id)
        if not record:
            raise ValidationError(f"Pending action '{action_id}' was not found or has expired.")
        return record

    def consume(self, action_id: str) -> PendingActionRecord:
        record = self.get(action_id)
        self._actions.pop(action_id, None)
        return record
