from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REDACT_FIELDS = {"name", "givenName", "familyName", "birthdate", "address1", "cityVillage", "country", "identifier"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if key in REDACT_FIELDS else _redact(val)) for key, val in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


@dataclass(slots=True)
class AuditEvent:
    user_id: str
    role: str
    intent: str
    action: str
    patient_uuid: str | None
    prompt: str | None
    endpoint: str | None
    request_payload: dict[str, Any] | None
    response_status: int | None
    outcome: str
    metadata: dict[str, Any]

    def serialize(self) -> str:
        body = asdict(self)
        body["timestamp"] = datetime.now(timezone.utc).isoformat()
        body["request_payload"] = _redact(body["request_payload"])
        return json.dumps(body, ensure_ascii=True)


class AuditLogger:
    def __init__(self, path: Path):
        self.path = path

    def log(self, event: AuditEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.serialize() + "\n")
