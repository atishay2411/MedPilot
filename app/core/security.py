from dataclasses import dataclass

from app.core.exceptions import AuthorizationError


ROLE_PERMISSIONS = {
    "clinician": {
        "read:patient",
        "read:clinical",
        "read:population",
        "read:metadata",
        "write:patient",
        "write:encounter",
        "write:observation",
        "write:condition",
        "write:allergy",
        "write:medication",
        "write:ingestion",
        "delete:observation",
        "delete:condition",
        "delete:allergy",
    },
    "nurse": {
        "read:patient",
        "read:clinical",
        "read:population",
        "write:observation",
        "write:allergy",
    },
    "admin": {"read:patient", "read:clinical", "read:population", "read:metadata", "write:admin"},
    "read-only": {"read:patient", "read:clinical", "read:population", "read:metadata"},
}


@dataclass(slots=True)
class Actor:
    user_id: str
    role: str


def ensure_permission(actor: Actor, permission: str) -> None:
    allowed = ROLE_PERMISSIONS.get(actor.role, set())
    if permission not in allowed:
        raise AuthorizationError(f"Role '{actor.role}' cannot perform '{permission}'.")
