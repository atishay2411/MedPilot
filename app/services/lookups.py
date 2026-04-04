from __future__ import annotations

from app.clients.openmrs import OpenMRSClient
from app.core.exceptions import ValidationError


class LookupService:
    def __init__(self, client: OpenMRSClient):
        self.client = client

    def resolve_uuid(self, entity: str, query: str | None) -> str:
        if not query:
            raise ValidationError(f"Missing lookup query for {entity}.")
        results = self.client.search(entity, query)
        if not results:
            raise ValidationError(f"No {entity} found for '{query}'.")
        return results[0]["uuid"]
