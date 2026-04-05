from __future__ import annotations

from typing import Any

from app.clients.openmrs import OpenMRSClient


class VisitService:
    def __init__(self, client: OpenMRSClient):
        self.client = client

    def list_for_patient(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get("/ws/rest/v1/visit", params={"patient": patient_uuid, "v": "full"})

    def list_types(self) -> list[dict[str, Any]]:
        result = self.client.get("/ws/rest/v1/visittype", params={"v": "default"})
        return result.get("results", [])

    def create(self, patient_uuid: str, visit_type_uuid: str, location_uuid: str | None = None) -> dict[str, Any]:
        from app.services.utils import now_iso
        payload: dict[str, Any] = {
            "patient": patient_uuid,
            "visitType": visit_type_uuid,
            "startDatetime": now_iso(),
        }
        if location_uuid:
            payload["location"] = location_uuid
        return self.client.post("/ws/rest/v1/visit", payload)

    def end_visit(self, visit_uuid: str) -> dict[str, Any]:
        from app.services.utils import now_iso
        return self.client.post(f"/ws/rest/v1/visit/{visit_uuid}", {"stopDatetime": now_iso()})

    def format_visit_summary(self, visits_result: dict[str, Any]) -> str:
        visits = visits_result.get("results", [])
        if not visits:
            return "No visits recorded."
        lines = []
        for v in visits[:20]:
            visit_type = (v.get("visitType") or {}).get("display", "Unknown Type")
            start = (v.get("startDatetime") or "")[:10]
            stop = (v.get("stopDatetime") or "")[:10] or "Ongoing"
            location = (v.get("location") or {}).get("display", "")
            loc_str = f" at {location}" if location else ""
            lines.append(f"- **{visit_type}**{loc_str}: {start} → {stop}")
        total = len(visits)
        shown = len(lines)
        suffix = f"\n\n*Showing {shown} of {total} visits.*" if total > shown else ""
        return "\n".join(lines) + suffix
