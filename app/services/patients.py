from __future__ import annotations

import random
from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.config import Settings
from app.core.exceptions import ValidationError
from app.models.domain import PatientRegistration
from app.services.utils import generate_openmrs_identifier


class PatientService:
    def __init__(self, client: OpenMRSClient, settings: Settings):
        self.client = client
        self.settings = settings

    def search(self, query: str) -> list[dict[str, Any]]:
        return self.client.get("/ws/rest/v1/patient", params={"q": query}).get("results", [])

    def get_demographics(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get(f"/ws/fhir2/R4/Patient/{patient_uuid}")

    def find_duplicate_candidates(self, registration: PatientRegistration) -> list[str]:
        results = self.search(f"{registration.given_name} {registration.family_name}")
        matches: list[str] = []
        for result in results:
            person = result.get("person", {})
            birthdate = person.get("birthdate")
            if birthdate and birthdate.startswith(registration.birthdate):
                matches.append(result.get("display", "Existing patient"))
        return matches

    def build_create_payload(self, registration: PatientRegistration) -> dict[str, Any]:
        identifier = generate_openmrs_identifier(random.randint(100000, 999999))
        return {
            "person": {
                "names": [{"givenName": registration.given_name, "familyName": registration.family_name}],
                "gender": registration.gender,
                "birthdate": registration.birthdate,
                "addresses": [
                    {
                        "address1": registration.address1 or "",
                        "cityVillage": registration.city_village or "",
                        "country": registration.country or "",
                    }
                ],
            },
            "identifiers": [
                {
                    "identifier": identifier,
                    "identifierType": self.settings.openmrs_identifier_type_uuid,
                    "location": self.settings.openmrs_location_uuid,
                    "preferred": True,
                }
            ],
        }

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/ws/rest/v1/patient", payload)

    def resolve_patient(self, query: str | None = None, patient_uuid: str | None = None) -> dict[str, Any]:
        if patient_uuid:
            patient = self.get_demographics(patient_uuid)
            return {
                "uuid": patient_uuid,
                "display": self.format_patient_display(patient),
                "resource": patient,
                "alternatives": [],
            }

        if not query:
            raise ValidationError("A patient name, identifier, or UUID is required for this action.")

        results = self.search(query)
        if not results:
            raise ValidationError(f"No patient matched '{query}'.")

        chosen = self._pick_best_match(query, results)
        return {
            "uuid": chosen["uuid"],
            "display": chosen.get("display", query),
            "resource": chosen,
            "alternatives": [item.get("display", "Unknown patient") for item in results[1:4]],
        }

    def build_create_payload_from_fhir(self, resource: dict[str, Any]) -> dict[str, Any]:
        name = (resource.get("name") or [{}])[0]
        address = (resource.get("address") or [{}])[0]
        registration = PatientRegistration(
            given_name=(name.get("given") or [""])[0],
            family_name=name.get("family", ""),
            gender=(resource.get("gender", "U") or "U")[:1].upper(),
            birthdate=resource.get("birthDate", ""),
            address1=((address.get("line") or [""]) or [""])[0],
            city_village=address.get("city"),
            country=address.get("country"),
        )
        return self.build_create_payload(registration)

    @staticmethod
    def format_patient_display(resource: dict[str, Any]) -> str:
        names = resource.get("name") or []
        if names:
            first = names[0]
            given = " ".join(first.get("given", []))
            family = first.get("family", "")
            joined = " ".join(part for part in [given, family] if part).strip()
            if joined:
                return joined
        return resource.get("display", resource.get("uuid", "Unknown patient"))

    @staticmethod
    def _pick_best_match(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        query_lower = query.lower()
        for result in results:
            if query_lower == result.get("display", "").lower():
                return result
            identifiers = result.get("identifiers", [])
            if any(query_lower == str(item.get("identifier", "")).lower() for item in identifiers):
                return result
        return results[0]
