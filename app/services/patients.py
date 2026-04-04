from __future__ import annotations

import random
from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.config import Settings
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
