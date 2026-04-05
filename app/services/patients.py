from __future__ import annotations

import random
import re
from typing import Any

from app.clients.openmrs import OpenMRSClient
from app.config import Settings
from app.core.exceptions import ExternalServiceError, ValidationError
from app.models.domain import PatientRegistration, PatientUpdateInput
from app.services.utils import generate_openmrs_identifier


class PatientService:
    def __init__(self, client: OpenMRSClient, settings: Settings):
        self.client = client
        self.settings = settings

    def search(self, query: str, *, search_mode: str = "default") -> list[dict[str, Any]]:
        normalized_query = query.strip()
        if not normalized_query:
            return self.list_all()
        if normalized_query.lower() in {"all", "all patients", "everyone", "*"}:
            return self.list_all()
        if search_mode == "starts_with":
            return self._search_by_name_filter(normalized_query, mode="starts_with")
        if search_mode == "contains":
            return self._search_by_name_filter(normalized_query, mode="contains")
        results = self.client.get("/ws/rest/v1/patient", params={"q": normalized_query}).get("results", [])
        if results:
            return results
        if self._looks_like_identifier_or_uuid(normalized_query):
            return self.search_by_identifier(normalized_query)
        return results

    def list_all(self, *, limit: int = 100) -> list[dict[str, Any]]:
        bundle = self.client.get("/ws/fhir2/R4/Patient", params={"_count": limit})
        results: list[dict[str, Any]] = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            normalized = self._normalize_fhir_patient(resource)
            if normalized.get("uuid"):
                results.append(normalized)
        return results

    def get_demographics(self, patient_uuid: str) -> dict[str, Any]:
        return self.client.get(f"/ws/fhir2/R4/Patient/{patient_uuid}")

    def search_by_identifier(self, identifier: str) -> list[dict[str, Any]]:
        """Look up a patient by identifier number or UUID.

        This is the public, explicitly-named method that backs the
        ``search_by_identifier`` intent. Previously an internal helper.
        """
        query = identifier.strip()
        if not query:
            return []

        if self._looks_like_uuid(query):
            try:
                resource = self.get_demographics(query)
                if resource:
                    return [self._normalize_fhir_patient(resource)]
            except ExternalServiceError:
                # UUID lookup failed — fall through to identifier search.
                pass

        try:
            bundle = self.client.get("/ws/fhir2/R4/Patient", params={"identifier": query})
        except ExternalServiceError:
            return []

        results: list[dict[str, Any]] = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            normalized = self._normalize_fhir_patient(resource)
            if normalized.get("uuid"):
                results.append(normalized)
        return results

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

    def build_update_payload(self, update: PatientUpdateInput) -> dict[str, Any]:
        """Build a partial update payload from a PatientUpdateInput.

        Only fields that are explicitly supplied (non-None) are included so that
        omitted fields are not overwritten to blank values on the server.
        """
        person: dict[str, Any] = {}
        if update.given_name is not None or update.family_name is not None:
            name_entry: dict[str, Any] = {}
            if update.given_name is not None:
                name_entry["givenName"] = update.given_name
            if update.family_name is not None:
                name_entry["familyName"] = update.family_name
            person["names"] = [name_entry]
        if update.gender is not None:
            person["gender"] = update.gender
        if update.birthdate is not None:
            person["birthdate"] = update.birthdate
        address: dict[str, Any] = {}
        if update.address1 is not None:
            address["address1"] = update.address1
        if update.city_village is not None:
            address["cityVillage"] = update.city_village
        if update.country is not None:
            address["country"] = update.country
        if address:
            person["addresses"] = [address]
        return {"person": person} if person else {}

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/ws/rest/v1/patient", payload)

    def update(self, patient_uuid: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to patient/{uuid} — OpenMRS REST update pattern."""
        return self.client.post(f"/ws/rest/v1/patient/{patient_uuid}", payload)

    def delete(self, patient_uuid: str, *, purge: bool = False) -> dict[str, Any]:
        response = self.client.delete(
            f"/ws/rest/v1/patient/{patient_uuid}",
            params={"purge": "true"} if purge else None,
        )
        if response:
            return response
        return {"deleted": True, "patient_uuid": patient_uuid, "purge": purge}

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

        # Strip common prefixes the LLM sometimes includes in patient_query
        _lower = query.lower().strip()
        for _prefix in ("patient ", "pt ", "for patient ", "for pt "):
            if _lower.startswith(_prefix):
                query = query[len(_prefix):].strip()
                break

        results = self.search(query)
        if not results and self._looks_like_identifier_or_uuid(query):
            results = self.search_by_identifier(query)
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

    @staticmethod
    def _looks_like_uuid(query: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-fA-F-]{36}", query.strip()))

    @classmethod
    def _looks_like_identifier_or_uuid(cls, query: str) -> bool:
        normalized = query.strip()
        return cls._looks_like_uuid(normalized) or bool(re.fullmatch(r"[A-Za-z0-9-]{3,}", normalized))

    def _search_by_name_filter(self, query: str, *, mode: str) -> list[dict[str, Any]]:
        combined: list[dict[str, Any]] = []
        seen: set[str] = set()

        for result in self.client.get("/ws/rest/v1/patient", params={"q": query}).get("results", []):
            uuid = str(result.get("uuid", ""))
            if uuid and uuid not in seen:
                seen.add(uuid)
                combined.append(result)

        try:
            bundle = self.client.get("/ws/fhir2/R4/Patient", params={"name": query, "_count": 100})
        except ExternalServiceError:
            bundle = {}
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            normalized = self._normalize_fhir_patient(resource)
            uuid = str(normalized.get("uuid", ""))
            if uuid and uuid not in seen:
                seen.add(uuid)
                combined.append(normalized)

        return [result for result in combined if self._matches_name_mode(result, query, mode=mode)]

    def _normalize_fhir_patient(self, resource: dict[str, Any]) -> dict[str, Any]:
        identifiers = [
            {"identifier": item.get("value", "")}
            for item in resource.get("identifier", [])
            if item.get("value")
        ]
        return {
            "uuid": resource.get("id"),
            "display": self.format_patient_display(resource),
            "identifiers": identifiers,
            "resource": resource,
        }

    @staticmethod
    def _matches_name_mode(result: dict[str, Any], query: str, *, mode: str) -> bool:
        display = str(result.get("display", "")).strip().lower()
        if not display:
            return False

        query_lower = query.strip().lower()
        display_tokens = [token for token in re.split(r"\s+", display) if token]
        query_tokens = [token for token in re.split(r"\s+", query_lower) if token]

        if mode == "starts_with":
            if len(query_tokens) == 1:
                return any(token.startswith(query_tokens[0]) for token in display_tokens)
            return display.startswith(query_lower) or all(
                any(token.startswith(part) for token in display_tokens)
                for part in query_tokens
            )
        if mode == "contains":
            return query_lower in display
        return query_lower == display
