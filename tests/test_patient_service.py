from types import SimpleNamespace

from app.services.patients import PatientService


class FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []
        self.delete_calls: list[tuple[str, dict | None]] = []

    def get(self, path: str, *, params=None):
        self.calls.append((path, params))
        if path == "/ws/rest/v1/patient":
            return {
                "results": [
                    {"uuid": "rest-1", "display": "Nesh Test"},
                    {"uuid": "rest-2", "display": "Maria Santos"},
                ]
            }
        if path == "/ws/fhir2/R4/Patient":
            return {
                "entry": [
                    {
                        "resource": {
                            "id": "fhir-1",
                            "name": [{"given": ["Nora"], "family": "Lane"}],
                            "identifier": [{"value": "10001YY"}],
                        }
                    }
                ]
            }
        if path == "/ws/fhir2/R4/Patient/12345678-1234-1234-1234-1234567890ab":
            return {
                "id": "12345678-1234-1234-1234-1234567890ab",
                "name": [{"given": ["Nesh"], "family": "Test"}],
                "identifier": [{"value": "10001YY"}],
            }
        raise AssertionError(f"Unexpected path: {path}")

    def delete(self, path: str, *, params=None):
        self.delete_calls.append((path, params))
        return {}


def test_patient_service_prefix_search_filters_results():
    service = PatientService(FakeClient(), SimpleNamespace())

    results = service.search("N", search_mode="starts_with")

    assert [item["display"] for item in results] == ["Nesh Test", "Nora Lane"]


def test_patient_service_contains_search_filters_results():
    service = PatientService(FakeClient(), SimpleNamespace())

    results = service.search("esh", search_mode="contains")

    assert [item["display"] for item in results] == ["Nesh Test"]


def test_patient_service_search_all_uses_fhir_list():
    service = PatientService(FakeClient(), SimpleNamespace())

    results = service.search("all")

    assert [item["display"] for item in results] == ["Nora Lane"]


def test_patient_service_search_by_identifier_uses_fhir_lookup():
    """search_by_identifier is the renamed public API (was search_by_identifier_or_uuid)."""
    service = PatientService(FakeClient(), SimpleNamespace())

    results = service.search_by_identifier("12345678-1234-1234-1234-1234567890ab")

    assert [item["display"] for item in results] == ["Nesh Test"]


def test_patient_service_delete_passes_purge_flag():
    client = FakeClient()
    service = PatientService(client, SimpleNamespace())

    result = service.delete("patient-uuid-1", purge=True)

    assert result["deleted"] is True
    assert client.delete_calls == [("/ws/rest/v1/patient/patient-uuid-1", {"purge": "true"})]


def test_patient_service_build_update_payload_returns_only_supplied_fields():
    from app.models.domain import PatientUpdateInput
    service = PatientService(FakeClient(), SimpleNamespace())

    update = PatientUpdateInput(patient_uuid="uuid-1", given_name="Maria", gender="F")
    payload = service.build_update_payload(update)

    person = payload["person"]
    assert person["names"][0]["givenName"] == "Maria"
    assert person["gender"] == "F"
    # Unset fields must NOT appear so they don't overwrite existing data
    assert "birthdate" not in person
    assert "addresses" not in person


def test_patient_service_build_update_payload_empty_when_no_fields():
    from app.models.domain import PatientUpdateInput
    service = PatientService(FakeClient(), SimpleNamespace())

    update = PatientUpdateInput(patient_uuid="uuid-1")
    payload = service.build_update_payload(update)

    assert payload == {}
