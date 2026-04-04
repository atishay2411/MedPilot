from types import SimpleNamespace

from app.services.patients import PatientService


class FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []

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
        raise AssertionError(f"Unexpected path: {path}")


def test_patient_service_prefix_search_filters_results():
    service = PatientService(FakeClient(), SimpleNamespace())

    results = service.search("N", search_mode="starts_with")

    assert [item["display"] for item in results] == ["Nesh Test", "Nora Lane"]


def test_patient_service_contains_search_filters_results():
    service = PatientService(FakeClient(), SimpleNamespace())

    results = service.search("esh", search_mode="contains")

    assert [item["display"] for item in results] == ["Nesh Test"]
