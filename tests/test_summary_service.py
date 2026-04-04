from app.services.summaries import SummaryService


class StubPatients:
    @staticmethod
    def format_patient_display(resource):
        return "Maria Santos"


class StubObservations:
    @staticmethod
    def extract_observation_snapshot(resource):
        code = resource["code"]["coding"][0]
        value = resource["valueQuantity"]
        return {
            "uuid": resource["id"],
            "display": code["display"],
            "value": value["value"],
            "unit": value["unit"],
            "effectiveDateTime": resource["effectiveDateTime"],
        }


def test_summary_service_builds_narrative_and_analysis():
    service = SummaryService(StubPatients(), StubObservations(), None, None, None, None)
    snapshot = {
        "patient": {"id": "patient-1", "gender": "female", "birthDate": "1985-03-22", "name": [{"given": ["Maria"], "family": "Santos"}]},
        "observations": {
            "entry": [
                {
                    "resource": {
                        "id": "obs-1",
                        "effectiveDateTime": "2026-04-04T10:00:00Z",
                        "code": {"coding": [{"display": "Systolic blood pressure"}]},
                        "valueQuantity": {"value": 150, "unit": "mmHg"},
                    }
                }
            ]
        },
        "conditions": {
            "entry": [
                {
                    "resource": {
                        "id": "cond-1",
                        "clinicalStatus": {"coding": [{"code": "active"}]},
                        "code": {"coding": [{"display": "Type 2 Diabetes"}]},
                    }
                }
            ]
        },
        "allergies": {"entry": []},
        "medications": {"entry": []},
        "encounters": {"entry": []},
    }

    brief = service.build_clinical_brief(snapshot)

    assert "Maria Santos" in brief["narrative"]
    assert any("Elevated systolic blood pressure" in point for point in brief["analysis"])
    assert brief["highlights"]["active_conditions"] == ["Type 2 Diabetes"]
