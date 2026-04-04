from pathlib import Path

from app.parsers.patient_pdf import parse_patient_pdf


def test_parse_patient_pdf_extracts_expected_sections():
    pdf_path = Path("/Users/atishayjain/Desktop/Hackathon/scarlethacks26/openmrs/EHR Lab/patient_record.pdf")
    parsed = parse_patient_pdf(pdf_path)

    assert parsed.encounter_type_name == "Vitals"
    assert parsed.location_name == "Outpatient Clinic"
    assert parsed.provider_name == "Super User"
    assert parsed.conditions[0]["condition_name"] == "Diabetes mellitus"
    assert parsed.observations["Weight (kg)"] == 77
    assert parsed.allergies[0]["allergen_name"] == "Aspirin"
    assert parsed.medications[0]["drug_name"] == "Metformin"
