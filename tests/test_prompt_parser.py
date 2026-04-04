from app.services.prompt_parser import PromptParser


def test_prompt_parser_extracts_create_patient_entities():
    parser = PromptParser()
    parsed = parser.parse("Add patient John Doe, male, born 1990-05-12, Nairobi")

    assert parsed.intent == "create_patient"
    assert parsed.write is True
    assert parsed.entities["given_name"] == "John"
    assert parsed.entities["family_name"] == "Doe"
    assert parsed.entities["birthdate"] == "1990-05-12"
    assert parsed.entities["gender"] == "M"


def test_prompt_parser_uses_context_for_this_patient_analysis():
    parser = PromptParser()
    parsed = parser.parse("Analyze this patient for urgent issues")

    assert parsed.intent == "patient_analysis"
    assert parsed.entities["patient_query"] is None


def test_prompt_parser_extracts_blood_pressure_observation():
    parser = PromptParser()
    parsed = parser.parse("Record blood pressure 140/90 for patient Maria Santos")

    assert parsed.intent == "create_observation"
    assert len(parsed.entities["observations"]) == 2
    assert parsed.entities["observations"][0]["display"] == "Systolic blood pressure"
    assert parsed.entities["observations"][0]["value"] == 140.0


def test_prompt_parser_extracts_medication_order_entities():
    parser = PromptParser()
    parsed = parser.parse("Prescribe Metformin 500 tablet oral twice daily for 30 days for patient Maria Santos")

    assert parsed.intent == "create_medication"
    assert parsed.entities["drug_name"] == "Metformin"
    assert parsed.entities["dose"] == 500.0
    assert parsed.entities["route_name"] == "Oral"
    assert parsed.entities["frequency_name"] == "Twice Daily"
    assert parsed.entities["quantity"] == 60


def test_prompt_parser_extracts_patient_intake_bundle():
    parser = PromptParser()
    parsed = parser.parse(
        "Add a new patient named John Doe, male, born 1990-05-12, with diabetes and hypertension, allergies penicillin and aspirin, blood pressure 140/90, weight 80, prescribe Metformin 500 tablet oral twice daily for 30 days"
    )

    assert parsed.intent == "patient_intake"
    assert parsed.entities["given_name"] == "John"
    assert len(parsed.entities["conditions"]) == 2
    assert parsed.entities["conditions"][0]["condition_name"] == "diabetes"
    assert len(parsed.entities["allergies"]) == 2
    assert len(parsed.entities["observations"]) >= 3
    assert len(parsed.entities["medications"]) == 1


def test_prompt_parser_extracts_medication_dispense_entities():
    parser = PromptParser()
    parsed = parser.parse("Record dispensing of 20 tablets of Paracetamol for patient Maria Santos")

    assert parsed.intent == "create_medication_dispense"
    assert parsed.entities["drug_name"] == "Paracetamol"
    assert parsed.entities["quantity"] == 20.0
