from app.services.utils import generate_openmrs_identifier


def test_generate_openmrs_identifier_appends_check_character():
    identifier = generate_openmrs_identifier(10023)
    assert identifier.startswith("10023")
    assert len(identifier) == 6
