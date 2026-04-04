from fastapi import APIRouter, HTTPException, UploadFile, File
import shutil, os, tempfile
from backend.services.pdf_parser import parse_pdf
from backend.services import openmrs

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post("/parse")
async def parse_only(file: UploadFile = File(...)):
    """
    Upload a patient record PDF and return the parsed structured data.
    Does NOT write to OpenMRS — for preview/reconciliation only.
    """
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        parsed = parse_pdf(tmp_path)
        return {"success": True, "parsed": parsed}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/commit")
async def commit_to_openmrs(patient_id: str, parsed: dict):
    """
    After clinician reviews parsed data, commit it to OpenMRS.
    Requires patient_id and the parsed data dict.
    """
    patient_uuid = openmrs.get_patient_uuid(patient_id)
    if not patient_uuid:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    results = {"encounter": None, "allergies": [], "conditions": [], "observations": [], "medications": []}

    # Create encounter
    encounter_uuid = openmrs.create_encounter(
        patient_uuid,
        encounter_type_name=parsed.get("encounter_type_name", "Clinic Visit"),
        location_name=parsed.get("location_name", "Unknown Location"),
        provider_name=parsed.get("provider_name", ""),
        encounter_role_name=parsed.get("encounter_role_name", "Unknown")
    )
    results["encounter"] = encounter_uuid

    # Allergies
    for allergy in parsed.get("allergies", []):
        res = openmrs.add_allergy(
            patient_uuid,
            allergy.get("allergen_name"),
            allergy.get("severity_name"),
            allergy.get("reaction_name"),
            allergy.get("comment", "")
        )
        results["allergies"].append({"allergen": allergy.get("allergen_name"), **res})

    # Conditions
    for cond in parsed.get("conditions", []):
        res = openmrs.add_condition(
            patient_uuid,
            cond.get("condition_name"),
            cond.get("clinical_status", "ACTIVE"),
            cond.get("verification_status", "CONFIRMED"),
            cond.get("onset_date")
        )
        results["conditions"].append({"condition": cond.get("condition_name"), **res})

    # Observations (vitals)
    for concept_name, value in parsed.get("observations", {}).items():
        res = openmrs.add_observation(patient_uuid, concept_name, value)
        results["observations"].append({"concept": concept_name, **res})

    # Medications
    if encounter_uuid:
        for med in parsed.get("medications", []):
            res = openmrs.add_medication(patient_uuid, encounter_uuid, med)
            results["medications"].append({"drug": med.get("drug_name"), **res})

    return {"success": True, "results": results}
