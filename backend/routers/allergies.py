from fastapi import APIRouter
from pydantic import BaseModel
from backend.services import openmrs

router = APIRouter(prefix="/api/allergies", tags=["allergies"])


class AllergyIn(BaseModel):
    patient_uuid: str
    allergen_name: str
    severity_name: str
    reaction_name: str
    comment: str = ""


@router.get("/{patient_uuid}")
def get_allergies(patient_uuid: str):
    return {"allergies": openmrs.get_allergies(patient_uuid)}


@router.post("")
def add_allergy(allergy: AllergyIn):
    return openmrs.add_allergy(
        allergy.patient_uuid, allergy.allergen_name,
        allergy.severity_name, allergy.reaction_name, allergy.comment
    )


@router.delete("/{patient_uuid}/{allergy_uuid}")
def delete_allergy(patient_uuid: str, allergy_uuid: str):
    return openmrs.delete_allergy(patient_uuid, allergy_uuid)
