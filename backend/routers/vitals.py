from fastapi import APIRouter
from pydantic import BaseModel
from backend.services import openmrs

router = APIRouter(prefix="/api/vitals", tags=["vitals"])


class VitalIn(BaseModel):
    patient_uuid: str
    concept_name: str
    value: float


@router.get("/{patient_uuid}")
def get_vitals(patient_uuid: str):
    return {"vitals": openmrs.get_vitals(patient_uuid)}


@router.post("")
def add_vital(vital: VitalIn):
    return openmrs.add_observation(vital.patient_uuid, vital.concept_name, vital.value)
