from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.services import openmrs

router = APIRouter(prefix="/api/medications", tags=["medications"])


class MedicationIn(BaseModel):
    patient_uuid: str
    encounter_uuid: str
    drug_name: str
    dose: Optional[float] = None
    dose_units_name: Optional[str] = None
    route_name: Optional[str] = None
    frequency_name: Optional[str] = None
    duration: Optional[int] = None
    duration_units_name: Optional[str] = None
    quantity: Optional[int] = None
    quantity_units_name: Optional[str] = None
    care_setting_name: str = "Outpatient"
    orderer_name: Optional[str] = None
    num_refills: int = 0


@router.get("/{patient_uuid}")
def get_medications(patient_uuid: str):
    return {"medications": openmrs.get_medications(patient_uuid)}


@router.post("")
def add_medication(med: MedicationIn):
    return openmrs.add_medication(med.patient_uuid, med.encounter_uuid, med.dict())
