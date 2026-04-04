from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.services import openmrs

router = APIRouter(prefix="/api/conditions", tags=["conditions"])


class ConditionIn(BaseModel):
    patient_uuid: str
    condition_name: str
    clinical_status: str = "ACTIVE"
    verification_status: str = "CONFIRMED"
    onset_date: Optional[str] = None


@router.get("/{patient_uuid}")
def get_conditions(patient_uuid: str):
    return {"conditions": openmrs.get_conditions(patient_uuid)}


@router.post("")
def add_condition(condition: ConditionIn):
    return openmrs.add_condition(
        condition.patient_uuid, condition.condition_name,
        condition.clinical_status, condition.verification_status, condition.onset_date
    )


@router.patch("/{condition_uuid}")
def update_condition(condition_uuid: str, clinical_status: str):
    return openmrs.update_condition(condition_uuid, clinical_status)


@router.delete("/{condition_uuid}")
def delete_condition(condition_uuid: str):
    return openmrs.delete_condition(condition_uuid)
