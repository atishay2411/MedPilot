from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services import openmrs

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.get("/search")
def search(q: str):
    results = openmrs.search_patients(q)
    return {"patients": results}


@router.get("/{patient_uuid}")
def get_patient(patient_uuid: str):
    p = openmrs.get_patient(patient_uuid)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p


@router.post("")
def create_patient(data: dict):
    return openmrs.create_patient(data)
