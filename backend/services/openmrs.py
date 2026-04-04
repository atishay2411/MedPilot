"""
backend/services/openmrs.py
OpenMRS REST API client — adapted from lab EHR_script.py
"""
import os
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OPENMRS_BASE_URL", "http://localhost:8080/openmrs")
USERNAME = os.getenv("OPENMRS_USER", "admin")
PASSWORD = os.getenv("OPENMRS_PASS", "Admin123")

session = requests.Session()
session.auth = HTTPBasicAuth(USERNAME, PASSWORD)
session.headers.update({"Content-Type": "application/json"})


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def get_uuid(entity: str, name: str):
    """Search entity by name and return its UUID."""
    if not name:
        return None
    url = f"{BASE_URL}/ws/rest/v1/{entity}?q={name}"
    resp = session.get(url)
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        if results:
            return results[0].get("uuid")
    return None


# ─────────────────────────────────────────────
# Patients
# ─────────────────────────────────────────────

def search_patients(query: str):
    url = f"{BASE_URL}/ws/rest/v1/patient?q={query}&v=full"
    resp = session.get(url)
    if resp.status_code == 200:
        return resp.json().get("results", [])
    return []


def get_patient(patient_uuid: str):
    url = f"{BASE_URL}/ws/rest/v1/patient/{patient_uuid}?v=full"
    resp = session.get(url)
    if resp.status_code == 200:
        return resp.json()
    return None


def get_patient_uuid(openmrs_id: str):
    results = search_patients(openmrs_id)
    if results:
        return results[0].get("uuid")
    return None


def create_patient(data: dict):
    """Register a new patient."""
    url = f"{BASE_URL}/ws/rest/v1/patient"
    resp = session.post(url, json=data)
    if resp.status_code in [200, 201]:
        return {"success": True, "uuid": resp.json().get("uuid"), "data": resp.json()}
    return {"success": False, "error": resp.text}


# ─────────────────────────────────────────────
# Encounters
# ─────────────────────────────────────────────

def create_encounter(patient_uuid: str, encounter_type_name: str,
                     location_name: str, provider_name: str,
                     encounter_role_name: str = "Unknown"):
    encounter_type_uuid = get_uuid("encountertype", encounter_type_name)
    location_uuid = get_uuid("location", location_name)
    provider_uuid = get_uuid("provider", provider_name)
    encounter_role_uuid = get_uuid("encounterrole", encounter_role_name)

    if None in [encounter_type_uuid, location_uuid, provider_uuid, encounter_role_uuid]:
        return None

    payload = {
        "encounterDatetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
        "patient": patient_uuid,
        "encounterType": encounter_type_uuid,
        "location": location_uuid,
        "encounterProviders": [
            {"provider": provider_uuid, "encounterRole": encounter_role_uuid}
        ]
    }
    resp = session.post(f"{BASE_URL}/ws/rest/v1/encounter", json=payload)
    if resp.status_code == 201:
        return resp.json().get("uuid")
    return None


# ─────────────────────────────────────────────
# Vitals / Observations
# ─────────────────────────────────────────────

def get_vitals(patient_uuid: str):
    url = f"{BASE_URL}/ws/rest/v1/obs?patient={patient_uuid}&v=full"
    resp = session.get(url)
    if resp.status_code == 200:
        return resp.json().get("results", [])
    return []


def add_observation(patient_uuid: str, concept_name: str, value):
    concept_uuid = get_uuid("concept", concept_name)
    if not concept_uuid:
        return {"success": False, "error": f"Concept '{concept_name}' not found"}

    payload = {
        "person": patient_uuid,
        "concept": concept_uuid,
        "obsDatetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
        "value": value
    }
    resp = session.post(f"{BASE_URL}/ws/rest/v1/obs", json=payload)
    if resp.status_code in [200, 201]:
        return {"success": True, "data": resp.json()}
    return {"success": False, "error": resp.text}


# ─────────────────────────────────────────────
# Conditions
# ─────────────────────────────────────────────

def get_conditions(patient_uuid: str):
    url = f"{BASE_URL}/ws/rest/v1/condition?patient={patient_uuid}&v=full"
    resp = session.get(url)
    if resp.status_code == 200:
        return resp.json().get("results", [])
    return []


def add_condition(patient_uuid: str, condition_name: str,
                  clinical_status: str = "ACTIVE",
                  verification_status: str = "CONFIRMED",
                  onset_date: str = None):
    condition_uuid = get_uuid("concept", condition_name)
    if not condition_uuid:
        return {"success": False, "error": f"Condition concept '{condition_name}' not found"}

    payload = {
        "patient": patient_uuid,
        "condition": {"coded": condition_uuid},
        "clinicalStatus": clinical_status,
        "verificationStatus": verification_status,
        "onsetDate": onset_date or datetime.now().strftime("%Y-%m-%dT00:00:00.000+0000")
    }
    resp = session.post(f"{BASE_URL}/ws/rest/v1/condition", json=payload)
    if resp.status_code in [200, 201]:
        return {"success": True, "data": resp.json()}
    return {"success": False, "error": resp.text}


def update_condition(condition_uuid: str, clinical_status: str):
    url = f"{BASE_URL}/ws/rest/v1/condition/{condition_uuid}"
    resp = session.post(url, json={"clinicalStatus": clinical_status})
    if resp.status_code in [200, 201]:
        return {"success": True}
    return {"success": False, "error": resp.text}


def delete_condition(condition_uuid: str):
    url = f"{BASE_URL}/ws/rest/v1/condition/{condition_uuid}"
    resp = session.delete(url)
    if resp.status_code in [200, 204]:
        return {"success": True}
    return {"success": False, "error": resp.text}


# ─────────────────────────────────────────────
# Allergies
# ─────────────────────────────────────────────

def get_allergies(patient_uuid: str):
    url = f"{BASE_URL}/ws/rest/v1/patient/{patient_uuid}/allergy?v=full"
    resp = session.get(url)
    if resp.status_code == 200:
        return resp.json().get("results", [])
    return []


def add_allergy(patient_uuid: str, allergen_name: str,
                severity_name: str, reaction_name: str, comment: str = ""):
    allergen_uuid = get_uuid("concept", allergen_name)
    severity_uuid = get_uuid("concept", severity_name)
    reaction_uuid = get_uuid("concept", reaction_name)

    if None in [allergen_uuid, severity_uuid, reaction_uuid]:
        return {"success": False, "error": "One or more concept UUIDs not found"}

    # Dedup check
    existing = get_allergies(patient_uuid)
    for allergy in existing:
        existing_uuid = allergy.get("allergen", {}).get("codedAllergen", {}).get("uuid")
        if existing_uuid == allergen_uuid:
            return {"success": False, "error": f"Allergy '{allergen_name}' already exists"}

    payload = {
        "allergen": {"allergenType": "DRUG", "codedAllergen": {"uuid": allergen_uuid}},
        "severity": {"uuid": severity_uuid},
        "reactions": [{"reaction": {"uuid": reaction_uuid}}],
        "comment": comment or f"{severity_name} {reaction_name} reaction to {allergen_name}"
    }
    resp = session.post(f"{BASE_URL}/ws/rest/v1/patient/{patient_uuid}/allergy", json=payload)
    if resp.status_code in [200, 201]:
        return {"success": True, "data": resp.json()}
    return {"success": False, "error": resp.text}


def delete_allergy(patient_uuid: str, allergy_uuid: str):
    url = f"{BASE_URL}/ws/rest/v1/patient/{patient_uuid}/allergy/{allergy_uuid}"
    resp = session.delete(url)
    if resp.status_code in [200, 204]:
        return {"success": True}
    return {"success": False, "error": resp.text}


# ─────────────────────────────────────────────
# Medications
# ─────────────────────────────────────────────

def get_medications(patient_uuid: str):
    url = f"{BASE_URL}/ws/rest/v1/order?patient={patient_uuid}&v=full"
    resp = session.get(url)
    if resp.status_code == 200:
        return resp.json().get("results", [])
    return []


def add_medication(patient_uuid: str, encounter_uuid: str, med_conf: dict):
    drug_name = med_conf.get("drug_name")
    drug_uuid = get_uuid("drug", drug_name)
    if not drug_uuid:
        return {"success": False, "error": f"Drug '{drug_name}' not found"}

    drug_resp = session.get(f"{BASE_URL}/ws/rest/v1/drug/{drug_uuid}")
    concept_uuid = None
    if drug_resp.status_code == 200:
        concept_uuid = drug_resp.json().get("concept", {}).get("uuid")

    if not concept_uuid:
        return {"success": False, "error": f"Concept for drug '{drug_name}' not found"}

    # Dedup
    existing = get_medications(patient_uuid)
    for order in existing:
        if drug_name.lower() in order.get("display", "").lower() and order.get("action") == "NEW":
            return {"success": False, "error": f"Medication '{drug_name}' already active"}

    payload = {
        "type": "drugorder",
        "patient": patient_uuid,
        "encounter": encounter_uuid,
        "drug": drug_uuid,
        "concept": concept_uuid,
        "careSetting": get_uuid("caresetting", med_conf.get("care_setting_name", "Outpatient")),
        "orderer": get_uuid("provider", med_conf.get("orderer_name")),
        "dose": med_conf.get("dose"),
        "doseUnits": get_uuid("concept", med_conf.get("dose_units_name")),
        "route": get_uuid("concept", med_conf.get("route_name")),
        "frequency": get_uuid("concept", med_conf.get("frequency_name")),
        "duration": med_conf.get("duration"),
        "durationUnits": get_uuid("concept", med_conf.get("duration_units_name")),
        "quantity": med_conf.get("quantity"),
        "quantityUnits": get_uuid("concept", med_conf.get("quantity_units_name")),
        "numRefills": med_conf.get("num_refills", 0)
    }
    resp = session.post(f"{BASE_URL}/ws/rest/v1/order", json=payload)
    if resp.status_code in [200, 201]:
        return {"success": True, "data": resp.json()}
    return {"success": False, "error": resp.text}
