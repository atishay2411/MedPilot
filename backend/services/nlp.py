"""
backend/services/nlp.py
OpenAI GPT-powered clinical intent resolver for MedPilot.
Supports multi-turn conversation and 24 clinical intent types.
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are MedPilot, an AI clinical workflow assistant integrated with OpenMRS.
You help clinicians manage patient records using natural language.

You MUST always respond with a valid JSON object (no markdown, no extra text).

## Supported Intents:
- search_patient: find patients by name or ID
- get_patient: get full patient details
- create_patient: register a new patient
- get_vitals: retrieve patient vitals/observations
- add_vital: add a new vital observation
- get_conditions: list patient conditions
- add_condition: add a new condition
- update_condition: change condition status (ACTIVE/INACTIVE)
- delete_condition: remove a condition
- get_allergies: list patient allergies
- add_allergy: add a new allergy
- delete_allergy: remove an allergy
- get_medications: list patient medications
- add_medication: prescribe a new medication
- ingest_pdf: upload and parse a PDF patient record
- create_encounter: create a clinical encounter
- general: general information or greeting

## Response Format:
{
  "intent": "<intent_name>",
  "message": "<friendly human-readable response to show the user>",
  "params": {
    // all extracted parameters relevant to the intent
    // e.g., "patient_id": "10003A6", "condition_name": "Hypertension"
  },
  "requires_confirmation": true/false,
  "confirmation_message": "<if requires_confirmation, what to confirm>"
}

## Rules:
- Set "requires_confirmation": true for any add/update/delete action
- Extract as many parameters as possible from the user message
- If a patient was mentioned earlier in the conversation, remember their ID
- Be concise and professional in your "message" field
- For vitals, map common terms: weight→"Weight (kg)", height→"Height", 
  blood pressure→"Systolic blood pressure"/"Diastolic blood pressure",
  temperature→"Temperature (C)", pulse→"Pulse"
"""


def resolve_intent(conversation_history: list[dict]) -> dict:
    """
    Resolve the clinical intent from a conversation history.
    conversation_history: list of {"role": "user"/"assistant", "content": "..."}
    Returns a parsed JSON dict.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "intent": "general",
            "message": raw,
            "params": {},
            "requires_confirmation": False
        }


def extract_patient_context(conversation_history: list[dict]) -> str | None:
    """
    Extract the most recently referenced patient ID from conversation history.
    Used to maintain context across turns.
    """
    for msg in reversed(conversation_history):
        if msg["role"] == "assistant":
            try:
                data = json.loads(msg["content"])
                patient_id = data.get("params", {}).get("patient_id")
                if patient_id:
                    return patient_id
            except Exception:
                pass
    return None
