from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.models.domain import ParsedIntent
from app.services.observations import VITALS_CODE_MAP


VITAL_SYNONYMS = {
    "weight": ("Weight (kg)", "kg"),
    "height": ("Height", "cm"),
    "temperature": ("Temperature", "C"),
    "temp": ("Temperature", "C"),
    "respiratory rate": ("Respiratory Rate", "breaths/min"),
    "respirations": ("Respiratory Rate", "breaths/min"),
    "spo2": ("Oxygen Saturation (SpO2)", "%"),
    "oxygen saturation": ("Oxygen Saturation (SpO2)", "%"),
}


FREQUENCY_MULTIPLIERS = {
    "daily": 1,
    "once daily": 1,
    "twice daily": 2,
    "three times daily": 3,
    "four times daily": 4,
}


class PromptParser:
    def parse(self, prompt: str, *, has_file: bool = False) -> ParsedIntent:
        lowered = prompt.lower().strip()

        if has_file and any(token in lowered for token in ["pdf", "ingest", "extract", "upload"]):
            return ParsedIntent(intent="ingest_pdf", write=True, confidence=0.97, entities={"patient_query": self._extract_patient_query(prompt)})

        if "health gorilla" in lowered and any(token in lowered for token in ["sync", "import", "retrieve"]):
            return ParsedIntent(intent="sync_health_gorilla", write=True, confidence=0.95, entities=self._extract_person_entities(prompt))

        if any(token in lowered for token in ["change patient", "switch patient", "use patient", "set patient"]):
            return ParsedIntent(intent="switch_patient", write=False, confidence=0.95, entities={"patient_query": self._extract_switch_patient_query(prompt)})

        if any(token in lowered for token in ["metadata", "capabilities", "capability statement"]):
            return ParsedIntent(intent="get_metadata", write=False, confidence=0.9, entities={})

        if self._is_patient_creation_request(lowered):
            intake_entities = self._extract_patient_intake_entities(prompt)
            if self._has_intake_content(intake_entities):
                return ParsedIntent(intent="patient_intake", write=True, confidence=0.94, entities=intake_entities)
            return ParsedIntent(intent="create_patient", write=True, confidence=0.95, entities=self._extract_registration_entities(prompt))

        if "encounter" in lowered and any(token in lowered for token in ["start", "create", "new"]):
            return ParsedIntent(intent="create_encounter", write=True, confidence=0.88, entities=self._extract_encounter_entities(prompt))

        if any(token in lowered for token in ["prescribe ", "add medication", "start medication"]):
            return ParsedIntent(intent="create_medication", write=True, confidence=0.86, entities=self._extract_medication_entities(prompt))

        if any(token in lowered for token in ["summarize", "summary", "analyze", "analysis", "overview"]):
            return ParsedIntent(intent="patient_analysis", write=False, confidence=0.9, entities={"patient_query": self._extract_patient_query(prompt)})

        if self._is_patient_search_request(lowered):
            return ParsedIntent(
                intent="search_patient",
                write=False,
                confidence=0.9,
                entities={
                    "patient_query": self._extract_patient_query(prompt) or self._strip_search_wrapper(prompt),
                    "search_mode": self._extract_search_mode(prompt),
                },
            )

        if any(token in lowered for token in ["show vitals", "show observations", "last hba1c", "what is the patient's", "show last"]) or self._contains_vital(lowered):
            if any(token in lowered for token in ["record ", "add ", "log "]) and self._contains_vital(lowered):
                return ParsedIntent(intent="create_observation", write=True, confidence=0.9, entities=self._extract_observation_entities(prompt))
            if any(token in lowered for token in ["update ", "correct "]) and self._contains_vital(lowered):
                return ParsedIntent(intent="update_observation", write=True, confidence=0.86, entities=self._extract_observation_entities(prompt))
            if any(token in lowered for token in ["delete ", "remove "]):
                return ParsedIntent(intent="delete_observation", write=True, confidence=0.85, entities={"patient_query": self._extract_patient_query(prompt), "observation_display": self._extract_vital_display(prompt)})
            return ParsedIntent(intent="get_observations", write=False, confidence=0.82, entities={"patient_query": self._extract_patient_query(prompt), "observation_display": self._extract_vital_display(prompt)})

        if "condition" in lowered or "diagnosis" in lowered or "problem list" in lowered:
            if any(token in lowered for token in ["add diagnosis", "add condition", "diagnosis:", "condition:"]):
                return ParsedIntent(intent="create_condition", write=True, confidence=0.93, entities=self._extract_condition_create_entities(prompt))
            if any(token in lowered for token in ["mark ", "set ", "change "]) and any(token in lowered for token in ["inactive", "resolved", "active"]):
                return ParsedIntent(intent="update_condition", write=True, confidence=0.92, entities=self._extract_condition_update_entities(prompt))
            if any(token in lowered for token in ["delete ", "remove "]):
                return ParsedIntent(intent="delete_condition", write=True, confidence=0.91, entities=self._extract_named_delete_entities(prompt, "condition"))
            return ParsedIntent(intent="get_conditions", write=False, confidence=0.85, entities={"patient_query": self._extract_patient_query(prompt)})

        if "allergy" in lowered:
            if any(token in lowered for token in ["add allergy", "record allergy"]):
                return ParsedIntent(intent="create_allergy", write=True, confidence=0.9, entities=self._extract_allergy_entities(prompt))
            if "severity" in lowered and any(token in lowered for token in ["change ", "update "]):
                return ParsedIntent(intent="update_allergy", write=True, confidence=0.88, entities=self._extract_allergy_update_entities(prompt))
            if any(token in lowered for token in ["delete ", "remove "]):
                return ParsedIntent(intent="delete_allergy", write=True, confidence=0.9, entities=self._extract_named_delete_entities(prompt, "allergy"))
            return ParsedIntent(intent="get_allergies", write=False, confidence=0.84, entities={"patient_query": self._extract_patient_query(prompt)})

        if any(token in lowered for token in ["medication", "medications", "active orders", "prescribe", "discontinue", "stop ", "dispense", "dispensing"]):
            if any(token in lowered for token in ["dispense history", "medication dispense history", "show dispenses"]):
                return ParsedIntent(intent="get_medication_dispense", write=False, confidence=0.86, entities={"patient_query": self._extract_patient_query(prompt)})
            if any(token in lowered for token in ["record dispensing", "dispense ", "record dispense"]):
                return ParsedIntent(intent="create_medication_dispense", write=True, confidence=0.87, entities=self._extract_dispense_entities(prompt))
            if any(token in lowered for token in ["stop ", "discontinue "]):
                return ParsedIntent(intent="update_medication", write=True, confidence=0.88, entities=self._extract_medication_stop_entities(prompt))
            return ParsedIntent(intent="get_medications", write=False, confidence=0.82, entities={"patient_query": self._extract_patient_query(prompt)})

        return ParsedIntent(intent="search_patient", write=False, confidence=0.45, entities={"patient_query": self._strip_command(prompt)})

    @staticmethod
    def _is_patient_creation_request(lowered_prompt: str) -> bool:
        create_markers = [
            "register patient",
            "register a patient",
            "add patient",
            "add a patient",
            "create patient",
            "create a patient",
            "new patient",
            "patient named",
        ]
        return any(marker in lowered_prompt for marker in create_markers)

    @staticmethod
    def _is_patient_search_request(lowered_prompt: str) -> bool:
        search_markers = [
            "find patient",
            "search patient",
            "look up patient",
            "is there a patient",
            "is there any patient",
            "do we have a patient",
            "do we have any patient",
            "patient called",
            "patient named",
            "patient whose name is",
            "whose name is",
        ]
        if any(marker in lowered_prompt for marker in search_markers):
            return True
        if re.search(r"^(find|search|look up)\b", lowered_prompt) and "patient" in lowered_prompt:
            return True
        return False

    @staticmethod
    def _has_intake_content(entities: dict[str, Any]) -> bool:
        return any(
            entities.get(key)
            for key in ["conditions", "allergies", "observations", "medications", "dispenses"]
        )

    @staticmethod
    def _contains_vital(lowered_prompt: str) -> bool:
        return any(token in lowered_prompt for token in list(VITAL_SYNONYMS) + ["blood pressure", "bp"])

    @staticmethod
    def _strip_command(prompt: str) -> str:
        return re.sub(r"^(find|search|look up|show|summarize|analyze)\s+", "", prompt, flags=re.IGNORECASE).strip()

    @staticmethod
    def _strip_search_wrapper(prompt: str) -> str:
        cleaned = prompt.strip().rstrip("?.!")
        patterns = [
            r"^(?:find|search|look up)\s+(?:any\s+|related\s+|matching\s+|possible\s+|similar\s+)*patient\s+(?:whose\s+name\s+(?:is|starts with|beginning with|begins with)\s+|called\s+|named\s+)?",
            r"^(?:is there|is there any|do we have|do we have any)\s+(?:a\s+|any\s+)?patient\s+(?:called\s+|named\s+)?",
            r"^patient\s+(?:called|named)\s+",
            r"^(?:find|search|look up)\s+.*?\bwhose\s+name\s+is\s+",
        ]
        for pattern in patterns:
            updated = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
            if updated != cleaned:
                cleaned = updated
                break
        return cleaned.strip()

    @staticmethod
    def _extract_patient_query(prompt: str) -> str | None:
        if re.search(r"\bthis patient\b", prompt, re.IGNORECASE):
            return None
        patterns = [
            r"(?:patient\s+whose\s+name\s+(?:starts with|beginning with|begins with)\s+)([A-Za-z0-9][A-Za-z0-9'\-]*)",
            r"(?:patient\s+whose\s+name\s+is\s+)([A-Za-z0-9][A-Za-z0-9'\-]+(?:\s+[A-Za-z0-9][A-Za-z0-9'\-]+){0,3})",
            r"(?:patient\s+(?:called|named)\s+)([A-Za-z0-9][A-Za-z0-9'\-]+(?:\s+[A-Za-z0-9][A-Za-z0-9'\-]+){0,3})",
            r"(?:whose\s+name\s+(?:starts with|beginning with|begins with)\s+)([A-Za-z0-9][A-Za-z0-9'\-]*)",
            r"(?:whose\s+name\s+is\s+)([A-Za-z0-9][A-Za-z0-9'\-]+(?:\s+[A-Za-z0-9][A-Za-z0-9'\-]+){0,3})",
            r"(?:is there|is there any|do we have|do we have any)\s+(?:a\s+|any\s+)?patient\s+(?:called\s+|named\s+)?([A-Za-z0-9][A-Za-z0-9'\-]+(?:\s+[A-Za-z0-9][A-Za-z0-9'\-]+){0,3})",
            r"(?:patient\s+id)\s+([A-Za-z0-9\-]+)",
            r"(?:patient|for|of)\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){1,3})",
            r"(?:from\s+health gorilla\s+)([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,3})",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip("?.!")
        return None

    @staticmethod
    def _extract_search_mode(prompt: str) -> str:
        lowered = prompt.lower()
        if any(token in lowered for token in ["starts with", "beginning with", "begins with", "start with"]):
            return "starts_with"
        if "contains" in lowered:
            return "contains"
        return "default"

    @staticmethod
    def _extract_switch_patient_query(prompt: str) -> str | None:
        patterns = [
            r"(?:change|switch|use|set)\s+patient\s+(?:to\s+)?([A-Za-z0-9][A-Za-z0-9 '\-]+)",
            r"(?:change|switch)\s+to\s+patient\s+([A-Za-z0-9][A-Za-z0-9 '\-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".")
        return None

    def _extract_person_entities(self, prompt: str) -> dict[str, Any]:
        name_match = re.search(r"(?:sync|import|retrieve)\s+(?:patient\s+)?([A-Za-z][A-Za-z'\-]+)\s+([A-Za-z][A-Za-z'\-]+)", prompt, re.IGNORECASE)
        entities = {
            "given_name": name_match.group(1) if name_match else None,
            "family_name": name_match.group(2) if name_match else None,
            "birthdate": self._extract_birthdate(prompt),
        }
        return entities

    def _extract_registration_entities(self, prompt: str) -> dict[str, Any]:
        name_match = re.search(
            r"(?:register|add|create)\s+(?:a\s+)?(?:new\s+)?patient(?:\s+named)?\s+([A-Za-z0-9][A-Za-z0-9'\-]+)(?:\s+([A-Za-z0-9][A-Za-z0-9'\-]+))?(?=,|\s+who\b|\s+that\b|\s+born\b|\s+was born\b|\s+dob\b|\s+with\b|\s+in\b|$)",
            prompt,
            re.IGNORECASE,
        )
        gender = "U"
        gender_map = {"female": "F", "male": "M", "other": "O", "unknown": "U"}
        for word, code in gender_map.items():
            if re.search(rf"\b{word}\b", prompt, re.IGNORECASE):
                gender = code
        tail_parts = [part.strip() for part in prompt.split(",")]
        city = None
        if len(tail_parts) > 1:
            candidate = tail_parts[-1]
            if candidate and not re.search(r"\b(male|female|other|unknown|born|birth|dob|with|allerg|condition|disease|weight|height|temperature|bp|blood pressure|prescribe|dispense)\b", candidate, re.IGNORECASE):
                city = candidate
        if not city:
            city_match = re.search(r"\bin\s+([A-Za-z][A-Za-z '\-]+)\s*$", prompt, re.IGNORECASE)
            if city_match:
                city = city_match.group(1).strip().rstrip(".")
        return {
            "given_name": name_match.group(1) if name_match else None,
            "family_name": name_match.group(2) if name_match and name_match.group(2) else "Unknown",
            "birthdate": self._extract_birthdate(prompt),
            "gender": gender,
            "city_village": city,
        }

    @staticmethod
    def _extract_birthdate(prompt: str) -> str | None:
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", prompt)
        if iso_match:
            return iso_match.group(1)

        candidates: list[str] = []
        patterns = [
            r"(?:born(?:\s+on)?|birth(?:date)?|dob(?:\s+is|:)?|was born on)\s+([A-Za-z0-9,\-/ ]+)",
            r"\b(on\s+)?(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b",
            r"\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b",
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, prompt, re.IGNORECASE):
                value = next((group for group in match.groups()[::-1] if group), "")
                value = re.split(r"(?:,?\s+(?:with|and|who|that)\b|[.;])", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                if value:
                    candidates.append(value)

        for candidate in candidates:
            normalized = PromptParser._normalize_date_candidate(candidate)
            if normalized:
                return normalized
        return None

    @staticmethod
    def _normalize_date_candidate(value: str) -> str | None:
        cleaned = re.sub(r"\b(on|the)\b", "", value, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
        for fmt in (
            "%d %b %Y",
            "%d %B %Y",
            "%b %d %Y",
            "%B %d %Y",
            "%b %d, %Y",
            "%B %d, %Y",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
        ):
            try:
                return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    def _extract_patient_intake_entities(self, prompt: str) -> dict[str, Any]:
        registration = self._extract_registration_entities(prompt)
        observations = self._extract_all_observations(prompt)
        medications = self._extract_medication_list(prompt)
        dispenses = self._extract_dispense_list(prompt)
        conditions = self._extract_condition_list(prompt)
        allergies = self._extract_allergy_list(prompt)
        encounter = None
        if medications or dispenses or re.search(r"\b(encounter|clinic|location|visit)\b", prompt, re.IGNORECASE):
            encounter = self._extract_encounter_entities(prompt)
        return {
            **registration,
            "conditions": conditions,
            "allergies": allergies,
            "observations": observations,
            "medications": medications,
            "dispenses": dispenses,
            "encounter": encounter,
            "patient_query": None,
        }

    def _extract_encounter_entities(self, prompt: str) -> dict[str, Any]:
        location_match = re.search(r"\bat\s+([A-Za-z][A-Za-z '\-]+)", prompt, re.IGNORECASE)
        return {
            "patient_query": self._extract_patient_query(prompt),
            "encounter_type_name": "Vitals",
            "location_name": location_match.group(1).strip() if location_match else "Outpatient Clinic",
            "provider_name": "Super User",
            "encounter_role_name": "Clinician",
        }

    def _extract_condition_create_entities(self, prompt: str) -> dict[str, Any]:
        stripped = re.sub(r"^(add diagnosis|add condition|diagnosis:|condition:)\s*", "", prompt, flags=re.IGNORECASE).strip()
        name = re.split(r",|\bonset\b|\bactive\b|\binactive\b|\bresolved\b|\bconfirmed\b|\bunconfirmed\b", stripped, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        onset_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", prompt)
        clinical_status = "active"
        for token in ["inactive", "resolved", "active"]:
            if re.search(rf"\b{token}\b", prompt, re.IGNORECASE):
                clinical_status = token
                break
        verification = "confirmed"
        for token in ["confirmed", "unconfirmed", "provisional", "refuted"]:
            if re.search(rf"\b{token}\b", prompt, re.IGNORECASE):
                verification = token
                break
        return {
            "patient_query": self._extract_patient_query(prompt),
            "condition_name": name,
            "clinical_status": clinical_status,
            "verification_status": verification,
            "onset_date": onset_match.group(1) if onset_match else None,
        }

    def _extract_condition_list(self, prompt: str) -> list[dict[str, Any]]:
        patterns = [
            r"(?:diagnoses?|diseases?|conditions?)\s*[:=]?\s*(.+?)(?=,\s*(?:allerg|weight|height|blood pressure|bp|temperature|temp|spo2|respiratory|prescribe|medication|dispense|route|dose|frequency|$)|$)",
            r"\bwith\s+(.+?)(?=,\s*(?:allerg|weight|height|blood pressure|bp|temperature|temp|spo2|respiratory|prescribe|medication|dispense|born|male|female|other|unknown|city|country|at |$)|$)",
        ]
        raw_section = None
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                raw_section = match.group(1).strip()
                break
        if not raw_section:
            return []

        raw_section = re.sub(r"\b(patient|named|male|female|born)\b.*$", "", raw_section, flags=re.IGNORECASE).strip()
        items = self._split_clinical_list(raw_section)
        conditions = []
        for item in items:
            cleaned = re.sub(r"^(diagnosis|condition|disease)\s*", "", item, flags=re.IGNORECASE).strip(" .")
            if cleaned:
                conditions.append(
                    {
                        "condition_name": cleaned,
                        "clinical_status": "active",
                        "verification_status": "confirmed",
                        "onset_date": None,
                    }
                )
        return conditions

    def _extract_condition_update_entities(self, prompt: str) -> dict[str, Any]:
        match = re.search(r"(?:mark|set|change)\s+(.+?)\s+(?:as|to)\s+(active|inactive|resolved)", prompt, re.IGNORECASE)
        return {
            "patient_query": self._extract_patient_query(prompt),
            "condition_name": match.group(1).replace("condition", "").strip() if match else None,
            "status": match.group(2).lower() if match else "inactive",
        }

    def _extract_named_delete_entities(self, prompt: str, noun: str) -> dict[str, Any]:
        match = re.search(rf"(?:delete|remove)\s+(?:the\s+)?(.+?)\s+{noun}", prompt, re.IGNORECASE)
        return {
            "patient_query": self._extract_patient_query(prompt),
            "name": match.group(1).strip() if match else None,
        }

    def _extract_allergy_entities(self, prompt: str) -> dict[str, Any]:
        stripped = re.sub(r"^(add allergy|record allergy)\s*:?\s*", "", prompt, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"\s+for\s+patient\s+.*$", "", stripped, flags=re.IGNORECASE).strip()
        pieces = [part.strip() for part in stripped.split(",") if part.strip()]
        return {
            "patient_query": self._extract_patient_query(prompt),
            "allergen_name": pieces[0] if pieces else stripped,
            "severity": pieces[1].split()[0].lower() if len(pieces) > 1 else "moderate",
            "reaction": pieces[2] if len(pieces) > 2 else "rash",
            "comment": pieces[3] if len(pieces) > 3 else None,
        }

    def _extract_allergy_list(self, prompt: str) -> list[dict[str, Any]]:
        patterns = [
            r"(?:allerg(?:y|ies))\s*[:=]?\s*(.+?)(?=,\s*(?:weight|height|blood pressure|bp|temperature|temp|spo2|respiratory|prescribe|medication|dispense|$)|$)",
            r"allergic to\s+(.+?)(?=,\s*(?:weight|height|blood pressure|bp|temperature|temp|spo2|respiratory|prescribe|medication|dispense|$)|$)",
        ]
        raw_section = None
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                raw_section = match.group(1).strip()
                break
        if not raw_section:
            return []
        items = self._split_clinical_list(raw_section)
        allergies = []
        for item in items:
            allergies.append(self._parse_allergy_item(item))
        return [item for item in allergies if item["allergen_name"]]

    def _extract_allergy_update_entities(self, prompt: str) -> dict[str, Any]:
        match = re.search(r"(?:change|update)\s+(.+?)\s+(?:reaction\s+)?severity\s+to\s+([A-Za-z]+)", prompt, re.IGNORECASE)
        return {
            "patient_query": self._extract_patient_query(prompt),
            "allergen_name": match.group(1).replace("allergy", "").strip() if match else None,
            "severity": match.group(2).lower() if match else "moderate",
        }

    def _extract_medication_stop_entities(self, prompt: str) -> dict[str, Any]:
        match = re.search(r"(?:stop|discontinue)\s+(?:the\s+)?(.+?)(?:\s+order|\s+medication|$)", prompt, re.IGNORECASE)
        return {
            "patient_query": self._extract_patient_query(prompt),
            "drug_name": match.group(1).strip() if match else None,
            "status": "stopped",
        }

    def _extract_medication_entities(self, prompt: str) -> dict[str, Any]:
        medications = self._extract_medication_list(prompt)
        if not medications:
            return {"patient_query": self._extract_patient_query(prompt)}
        return {"patient_query": self._extract_patient_query(prompt), **medications[0]}

    def _extract_medication_list(self, prompt: str) -> list[dict[str, Any]]:
        prompt_without_patient = re.sub(r"\s+for\s+patient\s+.*$", "", prompt, flags=re.IGNORECASE).strip()
        pattern = re.compile(
            r"(?:prescribe|add medication|start medication)\s+(.+?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|tablet|tablets|tab|tabs|ml)\s+([A-Za-z]+)\s+(.+?)\s+for\s+(\d+)\s+(day|days|week|weeks|month|months)",
            re.IGNORECASE,
        )
        medications = []
        for match in pattern.finditer(prompt_without_patient):
            drug_name = match.group(1).strip()
            dose = float(match.group(2))
            dose_unit = match.group(3)
            route = match.group(4).capitalize()
            frequency_raw = match.group(5).strip().lower()
            duration = int(match.group(6))
            duration_unit = match.group(7).capitalize()
            normalized_frequency = " ".join(part.capitalize() for part in frequency_raw.split())
            multiplier = FREQUENCY_MULTIPLIERS.get(frequency_raw, 1)
            quantity = duration * multiplier
            medications.append(
                {
                    "drug_name": drug_name,
                    "concept_name": drug_name,
                    "dose": dose,
                    "dose_units_name": "Tablet" if dose_unit.lower() in {"tablet", "tablets", "tab", "tabs"} else dose_unit,
                    "route_name": route,
                    "frequency_name": normalized_frequency,
                    "duration": duration,
                    "duration_units_name": duration_unit,
                    "quantity": quantity,
                    "quantity_units_name": "Tablet" if route.lower() == "oral" else dose_unit,
                    "care_setting_name": "Outpatient",
                    "orderer_name": "Super User",
                    "encounter_type_name": "Vitals",
                    "location_name": "Outpatient Clinic",
                    "provider_name": "Super User",
                    "encounter_role_name": "Clinician",
                }
            )
        return medications

    def _extract_dispense_entities(self, prompt: str) -> dict[str, Any]:
        dispenses = self._extract_dispense_list(prompt)
        if not dispenses:
            return {"patient_query": self._extract_patient_query(prompt)}
        return {"patient_query": self._extract_patient_query(prompt), **dispenses[0]}

    def _extract_dispense_list(self, prompt: str) -> list[dict[str, Any]]:
        pattern = re.compile(
            r"(?:record dispensing(?: of)?|dispense|record dispense(?: of)?)\s+(\d+(?:\.\d+)?)\s+(tablet|tablets|capsule|capsules|ml|dose|doses)\s+of\s+(.+?)(?:\s+for\s+patient|\s*$)",
            re.IGNORECASE,
        )
        results = []
        for match in pattern.finditer(prompt):
            quantity = float(match.group(1))
            unit = match.group(2).capitalize()
            drug_name = match.group(3).strip()
            results.append(
                {
                    "drug_name": drug_name,
                    "quantity": quantity,
                    "unit": unit,
                    "when_handed_over": now_iso_like_date(prompt),
                    "dosage_text": f"Dispensed {quantity:g} {unit.lower()} of {drug_name}.",
                }
            )
        return results

    def _extract_vital_display(self, prompt: str) -> str | None:
        lowered = prompt.lower()
        if "blood pressure" in lowered or re.search(r"\bbp\b", lowered):
            return "Systolic blood pressure"
        for token, (display, _) in VITAL_SYNONYMS.items():
            if token in lowered:
                return display
        return None

    def _extract_observation_entities(self, prompt: str) -> dict[str, Any]:
        return {"patient_query": self._extract_patient_query(prompt), "observations": self._extract_all_observations(prompt)}

    def _extract_all_observations(self, prompt: str) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        bp_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", prompt)
        if bp_match:
            observations.extend(
                [
                    {
                        "display": "Systolic blood pressure",
                        "code": VITALS_CODE_MAP["Systolic blood pressure"],
                        "value": float(bp_match.group(1)),
                        "unit": "mmHg",
                    },
                    {
                        "display": "Diastolic blood pressure",
                        "code": VITALS_CODE_MAP["Diastolic blood pressure"],
                        "value": float(bp_match.group(2)),
                        "unit": "mmHg",
                    },
                ]
            )

        measurement_patterns = [
            (r"weight\s+(\d+(?:\.\d+)?)", "Weight (kg)", "kg"),
            (r"height\s+(\d+(?:\.\d+)?)", "Height", "cm"),
            (r"(?:temperature|temp)\s+(\d+(?:\.\d+)?)", "Temperature", "C"),
            (r"(?:spo2|oxygen saturation)\s+(\d+(?:\.\d+)?)", "Oxygen Saturation (SpO2)", "%"),
            (r"(?:respiratory rate|respirations?)\s+(\d+(?:\.\d+)?)", "Respiratory Rate", "breaths/min"),
        ]
        lowered = prompt.lower()
        for pattern, display, unit in measurement_patterns:
            match = re.search(pattern, lowered, re.IGNORECASE)
            if match:
                observations.append(
                    {
                        "display": display,
                        "code": VITALS_CODE_MAP[display],
                        "value": float(match.group(1)),
                        "unit": unit,
                    }
                )
        return observations

    @staticmethod
    def _split_clinical_list(raw: str) -> list[str]:
        normalized = re.sub(r"\band\b", ",", raw, flags=re.IGNORECASE)
        parts = [part.strip(" .") for part in re.split(r",|;", normalized) if part.strip(" .")]
        return parts

    @staticmethod
    def _parse_allergy_item(item: str) -> dict[str, Any]:
        severity = "moderate"
        reaction = "rash"
        cleaned = item.strip(" .")
        severity_match = re.search(r"\b(mild|moderate|severe)\b", cleaned, re.IGNORECASE)
        if severity_match:
            severity = severity_match.group(1).lower()
            cleaned = re.sub(rf"\b{severity_match.group(1)}\b", "", cleaned, flags=re.IGNORECASE).strip()
        reaction_match = re.search(r"\b(rash|anaphylaxis|fever|swelling|cough|hives|itching|nausea)\b", cleaned, re.IGNORECASE)
        if reaction_match:
            reaction = reaction_match.group(1).lower()
            cleaned = re.sub(rf"\b{reaction_match.group(1)}\b", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(allergy to|allergic to|allergy)\s*", "", cleaned, flags=re.IGNORECASE).strip()
        return {
            "allergen_name": cleaned,
            "severity": severity,
            "reaction": reaction,
            "comment": None,
        }


def now_iso_like_date(prompt: str) -> str:
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", prompt)
    if match:
        return f"{match.group(1)}T00:00:00Z"
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
