from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import ValidationError
from app.models.domain import PdfParseResult


def _make_iso_date(value: str) -> str:
    try:
        date_obj = datetime.strptime(value, "%Y-%m-%d")
        return date_obj.strftime("%Y-%m-%dT00:00:00.000+0000")
    except ValueError:
        return value


def _parse_numeric_and_unit(raw: str) -> tuple[int | float | None, str | None]:
    cleaned = raw.strip()
    match = re.match(r"(\d+(?:\.\d+)?)\s*(.*)", cleaned)
    if not match:
        return None, cleaned or None
    number = float(match.group(1))
    if number.is_integer():
        number = int(number)
    return number, match.group(2).strip() or None


def parse_patient_pdf(path: str | Path) -> PdfParseResult:
    file_path = Path(path)
    if not file_path.exists():
        raise ValidationError(f"PDF file not found: {file_path}")

    reader = PdfReader(str(file_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]

    parsed = PdfParseResult()
    current_section: str | None = None
    current_medication: dict[str, object] | None = None
    last_medication_key: str | None = None

    for line in lines:
        if line in {"Patient Demographics", "Problem List", "Vital Signs", "Allergies", "Medications"}:
            current_section = line
            if current_medication:
                parsed.medications.append(current_medication)
                current_medication = None
                last_medication_key = None
            continue

        if current_section == "Patient Demographics":
            if line.startswith("Encounter Type"):
                parsed.encounter_type_name = line.split("Encounter Type", 1)[1].strip()
            elif line.startswith("Location"):
                parsed.location_name = line.split("Location", 1)[1].strip()
            elif line.startswith("Provider"):
                parsed.provider_name = line.split("Provider", 1)[1].strip()
            elif line.startswith("Encounter Role"):
                parsed.encounter_role_name = line.split("Encounter Role", 1)[1].strip()
            elif line.startswith("Name"):
                parsed.name = line.split("Name", 1)[1].strip()
            elif line.startswith("Age"):
                parsed.age = line.split("Age", 1)[1].strip()
            elif line.startswith("Gender"):
                parsed.gender = line.split("Gender", 1)[1].strip()
            continue

        if current_section == "Problem List":
            if re.search(r"\d{4}-\d{2}-\d{2}", line):
                parts = line.split()
                parsed.conditions.append(
                    {
                        "condition_name": " ".join(parts[:-3]),
                        "clinical_status": parts[-3].lower(),
                        "verification_status": parts[-2].lower(),
                        "onset_date": _make_iso_date(parts[-1]),
                    }
                )
            continue

        if current_section == "Vital Signs":
            if line.startswith("Weight"):
                parsed.observations["Weight (kg)"] = int(re.search(r"(\d+)", line).group(1))
            elif line.startswith("Height"):
                parsed.observations["Height"] = int(re.search(r"(\d+)", line).group(1))
            elif line.startswith("Blood Pressure"):
                systolic, diastolic = line.split("Blood Pressure", 1)[1].strip().split("/")
                parsed.observations["Systolic blood pressure"] = int(systolic.strip())
                parsed.observations["Diastolic blood pressure"] = int(diastolic.strip())
            continue

        if current_section == "Allergies":
            if line.lower().startswith("allergen") and "severity" in line.lower():
                continue
            parts = line.split()
            if len(parts) >= 4:
                parsed.allergies.append(
                    {
                        "allergen_name": parts[0],
                        "severity_name": parts[1],
                        "reaction_name": parts[2],
                        "comment": " ".join(parts[3:]),
                    }
                )
            elif parsed.allergies:
                parsed.allergies[-1]["comment"] += f" {line}"
            continue

        if current_section == "Medications":
            if line.startswith("Drug:"):
                if current_medication:
                    parsed.medications.append(current_medication)
                current_medication = {"drug_name": line.split(":", 1)[1].strip()}
                last_medication_key = None
                continue
            if not current_medication:
                continue
            if ":" in line:
                key, value = [item.strip() for item in line.split(":", 1)]
                last_medication_key = key
                if key == "Concept":
                    current_medication["concept_name"] = value
                elif key == "Dose":
                    number, unit = _parse_numeric_and_unit(value)
                    current_medication["dose"] = number
                    current_medication["dose_units_name"] = unit
                elif key == "Route":
                    current_medication["route_name"] = value
                elif key == "Frequency":
                    current_medication["frequency_name"] = value
                elif key == "Duration":
                    number, unit = _parse_numeric_and_unit(value)
                    current_medication["duration"] = number
                    current_medication["duration_units_name"] = unit
                elif key == "Quantity":
                    number, unit = _parse_numeric_and_unit(value)
                    current_medication["quantity"] = number
                    current_medication["quantity_units_name"] = unit
                elif key == "Refills":
                    current_medication["num_refills"] = int(value)
                elif key == "Care Setting":
                    current_medication["care_setting_name"] = value
                elif key == "Orderer":
                    current_medication["orderer_name"] = value
            elif last_medication_key == "Frequency":
                current_medication["frequency_name"] = f"{current_medication.get('frequency_name', '')} {line}".strip()

    if current_medication:
        parsed.medications.append(current_medication)

    return parsed
