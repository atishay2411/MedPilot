"""Deterministic intent classifier and entity extractor.

This module provides pattern-based intent classification and entity extraction
that runs BEFORE the LLM, ensuring reliable routing for common queries even
when using small models like llama3.2 that cannot produce structured output well.

The LLM is used as a fallback only when patterns don't match.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.services.llm_reasoning import ConversationalDecision


# ── Capabilities description ──────────────────────────────────────────

CAPABILITIES_TEXT = """\
I'm **MedPilot**, your AI clinical copilot for OpenMRS. Here's what I can help with:

**🔍 Patient Management**
• Search, list, or count patients — *"List all patients"*, *"How many patients are there?"*
• Register new patients — *"Add a patient named John Doe, born Jan 1 1990"*
• Update demographics — *"Update Maria Santos gender to F"*
• Delete patient records — *"Delete patient John Doe"*

**📋 Clinical Data (per patient)**
• View & record vitals/observations — *"Show Maria's vitals"*, *"Record BP 120/80 for Maria"*
• View & manage conditions — *"Show conditions for Maria Santos"*
• View & manage allergies — *"Add penicillin allergy for Maria"*
• View & manage medications — *"Prescribe metformin 500mg for Maria"*
• Create encounters — *"Create an encounter for Maria"*
• Patient analysis & summaries — *"Summarize Maria Santos"*

**⚙️ System**
• FHIR capability statement — *"Show FHIR metadata"*
• PDF document ingestion
• Health Gorilla sync

Just ask in natural language! I work directly with your connected OpenMRS instance."""


# ── Date parsing ──────────────────────────────────────────────────────

_MONTH_MAP: dict[str, str] = {}
for _i, _names in enumerate([
    ("january", "jan"), ("february", "feb"), ("march", "mar"),
    ("april", "apr"), ("may",), ("june", "jun"),
    ("july", "jul"), ("august", "aug"), ("september", "sep"),
    ("october", "oct"), ("november", "nov"), ("december", "dec"),
], start=1):
    for _name in _names:
        _MONTH_MAP[_name] = f"{_i:02d}"


def parse_date(text: str) -> str | None:
    """Best-effort date parsing into YYYY-MM-DD."""
    text = text.strip().rstrip(".!?,;:")

    # ISO format already
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text

    # "24 April 2000", "24 Apr 2000"
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if m:
        day, month_str, year = m.groups()
        month_num = _MONTH_MAP.get(month_str.lower())
        if month_num:
            return f"{year}-{month_num}-{int(day):02d}"

    # "April 24, 2000", "Apr 24 2000"
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2})\s*,?\s*(\d{4})", text)
    if m:
        month_str, day, year = m.groups()
        month_num = _MONTH_MAP.get(month_str.lower())
        if month_num:
            return f"{year}-{month_num}-{int(day):02d}"

    # MM/DD/YYYY or DD/MM/YYYY — assume MM/DD/YYYY (US convention)
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", text)
    if m:
        a, b, year = m.groups()
        if int(a) <= 12:
            return f"{year}-{int(a):02d}-{int(b):02d}"
        return f"{year}-{int(b):02d}-{int(a):02d}"

    return None


def _extract_date_from_text(text: str) -> tuple[str | None, str]:
    """Extract a date from mixed text. Returns (date_str, text_with_date_removed)."""
    # Match "born on DD Mon YYYY" / "birthdate: DD Mon YYYY" / "dob: ..."
    patterns = [
        r",?\s*(?:birth\s*date|dob|born(?:\s+on)?)\s*[:=]?\s*(.+?)(?:\s*$|,)",
        r",?\s*(?:birth\s*date|dob|born(?:\s+on)?)\s+(.+?)(?:\s*$|,)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            date = parse_date(m.group(1).strip())
            if date:
                cleaned = text[:m.start()] + text[m.end():]
                return date, cleaned.strip().rstrip(",. ")
    return None, text


def _parse_name(text: str) -> tuple[str, str] | None:
    """Parse given_name and family_name from a name string."""
    # Remove common prefixes like "name:", "full name:", "named"
    text = re.sub(r"^(?:full\s+)?name\s*[:=]\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^named?\s+", "", text, flags=re.I).strip()
    text = re.sub(r"^called\s+", "", text, flags=re.I).strip()
    text = text.strip().rstrip(".!?,;:")

    parts = text.split()
    if len(parts) >= 2:
        return parts[0].title(), " ".join(p.title() for p in parts[1:])
    if len(parts) == 1 and parts[0]:
        return parts[0].title(), ""
    return None


# ── Pattern definitions ───────────────────────────────────────────────

_HELP_RE = re.compile(
    r"(?:what\s+(?:can|do)\s+you|help|capabilities|"
    r"what\s+are\s+(?:your\s+)?(?:capabilities|features)|"
    r"how\s+can\s+you\s+help|"
    r"what\s+(?:features?|actions?|things?)\s+(?:do\s+you|can\s+you|are))",
    re.I,
)

_COUNT_RE = re.compile(
    r"(?:how\s+many\s+patients?|count\s+(?:all\s+)?patients?|"
    r"patient\s+count|total\s+patients?|number\s+of\s+patients?|"
    r"how\s+many\s+(?:people|records?))",
    re.I,
)

_LIST_ALL_RE = re.compile(
    r"(?:list\s+(?:all\s+)?patients?|show\s+(?:all\s+|me\s+(?:all\s+)?)?patients?|"
    r"(?:tell|show)\s+me\s+about\s+all\s+(?:the\s+)?patients?|"
    r"view\s+(?:all\s+)?patients?|get\s+(?:all\s+)?(?:the\s+)?patients?|"
    r"all\s+patients?|who\s+are\s+(?:all\s+)?(?:the\s+|my\s+)?patients?)",
    re.I,
)

_CREATE_PATIENT_RE = re.compile(
    r"(?:add|create|register|new)\s+(?:a\s+)?patient\s+(?:named?\s+|called\s+)?(.+)",
    re.I,
)

_SEARCH_RE = re.compile(
    r"(?:search|find|look\s*up)\s+(?:for\s+)?(?:patient\s+)?(.+)",
    re.I,
)

_SWITCH_RE = re.compile(
    r"(?:switch|change)\s+(?:to\s+|the\s+)?(?:patient\s+)?(?:to\s+)?(.+)",
    re.I,
)

_VIEW_INTENTS: list[tuple[str, re.Pattern[str]]] = [
    ("get_observations", re.compile(
        r"(?:show|view|get|see|check|display)\s+(?:(?:the|my|their|me)\s+)?(?:vitals?|observations?|vital\s+signs?)", re.I)),
    ("get_conditions", re.compile(
        r"(?:show|view|get|see|check|display)\s+(?:(?:the|my|their|me)\s+)?(?:conditions?|problems?|diagnos\w*|problem\s+list)", re.I)),
    ("get_allergies", re.compile(
        r"(?:show|view|get|see|check|display)\s+(?:(?:the|my|their|me)\s+)?(?:allerg\w*)", re.I)),
    ("get_medications", re.compile(
        r"(?:show|view|get|see|check|display)\s+(?:(?:the|my|their|me)\s+)?(?:medic\w*|drugs?|prescri\w*)", re.I)),
]

_ANALYSIS_RE = re.compile(
    r"(?:analy[sz]e|summarize|summary\s+(?:of|for)?|"
    r"patient\s+analysis|clinical\s+(?:summary|analysis))\s*(?:(?:of|for)\s+)?(.+)?",
    re.I,
)

_DELETE_PATIENT_RE = re.compile(
    r"(?:delete|remove|void)\s+(?:(?:the\s+)?patient\s+)?(.+)",
    re.I,
)

_METADATA_RE = re.compile(
    r"(?:fhir\s+(?:metadata|capabilities?|statement)|"
    r"show\s+metadata|capability\s+statement|server\s+metadata)",
    re.I,
)


# ── Main classifier ──────────────────────────────────────────────────

def try_deterministic_classify(
    prompt: str,
    *,
    pending_intent: str | None = None,
    collected_entities: dict[str, Any] | None = None,
) -> ConversationalDecision | None:
    """Try to classify common intents via pattern matching.

    Returns a ConversationalDecision if a match is found, or None to
    fall through to the LLM.

    Args:
        prompt: Raw user message.
        pending_intent: If a clarification slot is active, the pending intent.
        collected_entities: Already-collected entities from the clarification slot.
    """
    lower = prompt.lower().strip()

    # ── Clarification follow-up (takes priority) ──────────────────────
    if pending_intent and collected_entities:
        resolved = _try_resolve_clarification(prompt, lower, pending_intent, collected_entities)
        if resolved:
            return resolved

    # ── Capabilities / help ───────────────────────────────────────────
    if _HELP_RE.search(lower):
        return ConversationalDecision(
            mode="inform",
            response_message=CAPABILITIES_TEXT,
            scope="global",
            confidence=1.0,
        )

    # ── FHIR metadata ────────────────────────────────────────────────
    if _METADATA_RE.search(lower):
        return ConversationalDecision(
            mode="action",
            intent="get_metadata",
            scope="global",
            confidence=1.0,
            entities={},
            response_message="Fetching FHIR capability statement...",
        )

    # ── Count patients ────────────────────────────────────────────────
    if _COUNT_RE.search(lower):
        return ConversationalDecision(
            mode="action",
            intent="count_patients",
            scope="global",
            confidence=1.0,
            entities={},
            response_message="Counting patients in OpenMRS...",
        )

    # ── List all patients ─────────────────────────────────────────────
    if _LIST_ALL_RE.search(lower):
        return ConversationalDecision(
            mode="action",
            intent="search_patient",
            scope="global",
            confidence=1.0,
            entities={"patient_query": None},
            response_message="Listing all patients...",
        )

    # ── Create patient ────────────────────────────────────────────────
    m = _CREATE_PATIENT_RE.search(lower)
    if m:
        return _parse_create_patient(m.group(1).strip(), prompt)

    # ── Delete patient ────────────────────────────────────────────────
    m = _DELETE_PATIENT_RE.search(lower)
    if m:
        patient_text = m.group(1).strip()
        purge = "purge" in lower
        if purge:
            patient_text = re.sub(r"\s*purge\s*", " ", patient_text).strip()
        return ConversationalDecision(
            mode="action",
            intent="delete_patient",
            scope="global",
            confidence=0.9,
            entities={"patient_query": patient_text, "purge": purge},
            response_message=f"Looking up patient to delete...",
        )

    # ── Switch patient ────────────────────────────────────────────────
    m = _SWITCH_RE.search(lower)
    if m:
        return ConversationalDecision(
            mode="action",
            intent="switch_patient",
            scope="global",
            confidence=0.9,
            entities={"patient_query": m.group(1).strip()},
            response_message=f"Switching patient...",
        )

    # ── Search patient ────────────────────────────────────────────────
    m = _SEARCH_RE.search(lower)
    if m:
        query = m.group(1).strip()
        if query:
            return ConversationalDecision(
                mode="action",
                intent="search_patient",
                scope="global",
                confidence=0.9,
                entities={"patient_query": query},
                response_message=f"Searching for '{query}'...",
            )

    # ── View clinical data (vitals, conditions, allergies, meds) ──────
    for intent, pattern in _VIEW_INTENTS:
        m = pattern.search(lower)
        if m:
            patient_query = _extract_patient_from_tail(lower[m.end():])
            return ConversationalDecision(
                mode="action",
                intent=intent,
                scope="patient",
                confidence=0.9,
                entities={"patient_query": patient_query} if patient_query else {},
                response_message=f"Fetching {intent.replace('get_', '')}...",
            )

    # ── Patient analysis ──────────────────────────────────────────────
    m = _ANALYSIS_RE.search(lower)
    if m:
        patient_query = (m.group(1) or "").strip() if m.lastindex else None
        patient_query = patient_query or None
        return ConversationalDecision(
            mode="action",
            intent="patient_analysis",
            scope="patient",
            confidence=0.9,
            entities={"patient_query": patient_query} if patient_query else {},
            response_message="Running patient analysis...",
        )

    # No deterministic match — fall through to LLM
    return None


# ── Internal helpers ──────────────────────────────────────────────────

def _extract_patient_from_tail(text: str) -> str | None:
    """Extract a patient name from trailing text like 'for Maria Santos'."""
    m = re.search(r"(?:for|of|on)\s+(?:patient\s+)?(.+)", text.strip(), re.I)
    if m:
        name = m.group(1).strip().rstrip(".!?,;:")
        if name:
            return name
    # Check if there's just a name at the end
    text = text.strip().rstrip(".!?,;:")
    if text and not re.match(r"^(?:please|now|again|today)$", text, re.I):
        words = text.split()
        if 1 <= len(words) <= 4 and all(w[0].isupper() or w[0] == "'" for w in words if w):
            return text
    return None


def _parse_create_patient(name_text: str, original_prompt: str) -> ConversationalDecision:
    """Parse a create_patient intent with optional DOB from the matched text."""
    entities: dict[str, Any] = {}

    # Try to extract DOB from the text
    date, name_text = _extract_date_from_text(name_text)
    if date:
        entities["birthdate"] = date

    # Also look in the original prompt for DOB patterns
    if not date:
        date, _ = _extract_date_from_text(original_prompt)
        if date:
            entities["birthdate"] = date

    parsed = _parse_name(name_text)
    if parsed:
        entities["given_name"] = parsed[0]
        entities["family_name"] = parsed[1] if parsed[1] else None

    # All required fields present?
    if all(entities.get(f) for f in ("given_name", "family_name", "birthdate")):
        return ConversationalDecision(
            mode="action",
            intent="create_patient",
            scope="global",
            confidence=1.0,
            entities=entities,
            response_message=f"Registering patient {entities['given_name']} {entities['family_name']}...",
        )

    # Partial match — ask for missing fields
    missing = [f for f in ("given_name", "family_name", "birthdate") if not entities.get(f)]
    display_name = f"{entities.get('given_name', '?')} {entities.get('family_name', '')}".strip()
    if "birthdate" in missing and entities.get("given_name"):
        question = f"I have the name **{display_name}**. What is their date of birth?"
    elif "family_name" in missing and entities.get("given_name"):
        question = f"I have the first name **{entities['given_name']}**. What is their last name and date of birth?"
    else:
        question = "I need the patient's full name and birthdate. Could you provide those details?"

    return ConversationalDecision(
        mode="clarify",
        intent="create_patient",
        scope="global",
        confidence=0.95,
        entities=entities,
        missing_fields=missing,
        clarifying_question=question,
        response_message=question,
    )


def _try_resolve_clarification(
    prompt: str,
    lower: str,
    pending_intent: str,
    collected_entities: dict[str, Any],
) -> ConversationalDecision | None:
    """Try to deterministically resolve a pending clarification with the new user answer."""
    if pending_intent != "create_patient":
        return None

    entities = dict(collected_entities)

    # ── Parse "Name: X, Birthdate: Y" combined format ──────────────────
    name_m = re.search(r"(?:full\s+)?name\s*[:=]\s*([^,]+)", lower)
    if name_m:
        parsed = _parse_name(name_m.group(1).strip())
        if parsed:
            entities["given_name"] = parsed[0]
            entities["family_name"] = parsed[1] if parsed[1] else entities.get("family_name")

    dob_m = re.search(r"(?:birth\s*date|dob|born(?:\s+on)?)\s*[:=]?\s*(.+?)(?:\s*$|,)", lower)
    if dob_m:
        date = parse_date(dob_m.group(1).strip())
        if date:
            entities["birthdate"] = date

    # ── If the whole message is just a date ─────────────────────────────
    if not entities.get("birthdate"):
        date = parse_date(prompt.strip())
        if date:
            entities["birthdate"] = date

    # ── "Full name is X Y, born on Z" format ───────────────────────────
    if not entities.get("birthdate") or not entities.get("given_name"):
        m = re.search(
            r"(?:full\s+)?name\s*(?:is|:=?)\s*([^,]+?)[\s,]+(?:birth\s*date|dob|born(?:\s+on)?)\s*[:=]?\s*(.+)",
            lower,
        )
        if m:
            parsed = _parse_name(m.group(1).strip())
            if parsed:
                entities["given_name"] = parsed[0]
                entities["family_name"] = parsed[1] if parsed[1] else entities.get("family_name")
            date = parse_date(m.group(2).strip())
            if date:
                entities["birthdate"] = date

    # ── Check if all required fields now satisfied ─────────────────────
    if all(entities.get(f) for f in ("given_name", "family_name", "birthdate")):
        return ConversationalDecision(
            mode="action",
            intent="create_patient",
            scope="global",
            confidence=1.0,
            entities=entities,
            response_message=f"Registering patient {entities['given_name']} {entities['family_name']}...",
        )

    # Still missing fields — continue clarification
    missing = [f for f in ("given_name", "family_name", "birthdate") if not entities.get(f)]
    if missing:
        display_name = f"{entities.get('given_name', '?')} {entities.get('family_name', '')}".strip()
        if "birthdate" in missing and entities.get("given_name"):
            question = f"I have the name **{display_name}**. What is their date of birth?"
        else:
            question = f"I still need: {', '.join(missing)}. Could you provide those?"
        return ConversationalDecision(
            mode="clarify",
            intent="create_patient",
            scope="global",
            confidence=0.95,
            entities=entities,
            missing_fields=missing,
            clarifying_question=question,
            response_message=question,
        )

    return None


def sanitize_response_message(message: str) -> str:
    """Strip leaked session context from LLM response messages.

    Small LLMs sometimes regurgitate raw prompt context into their response.
    This detects and removes those artifacts.
    """
    # Pattern: response contains "[intent: ..." or "[USER]:" or "[ASSISTANT]:"
    if "[intent:" in message or "[USER]:" in message or "[ASSISTANT]:" in message:
        # Find the first occurrence of leaked context and truncate
        for marker in ("[intent:", "[USER]:", "[ASSISTANT]:"):
            idx = message.find(marker)
            if idx > 0:
                message = message[:idx].strip()
                break
    # Strip trailing brackets/artifacts
    message = re.sub(r"\s*\[.*$", "", message)
    return message.strip()
