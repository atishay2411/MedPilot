from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.core.audit import AuditEvent, AuditLogger
from app.core.confirmation import ConfirmationRequest, ensure_confirmation
from app.core.exceptions import ExternalServiceError, ValidationError
from app.core.security import Actor, ensure_permission
from app.models.common import (
    ChatHistoryTurn,
    ChatResponseEnvelope,
    ChatSessionRecord,
    EntityResult,
    PendingActionRecord,
    PendingClarificationSlot,
    PendingWorkflowState,
    WorkflowStep,
)
from app.models.domain import EncounterInput, ObservationInput, ObservationUpdateInput, PatientRegistration, PatientUpdateInput
from app.services.allergies import AllergyService
from app.services.capabilities import extract_entities, get_capability, handler_map, is_global_intent
from app.services.chat_sessions import ChatSessionStore
from app.services.conditions import ConditionService
from app.services.encounters import EncounterService
from app.services.ingestion import IngestionService
from app.services.llm_reasoning import ConversationalDecision, LLMReasoningService
from app.services.medications import MedicationService
from app.services.observations import ObservationService
from app.services.patients import PatientService
from app.services.pending_actions import PendingActionStore
from app.services.population import PopulationService
from app.services.summaries import SummaryService
from app.services.utils import now_iso

# ---------------------------------------------------------------------------
# Module-level compiled regexes for deterministic patient-creation enrichment
# ---------------------------------------------------------------------------

_NAME_CREATION_PATTERNS = (
    re.compile(
        r"(?:enter|add|register|create|admit|intake)\s+(?:a\s+)?patient\s+(?:name[d]?\s+)?"
        r"([A-Z][a-zA-Z'\-]*)\s+([A-Z][a-zA-Z'\-]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpatient\s+(?:name[d]?\s+)?([A-Z][a-zA-Z'\-]*)\s+([A-Z][a-zA-Z'\-]*)",
        re.IGNORECASE,
    ),
)

_MONTH_MAP: dict[str, str] = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

_MONTH_NAMES = "|".join(_MONTH_MAP)

_BIRTHDATE_DMY = re.compile(
    rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAMES})[,\s]+(\d{{4}})",
    re.IGNORECASE,
)
_BIRTHDATE_MDY = re.compile(
    rf"({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})",
    re.IGNORECASE,
)
_BIRTHDATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

_CONDITION_PATTERN = re.compile(
    r"(?:has|have|suffering\s+from|diagnosed\s+with|presenting\s+with|history\s+of)\s+"
    r"([A-Za-z][A-Za-z0-9\s\-]*)",
    re.IGNORECASE,
)

_COUNTRY_BORN_PATTERN = re.compile(
    r"born\s+in\s+([A-Z][a-zA-Z\s]+?)(?:[,.]|\s+and|\s+he|\s+she|\s+who|$)",
    re.IGNORECASE,
)

# Phrases that indicate patient creation even when the LLM misses the intent
_PATIENT_CREATION_PHRASES = (
    "patient name", "register patient", "add patient", "create patient",
    "enter patient", "admit patient", "new patient", "intake patient",
)


class ChatAgentService:
    def __init__(
        self,
        reasoning: LLMReasoningService,
        sessions: ChatSessionStore,
        pending_store: PendingActionStore,
        audit: AuditLogger,
        patients: PatientService,
        summaries: SummaryService,
        observations: ObservationService,
        conditions: ConditionService,
        allergies: AllergyService,
        medications: MedicationService,
        encounters: EncounterService,
        ingestion: IngestionService,
        population: PopulationService,
    ):
        self.reasoning = reasoning
        self.sessions = sessions
        self.pending_store = pending_store
        self.audit = audit
        self.patients = patients
        self.summaries = summaries
        self.observations = observations
        self.conditions = conditions
        self.allergies = allergies
        self.medications = medications
        self.encounters = encounters
        self.ingestion = ingestion
        self.population = population

    _HANDLER_MAP = handler_map()

    def handle_message(
        self,
        prompt: str,
        actor: Actor,
        *,
        session_id: str | None = None,
        patient_uuid: str | None = None,
        attachment_path: str | None = None,
    ) -> ChatResponseEnvelope:
        session = self.sessions.get_or_create(session_id)
        self.sessions.append_turn(
            session,
            ChatHistoryTurn(role="user", content=prompt, patient_uuid=session.current_patient_uuid),
        )

        effective_patient_uuid = patient_uuid or session.current_patient_uuid

        # ── Pass 1: primary LLM decision ──────────────────────────────
        decision = self.reasoning.generate_conversational_response(
            prompt,
            session_state=session.snapshot(),
            has_file=bool(attachment_path),
        )

        # ── Pass 2 (conditional): run fallback only when intent is missing/unsupported ──
        # Replaces the old 3-pass repair → operational_resolver chain.
        if decision.mode != "inform" and not self._is_supported_intent(decision.intent):
            decision = self.reasoning.run_fallback_decision(
                prompt,
                decision,
                session_state=session.snapshot(),
                has_file=bool(attachment_path),
            )

        # ── Pass 3 (deterministic): entity enrichment for patient-creation intents ──
        # Handles natural-language phrasing the LLM frequently misses:
        #   "Enter a patient name Sahil Singh"  →  given_name/family_name
        #   "born at 12th December 2000"        →  birthdate (YYYY-MM-DD)
        #   "He is born in India"               →  country
        #   "he has stage-2 asthma"             →  conditions list
        # Also upgrades intent create_patient → patient_intake when conditions are
        # present, and upgrades mode clarify → action when all required fields are met.
        if decision.intent in ("create_patient", "patient_intake") or (
            decision.mode in ("action", "clarify")
            and self._looks_like_patient_creation(prompt)
        ):
            decision = self._enrich_patient_creation_decision(prompt, decision)

        # ── Clarification resolution (when a slot is in-flight) ───────
        has_pending = bool(session.pending_workflow or session.pending_clarification)
        if has_pending:
            decision = self.reasoning.resolve_clarification_answer(
                prompt,
                decision,
                session_state=session.snapshot(),
            )

        # ── Scope-based stale context clearance ───────────────────────
        # If the user started a fresh global query (not a follow-up) and
        # no pending slot is in-flight, clear any carry-over state so
        # earlier bad turns don't bias this new request.
        is_global = decision.scope == "global" or is_global_intent(decision.intent or "")
        if is_global and not has_pending and decision.mode == "action":
            self.sessions.clear_stale_context(session)

        workflow: list[WorkflowStep] = [
            WorkflowStep(
                status="completed",
                title="Conversational AI",
                detail=f"mode={decision.mode}, intent={decision.intent or 'none'}, scope={decision.scope} ({decision.confidence:.0%})",
            )
        ]

        # ── CLARIFY: assistant needs more info ─────────────────────────
        if decision.mode == "clarify":
            response = self._handle_clarify(decision, session, workflow)
            response.session_id = session.id
            response.scope = decision.scope
            self._update_session_after_response(
                session, response, decision.intent or "clarify",
                clarification_slot=self._build_clarification_slot(prompt, session, decision),
                pending_workflow=self._build_pending_workflow_state(prompt, session, decision),
                is_global=is_global,
            )
            self._log(actor, decision.intent or "clarify", effective_patient_uuid, prompt, "clarify")
            return response

        # ── INFORM: pure information, no system action ─────────────────
        if decision.mode == "inform":
            response = self._handle_inform(decision, session, workflow)
            response.session_id = session.id
            response.scope = "global" if is_global else "patient"
            self._update_session_after_response(session, response, "inform", pending_workflow=None, is_global=True)
            self._log(actor, "inform", effective_patient_uuid, prompt, "inform")
            return response

        # ── ACTION: dispatch to the appropriate domain handler ─────────
        intent = decision.intent or "inform"
        capability = get_capability(intent)
        if not capability:
            # Unknown intent from LLM — treat as inform
            response = ChatResponseEnvelope(
                session_id=session.id,
                intent="inform",
                message=decision.response_message or f"I received an unsupported intent '{intent}'. Could you rephrase?",
                workflow=workflow,
                scope="global",
            )
            self._update_session_after_response(session, response, "inform", pending_workflow=None, is_global=True)
            return response

        # Narrow entities to only the fields this capability knows about
        entities = extract_entities(decision.entities or {}, capability)

        # For global-scoped capabilities, do not inherit the active patient uuid
        resolved_patient_uuid = None if is_global_intent(intent) else effective_patient_uuid

        handler_name = self._HANDLER_MAP.get(intent)
        handler = getattr(self, handler_name)
        try:
            response = handler(
                prompt,
                entities,
                actor,
                workflow,
                patient_uuid=resolved_patient_uuid,
                attachment_path=attachment_path,
                session=session,
                llm_message=decision.response_message,
            )
        except ValidationError as exc:
            error_msg = str(exc)
            response = ChatResponseEnvelope(
                session_id=session.id,
                intent=intent,
                message=f"I ran into an issue: {error_msg}",
                workflow=workflow,
            )
            self._update_session_after_response(session, response, intent, pending_workflow=None, is_global=is_global)
            self._log(actor, intent, effective_patient_uuid, prompt, "error:validation")
            return response
        except ExternalServiceError as exc:
            workflow.append(WorkflowStep(status="blocked", title="OpenMRS unreachable", detail=str(exc)))
            response = ChatResponseEnvelope(
                session_id=session.id,
                intent=intent,
                message="The OpenMRS server returned an error while processing your request. Please try again.",
                workflow=workflow,
            )
            self._update_session_after_response(session, response, intent, pending_workflow=None, is_global=is_global)
            self._log(actor, intent, effective_patient_uuid, prompt, f"error:ExternalServiceError")
            return response
        except Exception as exc:
            workflow.append(WorkflowStep(status="blocked", title="Execution failed", detail=f"{type(exc).__name__}: unexpected error."))
            response = ChatResponseEnvelope(
                session_id=session.id,
                intent=intent,
                message="I ran into an unexpected application error while trying to complete that request.",
                workflow=workflow,
            )
            self._update_session_after_response(session, response, intent, pending_workflow=None, is_global=is_global)
            self._log(actor, intent, effective_patient_uuid, prompt, f"error:{type(exc).__name__}")
            return response

        response.session_id = session.id
        response.scope = decision.scope
        self._update_session_after_response(
            session, response, intent, pending_workflow=None,
            is_global=is_global,
        )
        self._log(
            actor, intent,
            response.patient_context.get("uuid") if response.patient_context else effective_patient_uuid,
            prompt,
            "preview" if response.pending_action else "completed",
        )
        return response


    def confirm_action(self, action_id: str, actor: Actor, *, destructive_confirm_text: str | None = None) -> ChatResponseEnvelope:
        record = self.pending_store.consume(action_id)
        ensure_permission(actor, record.permission)
        ensure_confirmation(ConfirmationRequest(confirmed=True, destructive_confirm_text=destructive_confirm_text), destructive=record.destructive)
        session = self.sessions.get(record.session_id) if record.session_id else None

        workflow = [
            WorkflowStep(status="completed", title="Pending action loaded", detail=record.action),
            WorkflowStep(status="completed", title="Confirmation accepted", detail="Write safety requirements satisfied."),
        ]
        result: Any
        response_patient_context = {"uuid": record.patient_uuid} if record.patient_uuid else None

        if record.intent == "create_patient":
            result = self.patients.create(record.payload)
            names = (((record.payload.get("person") or {}).get("names")) or [{}])[0]
            response_patient_context = {
                "uuid": result.get("uuid"),
                "display": " ".join(part for part in [names.get("givenName"), names.get("familyName")] if part).strip() or result.get("display"),
            }
        elif record.intent == "patient_intake":
            self._ensure_intake_permissions(actor, record.metadata)
            result = self._execute_patient_intake(record)
            created_patient_uuid = result[0]["detail"] if result else None
            registration = record.metadata.get("registration", {})
            response_patient_context = {
                "uuid": created_patient_uuid,
                "display": f"{registration.get('given_name', '')} {registration.get('family_name', '')}".strip(),
            }
        elif record.intent == "create_encounter":
            result = self.encounters.create_rest(record.payload)
        elif record.intent == "create_condition":
            result = self.conditions.create(record.payload)
        elif record.intent == "update_condition":
            result = self.conditions.patch_status(record.payload["condition_uuid"], record.payload["status"])
        elif record.intent == "delete_condition":
            result = self.conditions.delete(record.payload["condition_uuid"])
        elif record.intent == "create_allergy":
            result = self.allergies.create(record.patient_uuid, record.payload)
        elif record.intent == "update_allergy":
            result = self.allergies.patch_severity(record.payload["allergy_uuid"], record.payload["severity"])
        elif record.intent == "delete_allergy":
            result = self.allergies.delete(record.payload["allergy_uuid"])
        elif record.intent in {"create_observation", "update_observation"}:
            result = self._execute_observation_action(record)
        elif record.intent == "delete_observation":
            result = self.observations.delete(record.payload["observation_uuid"])
        elif record.intent == "update_medication":
            result = self.medications.patch_status(record.payload["medication_uuid"], record.payload["status"])
        elif record.intent == "create_medication":
            encounter = self.encounters.create_rest(record.metadata["encounter_payload"])
            result = {
                "encounter": encounter,
                "order": self.medications.create(self.medications.build_create_payload(record.patient_uuid, encounter["uuid"], record.payload)),
            }
        elif record.intent == "create_medication_dispense":
            result = self.medications.create_dispense(
                record.patient_uuid,
                record.payload["medication_reference"],
                record.payload["quantity"],
                record.payload["unit"],
                record.payload["when_handed_over"],
                record.payload["dosage_text"],
            )
        elif record.intent == "delete_patient":
            result = self.patients.delete(record.patient_uuid or record.payload["patient_uuid"], purge=bool(record.metadata.get("purge")))
            if session and session.current_patient_uuid == record.patient_uuid:
                response_patient_context = {"uuid": None, "display": None}
        elif record.intent == "update_patient":
            result = self.patients.update(record.patient_uuid or record.payload.get("patient_uuid", ""), record.payload)
            response_patient_context = {
                "uuid": record.patient_uuid,
                "display": record.metadata.get("patient_display"),
            }
        elif record.intent == "ingest_pdf":
            result = [item.model_dump() for item in self.ingestion.ingest_pdf(record.patient_uuid, record.metadata["file_path"])]
            Path(record.metadata["file_path"]).unlink(missing_ok=True)
        elif record.intent == "sync_health_gorilla":
            result = [item.model_dump() for item in self.ingestion.sync_health_gorilla(record.metadata["match_resource"], record.metadata["conditions"])]
        else:
            raise ValidationError(f"Unsupported confirmation workflow for '{record.intent}'.")


        workflow.append(WorkflowStep(status="completed", title="Workflow executed", detail=record.endpoint))
        self._log(actor, record.intent, record.patient_uuid, record.prompt, "executed")
        response = ChatResponseEnvelope(
            session_id=record.session_id,
            intent=record.intent,
            message=f"✅ {record.action} completed successfully.",
            workflow=workflow,
            patient_context=response_patient_context,
            data=result,
        )
        if session:
            self._update_session_after_response(session, response, record.intent, pending_workflow=None)
        return response

    # ── Helpers ────────────────────────────────────────────────────────

    def _resolve_patient_context(
        self,
        entities: dict[str, Any],
        explicit_patient_uuid: str | None,
        workflow: list[WorkflowStep],
        session: ChatSessionRecord | None = None,
    ) -> dict[str, Any]:
        fallback_patient_uuid = explicit_patient_uuid or (session.current_patient_uuid if session else None)
        patient = self.patients.resolve_patient(entities.get("patient_query"), fallback_patient_uuid)
        workflow.append(WorkflowStep(status="completed", title="Patient resolved", detail=f"{patient['display']} ({patient['uuid']})"))
        context = {"uuid": patient["uuid"], "display": patient["display"], "alternatives": patient["alternatives"]}
        return context

    def _preview_response(self, *, intent: str, message: str, workflow: list[WorkflowStep], pending: PendingActionRecord, patient_context: dict[str, Any] | None = None, data: Any = None, summary: str | None = None) -> ChatResponseEnvelope:
        workflow.append(WorkflowStep(status="requires_confirmation", title="Awaiting confirmation", detail="The assistant prepared the exact payload and is waiting for explicit approval."))
        return ChatResponseEnvelope(
            session_id=pending.session_id,
            intent=intent,
            message=message,
            workflow=workflow,
            patient_context=patient_context,
            data=data,
            summary=summary,
            pending_action={
                "id": pending.id,
                "action": pending.action,
                "endpoint": pending.endpoint,
                "destructive": pending.destructive,
                "payload": pending.payload,
                "metadata": pending.metadata,
            },
        )

    @staticmethod
    def _is_supported_intent(intent: str | None) -> bool:
        return bool(intent and get_capability(intent))

    @staticmethod
    def _looks_like_patient_creation(prompt: str) -> bool:
        """Return True when the prompt is clearly about registering a new patient."""
        lower = prompt.lower()
        return any(phrase in lower for phrase in _PATIENT_CREATION_PHRASES)

    @staticmethod
    def _enrich_patient_creation_decision(
        prompt: str,
        decision: "ConversationalDecision",
    ) -> "ConversationalDecision":
        """Deterministic enricher for patient create/intake decisions.

        Fills entity gaps left by the LLM when conversational phrasing is used,
        then upgrades intent (create_patient → patient_intake) and mode
        (clarify → action) when all required registration fields are satisfied.
        Does NOT overwrite values already correctly extracted by the LLM.
        """
        from app.services.llm_reasoning import ConversationalDecision  # local import avoids circular

        entities: dict[str, Any] = dict(decision.entities or {})
        changed = False

        # ── 1. Given / family name ────────────────────────────────────────
        if not (entities.get("given_name") and entities.get("family_name")):
            for pat in _NAME_CREATION_PATTERNS:
                m = pat.search(prompt)
                if m:
                    if not entities.get("given_name"):
                        entities["given_name"] = m.group(1).strip().title()
                        changed = True
                    if not entities.get("family_name"):
                        entities["family_name"] = m.group(2).strip().title()
                        changed = True
                    break

        # ── 2. Birthdate ──────────────────────────────────────────────────
        if not entities.get("birthdate"):
            m = _BIRTHDATE_DMY.search(prompt)
            if m:
                day, mon, year = m.group(1), m.group(2).lower(), m.group(3)
                entities["birthdate"] = f"{year}-{_MONTH_MAP.get(mon, '01')}-{day.zfill(2)}"
                changed = True
            else:
                m = _BIRTHDATE_MDY.search(prompt)
                if m:
                    mon, day, year = m.group(1).lower(), m.group(2), m.group(3)
                    entities["birthdate"] = f"{year}-{_MONTH_MAP.get(mon, '01')}-{day.zfill(2)}"
                    changed = True
                else:
                    m = _BIRTHDATE_ISO.search(prompt)
                    if m:
                        entities["birthdate"] = m.group(0)
                        changed = True

        # ── 3. Gender from pronouns ───────────────────────────────────────
        if not entities.get("gender"):
            if re.search(r'\bhe\b|\bhis\b|\bhim\b', prompt, re.IGNORECASE):
                entities["gender"] = "M"
            elif re.search(r'\bshe\b|\bher\b', prompt, re.IGNORECASE):
                entities["gender"] = "F"
            else:
                entities["gender"] = "U"
            changed = True

        # ── 4. Country from "born in <Country>" ───────────────────────────
        if not entities.get("country"):
            m = _COUNTRY_BORN_PATTERN.search(prompt)
            if m:
                entities["country"] = m.group(1).strip().title()
                changed = True

        # ── 5. Conditions from "has/diagnosed with X" ─────────────────────
        if not entities.get("conditions"):
            conditions: list[dict[str, Any]] = []
            seen: set[str] = set()
            for m in _CONDITION_PATTERN.finditer(prompt):
                raw = m.group(1).strip()
                # Stop at sentence boundaries and trailing conjunctions
                raw = re.split(r'[.!?]', raw)[0]
                # Stop at 'born' keyword (e.g. "Type-2 Diabetes born in India" → "Type-2 Diabetes")
                raw = re.sub(r'\s+born\b.*$', '', raw, flags=re.IGNORECASE)
                raw = re.sub(r'\s+(and|or|but|while|who|that)\b.*$', '', raw, flags=re.IGNORECASE).strip().rstrip(".,;:")
                key = raw.lower()
                # Skip single-word false positives that aren't clinical terms
                skip = {"no", "not", "none", "never", "a", "an", "the", "born", "india", "pakistan"}
                if raw and len(raw) > 2 and key not in seen and key not in skip:
                    conditions.append({
                        "condition_name": raw,
                        "clinical_status": "active",
                        "verification_status": "confirmed",
                    })
                    seen.add(key)
            if conditions:
                entities["conditions"] = conditions
                changed = True

        if not changed:
            return decision

        # ── 6. Upgrade intent: create_patient → patient_intake when conditions present
        intent = decision.intent or "patient_intake"
        if intent == "create_patient" and entities.get("conditions"):
            intent = "patient_intake"
        # Ensure a valid creation intent is set
        if intent not in ("create_patient", "patient_intake"):
            intent = "patient_intake" if entities.get("conditions") else "create_patient"

        # ── 7. Upgrade mode: clarify → action when required fields are now present
        mode = decision.mode
        if mode == "clarify" and all([
            entities.get("given_name"),
            entities.get("family_name"),
            entities.get("birthdate"),
        ]):
            mode = "action"

        return ConversationalDecision(
            mode=mode,
            intent=intent,
            write=True,
            confidence=decision.confidence,
            scope="global",
            entities=entities,
            missing_fields=[] if mode == "action" else list(decision.missing_fields or []),
            clarifying_question=None if mode == "action" else decision.clarifying_question,
            response_message=decision.response_message,
        )

    @staticmethod
    def _build_pending_workflow_state(
        prompt: str,
        session: ChatSessionRecord,
        decision: ConversationalDecision,
    ) -> PendingWorkflowState | None:
        if decision.mode != "clarify":
            return None
        return PendingWorkflowState(
            intent=decision.intent,
            original_prompt=prompt,
            collected_entities=decision.entities or {},
            missing_fields=list(decision.missing_fields or []),
            clarifying_question=decision.clarifying_question or decision.response_message,
            patient_uuid=session.current_patient_uuid,
            patient_display=session.current_patient_display,
        )

    @staticmethod
    def _build_clarification_slot(
        prompt: str,
        session: ChatSessionRecord,
        decision: ConversationalDecision,
    ) -> PendingClarificationSlot | None:
        """Build a structured clarification slot from a clarify-mode decision.

        Returns None when the decision is not in clarify mode.
        The turn_count increments from the existing slot so we can cap loops.
        """
        if decision.mode != "clarify":
            return None
        existing_turn = (
            session.pending_clarification.turn_count
            if session.pending_clarification
            else 0
        )
        return PendingClarificationSlot(
            question=decision.clarifying_question or decision.response_message,
            intent=decision.intent,
            collected_entities=decision.entities or {},
            missing_fields=list(decision.missing_fields or []),
            patient_uuid=session.current_patient_uuid,
            patient_display=session.current_patient_display,
            turn_count=existing_turn + 1,
        )

    @staticmethod
    def _build_encounter_input(patient_uuid: str, entities: dict[str, Any]) -> EncounterInput:
        return EncounterInput(
            patient_uuid=patient_uuid,
            encounter_type_name=entities.get("encounter_type_name", "Vitals"),
            location_name=entities.get("location_name", "Outpatient Clinic"),
            provider_name=entities.get("provider_name", "Super User"),
            encounter_role_name=entities.get("encounter_role_name", "Clinician"),
            encounter_datetime=now_iso(),
        )

    def _handle_clarify(self, decision: ConversationalDecision, session: ChatSessionRecord, workflow: list[WorkflowStep]) -> ChatResponseEnvelope:
        """Return a clarifying question to the user without executing any action."""
        workflow.append(WorkflowStep(
            status="planned",
            title="Awaiting your answer",
            detail="The assistant needs more information before proceeding.",
        ))
        return ChatResponseEnvelope(
            session_id=session.id,
            intent="clarify",
            message=decision.clarifying_question or decision.response_message,
            workflow=workflow,
            patient_context={"uuid": session.current_patient_uuid, "display": session.current_patient_display} if session.current_patient_uuid else None,
        )

    def _handle_inform(self, decision: ConversationalDecision, session: ChatSessionRecord, workflow: list[WorkflowStep]) -> ChatResponseEnvelope:
        """Return an informational response (no action taken)."""
        workflow.append(WorkflowStep(
            status="completed",
            title="Informational response",
            detail="No system action required.",
        ))
        return ChatResponseEnvelope(
            session_id=session.id,
            intent="inform",
            message=decision.response_message,
            workflow=workflow,
            patient_context={"uuid": session.current_patient_uuid, "display": session.current_patient_display} if session.current_patient_uuid else None,
        )

    # ── Domain handlers ───────────────────────────────────────────────

    def _handle_switch_patient(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, session: ChatSessionRecord, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:patient")
        patient = self.patients.resolve_patient(entities.get("patient_query"), None)
        workflow.append(WorkflowStep(status="completed", title="Patient switched", detail=f"Active patient set to {patient['display']}."))
        # switch_patient is the only intent that intentionally mutates the active
        # session patient; the update is applied via _update_session_after_response
        # because we explicitly pass patient_context with the new UUID.
        return ChatResponseEnvelope(
            session_id=session.id,
            intent="switch_patient",
            message=f"Switched active patient to **{patient['display']}**. Future queries will refer to this patient.",
            workflow=workflow,
            patient_context={"uuid": patient["uuid"], "display": patient["display"], "alternatives": patient["alternatives"]},
            data={"alternatives": patient["alternatives"]},
        )

    def _handle_get_metadata(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:metadata")
        data = self.patients.client.get("/ws/fhir2/R4/metadata")
        workflow.append(WorkflowStep(status="completed", title="FHIR metadata fetched", detail="CapabilityStatement retrieved."))
        return ChatResponseEnvelope(intent="get_metadata", message="Here is the OpenMRS FHIR capability statement.", workflow=workflow, data=data)

    def _handle_search_patient(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:patient")
        query = entities.get("patient_query") or prompt
        search_mode = entities.get("search_mode", "default")
        results = self.patients.search(query, search_mode=search_mode)
        workflow.append(WorkflowStep(status="completed", title="Patient search", detail=f"{len(results)} record(s) found."))
        # Do NOT set patient_context for search results — resolving a list of
        # patients should not silently switch the session's active patient.
        # Only switch_patient does that deliberately.
        summary = self._format_search_summary(query, results, search_mode=search_mode)
        return ChatResponseEnvelope(
            intent="search_patient",
            message=summary,
            workflow=workflow,
            patient_context=None,
            data=results,
            summary=summary,
        )

    def _handle_count_patients(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:population")
        query = (entities.get("patient_query") or "").strip()
        search_mode = entities.get("search_mode", "default")
        if query:
            results = self.patients.search(query, search_mode=search_mode)
            count = len(results)
            if search_mode == "starts_with":
                message = f"I found {count} patients whose names start with '{query}'."
            elif search_mode == "contains":
                message = f"I found {count} patients whose names contain '{query}'."
            else:
                message = f"I found {count} patients matching '{query}'."
            data: Any = {"label": "patients", "count": count, "matches": results}
        else:
            data = self.population.count_patients()
            count = int(data.get("count", 0))
            message = f"There are {count} patients available in the connected OpenMRS search scope."
        workflow.append(WorkflowStep(status="completed", title="Population query executed", detail=f"{count} patient record(s) counted."))
        return ChatResponseEnvelope(intent="count_patients", message=message, workflow=workflow, data=data, summary=message)

    @staticmethod
    def _format_search_summary(query: str, results: list[dict[str, Any]], *, search_mode: str = "default") -> str:
        if not results:
            return f"No patients found matching '{query}'."
        displays = [str(item.get("display", "Unknown")).strip() for item in results if str(item.get("display", "")).strip()]
        shown = displays[:5]
        names = ", ".join(shown)
        extra = max(0, len(displays) - len(shown))
        suffix = f", and {extra} more" if extra else ""
        if len(results) == 1:
            return f"Found 1 match: **{shown[0]}**."
        mode_desc = ""
        if search_mode == "starts_with":
            mode_desc = f" whose names start with '{query}'"
        elif search_mode == "contains":
            mode_desc = f" whose names contain '{query}'"
        return f"Found {len(results)} patients{mode_desc}: {names}{suffix}."

    def _handle_patient_analysis(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        brief = self.summaries.summarize_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Chart aggregated", detail="Demographics, conditions, allergies, medications, vitals reviewed."))
        llm_summary = self.reasoning.render_clinical_summary(context["display"], brief, session_state=session.snapshot() if session else None)
        return ChatResponseEnvelope(
            intent="patient_analysis",
            message=f"Clinical analysis for **{context['display']}**:",
            workflow=workflow,
            patient_context=context,
            data=brief["highlights"],
            summary=llm_summary or (brief["narrative"] + " " + " ".join(brief["analysis"])),
            evidence=brief["evidence"],
        )

    def _handle_get_observations(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        display = entities.get("observation_display")
        if display:
            resource = self.observations.find_latest_by_display(context["uuid"], display)
            if not resource:
                raise ValidationError(f"No {display} observation found for {context['display']}.")
            snapshot = self.observations.extract_observation_snapshot(resource)
            workflow.append(WorkflowStep(status="completed", title="Observation resolved", detail=f"Latest {display} loaded."))
            return ChatResponseEnvelope(
                intent="get_observations",
                message=f"Latest **{display}** for {context['display']}: **{snapshot['value']} {snapshot['unit']}**.",
                workflow=workflow,
                patient_context=context,
                data=snapshot,
            )
        data = self.observations.list_for_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Observations fetched", detail="Full observation history returned."))
        return ChatResponseEnvelope(intent="get_observations", message=f"Observations for **{context['display']}**:", workflow=workflow, patient_context=context, data=data)

    def _handle_get_conditions(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        data = self.conditions.list_for_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Conditions fetched", detail="Problem list returned."))
        return ChatResponseEnvelope(intent="get_conditions", message=f"Conditions for **{context['display']}**:", workflow=workflow, patient_context=context, data=data)

    def _handle_get_allergies(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        data = self.allergies.list_for_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Allergies fetched", detail="Allergy list returned."))
        return ChatResponseEnvelope(intent="get_allergies", message=f"Allergies for **{context['display']}**:", workflow=workflow, patient_context=context, data=data)

    def _handle_get_medications(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        data = self.medications.list_for_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Medications fetched", detail="Medication list returned."))
        return ChatResponseEnvelope(intent="get_medications", message=f"Medications for **{context['display']}**:", workflow=workflow, patient_context=context, data=data)

    def _handle_get_medication_dispense(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        data = self.medications.medication_dispense(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Dispense history fetched", detail="Medication dispense records returned."))
        return ChatResponseEnvelope(intent="get_medication_dispense", message=f"Dispense history for **{context['display']}**:", workflow=workflow, patient_context=context, data=data)

    def _handle_create_patient(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:patient")
        if not all([entities.get("given_name"), entities.get("family_name"), entities.get("birthdate")]):
            raise ValidationError("I need the patient's full name and birthdate to register them. Could you provide those details?")
        registration = PatientRegistration(
            given_name=entities["given_name"],
            family_name=entities["family_name"],
            gender=entities.get("gender", "U"),
            birthdate=entities["birthdate"],
            city_village=entities.get("city_village"),
            country=entities.get("country"),
        )
        payload = self.patients.build_create_payload(registration)
        duplicates = self.patients.find_duplicate_candidates(registration)
        session = _.get("session")
        pending = self.pending_store.create(
            action_kind="write",
            intent="create_patient",
            action="Create Patient",
            permission="write:patient",
            endpoint="POST /ws/rest/v1/patient",
            payload=payload,
            prompt=prompt,
            session_id=session.id if session else None,
            metadata={"duplicate_warnings": duplicates},
        )
        return self._preview_response(
            intent="create_patient",
            message=f"I've prepared to register **{registration.given_name} {registration.family_name}** (DOB: {registration.birthdate}, Gender: {registration.gender}).",
            workflow=workflow,
            pending=pending,
            data={"duplicate_warnings": duplicates},
            summary=f"Patient: {registration.given_name} {registration.family_name}, DOB {registration.birthdate}, gender {registration.gender}.",
        )

    def _handle_delete_patient(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "delete:patient")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        purge = bool(entities.get("purge"))
        endpoint = f"DELETE /ws/rest/v1/patient/{context['uuid']}" + ("?purge=true" if purge else "")
        pending = self.pending_store.create(
            action_kind="write",
            intent="delete_patient",
            action="Delete Patient",
            permission="delete:patient",
            endpoint=endpoint,
            payload={"patient_uuid": context["uuid"]},
            patient_uuid=context["uuid"],
            destructive=True,
            prompt=prompt,
            session_id=session.id if session else None,
            metadata={"purge": purge, "patient_display": context["display"]},
        )
        summary = "Hard purge requested." if purge else "Standard OpenMRS delete/void requested."
        return self._preview_response(
            intent="delete_patient",
            message=f"Found **{context['display']}**. This patient delete is destructive and requires confirmation.",
            workflow=workflow,
            pending=pending,
            patient_context=context,
            summary=summary,
        )

    def _handle_update_patient(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:patient")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        update = PatientUpdateInput(
            patient_uuid=context["uuid"],
            given_name=entities.get("given_name"),
            family_name=entities.get("family_name"),
            gender=entities.get("gender"),
            birthdate=entities.get("birthdate"),
            city_village=entities.get("city_village"),
        )
        payload = self.patients.build_update_payload(update)
        if not payload or not payload.get("person"):
            raise ValidationError("No demographic fields were specified to update. Please provide at least one field to change.")
        changed = []
        if entities.get("given_name"):
            changed.append(f"given name → {entities['given_name']}")
        if entities.get("family_name"):
            changed.append(f"family name → {entities['family_name']}")
        if entities.get("gender"):
            changed.append(f"gender → {entities['gender']}")
        if entities.get("birthdate"):
            changed.append(f"birthdate → {entities['birthdate']}")
        if entities.get("city_village"):
            changed.append(f"city → {entities['city_village']}")
        pending = self.pending_store.create(
            action_kind="write",
            intent="update_patient",
            action="Update Patient Demographics",
            permission="write:patient",
            endpoint=f"POST /ws/rest/v1/patient/{context['uuid']}",
            payload=payload,
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
            metadata={"patient_display": context["display"]},
        )
        summary = "; ".join(changed) if changed else "Demographics update."
        return self._preview_response(
            intent="update_patient",
            message=f"Ready to update **{context['display']}**: {summary}.",
            workflow=workflow,
            pending=pending,
            patient_context=context,
            summary=summary,
        )

    def _handle_search_by_identifier(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:patient")
        identifier = (entities.get("identifier") or "").strip()
        if not identifier:
            raise ValidationError("Please provide a patient identifier or UUID to search by.")
        results = self.patients.search_by_identifier(identifier)
        workflow.append(WorkflowStep(status="completed", title="Identifier lookup", detail=f"{len(results)} record(s) found."))
        context = {"uuid": results[0]["uuid"], "display": results[0].get("display")} if len(results) == 1 else None
        if not results:
            return ChatResponseEnvelope(
                intent="search_by_identifier",
                message=f"No patient found with identifier **{identifier}**.",
                workflow=workflow,
            )
        if len(results) == 1:
            return ChatResponseEnvelope(
                intent="search_by_identifier",
                message=f"Found patient **{results[0].get('display')}** with identifier `{identifier}`.",
                workflow=workflow,
                patient_context=context,
                data=results,
            )
        return ChatResponseEnvelope(
            intent="search_by_identifier",
            message=f"Found {len(results)} patients matching identifier `{identifier}`.",
            workflow=workflow,
            data=results,
        )

    def _handle_patient_intake(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:

        self._ensure_intake_permissions(actor, entities)
        if not all([entities.get("given_name"), entities.get("family_name"), entities.get("birthdate")]):
            raise ValidationError("Patient intake requires at least patient name and birthdate.")

        registration = PatientRegistration(
            given_name=entities["given_name"],
            family_name=entities["family_name"],
            gender=entities.get("gender", "U"),
            birthdate=entities["birthdate"],
            city_village=entities.get("city_village"),
            country=entities.get("country"),
        )
        patient_payload = self.patients.build_create_payload(registration)
        duplicates = self.patients.find_duplicate_candidates(registration)

        needs_encounter = bool(entities.get("medications") or entities.get("dispenses") or entities.get("encounter"))
        encounter_payload = None
        if needs_encounter:
            encounter_entities = entities.get("encounter") or {
                "encounter_type_name": "Vitals", "location_name": "Outpatient Clinic",
                "provider_name": "Super User", "encounter_role_name": "Clinician",
            }
            encounter_payload = self.encounters.build_rest_payload(
                EncounterInput(
                    patient_uuid="__PENDING_PATIENT_UUID__",
                    encounter_type_name=encounter_entities["encounter_type_name"],
                    location_name=encounter_entities["location_name"],
                    provider_name=encounter_entities["provider_name"],
                    encounter_role_name=encounter_entities["encounter_role_name"],
                    encounter_datetime=now_iso(),
                )
            )

        workflow.append(WorkflowStep(status="completed", title="Intake planned", detail=self._describe_intake_entities(entities)))
        session = _.get("session")

        # Clinical entities extracted from the prompt (may be empty lists)
        clinical_data = {
            "conditions": entities.get("conditions", []),
            "allergies": entities.get("allergies", []),
            "observations": entities.get("observations", []),
            "medications": entities.get("medications", []),
            "dispenses": entities.get("dispenses", []),
        }

        pending = self.pending_store.create(
            action_kind="workflow",
            intent="patient_intake",
            action="Create Patient Intake Workflow",
            permission="write:patient",
            endpoint="POST /patient + multi-entity clinical writes",
            # payload is what the user sees in the preview; include the full picture
            payload={
                "patient_payload": patient_payload,
                "clinical_data": clinical_data,
            },
            prompt=prompt,
            session_id=session.id if session else None,
            metadata={
                "registration": registration.model_dump(),
                "duplicate_warnings": duplicates,
                **clinical_data,
                "encounter_payload": encounter_payload,
            },
        )
        summary = (
            f"Patient **{registration.given_name} {registration.family_name}**, "
            f"{len(entities.get('conditions', []))} condition(s), {len(entities.get('allergies', []))} allergy/allergies, "
            f"{len(entities.get('observations', []))} observation(s), {len(entities.get('medications', []))} medication(s)."
        )
        return self._preview_response(
            intent="patient_intake",
            message="I've prepared a full intake workflow for this patient.",
            workflow=workflow, pending=pending,
            data={"duplicate_warnings": duplicates, "workflow_entities": pending.metadata},
            summary=summary,
        )

    def _handle_create_encounter(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:encounter")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        payload = self.encounters.build_rest_payload(self._build_encounter_input(context["uuid"], entities))
        pending = self.pending_store.create(
            action_kind="write", intent="create_encounter", action="Create Encounter",
            permission="write:encounter", endpoint="POST /ws/rest/v1/encounter",
            payload=payload, patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="create_encounter",
            message=f"Prepared a new encounter for **{context['display']}**.",
            workflow=workflow, pending=pending, patient_context=context,
            summary=f"Encounter at {entities.get('location_name', 'Outpatient Clinic')}.",
        )

    def _handle_create_observation(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:observation")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        observations = [
            ObservationInput(
                patient_uuid=context["uuid"],
                code=item["code"], display=item["display"],
                value=item["value"], unit=item["unit"],
                effective_datetime=now_iso(),
            ).model_dump()
            for item in entities.get("observations", [])
        ]
        if not observations:
            raise ValidationError("I couldn't extract a valid observation value. Please specify the vital sign and value.")
        pending = self.pending_store.create(
            action_kind="workflow" if len(observations) > 1 else "write",
            intent="create_observation",
            action="Create Observation" if len(observations) == 1 else "Create Observation Batch",
            permission="write:observation", endpoint="POST /ws/fhir2/R4/Observation",
            payload={"observations": observations}, patient_uuid=context["uuid"],
            prompt=prompt, session_id=session.id if session else None,
        )
        summary = "; ".join(f"**{item['display']}**: {item['value']} {item['unit']}" for item in observations)
        return self._preview_response(
            intent="create_observation",
            message=f"Prepared {len(observations)} observation(s) for **{context['display']}**.",
            workflow=workflow, pending=pending, patient_context=context, summary=summary,
        )

    def _handle_update_observation(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:observation")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        observations = entities.get("observations", [])
        if len(observations) != 1:
            raise ValidationError("Observation updates currently support one vital sign per request.")
        target = observations[0]
        current = self.observations.find_latest_by_display(context["uuid"], target["display"])
        if not current:
            raise ValidationError(f"No existing {target['display']} observation found for {context['display']}.")
        payload = ObservationUpdateInput(
            patient_uuid=context["uuid"], observation_uuid=current["id"],
            code=target["code"], display=target["display"],
            value=target["value"], unit=target["unit"], effective_datetime=now_iso(),
        ).model_dump()
        pending = self.pending_store.create(
            action_kind="write", intent="update_observation", action="Update Observation",
            permission="write:observation", endpoint=f"PUT /ws/fhir2/R4/Observation/{current['id']}",
            payload=payload, patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="update_observation",
            message=f"Ready to update **{target['display']}** for {context['display']}.",
            workflow=workflow, pending=pending, patient_context=context,
            summary=f"{target['display']} → {target['value']} {target['unit']}.",
        )

    def _handle_delete_observation(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "delete:observation")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        display = entities.get("observation_display")
        if not display:
            raise ValidationError("Please specify which observation or vital sign to remove.")
        current = self.observations.find_latest_by_display(context["uuid"], display)
        if not current:
            raise ValidationError(f"No {display} observation found for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write", intent="delete_observation", action="Delete Observation",
            permission="delete:observation", endpoint=f"DELETE /ws/fhir2/R4/Observation/{current['id']}",
            payload={"observation_uuid": current["id"], "pre_delete_resource": current},
            patient_uuid=context["uuid"], destructive=True, prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="delete_observation",
            message=f"Found the latest **{display}** reading for {context['display']}. ⚠ This delete cannot be undone.",
            workflow=workflow, pending=pending, patient_context=context,
        )

    def _handle_create_condition(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:condition")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        payload = self.conditions.build_create_payload(
            context["uuid"],
            entities.get("condition_name", ""),
            entities.get("clinical_status", "active"),
            entities.get("verification_status", "confirmed"),
            entities.get("onset_date"),
        )
        pending = self.pending_store.create(
            action_kind="write", intent="create_condition", action="Create Condition",
            permission="write:condition", endpoint="POST /ws/rest/v1/condition",
            payload=payload, patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="create_condition",
            message=f"Prepared to add **{entities.get('condition_name')}** for {context['display']}.",
            workflow=workflow, pending=pending, patient_context=context,
            summary=f"{entities.get('condition_name')} — status: {entities.get('clinical_status', 'active')}.",
        )

    def _handle_update_condition(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:condition")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        condition = self.conditions.find_by_name(context["uuid"], entities.get("condition_name", ""))
        if not condition:
            raise ValidationError(f"Could not find condition '{entities.get('condition_name')}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write", intent="update_condition", action="Update Condition",
            permission="write:condition", endpoint=f"PATCH /ws/fhir2/R4/Condition/{condition['id']}",
            payload={"condition_uuid": condition["id"], "status": entities.get("status", "inactive")},
            patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="update_condition",
            message=f"Ready to update **{entities.get('condition_name')}** status for {context['display']}.",
            workflow=workflow, pending=pending, patient_context=context,
            summary=f"Status → {entities.get('status', 'inactive')}.",
        )

    def _handle_delete_condition(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "delete:condition")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        condition = self.conditions.find_by_name(context["uuid"], entities.get("name", ""))
        if not condition:
            raise ValidationError(f"Could not find condition '{entities.get('name')}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write", intent="delete_condition", action="Delete Condition",
            permission="delete:condition", endpoint=f"DELETE /ws/fhir2/R4/Condition/{condition['id']}",
            payload={"condition_uuid": condition["id"], "pre_delete_resource": condition},
            patient_uuid=context["uuid"], destructive=True, prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="delete_condition",
            message=f"Found **{entities.get('name')}** for {context['display']}. ⚠ This delete cannot be undone.",
            workflow=workflow, pending=pending, patient_context=context,
        )

    def _handle_create_allergy(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:allergy")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        payload = self.allergies.build_rest_payload(
            entities.get("allergen_name", ""), entities.get("severity", "moderate"),
            entities.get("reaction", "rash"), entities.get("comment"),
        )
        pending = self.pending_store.create(
            action_kind="write", intent="create_allergy", action="Create Allergy",
            permission="write:allergy", endpoint=f"POST /ws/rest/v1/patient/{context['uuid']}/allergy",
            payload=payload, patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="create_allergy",
            message=f"Prepared allergy for **{context['display']}**: {entities.get('allergen_name')} ({entities.get('severity', 'moderate')}).",
            workflow=workflow, pending=pending, patient_context=context,
        )

    def _handle_update_allergy(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:allergy")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        allergy = self.allergies.find_by_allergen(context["uuid"], entities.get("allergen_name", ""))
        if not allergy:
            raise ValidationError(f"Could not find allergy '{entities.get('allergen_name')}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write", intent="update_allergy", action="Update Allergy",
            permission="write:allergy", endpoint=f"PATCH /ws/fhir2/R4/AllergyIntolerance/{allergy['id']}",
            payload={"allergy_uuid": allergy["id"], "severity": entities.get("severity", "moderate")},
            patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="update_allergy",
            message=f"Ready to update **{entities.get('allergen_name')}** severity for {context['display']}.",
            workflow=workflow, pending=pending, patient_context=context,
        )

    def _handle_delete_allergy(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "delete:allergy")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        allergy = self.allergies.find_by_allergen(context["uuid"], entities.get("name", ""))
        if not allergy:
            raise ValidationError(f"Could not find allergy '{entities.get('name')}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write", intent="delete_allergy", action="Delete Allergy",
            permission="delete:allergy", endpoint=f"DELETE /ws/fhir2/R4/AllergyIntolerance/{allergy['id']}",
            payload={"allergy_uuid": allergy["id"], "pre_delete_resource": allergy},
            patient_uuid=context["uuid"], destructive=True, prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="delete_allergy",
            message=f"Found **{entities.get('name')}** allergy for {context['display']}. ⚠ This delete cannot be undone.",
            workflow=workflow, pending=pending, patient_context=context,
        )

    def _handle_create_medication(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:medication")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        required = ["drug_name", "dose", "dose_units_name", "route_name", "frequency_name", "duration", "duration_units_name", "quantity", "quantity_units_name"]
        if any(entities.get(key) in {None, ""} for key in required):
            raise ValidationError("Medication ordering requires drug, dose, route, frequency, and duration.")
        encounter_payload = self.encounters.build_rest_payload(self._build_encounter_input(context["uuid"], entities))
        medication_payload = {k: entities[k] for k in ["drug_name", "concept_name", "dose", "dose_units_name", "route_name", "frequency_name", "duration", "duration_units_name", "quantity", "quantity_units_name", "care_setting_name", "orderer_name"] if k in entities}
        pending = self.pending_store.create(
            action_kind="workflow", intent="create_medication", action="Create Medication Order",
            permission="write:medication", endpoint="Create encounter + POST /ws/rest/v1/order",
            payload=medication_payload, patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
            metadata={"encounter_payload": encounter_payload},
        )
        summary = f"**{entities['drug_name']}** {entities['dose']} {entities['dose_units_name']} {entities['route_name']} {entities['frequency_name']} for {entities['duration']} {entities['duration_units_name']}."
        return self._preview_response(
            intent="create_medication",
            message=f"Prepared medication order for **{context['display']}**.",
            workflow=workflow, pending=pending, patient_context=context, summary=summary,
        )

    def _handle_create_medication_dispense(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:medication")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        if not entities.get("drug_name") or not entities.get("quantity"):
            raise ValidationError("Medication dispensing requires a drug name and quantity.")
        medication_reference = self.medications.resolve_medication_reference(context["uuid"], entities["drug_name"])
        pending = self.pending_store.create(
            action_kind="write", intent="create_medication_dispense", action="Create Medication Dispense",
            permission="write:medication", endpoint="POST /ws/fhir2/R4/MedicationDispense",
            payload={
                "medication_reference": medication_reference,
                "quantity": entities["quantity"], "unit": entities.get("unit", "Tablet"),
                "when_handed_over": entities.get("when_handed_over", now_iso()),
                "dosage_text": entities.get("dosage_text", f"Dispensed {entities['quantity']} {entities.get('unit', 'tablet')}(s) of {entities['drug_name']}."),
            },
            patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="create_medication_dispense",
            message=f"Prepared dispense: {entities['quantity']} {entities.get('unit', 'tablet')}(s) of **{entities['drug_name']}** for {context['display']}.",
            workflow=workflow, pending=pending, patient_context=context,
        )

    def _handle_update_medication(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:medication")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        medication = self.medications.find_by_name(context["uuid"], entities.get("drug_name", ""))
        if not medication:
            raise ValidationError(f"Could not find medication '{entities.get('drug_name')}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write", intent="update_medication", action="Update Medication",
            permission="write:medication", endpoint=f"PATCH /ws/fhir2/R4/MedicationRequest/{medication['id']}",
            payload={"medication_uuid": medication["id"], "status": entities.get("status", "stopped")},
            patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(
            intent="update_medication",
            message=f"Ready to set **{entities.get('drug_name')}** to {entities.get('status', 'stopped')} for {context['display']}.",
            workflow=workflow, pending=pending, patient_context=context,
        )

    def _handle_ingest_pdf(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, attachment_path: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:ingestion")
        if not attachment_path:
            raise ValidationError("PDF ingestion requires an attached PDF file.")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        parsed = self.ingestion.parse_pdf(attachment_path)
        workflow.append(WorkflowStep(status="completed", title="PDF extracted", detail="Structured data parsed from the attached document."))
        pending = self.pending_store.create(
            action_kind="workflow", intent="ingest_pdf", action="Ingest Patient PDF",
            permission="write:ingestion", endpoint="MULTI-STEP PDF INGESTION",
            payload=parsed.model_dump(), patient_uuid=context["uuid"], prompt=prompt,
            session_id=session.id if session else None,
            metadata={"file_path": attachment_path},
        )
        summary = f"Conditions: {len(parsed.conditions)}, allergies: {len(parsed.allergies)}, medications: {len(parsed.medications)}, observations: {len(parsed.observations)}."
        return self._preview_response(
            intent="ingest_pdf",
            message=f"Parsed the attached PDF for **{context['display']}**.",
            workflow=workflow, pending=pending, patient_context=context,
            data=parsed.model_dump(), summary=summary,
        )

    def _handle_sync_health_gorilla(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:ingestion")
        if not all([entities.get("given_name"), entities.get("family_name"), entities.get("birthdate")]):
            raise ValidationError("Health Gorilla sync requires first name, last name, and birthdate.")
        preview = self.ingestion.health_gorilla_preview(entities["given_name"], entities["family_name"], entities["birthdate"])
        if not preview["matches"]:
            return ChatResponseEnvelope(intent="sync_health_gorilla", message="No Health Gorilla patient match was found.", workflow=workflow, data=preview)
        match_resource = preview["matches"][0]["resource"]
        pending = self.pending_store.create(
            action_kind="workflow", intent="sync_health_gorilla", action="Sync Health Gorilla Patient",
            permission="write:ingestion", endpoint="Health Gorilla Patient + Condition Import",
            payload={"condition_count": len(preview["conditions"])}, prompt=prompt,
            session_id=session.id if session else None,
            metadata={"match_resource": match_resource, "conditions": preview["conditions"]},
        )
        workflow.append(WorkflowStep(status="completed", title="External chart previewed", detail=f"{len(preview['conditions'])} condition(s) prepared."))
        return self._preview_response(
            intent="sync_health_gorilla",
            message="Matched external patient and prepared the import workflow.",
            workflow=workflow, pending=pending,
            summary=f"Patient: {self.patients.format_patient_display(match_resource)}. Conditions: {len(preview['conditions'])}.",
            data=preview,
        )

    # ── Internal support ──────────────────────────────────────────────

    def _ensure_intake_permissions(self, actor: Actor, entities: dict[str, Any]) -> None:
        ensure_permission(actor, "write:patient")
        if entities.get("conditions"):
            ensure_permission(actor, "write:condition")
        if entities.get("allergies"):
            ensure_permission(actor, "write:allergy")
        if entities.get("observations"):
            ensure_permission(actor, "write:observation")
        if entities.get("medications") or entities.get("dispenses"):
            ensure_permission(actor, "write:medication")
            ensure_permission(actor, "write:encounter")

    @staticmethod
    def _describe_intake_entities(entities: dict[str, Any]) -> str:
        return (
            f"{len(entities.get('conditions', []))} condition(s), "
            f"{len(entities.get('allergies', []))} allergy/allergies, "
            f"{len(entities.get('observations', []))} observation(s), "
            f"{len(entities.get('medications', []))} medication(s), "
            f"{len(entities.get('dispenses', []))} dispense(s)."
        )

    def _execute_patient_intake(self, record: PendingActionRecord) -> list[dict[str, Any]]:
        metadata = record.metadata
        registration = metadata["registration"]
        created_patient = self.patients.create(record.payload["patient_payload"])
        patient_uuid = created_patient["uuid"]
        results: list[EntityResult] = [
            EntityResult(entity_type="patient", name=f"{registration['given_name']} {registration['family_name']}", outcome="success", detail=patient_uuid)
        ]

        encounter_uuid: str | None = None
        encounter_payload = metadata.get("encounter_payload")
        if encounter_payload:
            payload = {**encounter_payload, "patient": patient_uuid}
            encounter = self.encounters.create_rest(payload)
            encounter_uuid = encounter["uuid"]
            results.append(EntityResult(entity_type="encounter", name="Encounter", outcome="success", detail=encounter_uuid))

        for condition in metadata.get("conditions", []):
            try:
                payload = self.conditions.build_create_payload(patient_uuid, condition["condition_name"], condition.get("clinical_status", "active"), condition.get("verification_status", "confirmed"), condition.get("onset_date"))
                self.conditions.create(payload)
                results.append(EntityResult(entity_type="condition", name=condition["condition_name"], outcome="success", detail="Created"))
            except ValidationError as exc:
                results.append(EntityResult(entity_type="condition", name=condition["condition_name"], outcome="failed", detail=f"ValidationError: {exc}"))
            except ExternalServiceError as exc:
                results.append(EntityResult(entity_type="condition", name=condition["condition_name"], outcome="failed", detail=f"ExternalServiceError: {exc}"))
            except Exception as exc:
                results.append(EntityResult(entity_type="condition", name=condition["condition_name"], outcome="failed", detail=f"{type(exc).__name__}: {exc}"))

        for allergy in metadata.get("allergies", []):
            try:
                payload = self.allergies.build_rest_payload(allergy["allergen_name"], allergy["severity"], allergy["reaction"], allergy.get("comment"))
                self.allergies.create(patient_uuid, payload)
                results.append(EntityResult(entity_type="allergy", name=allergy["allergen_name"], outcome="success", detail="Created"))
            except ValidationError as exc:
                results.append(EntityResult(entity_type="allergy", name=allergy["allergen_name"], outcome="failed", detail=f"ValidationError: {exc}"))
            except ExternalServiceError as exc:
                results.append(EntityResult(entity_type="allergy", name=allergy["allergen_name"], outcome="failed", detail=f"ExternalServiceError: {exc}"))
            except Exception as exc:
                results.append(EntityResult(entity_type="allergy", name=allergy["allergen_name"], outcome="failed", detail=f"{type(exc).__name__}: {exc}"))

        for observation in metadata.get("observations", []):
            try:
                obs_input = ObservationInput(patient_uuid=patient_uuid, code=observation["code"], display=observation["display"], value=observation["value"], unit=observation["unit"], effective_datetime=now_iso())
                self.observations.create(self.observations.build_fhir_payload(obs_input))
                results.append(EntityResult(entity_type="observation", name=observation["display"], outcome="success", detail="Created"))
            except ValidationError as exc:
                results.append(EntityResult(entity_type="observation", name=observation["display"], outcome="failed", detail=f"ValidationError: {exc}"))
            except ExternalServiceError as exc:
                results.append(EntityResult(entity_type="observation", name=observation["display"], outcome="failed", detail=f"ExternalServiceError: {exc}"))
            except Exception as exc:
                results.append(EntityResult(entity_type="observation", name=observation["display"], outcome="failed", detail=f"{type(exc).__name__}: {exc}"))

        for medication in metadata.get("medications", []):
            try:
                if not encounter_uuid:
                    raise ValidationError("Medication orders require encounter context.")
                self.medications.create(self.medications.build_create_payload(patient_uuid, encounter_uuid, medication))
                results.append(EntityResult(entity_type="medication", name=medication["drug_name"], outcome="success", detail="Created"))
            except ValidationError as exc:
                results.append(EntityResult(entity_type="medication", name=medication["drug_name"], outcome="failed", detail=f"ValidationError: {exc}"))
            except ExternalServiceError as exc:
                results.append(EntityResult(entity_type="medication", name=medication["drug_name"], outcome="failed", detail=f"ExternalServiceError: {exc}"))
            except Exception as exc:
                results.append(EntityResult(entity_type="medication", name=medication["drug_name"], outcome="failed", detail=f"{type(exc).__name__}: {exc}"))


        for dispense in metadata.get("dispenses", []):
            try:
                medication_reference = self.medications.resolve_medication_reference(patient_uuid, dispense["drug_name"])
                self.medications.create_dispense(patient_uuid, medication_reference, dispense["quantity"], dispense["unit"], dispense["when_handed_over"], dispense["dosage_text"])
                results.append(EntityResult(entity_type="medication_dispense", name=dispense["drug_name"], outcome="success", detail="Created"))
            except ValidationError as exc:
                results.append(EntityResult(entity_type="medication_dispense", name=dispense["drug_name"], outcome="failed", detail=f"ValidationError: {exc}"))
            except ExternalServiceError as exc:
                results.append(EntityResult(entity_type="medication_dispense", name=dispense["drug_name"], outcome="failed", detail=f"ExternalServiceError: {exc}"))
            except Exception as exc:
                results.append(EntityResult(entity_type="medication_dispense", name=dispense["drug_name"], outcome="failed", detail=f"{type(exc).__name__}: {exc}"))


        return [item.model_dump() for item in results]

    def _execute_observation_action(self, record: PendingActionRecord) -> list[dict[str, Any]] | dict[str, Any]:
        payloads = record.payload.get("observations")
        if payloads:
            return [self.observations.create(self.observations.build_fhir_payload(ObservationInput.model_validate(payload))) for payload in payloads]
        return self.observations.update(ObservationUpdateInput.model_validate(record.payload))

    def _update_session_after_response(
        self,
        session: ChatSessionRecord,
        response: ChatResponseEnvelope,
        intent: str,
        *,
        clarification_slot: PendingClarificationSlot | None = None,
        pending_workflow: PendingWorkflowState | None = None,
        is_global: bool = False,
    ) -> None:
        # Patient-context update policy:
        # 1. Global queries (is_global=True) → NEVER update the active patient.
        #    A count_patients or search_patient listing must not silently
        #    overwrite who the clinician is currently looking at.
        # 2. switch_patient / create_patient / patient_intake → Always update
        #    (these intents exist specifically to change the active patient).
        # 3. All other patient-scoped handlers → Update only when the session
        #    had NO prior active patient (first-use auto-set after the first
        #    chart query). If a patient was already active, the UUID is the
        #    same so no write is needed.
        _patient_switching_intents = {"switch_patient", "create_patient", "patient_intake"}

        if not is_global and response.patient_context:
            new_uuid = response.patient_context.get("uuid")
            new_display = response.patient_context.get("display")
            if intent in _patient_switching_intents:
                self.sessions.set_current_patient(session, new_uuid, new_display)
            elif not session.current_patient_uuid and new_uuid:
                # First chart interaction — auto-set the patient for follow-ups
                self.sessions.set_current_patient(session, new_uuid, new_display)
        # Always persist last intent and slot state
        self.sessions.set_last_intent(session, intent)
        self.sessions.set_pending_clarification(session, clarification_slot)
        self.sessions.set_pending_workflow(session, pending_workflow)
        self.sessions.append_turn(
            session,
            ChatHistoryTurn(
                role="assistant",
                content=response.summary or response.message,
                intent=intent,
                patient_uuid=response.patient_context.get("uuid") if response.patient_context else session.current_patient_uuid,
            ),
        )
        refreshed = self.sessions.get(session.id)
        response.session_id = refreshed.id
        response.session_state = refreshed.snapshot()

    def _log(self, actor: Actor, intent: str, patient_uuid: str | None, prompt: str | None, outcome: str) -> None:
        self.audit.log(
            AuditEvent(
                user_id=actor.user_id, role=actor.role, intent=intent,
                action="chat_agent", patient_uuid=patient_uuid, prompt=prompt,
                endpoint=f"AGENT::{intent}", request_payload={"prompt": prompt, "patient_uuid": patient_uuid},
                response_status=200, outcome=outcome, metadata={},
            )
        )
