from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.audit import AuditEvent, AuditLogger
from app.core.confirmation import ConfirmationRequest, ensure_confirmation
from app.core.exceptions import ValidationError
from app.core.security import Actor, ensure_permission
from app.models.common import ChatHistoryTurn, ChatResponseEnvelope, ChatSessionRecord, EntityResult, PendingActionRecord, WorkflowStep
from app.models.domain import ObservationInput, ObservationUpdateInput, PatientRegistration
from app.services.allergies import AllergyService
from app.services.chat_sessions import ChatSessionStore
from app.services.conditions import ConditionService
from app.services.encounters import EncounterService
from app.services.ingestion import IngestionService
from app.services.llm_reasoning import LLMReasoningService
from app.services.medications import MedicationService
from app.services.observations import ObservationService
from app.services.patients import PatientService
from app.services.pending_actions import PendingActionStore
from app.services.prompt_parser import PromptParser
from app.services.summaries import SummaryService
from app.services.utils import now_iso


class ChatAgentService:
    def __init__(
        self,
        parser: PromptParser,
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
    ):
        self.parser = parser
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
        self.sessions.append_turn(session, ChatHistoryTurn(role="user", content=prompt, patient_uuid=session.current_patient_uuid))
        deterministic = self.parser.parse(prompt, has_file=bool(attachment_path))
        parsed = self.reasoning.resolve_intent(
            prompt,
            deterministic,
            has_file=bool(attachment_path),
            session_state=session.snapshot(),
        )
        workflow = [WorkflowStep(status="completed", title="Intent parsed", detail=f"{parsed.intent} ({parsed.confidence:.0%} confidence)")]
        if self.reasoning.enabled:
            mode = "LLM-assisted" if parsed.model_dump() != deterministic.model_dump() else "deterministic intent confirmed by LLM"
            workflow.append(WorkflowStep(status="completed", title="Reasoning layer", detail=mode))

        handlers = {
            "get_metadata": self._handle_get_metadata,
            "switch_patient": self._handle_switch_patient,
            "search_patient": self._handle_search_patient,
            "patient_analysis": self._handle_patient_analysis,
            "get_observations": self._handle_get_observations,
            "get_conditions": self._handle_get_conditions,
            "get_allergies": self._handle_get_allergies,
            "get_medications": self._handle_get_medications,
            "get_medication_dispense": self._handle_get_medication_dispense,
            "create_patient": self._handle_create_patient,
            "patient_intake": self._handle_patient_intake,
            "create_encounter": self._handle_create_encounter,
            "create_observation": self._handle_create_observation,
            "update_observation": self._handle_update_observation,
            "delete_observation": self._handle_delete_observation,
            "create_condition": self._handle_create_condition,
            "update_condition": self._handle_update_condition,
            "delete_condition": self._handle_delete_condition,
            "create_allergy": self._handle_create_allergy,
            "update_allergy": self._handle_update_allergy,
            "delete_allergy": self._handle_delete_allergy,
            "create_medication": self._handle_create_medication,
            "create_medication_dispense": self._handle_create_medication_dispense,
            "update_medication": self._handle_update_medication,
            "ingest_pdf": self._handle_ingest_pdf,
            "sync_health_gorilla": self._handle_sync_health_gorilla,
        }
        handler = handlers.get(parsed.intent, self._handle_search_patient)
        effective_patient_uuid = patient_uuid or session.current_patient_uuid
        response = handler(
            prompt,
            parsed.entities,
            actor,
            workflow,
            patient_uuid=effective_patient_uuid,
            attachment_path=attachment_path,
            session=session,
        )
        response.session_id = session.id
        self._update_session_after_response(session, response, parsed.intent)
        self._log(actor, parsed.intent, response.patient_context.get("uuid") if response.patient_context else effective_patient_uuid, prompt, "preview" if response.pending_action else "completed")
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
            message=f"{record.action} completed successfully.",
            workflow=workflow,
            patient_context=response_patient_context,
            data=result,
        )
        if session:
            self._update_session_after_response(session, response, record.intent)
        return response

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

    def _handle_switch_patient(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, session: ChatSessionRecord, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:patient")
        patient = self.patients.resolve_patient(entities.get("patient_query"), None)
        workflow.append(WorkflowStep(status="completed", title="Patient switched", detail=f"Current session patient set to {patient['display']} ({patient['uuid']})."))
        return ChatResponseEnvelope(
            session_id=session.id,
            intent="switch_patient",
            message=f"I switched the active patient to {patient['display']}. Future prompts can refer to this patient.",
            workflow=workflow,
            patient_context={"uuid": patient["uuid"], "display": patient["display"], "alternatives": patient["alternatives"]},
            data={"alternatives": patient["alternatives"]},
        )

    def _handle_get_metadata(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:metadata")
        data = self.patients.client.get("/ws/fhir2/R4/metadata")
        workflow.append(WorkflowStep(status="completed", title="FHIR metadata fetched", detail="CapabilityStatement retrieved from the OpenMRS FHIR endpoint."))
        return ChatResponseEnvelope(intent="get_metadata", message="Retrieved the OpenMRS FHIR capability statement.", workflow=workflow, data=data)

    def _handle_search_patient(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:patient")
        query = entities.get("patient_query") or prompt
        search_mode = entities.get("search_mode", "default")
        results = self.patients.search(query, search_mode=search_mode)
        workflow.append(WorkflowStep(status="completed", title="Patient search executed", detail=f"{len(results)} candidate record(s) returned."))
        context = {"uuid": results[0]["uuid"], "display": results[0].get("display")} if len(results) == 1 else None
        qualifier = self._search_mode_label(search_mode)
        summary = self._format_search_summary(query, results, search_mode=search_mode)
        return ChatResponseEnvelope(
            intent="search_patient",
            message=f"I found {len(results)} patient candidate(s) for '{query}'{qualifier}.",
            workflow=workflow,
            patient_context=context,
            data=results,
            summary=summary,
        )

    @staticmethod
    def _search_mode_label(search_mode: str) -> str:
        if search_mode == "starts_with":
            return " using prefix matching"
        if search_mode == "contains":
            return " using contains matching"
        return ""

    @staticmethod
    def _format_search_summary(query: str, results: list[dict[str, Any]], *, search_mode: str = "default") -> str:
        if not results:
            if search_mode == "starts_with":
                return f"I could not find any patients whose name starts with '{query}'."
            if search_mode == "contains":
                return f"I could not find any patients whose name contains '{query}'."
            return f"I could not find any patients matching '{query}'."

        displays = [str(item.get("display", "Unknown patient")).strip() for item in results if str(item.get("display", "")).strip()]
        shown = displays[:5]
        names = ", ".join(shown)
        extra_count = max(0, len(displays) - len(shown))
        suffix = f", and {extra_count} more" if extra_count else ""

        if len(results) == 1:
            return f"I found 1 match: {shown[0]}."
        if search_mode == "starts_with":
            return f"I found {len(results)} patients whose names start with '{query}': {names}{suffix}."
        if search_mode == "contains":
            return f"I found {len(results)} patients whose names contain '{query}': {names}{suffix}."
        return f"I found {len(results)} matching patients: {names}{suffix}."

    def _handle_patient_analysis(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        brief = self.summaries.summarize_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Chart aggregated", detail="Patient demographics, conditions, allergies, medications, vitals, and encounters reviewed."))
        llm_summary = self.reasoning.render_clinical_summary(context["display"], brief, session_state=session.snapshot() if session else None)
        return ChatResponseEnvelope(
            intent="patient_analysis",
            message=f"Here is the clinician-facing analysis for {context['display']}.",
            workflow=workflow,
            patient_context=context,
            data=brief["highlights"],
            summary=llm_summary or (brief["narrative"] + " Clinical analysis: " + " ".join(brief["analysis"])),
            evidence=brief["evidence"],
        )

    def _handle_get_observations(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        display = entities.get("observation_display")
        if display:
            resource = self.observations.find_latest_by_display(context["uuid"], display)
            if not resource:
                raise ValidationError(f"No {display} observation was found for {context['display']}.")
            snapshot = self.observations.extract_observation_snapshot(resource)
            workflow.append(WorkflowStep(status="completed", title="Observation resolved", detail=f"Latest {display} reading loaded."))
            return ChatResponseEnvelope(
                intent="get_observations",
                message=f"Latest {display} for {context['display']}: {snapshot['value']} {snapshot['unit']}.",
                workflow=workflow,
                patient_context=context,
                data=snapshot,
            )
        data = self.observations.list_for_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Observation history fetched", detail="Full patient observation bundle returned."))
        return ChatResponseEnvelope(intent="get_observations", message=f"Retrieved observations for {context['display']}.", workflow=workflow, patient_context=context, data=data)

    def _handle_get_conditions(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        data = self.conditions.list_for_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Problem list fetched", detail="FHIR Condition bundle returned."))
        return ChatResponseEnvelope(intent="get_conditions", message=f"Retrieved conditions for {context['display']}.", workflow=workflow, patient_context=context, data=data)

    def _handle_get_allergies(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        data = self.allergies.list_for_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Allergy list fetched", detail="FHIR allergy bundle returned."))
        return ChatResponseEnvelope(intent="get_allergies", message=f"Retrieved allergies for {context['display']}.", workflow=workflow, patient_context=context, data=data)

    def _handle_get_medications(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        data = self.medications.list_for_patient(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Medication list fetched", detail="FHIR medication request bundle returned."))
        return ChatResponseEnvelope(intent="get_medications", message=f"Retrieved medications for {context['display']}.", workflow=workflow, patient_context=context, data=data)

    def _handle_get_medication_dispense(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "read:clinical")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        data = self.medications.medication_dispense(context["uuid"])
        workflow.append(WorkflowStep(status="completed", title="Medication dispense history fetched", detail="FHIR medication dispense bundle returned."))
        return ChatResponseEnvelope(intent="get_medication_dispense", message=f"Retrieved dispense history for {context['display']}.", workflow=workflow, patient_context=context, data=data)

    def _handle_create_patient(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:patient")
        if not all([entities.get("given_name"), entities.get("family_name"), entities.get("birthdate")]):
            raise ValidationError("Patient creation requires a patient name and birthdate in the prompt. Gender is optional if it is not stated.")
        registration = PatientRegistration(**entities)
        payload = self.patients.build_create_payload(registration)
        duplicates = self.patients.find_duplicate_candidates(registration)
        pending = self.pending_store.create(
            action_kind="write",
            intent="create_patient",
            action="Create Patient",
            permission="write:patient",
            endpoint="POST /ws/rest/v1/patient",
            payload=payload,
            prompt=prompt,
            session_id=_.get("session").id if _.get("session") else None,
            metadata={"duplicate_warnings": duplicates},
        )
        return self._preview_response(
            intent="create_patient",
            message="I prepared a patient registration payload and checked for likely duplicates.",
            workflow=workflow,
            pending=pending,
            data={"duplicate_warnings": duplicates},
            summary=f"Patient preview: {registration.given_name} {registration.family_name}, DOB {registration.birthdate}, gender {registration.gender}.",
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
        )
        patient_payload = self.patients.build_create_payload(registration)
        duplicates = self.patients.find_duplicate_candidates(registration)

        needs_encounter = bool(entities.get("medications") or entities.get("dispenses") or entities.get("encounter"))
        encounter_payload = None
        if needs_encounter:
            encounter_entities = entities.get("encounter") or {
                "encounter_type_name": "Vitals",
                "location_name": "Outpatient Clinic",
                "provider_name": "Super User",
                "encounter_role_name": "Clinician",
            }
            encounter_payload = self.encounters.build_rest_payload(
                type(
                    "EncounterPayload",
                    (),
                    {
                        "patient_uuid": "__PENDING_PATIENT_UUID__",
                        "encounter_type_name": encounter_entities["encounter_type_name"],
                        "location_name": encounter_entities["location_name"],
                        "provider_name": encounter_entities["provider_name"],
                        "encounter_role_name": encounter_entities["encounter_role_name"],
                        "encounter_datetime": now_iso(),
                    },
                )()
            )

        workflow.append(WorkflowStep(status="completed", title="Intake workflow planned", detail=self._describe_intake_entities(entities)))
        pending = self.pending_store.create(
            action_kind="workflow",
            intent="patient_intake",
            action="Create Patient Intake Workflow",
            permission="write:patient",
            endpoint="POST /patient + multi-entity clinical writes",
            payload={"patient_payload": patient_payload},
            prompt=prompt,
            session_id=_.get("session").id if _.get("session") else None,
            metadata={
                "registration": registration.model_dump(),
                "duplicate_warnings": duplicates,
                "conditions": entities.get("conditions", []),
                "allergies": entities.get("allergies", []),
                "observations": entities.get("observations", []),
                "medications": entities.get("medications", []),
                "dispenses": entities.get("dispenses", []),
                "encounter_payload": encounter_payload,
                "assumptions": ["Family name defaults to 'Unknown' when omitted."] if entities.get("family_name") == "Unknown" else [],
            },
        )
        summary = (
            f"Patient {registration.given_name} {registration.family_name}, "
            f"{len(entities.get('conditions', []))} condition(s), {len(entities.get('allergies', []))} allergy/allergies, "
            f"{len(entities.get('observations', []))} observation(s), {len(entities.get('medications', []))} medication order(s), "
            f"{len(entities.get('dispenses', []))} dispense event(s)."
        )
        return self._preview_response(
            intent="patient_intake",
            message="I translated that intake request into a sequenced patient creation workflow.",
            workflow=workflow,
            pending=pending,
            data={"duplicate_warnings": duplicates, "workflow_entities": pending.metadata},
            summary=summary,
        )

    def _handle_create_encounter(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:encounter")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        payload = self.encounters.build_rest_payload(
            type(
                "EncounterPayload",
                (),
                {
                    "patient_uuid": context["uuid"],
                    "encounter_type_name": entities["encounter_type_name"],
                    "location_name": entities["location_name"],
                    "provider_name": entities["provider_name"],
                    "encounter_role_name": entities["encounter_role_name"],
                    "encounter_datetime": now_iso(),
                },
            )()
        )
        pending = self.pending_store.create(
            action_kind="write",
            intent="create_encounter",
            action="Create Encounter",
            permission="write:encounter",
            endpoint="POST /ws/rest/v1/encounter",
            payload=payload,
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="create_encounter", message=f"I prepared a new encounter for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=f"Encounter type {entities['encounter_type_name']} at {entities['location_name']}.")

    def _handle_create_observation(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:observation")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        observations = [
            ObservationInput(
                patient_uuid=context["uuid"],
                code=item["code"],
                display=item["display"],
                value=item["value"],
                unit=item["unit"],
                effective_datetime=now_iso(),
            ).model_dump()
            for item in entities.get("observations", [])
        ]
        if not observations:
            raise ValidationError("I could not extract a valid observation value from that prompt.")
        pending = self.pending_store.create(
            action_kind="workflow" if len(observations) > 1 else "write",
            intent="create_observation",
            action="Create Observation" if len(observations) == 1 else "Create Observation Batch",
            permission="write:observation",
            endpoint="POST /ws/fhir2/R4/Observation",
            payload={"observations": observations},
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        summary = "; ".join(f"{item['display']}: {item['value']} {item['unit']}" for item in observations)
        return self._preview_response(intent="create_observation", message=f"I prepared {len(observations)} observation write(s) for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=summary)

    def _handle_update_observation(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:observation")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        observations = entities.get("observations", [])
        if len(observations) != 1:
            raise ValidationError("Observation updates currently support one vital sign per prompt.")
        target = observations[0]
        current = self.observations.find_latest_by_display(context["uuid"], target["display"])
        if not current:
            raise ValidationError(f"No existing {target['display']} observation was found for {context['display']}.")
        payload = ObservationUpdateInput(
            patient_uuid=context["uuid"],
            observation_uuid=current["id"],
            code=target["code"],
            display=target["display"],
            value=target["value"],
            unit=target["unit"],
            effective_datetime=now_iso(),
        ).model_dump()
        pending = self.pending_store.create(
            action_kind="write",
            intent="update_observation",
            action="Update Observation",
            permission="write:observation",
            endpoint=f"PUT /ws/fhir2/R4/Observation/{current['id']}",
            payload=payload,
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="update_observation", message=f"I prepared an update for the latest {target['display']} reading for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=f"{target['display']} will be updated to {target['value']} {target['unit']}.")

    def _handle_delete_observation(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "delete:observation")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        display = entities.get("observation_display")
        if not display:
            raise ValidationError("Please specify which observation or vital sign to remove.")
        current = self.observations.find_latest_by_display(context["uuid"], display)
        if not current:
            raise ValidationError(f"I could not find a {display} observation for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write",
            intent="delete_observation",
            action="Delete Observation",
            permission="delete:observation",
            endpoint=f"DELETE /ws/fhir2/R4/Observation/{current['id']}",
            payload={"observation_uuid": current["id"], "pre_delete_resource": current},
            patient_uuid=context["uuid"],
            destructive=True,
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="delete_observation", message=f"I found the latest {display} reading for {context['display']}. This delete cannot be undone.", workflow=workflow, pending=pending, patient_context=context)

    def _handle_create_condition(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:condition")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        payload = self.conditions.build_create_payload(context["uuid"], entities["condition_name"], entities["clinical_status"], entities["verification_status"], entities.get("onset_date"))
        pending = self.pending_store.create(
            action_kind="write",
            intent="create_condition",
            action="Create Condition",
            permission="write:condition",
            endpoint="POST /ws/rest/v1/condition",
            payload=payload,
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="create_condition", message=f"I prepared a new diagnosis payload for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=f"{entities['condition_name']} will be added with status {entities['clinical_status']}.")

    def _handle_update_condition(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:condition")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        condition = self.conditions.find_by_name(context["uuid"], entities["condition_name"])
        if not condition:
            raise ValidationError(f"I could not find condition '{entities['condition_name']}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write",
            intent="update_condition",
            action="Update Condition",
            permission="write:condition",
            endpoint=f"PATCH /ws/fhir2/R4/Condition/{condition['id']}",
            payload={"condition_uuid": condition["id"], "status": entities["status"]},
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="update_condition", message=f"I prepared a status update for {entities['condition_name']} for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=f"Condition status will be set to {entities['status']}.")

    def _handle_delete_condition(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "delete:condition")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        condition = self.conditions.find_by_name(context["uuid"], entities["name"])
        if not condition:
            raise ValidationError(f"I could not find condition '{entities['name']}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write",
            intent="delete_condition",
            action="Delete Condition",
            permission="delete:condition",
            endpoint=f"DELETE /ws/fhir2/R4/Condition/{condition['id']}",
            payload={"condition_uuid": condition["id"], "pre_delete_resource": condition},
            patient_uuid=context["uuid"],
            destructive=True,
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="delete_condition", message=f"I found the matching condition for {context['display']}. This delete cannot be undone.", workflow=workflow, pending=pending, patient_context=context, summary=f"Condition slated for deletion: {entities['name']}.")

    def _handle_create_allergy(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:allergy")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        payload = self.allergies.build_rest_payload(entities["allergen_name"], entities["severity"], entities["reaction"], entities.get("comment"))
        pending = self.pending_store.create(
            action_kind="write",
            intent="create_allergy",
            action="Create Allergy",
            permission="write:allergy",
            endpoint=f"POST /ws/rest/v1/patient/{context['uuid']}/allergy",
            payload=payload,
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="create_allergy", message=f"I prepared an allergy write for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=f"Allergy: {entities['allergen_name']} with {entities['severity']} severity.")

    def _handle_update_allergy(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:allergy")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        allergy = self.allergies.find_by_allergen(context["uuid"], entities["allergen_name"])
        if not allergy:
            raise ValidationError(f"I could not find allergy '{entities['allergen_name']}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write",
            intent="update_allergy",
            action="Update Allergy",
            permission="write:allergy",
            endpoint=f"PATCH /ws/fhir2/R4/AllergyIntolerance/{allergy['id']}",
            payload={"allergy_uuid": allergy["id"], "severity": entities["severity"]},
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="update_allergy", message=f"I prepared an allergy severity update for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=f"Severity will be updated to {entities['severity']}.")

    def _handle_delete_allergy(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "delete:allergy")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        allergy = self.allergies.find_by_allergen(context["uuid"], entities["name"])
        if not allergy:
            raise ValidationError(f"I could not find allergy '{entities['name']}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write",
            intent="delete_allergy",
            action="Delete Allergy",
            permission="delete:allergy",
            endpoint=f"DELETE /ws/fhir2/R4/AllergyIntolerance/{allergy['id']}",
            payload={"allergy_uuid": allergy["id"], "pre_delete_resource": allergy},
            patient_uuid=context["uuid"],
            destructive=True,
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="delete_allergy", message=f"I found the matching allergy for {context['display']}. This delete cannot be undone.", workflow=workflow, pending=pending, patient_context=context)

    def _handle_create_medication(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:medication")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        required = ["drug_name", "dose", "dose_units_name", "route_name", "frequency_name", "duration", "duration_units_name", "quantity", "quantity_units_name"]
        if any(entities.get(key) in {None, ""} for key in required):
            raise ValidationError("Medication ordering requires drug, dose, route, frequency, and duration in the prompt.")

        encounter_payload = self.encounters.build_rest_payload(
            type(
                "EncounterPayload",
                (),
                {
                    "patient_uuid": context["uuid"],
                    "encounter_type_name": entities["encounter_type_name"],
                    "location_name": entities["location_name"],
                    "provider_name": entities["provider_name"],
                    "encounter_role_name": entities["encounter_role_name"],
                    "encounter_datetime": now_iso(),
                },
            )()
        )
        medication_payload = {
            "drug_name": entities["drug_name"],
            "concept_name": entities["concept_name"],
            "dose": entities["dose"],
            "dose_units_name": entities["dose_units_name"],
            "route_name": entities["route_name"],
            "frequency_name": entities["frequency_name"],
            "duration": entities["duration"],
            "duration_units_name": entities["duration_units_name"],
            "quantity": entities["quantity"],
            "quantity_units_name": entities["quantity_units_name"],
            "care_setting_name": entities["care_setting_name"],
            "orderer_name": entities["orderer_name"],
        }
        pending = self.pending_store.create(
            action_kind="workflow",
            intent="create_medication",
            action="Create Medication Order",
            permission="write:medication",
            endpoint="Create encounter + POST /ws/rest/v1/order",
            payload=medication_payload,
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
            metadata={"encounter_payload": encounter_payload, "assumptions": ["Encounter context will be created automatically before the order is placed."]},
        )
        summary = f"{entities['drug_name']} {entities['dose']} {entities['dose_units_name']} {entities['route_name']} {entities['frequency_name']} for {entities['duration']} {entities['duration_units_name']}."
        return self._preview_response(intent="create_medication", message=f"I prepared a medication ordering workflow for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=summary)

    def _handle_create_medication_dispense(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:medication")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        if not entities.get("drug_name") or not entities.get("quantity"):
            raise ValidationError("Medication dispensing requires a drug name and quantity.")
        medication_reference = self.medications.resolve_medication_reference(context["uuid"], entities["drug_name"])
        pending = self.pending_store.create(
            action_kind="write",
            intent="create_medication_dispense",
            action="Create Medication Dispense",
            permission="write:medication",
            endpoint="POST /ws/fhir2/R4/MedicationDispense",
            payload={
                "medication_reference": medication_reference,
                "quantity": entities["quantity"],
                "unit": entities["unit"],
                "when_handed_over": entities["when_handed_over"],
                "dosage_text": entities["dosage_text"],
            },
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="create_medication_dispense", message=f"I prepared a medication dispense record for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=f"{entities['quantity']} {entities['unit']} of {entities['drug_name']} will be dispensed.")

    def _handle_update_medication(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:medication")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        medication = self.medications.find_by_name(context["uuid"], entities["drug_name"])
        if not medication:
            raise ValidationError(f"I could not find medication '{entities['drug_name']}' for {context['display']}.")
        pending = self.pending_store.create(
            action_kind="write",
            intent="update_medication",
            action="Update Medication",
            permission="write:medication",
            endpoint=f"PATCH /ws/fhir2/R4/MedicationRequest/{medication['id']}",
            payload={"medication_uuid": medication["id"], "status": entities["status"]},
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
        )
        return self._preview_response(intent="update_medication", message=f"I prepared a medication status update for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, summary=f"{entities['drug_name']} will be set to {entities['status']}.")

    def _handle_ingest_pdf(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, patient_uuid: str | None = None, attachment_path: str | None = None, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:ingestion")
        if not attachment_path:
            raise ValidationError("PDF ingestion requires an attached PDF file.")
        context = self._resolve_patient_context(entities, patient_uuid, workflow, session=session)
        parsed = self.ingestion.parse_pdf(attachment_path)
        workflow.append(WorkflowStep(status="completed", title="PDF extracted", detail="Structured sections parsed from the attached patient record."))
        pending = self.pending_store.create(
            action_kind="workflow",
            intent="ingest_pdf",
            action="Ingest Patient PDF",
            permission="write:ingestion",
            endpoint="MULTI-STEP PDF INGESTION",
            payload=parsed.model_dump(),
            patient_uuid=context["uuid"],
            prompt=prompt,
            session_id=session.id if session else None,
            metadata={"file_path": attachment_path},
        )
        summary = (
            f"Conditions: {len(parsed.conditions)}, allergies: {len(parsed.allergies)}, "
            f"medications: {len(parsed.medications)}, observations: {len(parsed.observations)}."
        )
        return self._preview_response(intent="ingest_pdf", message=f"I parsed the attached PDF and prepared the ingestion workflow for {context['display']}.", workflow=workflow, pending=pending, patient_context=context, data=parsed.model_dump(), summary=summary)

    def _handle_sync_health_gorilla(self, prompt: str, entities: dict[str, Any], actor: Actor, workflow: list[WorkflowStep], *, session: ChatSessionRecord | None = None, **_: Any) -> ChatResponseEnvelope:
        ensure_permission(actor, "write:ingestion")
        if not all([entities.get("given_name"), entities.get("family_name"), entities.get("birthdate")]):
            raise ValidationError("Health Gorilla sync requires first name, last name, and birthdate in the prompt.")
        preview = self.ingestion.health_gorilla_preview(entities["given_name"], entities["family_name"], entities["birthdate"])
        if not preview["matches"]:
            return ChatResponseEnvelope(intent="sync_health_gorilla", message="No Health Gorilla patient match was found.", workflow=workflow, data=preview)
        match_resource = preview["matches"][0]["resource"]
        pending = self.pending_store.create(
            action_kind="workflow",
            intent="sync_health_gorilla",
            action="Sync Health Gorilla Patient",
            permission="write:ingestion",
            endpoint="Health Gorilla Patient + Condition Import",
            payload={"condition_count": len(preview["conditions"])},
            prompt=prompt,
            session_id=session.id if session else None,
            metadata={"match_resource": match_resource, "conditions": preview["conditions"]},
        )
        workflow.append(WorkflowStep(status="completed", title="External chart previewed", detail=f"{len(preview['conditions'])} condition(s) prepared from Health Gorilla."))
        return self._preview_response(intent="sync_health_gorilla", message="I matched the external patient and prepared the import workflow.", workflow=workflow, pending=pending, summary=f"Patient: {self.patients.format_patient_display(match_resource)}. Conditions queued: {len(preview['conditions'])}.", data=preview)

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
            f"{len(entities.get('medications', []))} medication order(s), "
            f"{len(entities.get('dispenses', []))} dispense event(s)."
        )

    def _execute_patient_intake(self, record: PendingActionRecord) -> list[dict[str, Any]]:
        metadata = record.metadata
        registration = metadata["registration"]
        created_patient = self.patients.create(record.payload["patient_payload"])
        patient_uuid = created_patient["uuid"]
        results: list[EntityResult] = [
            EntityResult(
                entity_type="patient",
                name=f"{registration['given_name']} {registration['family_name']}",
                outcome="success",
                detail=patient_uuid,
            )
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
                payload = self.conditions.build_create_payload(
                    patient_uuid,
                    condition["condition_name"],
                    condition.get("clinical_status", "active"),
                    condition.get("verification_status", "confirmed"),
                    condition.get("onset_date"),
                )
                self.conditions.create(payload)
                results.append(EntityResult(entity_type="condition", name=condition["condition_name"], outcome="success", detail="Created"))
            except Exception as exc:
                results.append(EntityResult(entity_type="condition", name=condition["condition_name"], outcome="failed", detail=str(exc)))

        for allergy in metadata.get("allergies", []):
            try:
                payload = self.allergies.build_rest_payload(allergy["allergen_name"], allergy["severity"], allergy["reaction"], allergy.get("comment"))
                self.allergies.create(patient_uuid, payload)
                results.append(EntityResult(entity_type="allergy", name=allergy["allergen_name"], outcome="success", detail="Created"))
            except Exception as exc:
                results.append(EntityResult(entity_type="allergy", name=allergy["allergen_name"], outcome="failed", detail=str(exc)))

        for observation in metadata.get("observations", []):
            try:
                payload = ObservationInput(
                    patient_uuid=patient_uuid,
                    code=observation["code"],
                    display=observation["display"],
                    value=observation["value"],
                    unit=observation["unit"],
                    effective_datetime=now_iso(),
                )
                self.observations.create(self.observations.build_fhir_payload(payload))
                results.append(EntityResult(entity_type="observation", name=observation["display"], outcome="success", detail="Created"))
            except Exception as exc:
                results.append(EntityResult(entity_type="observation", name=observation["display"], outcome="failed", detail=str(exc)))

        for medication in metadata.get("medications", []):
            try:
                if not encounter_uuid:
                    raise ValidationError("Medication orders require encounter context.")
                self.medications.create(self.medications.build_create_payload(patient_uuid, encounter_uuid, medication))
                results.append(EntityResult(entity_type="medication", name=medication["drug_name"], outcome="success", detail="Created"))
            except Exception as exc:
                results.append(EntityResult(entity_type="medication", name=medication["drug_name"], outcome="failed", detail=str(exc)))

        for dispense in metadata.get("dispenses", []):
            try:
                medication_reference = self.medications.resolve_medication_reference(patient_uuid, dispense["drug_name"])
                self.medications.create_dispense(
                    patient_uuid,
                    medication_reference,
                    dispense["quantity"],
                    dispense["unit"],
                    dispense["when_handed_over"],
                    dispense["dosage_text"],
                )
                results.append(EntityResult(entity_type="medication_dispense", name=dispense["drug_name"], outcome="success", detail="Created"))
            except Exception as exc:
                results.append(EntityResult(entity_type="medication_dispense", name=dispense["drug_name"], outcome="failed", detail=str(exc)))

        return [item.model_dump() for item in results]

    def _execute_observation_action(self, record: PendingActionRecord) -> list[dict[str, Any]] | dict[str, Any]:
        payloads = record.payload.get("observations")
        if payloads:
            return [self.observations.create(self.observations.build_fhir_payload(ObservationInput.model_validate(payload))) for payload in payloads]
        return self.observations.update(ObservationUpdateInput.model_validate(record.payload))

    def _update_session_after_response(self, session: ChatSessionRecord, response: ChatResponseEnvelope, intent: str) -> None:
        if response.patient_context:
            self.sessions.set_current_patient(session, response.patient_context.get("uuid"), response.patient_context.get("display"))
        self.sessions.set_last_intent(session, intent)
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
                user_id=actor.user_id,
                role=actor.role,
                intent=intent,
                action="chat_agent",
                patient_uuid=patient_uuid,
                prompt=prompt,
                endpoint=f"AGENT::{intent}",
                request_payload={"prompt": prompt, "patient_uuid": patient_uuid},
                response_status=200,
                outcome=outcome,
                metadata={},
            )
        )
