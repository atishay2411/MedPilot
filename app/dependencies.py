from functools import lru_cache

from app.clients.health_gorilla import HealthGorillaClient
from app.clients.openmrs import OpenMRSClient
from app.config import get_settings
from app.core.audit import AuditLogger
from app.llm.factory import build_llm_provider
from app.llm.base import LLMProvider
from app.services.allergies import AllergyService
from app.services.chat_agent import ChatAgentService
from app.services.chat_sessions import ChatSessionStore
from app.services.conditions import ConditionService
from app.services.encounters import EncounterService
from app.services.ingestion import IngestionService
from app.services.intents import IntentService
from app.services.lookups import LookupService
from app.services.medications import MedicationService
from app.services.llm_reasoning import LLMReasoningService
from app.services.observations import ObservationService
from app.services.pending_actions import PendingActionStore
from app.services.patients import PatientService
from app.services.population import PopulationService
from app.services.prompt_parser import PromptParser
from app.services.summaries import SummaryService


@lru_cache(maxsize=1)
def get_openmrs_client() -> OpenMRSClient:
    return OpenMRSClient(get_settings())


@lru_cache(maxsize=1)
def get_health_gorilla_client() -> HealthGorillaClient:
    return HealthGorillaClient(get_settings())


@lru_cache(maxsize=1)
def get_lookup_service() -> LookupService:
    return LookupService(get_openmrs_client())


@lru_cache(maxsize=1)
def get_patient_service() -> PatientService:
    return PatientService(get_openmrs_client(), get_settings())


@lru_cache(maxsize=1)
def get_encounter_service() -> EncounterService:
    return EncounterService(get_openmrs_client(), get_lookup_service())


@lru_cache(maxsize=1)
def get_observation_service() -> ObservationService:
    return ObservationService(get_openmrs_client())


@lru_cache(maxsize=1)
def get_condition_service() -> ConditionService:
    return ConditionService(get_openmrs_client(), get_lookup_service())


@lru_cache(maxsize=1)
def get_allergy_service() -> AllergyService:
    return AllergyService(get_openmrs_client(), get_lookup_service())


@lru_cache(maxsize=1)
def get_medication_service() -> MedicationService:
    return MedicationService(get_openmrs_client(), get_lookup_service())


@lru_cache(maxsize=1)
def get_summary_service() -> SummaryService:
    return SummaryService(
        get_patient_service(),
        get_observation_service(),
        get_condition_service(),
        get_allergy_service(),
        get_medication_service(),
        get_encounter_service(),
    )


@lru_cache(maxsize=1)
def get_population_service() -> PopulationService:
    return PopulationService(get_openmrs_client())


@lru_cache(maxsize=1)
def get_intent_service() -> IntentService:
    return IntentService(get_patient_service())


@lru_cache(maxsize=1)
def get_audit_logger() -> AuditLogger:
    return AuditLogger(get_settings().audit_log_path)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    return build_llm_provider(get_settings())


@lru_cache(maxsize=1)
def get_llm_reasoning_service() -> LLMReasoningService:
    return LLMReasoningService(get_llm_provider(), get_settings())


@lru_cache(maxsize=1)
def get_chat_session_store() -> ChatSessionStore:
    return ChatSessionStore(get_settings())


@lru_cache(maxsize=1)
def get_prompt_parser() -> PromptParser:
    return PromptParser()


@lru_cache(maxsize=1)
def get_pending_action_store() -> PendingActionStore:
    return PendingActionStore()


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        get_settings(),
        get_openmrs_client(),
        get_health_gorilla_client(),
        get_patient_service(),
        get_encounter_service(),
        get_observation_service(),
        get_condition_service(),
        get_allergy_service(),
        get_medication_service(),
    )


@lru_cache(maxsize=1)
def get_chat_agent_service() -> ChatAgentService:
    return ChatAgentService(
        get_prompt_parser(),
        get_llm_reasoning_service(),
        get_chat_session_store(),
        get_pending_action_store(),
        get_audit_logger(),
        get_patient_service(),
        get_summary_service(),
        get_observation_service(),
        get_condition_service(),
        get_allergy_service(),
        get_medication_service(),
        get_encounter_service(),
        get_ingestion_service(),
    )
