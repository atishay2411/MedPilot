from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "MedPilot"
    medpilot_env: str = "development"
    medpilot_host: str = "127.0.0.1"
    medpilot_port: int = 8000
    medpilot_secret_key: str = "change-me"
    medpilot_user_id: str = "demo-clinician"
    medpilot_user_role: str = "clinician"
    actions_require_confirmation: bool = True

    openmrs_base_url: str = "http://localhost:8080/openmrs"
    openmrs_username: str = "admin"
    openmrs_password: str = "Admin123"
    openmrs_identifier_type_uuid: str = "05a29f94-c0ed-11e2-94be-8c13b969e334"
    openmrs_location_uuid: str = "8d6c993e-c2cc-11de-8d13-0010c6dffd0f"

    health_gorilla_base_url: str = "https://sandbox.healthgorilla.com/fhir"
    health_gorilla_token: str | None = None
    health_gorilla_max_conditions: int = 20

    medpilot_llm_provider: str = "none"
    medpilot_llm_model: str | None = None
    medpilot_llm_timeout_seconds: float = 30.0
    medpilot_llm_reasoning_effort: str = "medium"
    medpilot_llm_max_output_tokens: int = 2000
    medpilot_llm_enable_intent_reasoning: bool = True
    medpilot_llm_enable_summary_reasoning: bool = True

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434/api"

    request_timeout_seconds: float = 10.0
    max_retries: int = 3
    audit_log_path: Path = Field(default=Path("data/audit/audit.log"))
    chat_sessions_path: Path = Field(default=Path("data/chat/sessions"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chat_sessions_path.mkdir(parents=True, exist_ok=True)
    return settings
