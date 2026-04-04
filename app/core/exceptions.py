class MedPilotError(Exception):
    """Base application exception."""


class ValidationError(MedPilotError):
    """Raised for invalid user or workflow input."""


class AuthorizationError(MedPilotError):
    """Raised when a role lacks permission."""


class ConfirmationError(MedPilotError):
    """Raised when a write action is missing confirmation."""


class ExternalServiceError(MedPilotError):
    """Raised when an upstream system fails."""


class LLMProviderError(MedPilotError):
    """Raised when a configured LLM provider fails."""
