from dataclasses import dataclass

from app.core.exceptions import ConfirmationError


@dataclass(slots=True)
class ConfirmationRequest:
    confirmed: bool = False
    destructive_confirm_text: str | None = None


def ensure_confirmation(request: ConfirmationRequest, destructive: bool = False) -> None:
    if not request.confirmed:
        raise ConfirmationError("This write action requires explicit confirmation.")
    if destructive and request.destructive_confirm_text != "DELETE":
        raise ConfirmationError("Destructive actions require typing DELETE.")
