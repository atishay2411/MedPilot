from __future__ import annotations

from datetime import datetime, timezone


VALID_ID_CHARS = "0123456789ACDEFGHJKLMNPRTUVWXY"


def generate_openmrs_identifier(base_number: int) -> str:
    base = str(base_number)
    total = 0
    factor = 2
    for char in reversed(base):
        code_point = VALID_ID_CHARS.index(char)
        addend = factor * code_point
        factor = 1 if factor == 2 else 2
        addend = (addend // len(VALID_ID_CHARS)) + (addend % len(VALID_ID_CHARS))
        total += addend
    remainder = total % len(VALID_ID_CHARS)
    check_code_point = (len(VALID_ID_CHARS) - remainder) % len(VALID_ID_CHARS)
    return f"{base}{VALID_ID_CHARS[check_code_point]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000+00:00")
