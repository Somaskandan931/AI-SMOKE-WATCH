import re

# Loose INDIA-style plate pattern, e.g. "TN38AB1234". Used only to flag
# obviously-wrong manual entries in the UI layer -- the backend still accepts
# any non-blank string per FR-10 (user can always override OCR / type freely).
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")


def looks_like_valid_plate(value: str) -> bool:
    cleaned = re.sub(r"[^A-Z0-9]", "", value.upper())
    return bool(PLATE_REGEX.match(cleaned))
