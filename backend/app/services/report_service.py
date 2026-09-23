"""
Report generation (PRD section 10 / FR-13).

Hard rule from the PRD (section 14.11): the report must describe a
*suspected* issue and must never claim a confirmed legal emission violation.
"""
import os
import urllib.parse
from datetime import datetime, timezone

AUTHORITY_HANDLE = os.getenv("AUTHORITY_HANDLE", "@ChennaiTrafficP")

REPORT_TEMPLATE = (
    "Suspected visible exhaust smoke was observed from a vehicle.\n\n"
    "Vehicle Registration: {registration}\n"
    "Location: {location}\n"
    "Date and Time: {timestamp}\n\n"
    "Visible exhaust smoke was detected in the attached evidence.\n"
    "Kindly review and take appropriate action if required.\n\n"
    "{authority_handle}"
)


def build_report(
    registration: str,
    location: str,
    timestamp_iso: str | None,
    smoke_confidence: float,
) -> dict:
    if timestamp_iso:
        try:
            dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    display_timestamp = dt.strftime("%d/%m/%Y, %I:%M %p")

    report_text = REPORT_TEMPLATE.format(
        registration=registration,
        location=location or "Location unavailable",
        timestamp=display_timestamp,
        authority_handle=AUTHORITY_HANDLE,
    )

    intent_url = "https://twitter.com/intent/tweet?" + urllib.parse.urlencode({"text": report_text})

    return {
        "registration": registration,
        "location": location or "Location unavailable",
        "timestamp": display_timestamp,
        "report_text": report_text,
        "authority_handle": AUTHORITY_HANDLE,
        "x_intent_url": intent_url,
    }
