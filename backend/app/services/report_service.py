"""
Report generation (PRD section 10 / FR-13).

Hard rule from the PRD (section 14.11): the report must describe a
*suspected* issue and must never claim a confirmed legal emission violation.
"""
import urllib.parse
from datetime import datetime, timedelta, timezone

from app import config

AUTHORITY_HANDLE = config.CIVIC_AUTHORITY_HANDLE
IST = timezone(timedelta(hours=5, minutes=30))
MAX_LOCATION_CHARS = 60  # keeps the post under X's 280-char limit

REPORT_TEMPLATE = (
    "Suspected visible exhaust smoke from a vehicle.\n\n"
    "Reg: {registration}\n"
    "Location: {location}\n"
    "Time: {timestamp}\n\n"
    "Please review and take action if required.\n\n"
    "{authority_handle}"
)


def build_report(
    registration: str,
    location: str,
    timestamp_iso: str | None,
    smoke_confidence: float,
) -> dict:
    dt = None
    if timestamp_iso:
        try:
            dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        except ValueError:
            dt = None
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)  # naive client time is assumed local (IST)
    else:
        dt = dt.astimezone(IST)

    display_timestamp = dt.strftime("%d/%m/%Y, %I:%M %p") + " IST"
    location = (location or "Location unavailable")[:MAX_LOCATION_CHARS]

    report_text = REPORT_TEMPLATE.format(
        registration=registration,
        location=location,
        timestamp=display_timestamp,
        authority_handle=AUTHORITY_HANDLE,
    )

    intent_url = "https://twitter.com/intent/tweet?" + urllib.parse.urlencode({"text": report_text})

    return {
        "registration": registration,
        "location": location,
        "timestamp": display_timestamp,
        "report_text": report_text,
        "authority_handle": AUTHORITY_HANDLE,
        "x_intent_url": intent_url,
    }
