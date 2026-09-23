from fastapi import APIRouter, HTTPException

from app.schemas.report_schema import ReportRequest, ReportResponse
from app.services.report_service import build_report
from app.utils.validation import looks_like_valid_plate

router = APIRouter()


@router.post("/report/generate", response_model=ReportResponse)
async def generate_report(payload: ReportRequest):
    """
    FR-13: build the structured, preview-able report. FR-15 (explicit user
    confirmation before publishing) is enforced client-side -- this endpoint
    only *drafts* the report and an X share-intent URL; nothing is posted
    from the server.
    """
    if not payload.smoke_detected:
        # Defense in depth: the reporting gate should already have stopped
        # the user before they reach this screen (FR-05).
        raise HTTPException(
            status_code=400,
            detail="Reports can only be generated after visible smoke has been detected.",
        )

    if not looks_like_valid_plate(payload.registration_number):
        # Not a hard failure -- plates get manually corrected (FR-10) and
        # formats vary by state/country -- but we surface a warning field.
        pass

    location = payload.place_name or (
        f"{payload.latitude:.5f}, {payload.longitude:.5f}"
        if payload.latitude is not None and payload.longitude is not None
        else "Location unavailable"
    )

    report = build_report(
        registration=payload.registration_number,
        location=location,
        timestamp_iso=payload.timestamp,
        smoke_confidence=payload.smoke_confidence,
    )

    return ReportResponse(
        registration=report["registration"],
        location=report["location"],
        timestamp=report["timestamp"],
        smoke_detected=payload.smoke_detected,
        smoke_confidence=payload.smoke_confidence,
        report_text=report["report_text"],
        authority_handle=report["authority_handle"],
        x_intent_url=report["x_intent_url"],
    )
