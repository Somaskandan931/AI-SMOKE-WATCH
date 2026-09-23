from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ReportRequest(BaseModel):
    registration_number: str = Field(..., min_length=3, max_length=20)
    smoke_detected: bool
    smoke_confidence: float = Field(ge=0.0, le=1.0)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_name: Optional[str] = None  # human-readable landmark/locality, if resolved
    timestamp: Optional[str] = None  # ISO 8601; server fills in if omitted

    @field_validator("registration_number")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("registration_number must not be blank")
        return v.upper()


class ReportResponse(BaseModel):
    registration: str
    location: str
    timestamp: str
    smoke_detected: bool
    smoke_confidence: float
    report_text: str
    authority_handle: str
    x_intent_url: str
