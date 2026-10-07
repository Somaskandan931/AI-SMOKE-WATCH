from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class DetectResponse(BaseModel):
    vehicle_detected: bool
    smoke_detected: bool
    vehicle_confidence: float = 0.0
    smoke_confidence: float = 0.0
    smoke_associated_with_vehicle: bool = False
    reporting_allowed: bool
    detections: List[Detection] = []
    mode: str  # "model" or "mock"
    message: str


class PlateDetectResponse(BaseModel):
    plate_found: bool
    box: Optional[BoundingBox] = None
    cropped_image_base64: Optional[str] = None
    mode: str
    message: str


class PlateOcrResponse(BaseModel):
    registration_number: Optional[str]
    confidence: float = 0.0
    ocr_available: bool
    mode: str
    message: str


class AnalyzeResponse(BaseModel):
    """One-shot result of /analyze: vehicle+smoke detection, plate
    location, and OCR, all run server-side against a single uploaded
    image so the client never has to re-upload the same photo for each
    stage. Field names deliberately mirror DetectResponse/PlateDetectResponse/
    PlateOcrResponse so existing client-side model parsing logic transfers
    with minimal changes.
    """

    # --- vehicle + smoke (same as DetectResponse) ---
    vehicle_detected: bool
    smoke_detected: bool
    vehicle_confidence: float = 0.0
    smoke_confidence: float = 0.0
    smoke_associated_with_vehicle: bool = False
    reporting_allowed: bool
    detections: List[Detection] = []
    mode: str  # vehicle/smoke stage: "model" or "mock"
    message: str

    # --- plate location + crop (same as PlateDetectResponse) ---
    # Plate detection/OCR only runs when reporting_allowed is True (no
    # point locating a plate for a report that can't be filed), so these
    # stay at their "not attempted" defaults otherwise.
    plate_found: bool = False
    plate_box: Optional[BoundingBox] = None
    plate_cropped_image_base64: Optional[str] = None
    plate_mode: Optional[str] = None
    plate_message: Optional[str] = None

    # --- OCR (same as PlateOcrResponse) ---
    registration_number: Optional[str] = None
    ocr_confidence: float = 0.0
    ocr_available: bool = False
    ocr_mode: Optional[str] = None
    ocr_message: Optional[str] = None
