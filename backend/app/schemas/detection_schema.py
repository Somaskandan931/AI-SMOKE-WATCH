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
