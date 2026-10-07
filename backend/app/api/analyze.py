from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.detection_schema import AnalyzeResponse, BoundingBox, Detection
from app.services.ocr_service import read_plate
from app.services.plate_detector import plate_detector
from app.services.yolo_service import yolo_service
from app.utils.image_processing import (
    InvalidImageError,
    decode_upload_to_bgr,
    encode_bgr_to_base64_jpeg,
)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)):
    """
    One-shot combined stage: vehicle+smoke detection (FR-03/04/05/06),
    plate location (FR-07/08) and OCR (FR-09/10) against a single
    uploaded image. Replaces the previous client flow of uploading the
    same vehicle photo three times (/detect, then /plate/detect, then
    /plate/ocr) with one upload -- meaningfully better UX on a mobile
    data connection, and removes one full round-trip's worth of latency
    before the user sees a result.

    Plate location/OCR are skipped (left at their "not attempted"
    defaults) when reporting_allowed is False, since there's no point
    reading a plate for a report that can't be filed anyway -- same gate
    the client already applies before showing the plate-capture screen.

    /detect, /plate/detect and /plate/ocr are kept as-is and still work
    standalone -- e.g. for the plate close-up retake flow, which
    genuinely is a second, different image.
    """
    try:
        image_bgr = await decode_upload_to_bgr(image)
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result = yolo_service.run(image_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    detections = [
        Detection(
            label=d.label,
            confidence=d.confidence,
            box=BoundingBox(x1=d.box[0], y1=d.box[1], x2=d.box[2], y2=d.box[3]),
        )
        for d in result["detections"]
    ]

    if result["reporting_allowed"]:
        message = "Visible exhaust smoke detected."
    elif not result["vehicle_detected"]:
        message = "No vehicle could be detected in this image. Try a clearer photo."
    elif not result["smoke_detected"]:
        message = "Visible exhaust smoke could not be detected. Reporting cannot continue."
    elif not result["smoke_associated_with_vehicle"]:
        message = "Smoke-like haze was found but doesn't appear to come from this vehicle."
    else:
        message = "Smoke confidence is below the reporting threshold."

    response = AnalyzeResponse(
        vehicle_detected=result["vehicle_detected"],
        smoke_detected=result["smoke_detected"],
        vehicle_confidence=result["vehicle_confidence"],
        smoke_confidence=result["smoke_confidence"],
        smoke_associated_with_vehicle=result["smoke_associated_with_vehicle"],
        reporting_allowed=result["reporting_allowed"],
        detections=detections,
        mode=result["mode"],
        message=message,
    )

    if not result["reporting_allowed"]:
        return response

    # Reuse the exact same decoded frame -- no re-read, no re-upload.
    box, plate_mode = plate_detector.detect(image_bgr)

    if box is None:
        response.plate_found = False
        response.plate_mode = plate_mode
        response.plate_message = "Could not locate a license plate. Try capturing it closer and in focus."
        return response

    cropped = plate_detector.crop(image_bgr, box)
    response.plate_found = True
    response.plate_box = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    response.plate_cropped_image_base64 = encode_bgr_to_base64_jpeg(cropped)
    response.plate_mode = plate_mode
    response.plate_message = "License plate located."

    registration, ocr_confidence, ocr_available = read_plate(cropped)
    response.ocr_available = ocr_available
    response.ocr_mode = plate_mode

    if not ocr_available:
        response.ocr_message = "OCR engine is not available on this server. Please enter the registration number manually."
    elif not registration:
        response.ocr_message = "OCR could not read the plate clearly. Please enter it manually or retake the photo."
    else:
        response.registration_number = registration
        response.ocr_confidence = round(ocr_confidence, 2)
        response.ocr_message = "Registration number extracted. Please verify it's correct."

    return response
