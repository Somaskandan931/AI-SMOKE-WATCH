from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.detection_schema import BoundingBox, PlateDetectResponse, PlateOcrResponse
from app.services.ocr_service import read_plate
from app.services.plate_detector import plate_detector
from app.utils.image_processing import (
    InvalidImageError,
    decode_upload_to_bgr,
    encode_bgr_to_base64_jpeg,
)

router = APIRouter()


@router.post("/plate/detect", response_model=PlateDetectResponse)
async def detect_plate(image: UploadFile = File(...)):
    """FR-07/FR-08: locate and crop the license plate region."""
    try:
        image_bgr = await decode_upload_to_bgr(image)
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))

    box, mode = plate_detector.detect(image_bgr)

    if box is None:
        return PlateDetectResponse(
            plate_found=False,
            mode=mode,
            message="Could not locate a license plate. Try capturing it closer and in focus.",
        )

    cropped = plate_detector.crop(image_bgr, box)
    return PlateDetectResponse(
        plate_found=True,
        box=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        cropped_image_base64=encode_bgr_to_base64_jpeg(cropped),
        mode=mode,
        message="License plate located.",
    )


@router.post("/plate/ocr", response_model=PlateOcrResponse)
async def ocr_plate(image: UploadFile = File(...)):
    """
    FR-09/FR-10: run OCR on a (usually already-cropped) plate image.
    If no OCR engine is installed, or OCR fails, `ocr_available=False` /
    `registration_number=None` is returned so the app falls back to manual
    entry rather than erroring out (Reliability NFR).
    """
    try:
        image_bgr = await decode_upload_to_bgr(image)
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Run plate detection first in case the caller sent the full frame
    # rather than a pre-cropped plate.
    box, det_mode = plate_detector.detect(image_bgr)
    target = plate_detector.crop(image_bgr, box) if box else image_bgr

    registration, confidence, ocr_available = read_plate(target)

    if not ocr_available:
        return PlateOcrResponse(
            registration_number=None,
            confidence=0.0,
            ocr_available=False,
            mode="unavailable",
            message="OCR engine is not available on this server. Please enter the registration number manually.",
        )

    if not registration:
        return PlateOcrResponse(
            registration_number=None,
            confidence=0.0,
            ocr_available=True,
            mode=det_mode,
            message="OCR could not read the plate clearly. Please enter it manually or retake the photo.",
        )

    return PlateOcrResponse(
        registration_number=registration,
        confidence=round(confidence, 2),
        ocr_available=True,
        mode=det_mode,
        message="Registration number extracted. Please verify it's correct.",
    )
