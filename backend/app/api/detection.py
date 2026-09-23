from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.detection_schema import BoundingBox, Detection, DetectResponse
from app.services.yolo_service import yolo_service
from app.utils.image_processing import InvalidImageError, decode_upload_to_bgr

router = APIRouter()


@router.post("/detect", response_model=DetectResponse)
async def detect(image: UploadFile = File(...)):
    """
    FR-03/FR-04/FR-05/FR-06: detect vehicle + visible exhaust smoke, check
    spatial association, and apply the confidence-threshold reporting gate.
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

    return DetectResponse(
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
