from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import detection, plate, reports
from app.services.plate_detector import plate_detector
from app.services.yolo_service import yolo_service

app = FastAPI(
    title="Vehicle Smoke Civic Reporting API",
    description="Detect. Verify. Report. Backend for the visible-exhaust-smoke reporting app.",
    version="0.1.0",
)

# Wide-open CORS for hackathon demo purposes -- tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router, prefix="/api", tags=["detection"])
app.include_router(plate.router, prefix="/api", tags=["plate"])
app.include_router(reports.router, prefix="/api", tags=["reports"])


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        # Kept as exactly "model"/"mock" for API-contract compatibility with
        # existing tests/clients. See yolo_tier for which tier of "model"
        # (unified vs two-model) actually loaded, and yolo_diagnostics for
        # the real class names read back from whichever weight file(s) loaded.
        "yolo_mode": "model" if yolo_service.mode in ("unified_model", "two_model") else "mock",
        "yolo_tier": yolo_service.mode,
        "yolo_diagnostics": yolo_service.diagnostics,
        "plate_mode": plate_detector.mode,
    }
