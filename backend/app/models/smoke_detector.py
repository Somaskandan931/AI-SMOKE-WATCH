"""
Thin re-export so `app/models/smoke_detector.py` matches the spec'd
project layout. The actual detection logic (real + mock inference,
class handling, association check) lives in app/services/yolo_service.py
since it's shared infrastructure (model loading, singleton) rather than
a pure data model. Import from here for the documented module path.
"""

from app.services.yolo_service import (  # noqa: F401
    Detection,
    DetectionResult,
    YoloService,
    iou,
    smoke_associated_with_vehicle,
    yolo_service,
)
