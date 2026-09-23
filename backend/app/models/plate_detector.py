"""
NOTE: this file is currently UNUSED. backend/app/api/plate.py imports the
plate detector from app.services.plate_detector (not this module) -- this
is a duplicate from an earlier session that was never wired in. The two
had drifted: this one already correctly read plate_best.pt via config.py,
while app/services/plate_detector.py was still looking for a nonexistent
"plate.pt" until that was fixed. Kept here for reference rather than
deleted (structure preserved); if you want a single source of truth,
delete this file and keep app/services/plate_detector.py.

License plate localization, kept as a separate model/stage from the
vehicle+smoke detector (per the spec: "License plate detection may
initially be implemented as a separate model to keep the smoke-detection
model focused").

Real mode: if ai/weights/plate_best.pt exists and ultralytics is
installed, runs a dedicated plate-detector YOLO model.

Mock mode: returns a plausible plate region (lower-center third of the
image, where a plate usually sits in a close-up photo) so OCR and the
rest of the pipeline can run without a trained plate model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app import config


@dataclass
class PlateDetection:
    box: Optional[tuple[float, float, float, float]]
    confidence: float
    mock_mode: bool


class PlateDetector:
    def __init__(self) -> None:
        self._model = None
        self.mock_mode = True
        # Respect the same FORCE_MOCK_INFERENCE / MOCK_MODE switch as
        # yolo_service.py, so the test suite (which forces mock mode via
        # tests/conftest.py) gets deterministic plate results too, instead
        # of this detector silently going real just because plate_best.pt
        # happens to be present on disk.
        if config.FORCE_MOCK_INFERENCE:
            return
        weights = config.WEIGHTS_DIR / "plate_best.pt"
        if weights.exists() and config.ultralytics_available():
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(weights))
                self.mock_mode = False
            except Exception:
                self._model = None
                self.mock_mode = True

    def detect(self, image: np.ndarray) -> PlateDetection:
        if self._model is not None:
            return self._real_detect(image)
        return self._mock_detect(image)

    def _real_detect(self, image: np.ndarray) -> PlateDetection:
        results = self._model(image, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return PlateDetection(box=None, confidence=0.0, mock_mode=False)
        best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = [float(v) for v in best.xyxy[0]]
        return PlateDetection(box=(x1, y1, x2, y2), confidence=float(best.conf[0]), mock_mode=False)

    def _mock_detect(self, image: np.ndarray) -> PlateDetection:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return PlateDetection(box=None, confidence=0.0, mock_mode=True)
        # Lower-center region — a reasonable heuristic for a close-up
        # plate photo, used only to produce a crop for mock OCR.
        box = (w * 0.2, h * 0.55, w * 0.8, h * 0.85)
        return PlateDetection(box=box, confidence=0.65, mock_mode=True)


plate_detector = PlateDetector()
