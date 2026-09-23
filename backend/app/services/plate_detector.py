"""
License plate detection (PRD section 8 / FR-07, FR-08).

Kept as a separate stage from smoke detection, per the PRD's note that plate
detection may be a distinct model. Looks for a trained plate weights file
via config.find_plate_weights() (checks ai/weights/plate_best.pt -- the
file this repo actually ships -- with PLATE_WEIGHTS_PATH still honored as
an override for a custom path/filename). Falls back to an OpenCV
contour/aspect-ratio heuristic (looks for a wide rectangular high-contrast
region -- typical of a plate) if no weights load, so the pipeline stays
demonstrable either way.
"""
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from app import config

logger = logging.getLogger(__name__)

# Explicit override wins if set; otherwise fall back to whatever
# config.find_plate_weights() actually finds on disk (plate_best.pt).
PLATE_WEIGHTS_PATH_OVERRIDE = os.getenv("PLATE_WEIGHTS_PATH")


class PlateDetector:
    def __init__(self):
        self.mode = "mock"
        self.model = None

        if config.FORCE_MOCK_INFERENCE:
            return

        weights_path = PLATE_WEIGHTS_PATH_OVERRIDE or config.find_plate_weights()
        if weights_path and os.path.exists(weights_path) and config.ultralytics_available():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(weights_path))
                self.mode = "model"
                logger.info("plate_detector: loaded %s, classes=%s", weights_path, self.model.names)
            except Exception as e:
                logger.warning("plate_detector: failed to load %s: %s", weights_path, e)
                self.model = None
                self.mode = "mock"

    def detect(self, image_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        if self.mode == "model" and self.model is not None:
            results = self.model.predict(image_bgr, verbose=False)
            for r in results:
                if len(r.boxes) > 0:
                    box = r.boxes[0]
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    return (x1, y1, x2, y2), "model"
            return None, "model"

        return self._detect_mock(image_bgr), "mock"

    def _detect_mock(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect_ratio = w / h
            area = w * h
            frame_area = image_bgr.shape[0] * image_bgr.shape[1]
            if 2.0 <= aspect_ratio <= 5.5 and 0.005 * frame_area <= area <= 0.4 * frame_area:
                return (x, y, x + w, y + h)
        return None

    def crop(self, image_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image_bgr.shape[1], x2)
        y2 = min(image_bgr.shape[0], y2)
        return image_bgr[y1:y2, x1:x2]


plate_detector = PlateDetector()
