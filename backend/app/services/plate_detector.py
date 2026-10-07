"""
License plate detection (PRD section 8 / FR-07, FR-08).

Kept as a separate stage from smoke detection, per the PRD's note that plate
detection may be a distinct model. Looks for a trained plate weights file
via config.find_plate_weights() (checks ai/weights/plate_best.pt). As of
this writing that file is NOT actually present in ai/weights/ (only a
README.md is), so in practice this always runs in "mock" mode -- see
ai/weights/README.md if you want to drop in a real plate_best.pt later;
no code changes will be needed once it's there.

Mock mode is a two-pass OpenCV pipeline, tried in order:
  1. Haar cascade (haarcascade_license_plate_rus_16stages.xml, bundled
     with opencv-python). Trained on Russian plates, but the aspect
     ratio/edge-density signature it looks for transfers reasonably well
     to other plate styles including Indian HSRP plates, and it's far
     less prone to false positives than raw contour-matching since it's
     an actual trained classifier, not just "biggest wide rectangle".
  2. Contour/aspect-ratio fallback (the original heuristic), now with a
     morphological closing pass first to merge the character-level edges
     Canny produces into one solid plate-shaped blob rather than a
     scatter of small contours -- this was the main reason the old
     version frequently missed real plates.
Real weights still take priority over both when present.
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

_CASCADE_FILENAME = "haarcascade_license_plate_rus_16stages.xml"


class PlateDetector:
    def __init__(self):
        self.mode = "mock"
        self.model = None
        self._cascade = None

        if not config.FORCE_MOCK_INFERENCE:
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

        if self.mode == "mock":
            cascade_path = os.path.join(cv2.data.haarcascades, _CASCADE_FILENAME)
            if os.path.exists(cascade_path):
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self._cascade = cascade
                else:
                    logger.warning("plate_detector: failed to load haar cascade %s", cascade_path)
            else:
                logger.warning("plate_detector: haar cascade not found at %s", cascade_path)

    def detect(self, image_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        if self.mode == "model" and self.model is not None:
            results = self.model.predict(image_bgr, verbose=False)
            for r in results:
                if len(r.boxes) > 0:
                    box = r.boxes[0]
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    return (x1, y1, x2, y2), "model"
            return None, "model"

        box = self._detect_cascade(image_bgr)
        if box is not None:
            return box, "mock"

        return self._detect_contour(image_bgr), "mock"

    def _detect_cascade(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self._cascade is None:
            return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        plates = self._cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 20)
        )
        if len(plates) == 0:
            return None
        # Prefer the largest detected region -- most likely to be the
        # plate the user actually framed, rather than a small false hit.
        x, y, w, h = max(plates, key=lambda p: p[2] * p[3])
        return (int(x), int(y), int(x + w), int(y + h))

    def _detect_contour(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)
        # Merge nearby edge fragments (individual characters, plate
        # border, mounting screws) into one solid blob per candidate
        # region before contour-finding -- without this, Canny output on
        # a real plate is a scatter of small contours and the "biggest
        # contour" logic below almost never lands on the plate itself.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
