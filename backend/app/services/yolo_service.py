"""
Vehicle + visible-exhaust-smoke detection.

Three tiers, checked in this order at startup (see config.py for the exact
filenames/thresholds each tier depends on):

1. UNIFIED MODEL -- ai/weights/best.pt or last.pt, produced by training
   ai/training/train.py on the merged dataset (ai/configs/smoke.yaml).
   Expected classes: "vehicle", "exhaust_smoke" directly. This is the
   PRD-correct model and takes priority the moment it exists.

2. TWO COMMUNITY MODELS -- ai/weights/vehicle_yolov8n.pt (stock COCO
   YOLOv8n; car/truck/bus/motorcycle relabeled to "vehicle") +
   ai/weights/smoke_yolov8s.pt (a community fire/smoke detector,
   domain-transferred from general smoke imagery, not exhaust-pipe
   specific; its "smoke" class relabeled to "exhaust_smoke"). Real
   inference on real weights, used automatically whenever no unified
   model has been trained yet.

3. MOCK -- an OpenCV heuristic (grey/hazy region above a dark contour),
   used only if neither tier above loads (ultralytics missing, or no
   weight files present at all). Mock results are always labeled
   `mode: "mock"` and never treated as real model output.

FR-06 spatial association is applied identically on top of whichever
tier produced the raw detections.

Startup logs which tier loaded and the *actual* class names read back
from each model file -- config.py's VEHICLE_CLASS_NAMES/SMOKE_CLASS_NAME
are the class names these weights were believed to use when documented,
not something re-verified against the binary files themselves, so a
mismatch is logged loudly instead of silently producing zero detections
forever.
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from app import config

logger = logging.getLogger(__name__)

SMOKE_CONFIDENCE_THRESHOLD = config.SMOKE_CONFIDENCE_THRESHOLD

VEHICLE_LABEL = "vehicle"
SMOKE_LABEL = "exhaust_smoke"


@dataclass
class BoxDetection:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2


class YoloService:
    def __init__(self):
        self.mode = "mock"          # "unified_model" | "two_model" | "mock"
        self.unified_model = None
        self.vehicle_model = None
        self.smoke_model = None
        self.diagnostics = {}
        self._try_load_models()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _try_load_models(self):
        if config.FORCE_MOCK_INFERENCE:
            self.mode = "mock"
            self.diagnostics = {"reason": "FORCE_MOCK_INFERENCE=1"}
            return

        if not config.ultralytics_available():
            self.mode = "mock"
            self.diagnostics = {"reason": "ultralytics not installed"}
            return

        if self._try_load_unified():
            return
        if self._try_load_two_model():
            return

        self.mode = "mock"
        self.diagnostics = {"reason": "no weight files found under ai/weights/"}

    def _try_load_unified(self) -> bool:
        unified_path = config.find_trained_weights()
        if unified_path is None:
            return False
        try:
            from ultralytics import YOLO

            self.unified_model = YOLO(str(unified_path))
            self.mode = "unified_model"
            names = self.unified_model.names
            self.diagnostics = {"weights": str(unified_path), "classes": names}
            logger.info("yolo_service: loaded unified model %s, classes=%s", unified_path, names)
            expected = {VEHICLE_LABEL, SMOKE_LABEL}
            found = set(names.values()) if isinstance(names, dict) else set(names)
            if not expected.issubset(found):
                logger.warning(
                    "yolo_service: unified model at %s is missing expected classes %s "
                    "(found %s) -- detection for the missing class(es) will never fire.",
                    unified_path, expected - found, found,
                )
            return True
        except Exception as e:
            logger.warning("yolo_service: failed to load unified model %s: %s", unified_path, e)
            self.unified_model = None
            return False

    def _try_load_two_model(self) -> bool:
        vehicle_path = config.find_vehicle_weights()
        smoke_path = config.find_smoke_weights()
        if vehicle_path is None or smoke_path is None:
            return False
        try:
            from ultralytics import YOLO

            self.vehicle_model = YOLO(str(vehicle_path))
            self.smoke_model = YOLO(str(smoke_path))
            self.mode = "two_model"

            vehicle_names = self.vehicle_model.names
            smoke_names = self.smoke_model.names
            self.diagnostics = {
                "vehicle_weights": str(vehicle_path),
                "vehicle_classes": vehicle_names,
                "smoke_weights": str(smoke_path),
                "smoke_classes": smoke_names,
            }
            logger.info("yolo_service: loaded vehicle model %s, classes=%s", vehicle_path, vehicle_names)
            logger.info("yolo_service: loaded smoke model %s, classes=%s", smoke_path, smoke_names)

            found_vehicle_names = set(vehicle_names.values()) if isinstance(vehicle_names, dict) else set(vehicle_names)
            if not (config.VEHICLE_CLASS_NAMES & found_vehicle_names):
                logger.warning(
                    "yolo_service: none of config.VEHICLE_CLASS_NAMES=%s appear in %s's actual "
                    "classes %s -- vehicle detection will never fire until this is corrected.",
                    config.VEHICLE_CLASS_NAMES, vehicle_path, found_vehicle_names,
                )
            found_smoke_names = {n.lower() for n in (smoke_names.values() if isinstance(smoke_names, dict) else smoke_names)}
            if config.SMOKE_CLASS_NAME.lower() not in found_smoke_names:
                logger.warning(
                    "yolo_service: config.SMOKE_CLASS_NAME=%r not found in %s's actual classes %s "
                    "-- smoke detection will never fire until this is corrected.",
                    config.SMOKE_CLASS_NAME, smoke_path, found_smoke_names,
                )
            return True
        except Exception as e:
            logger.warning("yolo_service: failed to load two-model setup (%s / %s): %s", vehicle_path, smoke_path, e)
            self.vehicle_model = None
            self.smoke_model = None
            return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, image_bgr: np.ndarray) -> Tuple[List[BoxDetection], str]:
        if self.mode == "unified_model" and self.unified_model is not None:
            return self._detect_unified(image_bgr), "model"
        if self.mode == "two_model" and self.vehicle_model is not None:
            return self._detect_two_model(image_bgr), "model"
        return self._detect_mock(image_bgr), "mock"

    def _detect_unified(self, image_bgr: np.ndarray) -> List[BoxDetection]:
        results = self.unified_model.predict(image_bgr, verbose=False)
        detections: List[BoxDetection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append(BoxDetection(label=label, confidence=conf, box=(x1, y1, x2, y2)))
        return detections

    def _detect_two_model(self, image_bgr: np.ndarray) -> List[BoxDetection]:
        detections: List[BoxDetection] = []

        for r in self.vehicle_model.predict(image_bgr, verbose=False, conf=config.VEHICLE_CONFIDENCE_THRESHOLD):
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name not in config.VEHICLE_CLASS_NAMES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append(BoxDetection(label=VEHICLE_LABEL, confidence=conf, box=(x1, y1, x2, y2)))

        for r in self.smoke_model.predict(image_bgr, verbose=False):
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name.lower() != config.SMOKE_CLASS_NAME.lower():
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append(BoxDetection(label=SMOKE_LABEL, confidence=conf, box=(x1, y1, x2, y2)))

        return detections

    def _detect_mock(self, image_bgr: np.ndarray) -> List[BoxDetection]:
        """
        Heuristic placeholder used only when no trained weights load at all.

        Vehicle: largest dark/solid contour in the lower 2/3 of the frame.
        Smoke: low-saturation, mid-to-high brightness hazy region
        (grey/white plume) directly above that vehicle contour.
        """
        h, w = image_bgr.shape[:2]
        detections: List[BoxDetection] = []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vehicle_box = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > (w * h) * 0.02:
                x, y, bw, bh = cv2.boundingRect(largest)
                vehicle_box = (x, y, x + bw, y + bh)
                area_frac = (bw * bh) / (w * h)
                vehicle_conf = float(min(0.95, 0.45 + area_frac))
                detections.append(
                    BoxDetection(label=VEHICLE_LABEL, confidence=round(vehicle_conf, 2), box=vehicle_box)
                )

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        haze_mask = ((sat < 60) & (val > 90) & (val < 235)).astype(np.uint8) * 255

        search_y2 = vehicle_box[1] if vehicle_box else int(h * 0.6)
        search_y1 = max(0, search_y2 - int(h * 0.35))
        region = haze_mask[search_y1:search_y2, :]

        if region.size > 0:
            haze_fraction = float(np.count_nonzero(region)) / region.size
            if haze_fraction > 0.18:
                ys, xs = np.nonzero(region)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = search_y1 + int(ys.min()), search_y1 + int(ys.max())
                smoke_conf = float(min(0.92, haze_fraction + 0.35))
                detections.append(
                    BoxDetection(label=SMOKE_LABEL, confidence=round(smoke_conf, 2), box=(x1, y1, x2, y2))
                )

        return detections

    # ------------------------------------------------------------------ #
    # Decision logic (FR-05, FR-06, section 14.9)
    # ------------------------------------------------------------------ #
    @staticmethod
    def boxes_associated(vehicle_box, smoke_box, image_shape) -> bool:
        """Smoke counts only if it overlaps or sits directly adjacent (above/behind)
        the vehicle's bounding box -- not just anywhere in the frame."""
        h, w = image_shape[:2]
        vx1, vy1, vx2, vy2 = vehicle_box
        sx1, sy1, sx2, sy2 = smoke_box

        margin_x = 0.25 * (vx2 - vx1)
        margin_y = 0.6 * (vy2 - vy1)
        ex1, ey1, ex2, ey2 = vx1 - margin_x, vy1 - margin_y, vx2 + margin_x, vy2 + margin_y

        overlap_x = max(0, min(ex2, sx2) - max(ex1, sx1))
        overlap_y = max(0, min(ey2, sy2) - max(ey1, sy1))
        return overlap_x > 0 and overlap_y > 0

    def run(self, image_bgr: np.ndarray):
        detections, mode = self.detect(image_bgr)

        vehicles = [d for d in detections if d.label == VEHICLE_LABEL]
        smokes = [d for d in detections if d.label == SMOKE_LABEL]

        vehicle = max(vehicles, key=lambda d: d.confidence, default=None)
        smoke = max(smokes, key=lambda d: d.confidence, default=None)

        vehicle_detected = vehicle is not None
        smoke_detected = smoke is not None
        associated = False

        if vehicle_detected and smoke_detected:
            associated = self.boxes_associated(vehicle.box, smoke.box, image_bgr.shape)

        smoke_confidence = smoke.confidence if smoke else 0.0
        vehicle_confidence = vehicle.confidence if vehicle else 0.0

        reporting_allowed = bool(
            vehicle_detected
            and smoke_detected
            and associated
            and smoke_confidence >= SMOKE_CONFIDENCE_THRESHOLD
        )

        return {
            "vehicle_detected": vehicle_detected,
            "smoke_detected": smoke_detected,
            "vehicle_confidence": round(vehicle_confidence, 2),
            "smoke_confidence": round(smoke_confidence, 2),
            "smoke_associated_with_vehicle": associated,
            "reporting_allowed": reporting_allowed,
            "detections": detections,
            "mode": mode,
        }


yolo_service = YoloService()
