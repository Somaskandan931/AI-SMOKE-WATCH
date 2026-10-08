"""Faster R-CNN builder (torchvision), single class 'smoke' + background."""
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from common import IMGSZ


def build(backbone="mobilenet", pretrained=True):
    kw = dict(min_size=IMGSZ, max_size=IMGSZ, box_score_thresh=0.001, box_nms_thresh=0.7,
              box_detections_per_img=300)
    w = "COCO_V1" if pretrained else None
    if backbone == "resnet50":
        m = fasterrcnn_resnet50_fpn(weights=w, weights_backbone=None, **kw)
    elif backbone == "mobilenet":
        m = fasterrcnn_mobilenet_v3_large_fpn(weights=w, weights_backbone=None, **kw)
    else:
        raise ValueError(f"unknown backbone {backbone}")
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)  # background + smoke
    return m
