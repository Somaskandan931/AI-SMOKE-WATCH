<!-- Working draft in the style of a Natl. Acad. Sci. Lett. short communication. Every {{...}} is filled ONLY from
     ai/benchmark/results/tables.md after the experiments are run. Do not type a number that is not in results/. -->

# Comparative Evaluation of Deep Learning Object Detectors for Visible Vehicle-Exhaust Smoke Detection and Mobile-Based Civic Reporting

R. Somaskandan¹ · {{co-authors}}
¹ {{affiliation}}

**Abstract** Visible smoke from vehicles can indicate abnormal combustion, but detecting it in ordinary images is hard because smoke resembles fog, dust, steam and shadows. This study compares {{N}} deep-learning detectors (YOLOv8n, YOLOv8s, YOLO11n, YOLO11s, Faster R-CNN) on one smoke dataset ({{train}}/{{val}}/{{test}} images in a source-grouped split) under a common protocol, with confidence thresholds chosen on validation data only. The best accuracy–efficiency trade-off was obtained by {{model}} (mAP@50 {{x}}, mAP@50–95 {{x}}, F1 {{x}}, {{x}} ms per image on {{hardware}}). The selected detector is integrated into SmokeWatch, a Flutter/FastAPI application that requires vehicle–smoke association, plate OCR and explicit user confirmation before a suspected-smoke report is shared. The system detects visible smoke only; it does not measure pollutants or establish a legal violation.

**Keywords** Vehicle smoke detection · Object detection · YOLO · Faster R-CNN · Mobile computing · Civic reporting

## 1 Introduction
Vehicle exhaust is a major source of urban air pollution; visibly excessive smoke may indicate an abnormal emission event. Hand-crafted colour/texture methods are sensitive to illumination and background, whereas learned detectors generalise better. The question addressed here is: *which existing detector gives the best trade-off between detection accuracy and computational cost for visible vehicle smoke under identical conditions?* Contributions: (i) a controlled multi-model benchmark; (ii) accuracy and deployment metrics (latency, size); (iii) hard-negative and difficult-case analysis; (iv) vehicle–smoke association; (v) integration in a mobile reporting workflow. {{Related-work paragraph: YOLO family [1,2,3], Faster R-CNN [4], smoke detection difficulty, mobile CV constraints.}}

## 2 Dataset and Methodology
**Dataset.** Smoke-only boxes merged from three Roboflow sources; the original export contained several augmented copies of the same photo in different splits ({{x}} of {{y}} test images shared a source with train/val), so a source-grouped split was built (Table 1). **Models and training.** Table 2 lists the detectors. All YOLO models: 640 px, 50 epochs, batch 8, seed 0, optimizer auto, lr0 0.01, weight decay 5e-4. Faster R-CNN ({{backbone}}): {{epochs}} epochs, SGD; deviations are due to the framework. **Evaluation.** Precision, recall, F1 (IoU 0.5), mAP@50, mAP@50–95; confidence threshold chosen on validation by maximum F1 then frozen; test set used once; 95% bootstrap CIs over test images. **Efficiency.** Mean/median/P95 latency (batch 1, {{hardware}}, {{threads}} threads), parameters, file size.

**Table 1** Dataset (grouped split) — {{from tables.md Table 1}}
**Table 2** Benchmark models — type, parameters, input size — {{from efficiency.json}}
**Fig. 1** SmokeWatch architecture (`System_architecture.png`, updated to show the benchmark-selected detector)

## 3 Results
**Table 3** Test results with 95% CI — {{tables.md Table 2}}. **Table 4** Efficiency — {{tables.md Table 3}}. **Figs. 2–5** benchmark bars, PR curves, threshold analysis, accuracy–latency. Discussion: {{write only after results: which model is Pareto-optimal, whether CIs overlap, why}}.

**Difficult cases.** {{Table 5 from hard_cases_summary.json; state n per category; these are small, hand-verified sets — report as indicative.}}
**Association.** {{Table 6 from association_summary.json (n labelled pairs); chosen rule and why.}}

## 4 Mobile Application
Flow: image → vehicle + smoke detection → spatial association → plate detection/OCR (user-editable) → location/timestamp → preview → explicit user confirmation → X share intent. {{Screenshots from /screenshots.}}

## 5 Limitations
Visible smoke ≠ pollutant concentration; detection ≠ legal determination; small dataset ({{n}} images); annotation bias; fog/dust/steam false positives; OCR errors; latency depends on device; single training seed (CIs reflect test-set sampling, not training variance); images in `photos for test/` {{origin: real or generated}}.

## 6 Conclusion
{{Answer the research question with the measured best trade-off and its numbers; state what was integrated; list future work: video/temporal cues, quantisation, more cities, hard negatives.}}

**Declarations** Conflict of interest: {{…}}. Data/code availability: {{repository, Roboflow sources and licences}}.

## References (verify each entry before submission)
1. Redmon J, Divvala S, Girshick R, Farhadi A (2016) You only look once: unified, real-time object detection. CVPR.
2. Jocher G, Chaurasia A, Qiu J (2023) Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
3. Khanam R, Hussain M (2024) YOLOv11: an overview of the key architectural enhancements. arXiv:2410.17725
4. Ren S, He K, Girshick R, Sun J (2015) Faster R-CNN: towards real-time object detection with region proposal networks. NeurIPS 28.
5. Howard A et al. (2019) Searching for MobileNetV3. ICCV.
6. Lin T-Y et al. (2014) Microsoft COCO: common objects in context. ECCV.
7. Flutter (flutter.dev); FastAPI (fastapi.tiangolo.com); {{OCR engine actually used}}; {{Roboflow dataset sources}}.
