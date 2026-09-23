"""
Targeted evaluation against the PRD's specified difficult conditions
(section 14.10): dust, fog, steam, shadows, low light, multiple vehicles,
partial smoke visibility. Point --dir at a folder of labeled hard-case
images (same YOLO label format) to break out metrics per condition tag
encoded in the filename, e.g. "dust_0001.jpg", "fog_0002.jpg".
"""
import argparse
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="../weights/best.pt")
    parser.add_argument("--dir", required=True, help="folder of hard-case images")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--smoke-class-name", default=None,
        help="Exact class name to count as 'smoke' (e.g. 'smoke' for "
             "smoke_only.yaml, 'exhaust_smoke' for smoke.yaml). Auto-detects "
             "by default: matches any class name containing 'smoke' "
             "(case-insensitive) -- override this if a model ever has more "
             "than one such class.",
    )
    args = parser.parse_args()

    model = YOLO(args.weights)
    by_condition = defaultdict(list)

    def is_smoke_class(name: str) -> bool:
        if args.smoke_class_name is not None:
            return name == args.smoke_class_name
        return "smoke" in name.lower()

    for img_path in Path(args.dir).glob("*.*"):
        condition = img_path.stem.split("_")[0]
        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        n_smoke = sum(1 for r in results for b in r.boxes if is_smoke_class(r.names[int(b.cls[0])]))
        by_condition[condition].append(n_smoke)

    for condition, counts in by_condition.items():
        avg = sum(counts) / len(counts) if counts else 0
        print(f"{condition}: avg smoke detections/image = {avg:.2f} over {len(counts)} images")