Drop hard-negative / difficult-case images here (fog, dust, steam, shadows, haze, low light, partial smoke,
multiple vehicles). Then run `python hard_cases.py init` in `ai/benchmark/`, check each row in
`hard_cases/manifest.csv` by eye, and set `verified=1`.

Optional: add a YOLO label file next to an image (same name, `.txt`) to get box-level precision/recall/F1.
A verified image with `has_smoke=0` means "no smoke anywhere" - every detection on it is a false positive.
