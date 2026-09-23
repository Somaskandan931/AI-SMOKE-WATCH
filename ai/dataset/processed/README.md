Roboflow-exported YOLO-format dataset goes here (images/train,
images/val, labels/train, labels/val, etc. — matches ../configs/smoke.yaml).

Populate it automatically one of two ways:

    # Single Roboflow project that already has both vehicle + exhaust_smoke classes:
    python ../download_from_roboflow.py --workspace <slug> --project <slug> --version <n>

    # Merge several small public datasets (no single ready-made one exists at
    # meaningful size — see ../sources.yaml's header for why) + your own
    # negative photos from ../raw/negatives/:
    python ../download_from_roboflow.py --sources ../sources.yaml

Both require `ROBOFLOW_API_KEY` — see the script's docstring — or export
manually from the Roboflow UI and drop the split folders in here yourself.
