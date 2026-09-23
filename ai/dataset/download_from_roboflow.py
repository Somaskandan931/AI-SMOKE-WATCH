"""
Pulls the vehicle + exhaust_smoke dataset(s) needed for ../configs/smoke.yaml
and drops them exactly where it already expects (ai/dataset/processed/
{images,labels}/{train,val,test}) -- so train.py works with zero path edits.

Two modes, both writing to the same processed/ layout:

1. SINGLE SOURCE (original behavior, unchanged) -- one Roboflow project
   that already contains both "vehicle" and "exhaust_smoke" classes:

       python download_from_roboflow.py \\
           --workspace your-workspace-slug --project vehicle-exhaust-smoke --version 1

2. MULTI-SOURCE MERGE (new) -- there is no single public dataset of
   meaningful size with both classes ready-made (LaSSoV / DB-Net's
   PoVSSeg are academic-only, "available on request" -- see
   sources.yaml's header for details), so this pulls several small
   public Roboflow Universe projects listed in sources.yaml, remaps
   each one's own class names onto the canonical {vehicle: 0,
   exhaust_smoke: 1} used by smoke.yaml (dropping any of a source's
   classes that aren't in its class_map), and merges everything into
   one dataset:

       python download_from_roboflow.py --sources sources.yaml

   It also folds in your own "no smoke" negative photos from
   ai/dataset/raw/negatives/ as background images with empty label
   files (PRD 14.3 -- negatives matter most for avoiding false
   positives). Anything with no split info from its source (negatives,
   or a source that only ships a "train" split) is deterministically
   split 80/10/10 into train/val/test.

Setup (either mode):
    pip install -r ../requirements.txt --break-system-packages
    export ROBOFLOW_API_KEY=your_key_here   # from roboflow.com/settings/api
    # (Universe datasets are public but still require your own free key.)

Either mode writes the same output: ai/dataset/processed/{images,labels}/
{train,val,test}, matching smoke.yaml's train:/val:/test: keys exactly --
no changes to smoke.yaml, train.py, validate.py, or evaluate.py needed.
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT.parents[1] / ".env")

PROCESSED_DIR = ROOT / "processed"
DOWNLOADS_CACHE = ROOT / "_roboflow_downloads"  # per-source raw exports, cached across reruns
NEGATIVES_DIR_DEFAULT = ROOT / "raw" / "negatives"

# Roboflow's YOLOv8 export names the validation split "valid"; our
# smoke.yaml (matching this repo's existing convention) calls it "val".
SPLIT_RENAME = {"train": "train", "valid": "val", "test": "test"}
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}  # for files with no split info
RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _require_roboflow():
    try:
        from roboflow import Roboflow
        return Roboflow
    except ImportError:
        raise SystemExit(
            "roboflow is not installed. Run: pip install -r ../requirements.txt --break-system-packages"
        )


def _require_api_key(api_key):
    if not api_key:
        raise SystemExit(
            "No Roboflow API key. Set ROBOFLOW_API_KEY or pass --api-key.\n"
            "Find yours at https://app.roboflow.com/settings/api\n"
            "(Universe datasets are public but still require your own free key to download.)"
        )


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _read_export_class_names(export_root: Path) -> dict:
    """Read the class id -> name mapping Roboflow writes into data.yaml."""
    data_yaml = export_root / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"No data.yaml found in {export_root} -- unexpected export format.")
    names = _load_yaml(data_yaml)["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: n for i, n in enumerate(names)}


# --------------------------------------------------------------------------- #
# Mode 1: single source (original behavior, unchanged)
# --------------------------------------------------------------------------- #

def download_single(workspace, project, version, api_key):
    Roboflow = _require_roboflow()
    _require_api_key(api_key)

    rf = Roboflow(api_key=api_key)
    project_ = rf.workspace(workspace).project(project)
    dataset = project_.version(version).download("yolov8")

    src_root = Path(dataset.location)
    print(f"\nDownloaded to {src_root}. Arranging into {PROCESSED_DIR} ...")

    for rf_split, target_split in SPLIT_RENAME.items():
        src_images = src_root / rf_split / "images"
        src_labels = src_root / rf_split / "labels"
        if not src_images.exists():
            continue  # e.g. "test" split is often absent -- that's fine, smoke.yaml marks it optional

        dst_images = PROCESSED_DIR / "images" / target_split
        dst_labels = PROCESSED_DIR / "labels" / target_split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for f in src_images.glob("*"):
            shutil.copy2(f, dst_images / f.name)
        for f in src_labels.glob("*"):
            shutil.copy2(f, dst_labels / f.name)

        print(f"  {rf_split} -> images/{target_split}, labels/{target_split} "
              f"({len(list(dst_images.glob('*')))} images)")

    print(
        f"\nDone. Dataset is ready at {PROCESSED_DIR}, matching ../configs/smoke.yaml.\n"
        f"Train with:\n  cd ../training && python train.py"
    )


# --------------------------------------------------------------------------- #
# Mode 2: multi-source merge
# --------------------------------------------------------------------------- #

def _download_source(rf, source: dict, cache_dir: Path) -> Path:
    """Download one sources.yaml entry, cached by name across reruns."""
    dest = cache_dir / source["name"]
    if (dest / "data.yaml").exists():
        print(f"  [{source['name']}] already downloaded, reusing {dest}")
        return dest

    project_ = rf.workspace(source["workspace"]).project(source["project"])
    dataset = project_.version(source["version"]).download("yolov8", location=str(dest))
    return Path(dataset.location)


def _remap_label(label_path: Path, id_to_name: dict, class_map: dict, canonical_ids: dict):
    """
    Translate one source's YOLO label file into canonical ids. Boxes for
    classes not present in this source's class_map are dropped (not
    mis-mapped). Returns (label_text, [canonical_ids_used]) or None if the
    image has zero boxes left after remapping -- callers should skip that
    image entirely rather than train on an unlabeled positive-looking photo.
    """
    kept_lines, kept_ids = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            src_name = id_to_name.get(int(parts[0]))
            canonical_name = class_map.get(src_name)
            canonical_id = canonical_ids.get(canonical_name)
            if canonical_id is None:
                continue
            kept_lines.append(" ".join([str(canonical_id)] + parts[1:]))
            kept_ids.append(canonical_id)
    if not kept_lines:
        return None
    return "\n".join(kept_lines) + "\n", kept_ids


def _stage_source(source: dict, export_root: Path, pool: dict, canonical_ids: dict):
    id_to_name = _read_export_class_names(export_root)
    class_map = source["class_map"]
    box_counts = defaultdict(int)
    image_count = 0

    # If this source's export only shipped a "train" folder (no "valid"/
    # "test"), its images have no real split info of their own -- route
    # them to "unassigned" so _flush_pool's 80/10/10 split actually
    # applies, instead of silently dumping 100% of them into train forever.
    # (Previously this always used the source's own rf_split name as the
    # target, so a train-only export always produced val=0/test=0 --
    # exactly what sources.yaml's and this module's own docstrings claimed
    # couldn't happen.)
    available_splits = [
        rf_split for rf_split, _ in SPLIT_RENAME.items()
        if (export_root / rf_split / "images").exists()
    ]
    train_only = available_splits == ["train"]

    for rf_split, target_split in SPLIT_RENAME.items():
        src_images = export_root / rf_split / "images"
        src_labels = export_root / rf_split / "labels"
        if not src_images.exists():
            continue
        destination = "unassigned" if train_only else target_split
        for img in src_images.glob("*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            remapped = _remap_label(src_labels / (img.stem + ".txt"), id_to_name, class_map, canonical_ids)
            if remapped is None:
                continue  # nothing left after dropping unmapped classes -- skip the image
            label_text, ids = remapped
            image_count += 1
            for cid in ids:
                box_counts[cid] += 1
            pool[destination].append((source["name"], img, label_text))

    id_to_canonical_name = {v: k for k, v in canonical_ids.items()}
    readable_counts = {id_to_canonical_name.get(k, k): v for k, v in box_counts.items()}
    split_note = " (train-only export -> will be 80/10/10 split)" if train_only else ""
    print(f"  [{source['name']}] staged {image_count} images, boxes by class: {readable_counts}{split_note}")
    return image_count


def _stage_negatives(negatives_dir: Path, pool: dict):
    """Fold in the user's own no-smoke vehicle photos as background images
    with empty label files (PRD 14.3 -- negatives matter most)."""
    if not negatives_dir.exists():
        print(f"  (no negatives dir at {negatives_dir} -- skipping; see PRD 14.3 on why "
              f"they matter, and ai/dataset/raw/negatives/README.md to add some)")
        return 0
    n = 0
    for img in negatives_dir.glob("*"):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        pool["unassigned"].append(("negatives", img, ""))  # empty label = valid YOLO "no objects"
        n += 1
    print(f"  [negatives] staged {n} background images from {negatives_dir}")
    return n


def _flush_pool(pool: dict) -> dict:
    """Write the staged pool to processed/images|labels/{train,val,test}.
    Anything with no split info (negatives, or a source that only shipped a
    "train" split) is deterministically split 80/10/10 with a fixed seed."""
    random.seed(RANDOM_SEED)
    unassigned = pool.pop("unassigned", [])
    random.shuffle(unassigned)
    n = len(unassigned)
    cut_train = int(n * SPLIT_RATIO["train"])
    cut_val = cut_train + int(n * SPLIT_RATIO["val"])
    pool["train"].extend(unassigned[:cut_train])
    pool["val"].extend(unassigned[cut_train:cut_val])
    pool["test"].extend(unassigned[cut_val:])

    totals = {}
    for split in ("train", "val", "test"):
        items = pool.get(split, [])
        dst_images = PROCESSED_DIR / "images" / split
        dst_labels = PROCESSED_DIR / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        for source_name, img_path, label_text in items:
            out_stem = f"{source_name}_{img_path.stem}"
            shutil.copy2(img_path, dst_images / f"{out_stem}{img_path.suffix}")
            (dst_labels / f"{out_stem}.txt").write_text(label_text)
        totals[split] = len(items)
    return totals


def download_merged(sources_path: str, api_key: str, negatives_dir: Path, processed_dir: Path = None):
    Roboflow = _require_roboflow()
    _require_api_key(api_key)

    global PROCESSED_DIR
    if processed_dir is not None:
        PROCESSED_DIR = processed_dir

    cfg = _load_yaml(sources_path)
    canonical_ids = {name: cid for cid, name in cfg["canonical_classes"].items()}
    # Loosened from requiring exactly {"vehicle", "exhaust_smoke"} -- a
    # sources.yaml is also allowed to target a single-class run (e.g. a
    # smoke-only fine-tune, when no source in the merge has usable
    # vehicle-class data -- see sources_smoke_only.yaml).
    allowed = {"vehicle", "exhaust_smoke", "smoke"}
    if not canonical_ids or not set(canonical_ids) <= allowed:
        raise SystemExit(f"sources.yaml canonical_classes must be a non-empty subset of {allowed}, got {set(canonical_ids)}")

    rf = Roboflow(api_key=api_key)
    DOWNLOADS_CACHE.mkdir(exist_ok=True)
    pool = defaultdict(list)

    print(f"Merging {len(cfg['sources'])} source(s) into {PROCESSED_DIR} ...")
    for source in cfg["sources"]:
        print(f"\n[{source['name']}] {source.get('url', '')}")
        try:
            export_root = _download_source(rf, source, DOWNLOADS_CACHE)
        except Exception as e:
            print(f"  SKIPPED -- download failed ({e}).\n"
                  f"  Check that workspace/project/version in sources.yaml still match "
                  f"{source.get('url')} -- see sources.yaml's header for how to re-verify.")
            continue
        _stage_source(source, export_root, pool, canonical_ids)

    _stage_negatives(negatives_dir, pool)
    totals = _flush_pool(pool)

    print(f"\nDone. Merged dataset ready at {PROCESSED_DIR}:")
    for split, n in totals.items():
        print(f"  {split}: {n} images")
    total_images = sum(totals.values())
    print(
        f"\n{total_images} images total -- report mAP/precision/recall from this honestly; "
        f"a few hundred images gives you a demo-credible model, not a production-grade one.\n"
        f"Train with:\n  cd ../training && python train.py"
    )


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Populate ai/dataset/processed/ for smoke.yaml.")
    parser.add_argument("--workspace", help="Roboflow workspace slug (single-source mode)")
    parser.add_argument("--project", help="Roboflow project slug (single-source mode)")
    parser.add_argument("--version", type=int, help="Dataset version number (single-source mode)")
    parser.add_argument("--sources", help="Path to a sources.yaml (multi-source merge mode)")
    parser.add_argument(
        "--negatives-dir", default=str(NEGATIVES_DIR_DEFAULT),
        help="Folder of your own no-smoke vehicle photos (multi-source mode only)",
    )
    parser.add_argument(
        "--processed-dir", default=None,
        help="Where to write the merged dataset (multi-source mode only). "
             "Defaults to ai/dataset/processed -- override this when using a "
             "sources.yaml whose config.yaml points elsewhere (e.g. "
             "sources_smoke_only.yaml -> processed_smoke_only, per smoke_only.yaml).",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Defaults to $ROBOFLOW_API_KEY",
    )
    args = parser.parse_args()

    if args.sources:
        processed_dir = Path(args.processed_dir) if args.processed_dir else None
        download_merged(args.sources, args.api_key, Path(args.negatives_dir), processed_dir=processed_dir)
    elif args.workspace and args.project and args.version:
        download_single(args.workspace, args.project, args.version, args.api_key)
    else:
        parser.error("Provide either --workspace/--project/--version, or --sources <sources.yaml>")


if __name__ == "__main__":
    main()