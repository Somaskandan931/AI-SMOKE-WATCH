"""Step 0: audit the existing train/val/test split for source-image leakage and build a leakage-free split.

Why: the Roboflow export contains several augmented copies of the same source photo (".rf.<hash>" suffixes,
flip_/rot_ prefixes). In the original split the same photo can sit in train AND test, which inflates every
metric. This script groups images by source (filename normalisation + near-duplicate perceptual hash, incl.
mirrored copies) and assigns whole groups to a single split.

    python prepare_dataset.py --audit-only      # just report the leakage in the current split
    python prepare_dataset.py                   # write ai/dataset/processed_smoke_only_grouped/
"""
import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageOps

from common import DATASET, IMG_EXT, RESULTS, SEED, SRC_DATASET, label_path, save_json

AUG = re.compile(r"(^|_)(?:flip_|rot_neg_\d+_|rot_\d+_)+")
RF = re.compile(r"\.rf\.[0-9a-fA-F]+$")


def name_group(path, merge_prefixes):
    s = RF.sub("", path.stem)
    s = AUG.sub(r"\1", s)
    for pre in merge_prefixes:          # e.g. video-frame datasets: whole source = one group
        if s.startswith(pre):
            return pre
    return s


def dhash_bits(path):
    im = Image.open(path).convert("L")
    out = []
    for img in (im, ImageOps.mirror(im)):
        a = np.asarray(img.resize((9, 8), Image.BILINEAR), dtype=np.int16)
        out.append((a[:, 1:] > a[:, :-1]).flatten())
    return out[0], out[1]


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def count_boxes(img_path):
    lp = label_path(img_path)
    return sum(1 for l in lp.read_text().splitlines() if len(l.split()) >= 5) if lp.exists() else 0


def leakage_report(split_of, group_of):
    splits_of_group = defaultdict(set)
    for n, s in split_of.items():
        splits_of_group[group_of[n]].add(s)
    rep = {"n_images": len(split_of), "n_groups": len(splits_of_group),
           "groups_spanning_multiple_splits": sum(1 for v in splits_of_group.values() if len(v) > 1)}
    for s in ("train", "val", "test"):
        total = sum(1 for n, x in split_of.items() if x == s)
        leaked = sum(1 for n, x in split_of.items()
                     if x == s and len(splits_of_group[group_of[n]] - {s}) > 0)
        rep[s] = {"images": total, "images_sharing_a_source_with_another_split": leaked}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.71, 0.19, 0.10], help="train val test")
    ap.add_argument("--hash-dist", type=int, default=5, help="max dHash Hamming distance (of 64) to call two images near-duplicates")
    ap.add_argument("--merge-prefix", nargs="*", default=["dungta_data2"],
                    help="filename prefixes whose images are merged into ONE group (video frames)")
    ap.add_argument("--force-train", nargs="*", default=["dungta_data2"],
                    help="group names always placed in train")
    args = ap.parse_args()

    # ---- collect every image of the ORIGINAL export -------------------------------------------------
    items = []  # (path, old_split)
    for s in ("train", "val", "test"):
        for p in sorted((SRC_DATASET / "images" / s).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                items.append((p, s))
    n = len(items)
    names = [p.name for p, _ in items]
    old_split = {p.name: s for p, s in items}
    print(f"{n} images found in {SRC_DATASET}")

    # ---- group: name-based + near-duplicate hash ----------------------------------------------------
    dsu = DSU(n)
    by_name = {}
    for i, (p, _) in enumerate(items):
        g = name_group(p, args.merge_prefix)
        if g in by_name:
            dsu.union(by_name[g], i)
        else:
            by_name[g] = i
    bits, bits_f = zip(*[dhash_bits(p) for p, _ in items])
    B, BF = np.stack(bits), np.stack(bits_f)
    d = (B[:, None, :] != B[None, :, :]).sum(-1)
    df = (B[:, None, :] != BF[None, :, :]).sum(-1)
    dm = np.minimum(d, df)
    ii, jj = np.where(np.triu(dm <= args.hash_dist, k=1))
    for a, b in zip(ii, jj):
        dsu.union(int(a), int(b))
    group_of = {names[i]: f"g{dsu.find(i)}" for i in range(n)}
    members = defaultdict(list)
    for nm, g in group_of.items():
        members[g].append(nm)
    sizes = sorted((len(v) for v in members.values()), reverse=True)
    print(f"{len(members)} source groups; largest group sizes: {sizes[:8]}")

    before = leakage_report(old_split, group_of)
    print("\nLEAKAGE IN ORIGINAL SPLIT:")
    for s in ("train", "val", "test"):
        print(f"  {s}: {before[s]['images']} images, "
              f"{before[s]['images_sharing_a_source_with_another_split']} share a source with another split")
    print(f"  groups spanning >1 split: {before['groups_spanning_multiple_splits']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        save_json({"before": before, "hash_dist": args.hash_dist}, RESULTS / "split_audit.json")
        print("\n(audit only - nothing written; saved results/split_audit.json)")
        return

    # ---- assign whole groups to splits --------------------------------------------------------------
    rng = random.Random(args.seed)
    path_of = {p.name: p for p, _ in items}
    forced = {g for g, v in members.items()
              if any(name_group(path_of[nm], args.merge_prefix) in args.force_train for nm in v)}
    order = list(members.items())
    rng.shuffle(order)
    order.sort(key=lambda kv: -len(kv[1]))
    target = dict(zip(("train", "val", "test"), [r * n for r in args.ratios]))
    cur = Counter()
    new_split = {}
    for g, v in sorted(order, key=lambda kv: kv[0] not in forced):  # forced groups first (stable)
        if g in forced:
            s = "train"
        else:
            s = max(target, key=lambda k: (target[k] - cur[k]) / target[k])
        cur[s] += len(v)
        for nm in v:
            new_split[nm] = s
    after = leakage_report(new_split, group_of)

    # ---- write the new dataset ----------------------------------------------------------------------
    if DATASET.exists():
        if not args.overwrite:
            raise SystemExit(f"{DATASET} exists - pass --overwrite to rebuild it")
        shutil.rmtree(DATASET)
    boxes = Counter()
    rows = []
    for nm in names:
        s = new_split[nm]
        src = path_of[nm]
        for sub in ("images", "labels"):
            (DATASET / sub / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, DATASET / "images" / s / nm)
        lp = label_path(src)
        if lp.exists():
            shutil.copy2(lp, DATASET / "labels" / s / lp.name)
        nb = count_boxes(src)
        boxes[s] += nb
        rows.append([nm, group_of[nm], old_split[nm], s, nb])
    with open(DATASET.parent / "grouped_split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "group", "old_split", "new_split", "n_boxes"])
        w.writerows(rows)

    summary = {"before": before, "after": after, "hash_dist": args.hash_dist, "seed": args.seed,
               "ratios": args.ratios,
               "counts": {s: {"images": cur[s], "smoke_instances": boxes[s]} for s in ("train", "val", "test")}}
    save_json(summary, RESULTS / "split_audit.json")
    print("\nNEW GROUPED SPLIT (no source shared between splits):")
    for s in ("train", "val", "test"):
        print(f"  {s}: {cur[s]} images, {boxes[s]} smoke boxes")
    print(f"  groups spanning >1 split: {after['groups_spanning_multiple_splits']}")
    print(f"Written to {DATASET}\nSaved results/split_audit.json and grouped_split_manifest.csv")


if __name__ == "__main__":
    main()
