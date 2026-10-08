"""Train the Faster R-CNN baseline (torchvision, COCO-pretrained) on the same grouped split.

    python train_frcnn.py --backbone mobilenet            # default, CPU-friendly
    python train_frcnn.py --backbone resnet50 --device cuda   # classic R50-FPN, needs a GPU in practice

    # Same budget as the four YOLO models (50 epochs, batch 8; lr scaled linearly with batch size):
    python train_frcnn.py --backbone mobilenet --device cuda --match-yolo
    # If batch 8 runs out of GPU memory (4 GB card), keep the epoch count and the original batch size:
    python train_frcnn.py --backbone mobilenet --device cuda --epochs 50

A previous run in runs/frcnn is moved to runs/frcnn_prev_<N>ep before training, never overwritten.
"""
import argparse
import csv
import json
import random
import time

import cv2
import numpy as np
import torch

from common import RUNS, SEED, clear_pred_cache, ensure_dirs, list_images, load_gt_boxes, norm_device, save_json, env_info
from eval_core import build_stats, load_split_gt, summarize
from frcnn_model import build


class SmokeDS(torch.utils.data.Dataset):
    def __init__(self, split, train):
        self.paths = list_images(split)
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes = load_gt_boxes(p, w, h).copy()
        if self.train:
            if random.random() < 0.5:                      # horizontal flip
                img = img[:, ::-1].copy()
                if len(boxes):
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.5:                      # brightness / contrast jitter
                img = np.clip(img.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20),
                              0, 255).astype(np.uint8)
        if len(boxes):
            boxes = boxes[((boxes[:, 2] - boxes[:, 0]) >= 2) & ((boxes[:, 3] - boxes[:, 1]) >= 2)]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
        return t, {"boxes": torch.from_numpy(boxes).float().reshape(-1, 4),
                   "labels": torch.ones(len(boxes), dtype=torch.int64)}


def collate(batch):
    return tuple(zip(*batch))


def eval_val(model, device, gts):
    model.eval()
    preds = {}
    with torch.no_grad():
        for p in list_images("val"):
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)
            out = model([t])[0]
            preds[p.name] = {"boxes": out["boxes"].cpu().numpy(), "scores": out["scores"].cpu().numpy()}
    return summarize(build_stats(preds, gts), 0.25)


def backup_previous(out_dir):
    """Move an existing run aside (e.g. runs/frcnn -> runs/frcnn_prev_30ep) so it is never overwritten."""
    if not (out_dir / "best.pt").exists():
        return None
    try:
        old_ep = json.load(open(out_dir / "train_args.json"))["epochs"]
    except Exception:
        old_ep = "unknown"
    dst = out_dir.parent / f"{out_dir.name}_prev_{old_ep}ep"
    n = 1
    while dst.exists():
        n += 1
        dst = out_dir.parent / f"{out_dir.name}_prev_{old_ep}ep_{n}"
    out_dir.rename(dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mobilenet", choices=["mobilenet", "resnet50"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--match-yolo", action="store_true",
                    help="use the YOLO budget: 50 epochs, batch 8, lr 0.01 (0.00125 per image, linear scaling)")
    args = ap.parse_args()
    if args.match_yolo:
        args.epochs, args.batch, args.lr = 50, 8, 0.01

    ensure_dirs()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    out_dir = RUNS / "frcnn"
    moved = backup_previous(out_dir)
    if moved:
        print(f"previous run kept at {moved}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(norm_device(args.device))

    model = build(args.backbone, pretrained=True).to(device)
    loader = torch.utils.data.DataLoader(SmokeDS("train", True), batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, collate_fn=collate,
                                         generator=torch.Generator().manual_seed(SEED))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(0.67 * args.epochs), int(0.9 * args.epochs)], gamma=0.1)
    warm = min(300, len(loader) - 1)
    gts = load_split_gt("val")

    cfg = {"backbone": args.backbone, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
           "optimizer": "SGD(momentum=0.9, wd=5e-4)", "schedule": "MultiStep(0.67,0.9; gamma 0.1) + linear warmup",
           "augmentation": "hflip + brightness/contrast jitter (no mosaic)", "imgsz": 640, "seed": SEED,
           "pretrained": "COCO_V1", "matched_to_yolo_budget": bool(args.match_yolo), "env": env_info()}
    save_json(cfg, out_dir / "train_args.json")

    log = open(out_dir / "results.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["epoch", "train_loss", "val_precision", "val_recall", "val_map50", "val_map50_95", "seconds"])
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        for it, (imgs, tgts) in enumerate(loader):
            if epoch == 0 and it < warm:
                f = 0.001 + (1 - 0.001) * it / warm
                for g in opt.param_groups:
                    g["lr"] = args.lr * f
            elif epoch == 0 and it == warm:
                for g in opt.param_groups:
                    g["lr"] = args.lr
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch")
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss); nb += 1
            if it % 20 == 0:
                print(f"epoch {epoch+1}/{args.epochs} it {it}/{len(loader)} loss {float(loss):.3f}", flush=True)
        sched.step()
        s = eval_val(model, device, gts)
        fit = 0.1 * s["ap50"] + 0.9 * s["ap50_95"]            # same fitness Ultralytics uses for best.pt
        wr.writerow([epoch + 1, run_loss / max(nb, 1), s["precision"], s["recall"], s["ap50"], s["ap50_95"],
                     round(time.time() - t0, 1)])
        log.flush()
        print(f"== epoch {epoch+1}: val mAP50 {s['ap50']:.4f}  mAP50-95 {s['ap50_95']:.4f}", flush=True)
        ck = {"backbone": args.backbone, "state": model.state_dict(), "epoch": epoch + 1, "fitness": fit}
        torch.save(ck, out_dir / "last.pt")
        if fit > best:
            best = fit
            torch.save(ck, out_dir / "best.pt")
    log.close()
    clear_pred_cache("frcnn")
    print(f"done. best fitness {best:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()