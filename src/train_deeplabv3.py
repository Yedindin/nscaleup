# -*- coding: utf-8 -*-
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from .dataset import SegDataset
import numpy as np

IGNORE_INDEX = 255

@torch.no_grad()
def fast_cm(y_true, y_pred, C, ignore=IGNORE_INDEX):
    y_true = y_true.view(-1).cpu()
    y_pred = y_pred.view(-1).cpu()
    valid = (y_true != ignore)
    y_true = y_true[valid]; y_pred = y_pred[valid]
    if y_true.numel() == 0:
        return np.zeros((C, C), dtype=np.int64)
    idx = (y_true * C + y_pred).to(torch.int64)
    cm = torch.bincount(idx, minlength=C*C).reshape(C, C).cpu().numpy()
    return cm

def miou_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    denom = tp + fp + fn
    valid = denom > 0
    iou = np.zeros_like(tp); iou[valid] = tp[valid] / denom[valid]
    miou = iou[valid].mean() if valid.any() else 0.0
    return float(miou), iou

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--train_split", required=True)
    ap.add_argument("--val_split",   required=True)
    ap.add_argument("--num_classes", type=int, default=4)   # 0..3
    ap.add_argument("--img_size", type=int, default=896)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--logdir", default="runs")
    ap.add_argument("--weights", type=str, default=None)  # optional init
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.logdir, exist_ok=True)

    train_ds = SegDataset(args.img_dir, args.mask_dir, args.train_split, args.img_size, True)
    val_ds   = SegDataset(args.img_dir, args.mask_dir, args.val_split,   args.img_size, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(256, args.num_classes, 1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, args.num_classes, 1)
    model.to(device)

    if args.weights:
        sd = torch.load(args.weights, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        model.load_state_dict(sd, strict=False)
        print(f"Loaded weights from {args.weights} (strict=False)")

    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda")

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        tot, ok, allpix = 0.0, 0, 0
        cm_total = None
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if train: opt.zero_grad(set_to_none=True)
                with autocast():
                    out = model(xb)
                    logits = out["out"]
                    loss = loss_fn(logits, yb)
                    if "aux" in out:
                        loss = loss + 0.4 * loss_fn(out["aux"], yb)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()

                tot += loss.item() * xb.size(0)
                pred = logits.argmax(1)
                valid = (yb != IGNORE_INDEX)
                ok += ((pred == yb) & valid).sum().item()
                allpix += valid.sum().item()
                if not train:
                    cm = fast_cm(yb, pred, args.num_classes)
                    cm_total = cm if cm_total is None else (cm_total + cm)
        avg_loss = tot / max(1, len(loader.dataset))
        acc = ok / max(1, allpix) if allpix > 0 else 0.0
        if train: return avg_loss, acc, None, None
        miou, ious = miou_from_cm(cm_total if cm_total is not None else np.zeros((args.num_classes, args.num_classes)))
        return avg_loss, acc, miou, ious

    best_loss, best_miou = 1e9, -1.0
    ck_loss = os.path.join(args.logdir, "best_loss.pt")
    ck_miou = os.path.join(args.logdir, "best_miou.pt")
    ck_last = os.path.join(args.logdir, "last.pt")

    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc, _, _ = run_epoch(train_loader, True)
        va_loss, va_acc, va_miou, _ = run_epoch(val_loader, False)
        print(f"[{ep:03d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f} mIoU={va_miou:.4f}")

        torch.save({"model": model.state_dict(), "epoch": ep}, ck_last)

        if va_loss < best_loss:
            best_loss = va_loss
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": float(best_loss)}, ck_loss)
            print(f"  -> save BEST(loss): {ck_loss}")
        if va_miou is not None and va_miou > best_miou:
            best_miou = va_miou
            torch.save({"model": model.state_dict(), "epoch": ep, "miou": float(best_miou)}, ck_miou)
            print(f"  -> save BEST(mIoU): {ck_miou}")

    print("done. last:", ck_last, "best_loss:", ck_loss, "best_miou:", ck_miou)

if __name__ == "__main__":
    main()
