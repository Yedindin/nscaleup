# -*- coding: utf-8 -*-
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
import numpy as np
from .dataset import SegDataset

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

def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    denom = tp + fp + fn
    valid = denom > 0
    iou = np.zeros_like(tp); iou[valid] = tp[valid] / denom[valid]
    miou = iou[valid].mean() if valid.any() else 0.0
    f1 = np.zeros_like(tp)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1[valid] = (2*tp[valid]) / (2*tp[valid] + fp[valid] + fn[valid])
    mf1 = f1[valid].mean() if valid.any() else 0.0
    pix_acc = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0
    return float(miou), iou, float(mf1), f1, float(pix_acc)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=896)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SegDataset(args.img_dir, args.mask_dir, args.split_file, args.img_size, is_train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(256, args.num_classes, 1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, args.num_classes, 1)
    try:
        sd = torch.load(args.weights, map_location='cpu', weights_only=True)  # torch>=2.4
    except TypeError:
        sd = torch.load(args.weights, map_location='cpu')  # 호환용 fallback
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    cm_total = None
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)["out"]
            pred = logits.argmax(1)
            cm = fast_cm(yb, pred, args.num_classes)
            cm_total = cm if cm_total is None else (cm_total + cm)

    miou, iou, mf1, f1, acc = metrics_from_cm(cm_total)
    np.set_printoptions(suppress=True, precision=6)
    print(f"Per-class IoU: {iou}")
    print(f"Per-class F1 : {f1}")
    print(f"mIoU={miou:.4f}, mF1={mf1:.4f}, PixelAcc={acc:.4f}")

if __name__ == "__main__":
    main()
