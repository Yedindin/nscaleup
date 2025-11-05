# -*- coding: utf-8 -*-
import argparse, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir, mask_dir, out_dir = Path(args.img_dir), Path(args.mask_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 목록(확장자 포함)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = [p.name for p in img_dir.iterdir() if p.suffix.lower() in exts]
    imgs.sort()
    # 마스크 존재하는 것만
    imgs = [n for n in imgs if (mask_dir / (Path(n).stem + ".png")).exists()]

    random.seed(args.seed)
    random.shuffle(imgs)
    n = len(imgs); nv = int(round(n * args.val_ratio))
    val = imgs[:nv]; train = imgs[nv:]

    (out_dir / "train.txt").write_text("\n".join(train), encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val), encoding="utf-8")
    print(f"total={n}, train={len(train)}, val={len(val)} -> {out_dir}")

if __name__ == "__main__":
    main()
