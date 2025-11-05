#!/usr/bin/env python3
# Auto-generated from Jupyter notebook: fornextstep.ipynb
# This script linearizes the notebook and exposes path-like variables via CLI args.
import argparse, sys, os
from pathlib import Path
import os, json, subprocess
import numpy as np
from PIL import Image, ImageDraw
import yaml

def _build_parser():
    parser = argparse.ArgumentParser(description="Run converted notebook as a CLI script")
    parser.add_argument("--input_path", type=str, default="", help="input path (set if needed)")
    parser.add_argument("--output_path", type=str, default="", help="output path (set if needed)")
    parser.add_argument("--workdir", type=str, default=".", help="working directory")
    parser.add_argument("--project_root", type=str, default="/home/piai/yejin/nscaleup", help="default from notebook")
    parser.add_argument("--image_path", type=str, default="/home/piai/yejin/nscaleup/data/images/img_00113.jpg", help="default from notebook")
    parser.add_argument("--label_json_dir", type=str, default="/home/piai/yejin/nscaleup/data/labels", help="default from notebook")
    parser.add_argument("--pred_save_dir", type=str, default="/home/piai/yejin/nscaleup/masks_3", help="default from notebook")
    parser.add_argument("--weights_path", type=str, default="/home/piai/yejin/nscaleup/runs/best_miou.pt", help="default from notebook")
    parser.add_argument("--tmp_work_dir", type=str, default="/home/piai/yejin/nscaleup/tmp_single_eval", help="default from notebook")
    parser.add_argument("--yaml_path", type=str, default="/home/piai/yejin/nscaleup/meta/classes_3.yaml", help="default from notebook")
    return parser

def _main():
    parser = _build_parser()
    args = parser.parse_args()

    global PROJECT_ROOT, IMAGE_PATH, LABEL_JSON_DIR, PRED_SAVE_DIR, WEIGHTS_PATH, TMP_WORK_DIR, YAML_PATH
    PROJECT_ROOT   = Path(args.project_root)
    IMAGE_PATH     = Path(args.image_path)
    LABEL_JSON_DIR = Path(args.label_json_dir)
    PRED_SAVE_DIR  = Path(args.pred_save_dir)
    WEIGHTS_PATH   = Path(args.weights_path)
    TMP_WORK_DIR   = Path(args.tmp_work_dir)
    YAML_PATH      = Path(args.yaml_path)


if __name__ == "__main__":
    _main()



# ---- code cell separator ----

from pathlib import Path
import os, json, subprocess
import numpy as np
from PIL import Image, ImageDraw
import yaml

# ===== eval 설정 =====
NUM_CLASSES = 4
IMG_SIZE    = 896
BATCH_SIZE  = 1

# ===== 클래스/색상 =====
def load_class_map(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    m = {}
    for k, v in y.get("map", {}).items():
        m[k] = int(v)
        # "1" -> 1 도 추가
        try:
            ki = int(k)
            m[ki] = int(v)
        except:
            pass
    # 배경 안전망
    m.setdefault("배경", 0); m.setdefault("background", 0); m.setdefault("0", 0); m.setdefault(0, 0)
    return m

CLASS_COLORS = [
    (0, 0, 0, 0),      # 0 배경 (투명)
    (255, 0, 0, 100),  # 1 지형
    (0, 255, 0, 100),  # 2 건축물
    (0, 0, 255, 100),  # 3 음영
]
CLASS_NAME_TO_ID = load_class_map(YAML_PATH)


# Stable Diffusion inpaint용 (흰색=수정)
CLASSES_TO_EDIT       = [1,3]
SD_MASK_WHITE_IS_EDIT = True

# ===== 유틸 =====
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def _load_mask_png(p: Path) -> np.ndarray:
    im = Image.open(p)
    if im.mode != "L":
        im = im.convert("L")
    return np.array(im, dtype=np.int32)

def _colored_rgba(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cid, col in enumerate(CLASS_COLORS[:NUM_CLASSES]):
        idx = (mask == cid)
        if idx.any():
            rgba[idx] = col
    return Image.fromarray(rgba, mode="RGBA")

def _overlay(base_rgb_img: Image.Image, rgba_mask_img: Image.Image) -> Image.Image:
    base = base_rgb_img.convert("RGBA")
    if rgba_mask_img.size != base.size:
        rgba_mask_img = rgba_mask_img.resize(base.size, Image.NEAREST)
    out = Image.alpha_composite(base, rgba_mask_img)
    return out.convert("RGB")

def _to_binary_mask(mask: np.ndarray, classes_to_edit: list[int], white_is_edit=True) -> Image.Image:
    edit = np.isin(mask, classes_to_edit).astype(np.uint8) * 255
    if not white_is_edit:
        edit = 255 - edit
    return Image.fromarray(edit, mode="L")

# ===== JSON → Mask (LabelMe/VIA) =====
def _rasterize_polygons(size_hw: tuple[int,int], polygons, name2id: dict) -> np.ndarray:
    H, W = size_hw
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        label = poly.get("label")
        if label is None and "region_attributes" in poly:
            ra = poly["region_attributes"]
            for k in ("label","class","name","category"):
                if k in ra: label = ra[k]; break
        if label is None:
            continue
        cid = name2id.get(label, None)
        if cid is None:
            continue
        pts = poly.get("points")
        if pts is None:
            sp = poly.get("shape_attributes", {})
            xs = sp.get("all_points_x"); ys = sp.get("all_points_y")
            if xs is not None and ys is not None and len(xs)==len(ys) and len(xs)>=3:
                pts = list(zip(xs, ys))
        if pts is None or len(pts) < 3:
            continue
        draw.polygon([tuple(map(float, p)) for p in pts], outline=cid, fill=cid)
    return np.array(mask, dtype=np.int32)

def _make_gt_mask_from_json(json_path: Path, image_size_hw: tuple[int,int]) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    H, W = image_size_hw
    if isinstance(data, dict) and "shapes" in data:  # LabelMe
        polys = [{"label": sh.get("label"), "points": sh.get("points")}
                 for sh in data.get("shapes", []) if isinstance(sh, dict)]
        return _rasterize_polygons((H, W), polys, CLASS_NAME_TO_ID)
    if isinstance(data, dict) and "regions" in data:  # VIA
        polys = [{"region_attributes": r.get("region_attributes", {}),
                  "shape_attributes":  r.get("shape_attributes", {})}
                 for r in data.get("regions", []) if isinstance(r, dict)]
        return _rasterize_polygons((H, W), polys, CLASS_NAME_TO_ID)
    raise ValueError(f"Unsupported JSON schema: {json_path}")

# ===== 단일 이미지용 eval 호출 (pred를 masks_3에 저장한다고 가정) =====
def infer_one_to_masks3(img_path: Path, weights: Path, tmp_work: Path, pred_save_dir: Path) -> Path:
    """
    - 임시 split/임시 mask_dir(dummmy 또는 json→mask) 준비
    - eval_deeplabv3를 PROJECT_ROOT에서 실행
    - 예측 마스크는 pred_save_dir/<stem>.png 에 생성된다고 가정하고 체크
    """
    stem = img_path.stem
    _ensure_dir(tmp_work)
    _ensure_dir(pred_save_dir)

    # 임시 split
    tmp_split = tmp_work / f"{stem}.txt"
    tmp_split.write_text(stem + "\n", encoding="utf-8")

    # 임시 mask_dir (GT json 있으면 rasterize, 없으면 0 dummy)
    tmp_mask_dir = tmp_work / "gt_for_eval"
    _ensure_dir(tmp_mask_dir)
    json_path = LABEL_JSON_DIR / f"{stem}.json"
    if json_path.exists():
        # 원본 이미지 크기 기준으로 rasterize
        img = _load_rgb(img_path)
        gt = _make_gt_mask_from_json(json_path, (img.size[1], img.size[0]))
    else:
        # dummy
        img = _load_rgb(img_path)
        gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
    Image.fromarray(gt.astype(np.uint8), "L").save(tmp_mask_dir / f"{stem}.png")

    # 환경
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env.get("PYTHONPATH",""))

    # eval 실행 (너가 준 인자만 사용)
    cmd = [
        "python", "-m", "src.eval_deeplabv3",
        "--img_dir",    str(img_path.parent),
        "--mask_dir",   str(tmp_mask_dir),
        "--split_file", str(tmp_split),
        "--num_classes", str(NUM_CLASSES),
        "--img_size",    str(IMG_SIZE),
        "--batch_size",  str(BATCH_SIZE),
        "--weights",     str(weights),
    ]
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    if p.stdout: print("[STDOUT]\n", p.stdout)
    if p.stderr: print("[STDERR]\n", p.stderr)
    if p.returncode != 0:
        raise RuntimeError("eval_deeplabv3 실행 실패 (STDERR/STDOUT 참고)")

    pred_png = pred_save_dir / f"{stem}.png"
    if pred_png.exists():
        print("[FOUND]", pred_png)
        return pred_png

    # 혹시 다른 이름/경로일 수 있으니 fallback 검색
    cands = list(pred_save_dir.glob(f"{stem}*.png"))
    if cands:
        print("[Fallback FOUND]", cands[0])
        return cands[0]

    raise FileNotFoundError(
        f"예측 마스크를 {pred_save_dir}에서 찾지 못했습니다.\n"
        f"eval 코드에서 실제 저장 위치/파일명을 확인해 주세요."
    )

# ---- code cell separator ----

# 출력 폴더: 이미지 파일명과 동일
assert IMAGE_PATH.exists(), f"IMAGE_PATH not found: {IMAGE_PATH}"
img = _load_rgb(IMAGE_PATH)
H, W = img.size[1], img.size[0]

OUT_DIR = PROJECT_ROOT / "fornextstep" / IMAGE_PATH.stem
_ensure_dir(OUT_DIR)

# 1) 원본 저장
img.save(OUT_DIR / "original.jpg", quality=95)

# 2) 예측 마스크 확보 (이미 masks_3에 있으면 바로 사용, 없으면 eval 실행)
pred_png = PRED_SAVE_DIR / f"{IMAGE_PATH.stem}.png"
if not pred_png.exists():
    pred_png = infer_one_to_masks3(IMAGE_PATH, WEIGHTS_PATH, TMP_WORK_DIR, PRED_SAVE_DIR)

pred = _load_mask_png(pred_png)
# 이미지 크기에 맞춤
if (pred.shape[1], pred.shape[0]) != img.size:
    pred = np.array(Image.fromarray(pred.astype(np.uint8), "L").resize(img.size, Image.NEAREST), dtype=np.int32)

# 3) 예측 시각화 저장
pred_rgba    = _colored_rgba(pred)
pred_overlay = _overlay(img, pred_rgba)
pred_rgba.save(OUT_DIR / "pred_colored.png")
pred_overlay.save(OUT_DIR / "pred_overlay.jpg", quality=95)

# 4) GT JSON 있으면 GT 오버레이 저장
json_path = LABEL_JSON_DIR / f"{IMAGE_PATH.stem}.json"
if json_path.exists():
    gt = _make_gt_mask_from_json(json_path, (H, W))
    if gt.shape != (H, W):
        gt = np.array(Image.fromarray(gt.astype(np.uint8), "L").resize((W, H), Image.NEAREST), dtype=np.int32)
    gt_rgba    = _colored_rgba(gt)
    gt_overlay = _overlay(img, gt_rgba)
    gt_overlay.save(OUT_DIR / "gt_overlay.jpg", quality=95)
else:
    print(f"[Info] GT JSON not found: {json_path.name} (gt_overlay 생략)")

# 5) Stable Diffusion inpaint용 이진 마스크 (흰색=수정영역)
sd_mask = _to_binary_mask(pred, CLASSES_TO_EDIT, white_is_edit=SD_MASK_WHITE_IS_EDIT)
if sd_mask.size != img.size:
    sd_mask = sd_mask.resize(img.size, Image.NEAREST)
sd_mask.save(OUT_DIR / "sd_mask.png")

print("Done. Saved to:", OUT_DIR)
for p in ["original.jpg",
          "gt_overlay.jpg" if json_path.exists() else None,
          "pred_colored.png",
          "pred_overlay.jpg",
          "sd_mask.png"]:
    if p: print(" -", p)
