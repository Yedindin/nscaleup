# -*- coding: utf-8 -*-
import json, yaml, argparse
from pathlib import Path
import numpy as np
import cv2

IGNORE_INDEX = 255

def load_map_yaml(p: Path):
    y = yaml.safe_load(p.read_text(encoding="utf-8"))
    id2name = y.get("classes", {})
    name2id = y.get("map", {})
    return id2name, name2id

def json_to_mask(j, H, W, name2id):
    """
    LabelMe 포맷 가정:
      j["shapes"] : [{label, points, shape_type}, ...]
    polygons/rectangles만 지원 (필요시 더 추가)
    """
    mask = np.zeros((H, W), np.uint8)  # 0=background
    shapes = j.get("shapes", [])
    for s in shapes:
        lab = str(s.get("label", "")).strip()
        cls = name2id.get(lab, None)
        if cls is None:  # 매핑에 없으면 건너뜀(배경 취급)
            continue
        pts = np.array(s.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            # rectangle 처리 (LabelMe는 rectangle도 points 2개)
            if s.get("shape_type") == "rectangle" and len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                x1, y1, x2, y2 = map(int, [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
                cv2.rectangle(mask, (x1, y1), (x2, y2), int(cls), thickness=-1)
            continue
        cv2.fillPoly(mask, [pts.astype(np.int32)], int(cls))
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, help="LabelMe json 폴더")
    ap.add_argument("--out_dir",    required=True, help="라벨맵 PNG 출력 폴더")
    ap.add_argument("--map_yaml",   required=True, help="라벨 매핑 yaml")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _, name2id = load_map_yaml(Path(args.map_yaml))

    jsons = sorted(labels_dir.rglob("*.json"))
    done, miss = 0, 0
    for jp in jsons:
        try:
            j = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] parse fail: {jp} -> {e}")
            miss += 1; continue

        H, W = int(j.get("imageHeight", 0)), int(j.get("imageWidth", 0))
        if H <= 0 or W <= 0:
            print(f"[skip] bad size: {jp}")
            miss += 1; continue

        m = json_to_mask(j, H, W, name2id)
        out_path = out_dir / (jp.stem + ".png")
        cv2.imwrite(str(out_path), m)
        done += 1
        if done % 50 == 0:
            print(f"[{done}] {out_path}")

    print(f"done={done}, skipped={miss}, out_dir={out_dir}")

if __name__ == "__main__":
    main()
