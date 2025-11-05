#!/usr/bin/env bash
# eval.sh — DeepLabV3+ 평가 스크립트
set -euo pipefail

# [옵션] GPU 지정
# export CUDA_VISIBLE_DEVICES=0

# 프로젝트 루트로 이동 (스크립트를 루트에 두지 않았다면 주석 해제)
# cd "$(dirname "$0")"

# PYTHONPATH 설정
export PYTHONPATH="/home/piai/yejin/nscaleup:${PYTHONPATH}"

# 기본값 (필요하면 아래 값만 수정)
IMG_DIR="data/images"
MASK_DIR="masks_3"
SPLIT_FILE="splits/val.txt"
NUM_CLASSES=4            # 배경 포함이면 4
IMG_SIZE=896
BATCH_SIZE=4
WEIGHTS="runs/3cls_20251001_080453/best_miou.pt"

# 인자(옵션)로 덮어쓰기: 사용법 예) ./eval.sh runs/xxx/best_miou.pt splits/test.txt
WEIGHTS="${1:-$WEIGHTS}"
SPLIT_FILE="${2:-$SPLIT_FILE}"

# 존재 체크
[[ -f "$WEIGHTS" ]] || { echo "ERR: weights not found -> $WEIGHTS"; exit 1; }
[[ -f "$SPLIT_FILE" ]] || { echo "ERR: split file not found -> $SPLIT_FILE"; exit 1; }

# 로그 디렉토리
mkdir -p logs
TS=$(date +'%Y%m%d_%H%M%S')
LOG="logs/eval_${TS}.log"

echo "[Eval] weights=$WEIGHTS | split=$SPLIT_FILE | log=$LOG"
python -m src.eval_deeplabv3 \
  --img_dir "$IMG_DIR" \
  --mask_dir "$MASK_DIR" \
  --split_file "$SPLIT_FILE" \
  --num_classes "$NUM_CLASSES" \
  --img_size "$IMG_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --weights "$WEIGHTS" | tee "$LOG"


# cd /home/piai/yejin/nscaleup
# chmod +x eval.sh
# ./eval.sh                            # 기본값으로 평가
# ./eval.sh runs/xxx/best_miou.pt      # 가중치만 교체
# ./eval.sh runs/xxx/best.pt splits/val.txt   # 가중치와 스플릿 둘 다 지정
