python -m tools.make_splits \
  --img_dir  data/images \
  --mask_dir masks_3 \
  --out_dir  splits \
  --val_ratio 0.2 --seed 42

TS=$(date +'%Y%m%d_%H%M%S'); LOGDIR="runs/3cls_$TS"; LOG="logs/3cls_$TS.log"; mkdir -p "$LOGDIR" logs
nohup python -m src.train_deeplabv3 \
  --img_dir data/images \
  --mask_dir masks_3 \
  --train_split splits/train.txt \
  --val_split   splits/val.txt \
  --num_classes 4 \
  --batch_size 6 \
  --epochs 80 \
  --img_size 896 \
  --lr 3e-4 \
  --logdir "$LOGDIR" \
  > "$LOG" 2>&1 &
