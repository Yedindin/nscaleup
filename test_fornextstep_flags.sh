#!/usr/bin/env bash
set -euo pipefail
python /home/piai/yejin/nscaleup/fornextstep.py \
  --project_root='/home/piai/yejin/nscaleup' \
  --image_path='/home/piai/yejin/nscaleup/data/images/img_00113.jpg' \
  --label_json_dir='/home/piai/yejin/nscaleup/data/labels' \
  --pred_save_dir='/home/piai/yejin/nscaleup/masks_3' \
  --weights_path='/home/piai/yejin/nscaleup/runs/best_miou.pt' \
  --tmp_work_dir='/home/piai/yejin/nscaleup/tmp_single_eval' \
  --yaml_path='/home/piai/yejin/nscaleup/meta/classes_3.yaml'
