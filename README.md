# nscaleup — 3-Class Semantic Segmentation (DeepLabV3+)

본 프로젝트는 5개 클래스 → 3개 클래스(+배경)으로 통합된 세그멘테이션 데이터셋을 기반으로  
DeepLabV3+ (ResNet-50, COCO pretrained) 모델을 학습·평가한 결과를 포함합니다.

---

## 프로젝트 구조

```
nscaleup/
├─ src/                     # 모델 학습/평가 코드
├─ tools/                   # 유틸리티 스크립트
│  └─ make_splits.py
├─ meta/
│  └─ classes_3.yaml        # 클래스 정의
├─ splits/                  # train/val 분할 파일
├─ runs/                    # 학습 로그 및 체크포인트
├─ masks_3/                 # GT 또는 예측 마스크
├─ fornextstep/             # 단일 이미지 추론 결과
├─ viz_eval_overlays/       # 시각화 결과
├─ exports/                 # 최신 best 모델 (심볼릭 링크)
├─ train.sh / eval.sh       # 학습·평가 실행 스크립트
└─ fornextstep.py           # 단일 이미지 추론 스크립트
```

---

## 환경 (Environment)

- Python 3.10  
- PyTorch 2.2 / torchvision 0.17  
- CUDA 12.1  
- NVIDIA Driver 535.274  

설치:
```bash
pip install -r requirements.txt
```

---

## 데이터 (Dataset)

- 총 이미지: **614장**
- 분할 방식: `tools/make_splits.py`로 train/val 분할 (`--val_ratio 0.2 --seed 42`)
- 실제 분할 수는 `splits/*.txt` 기준 (아래 명령으로 확인 가능)
  ```bash
  wc -l splits/train.txt splits/val.txt
  ```
- 이미지/마스크 경로: `data/images`, `masks_3`
- 클래스 정의: `meta/classes_3.yaml` (num_classes=4; background 포함)

---

## 라벨 통합 규칙 (5 → 3 Classes)

| 기존 ID | 기존 의미              | 변경 ID | 변경 의미  |
|---------:|------------------------|---------:|------------|
| 1 | 지형(조경O) | 1 | 지형 |
| 2 | 지형(조경X) | 1 | 지형 |
| 3 | 건축물(철거가능) | 2 | 건축물 |
| 4 | 건축물(철거불가능) | 2 | 건축물 |
| 5 | 음영 | 3 | 음영 |

`meta/classes_3.yaml` 예시:
```yaml
classes:
  0: background
  1: 1
  2: 2
  3: 3

map:
  "지형": 1
  "건축물": 2
  "음영": 3
  "1": 1
  "2": 2
  "3": 3
```

---

## 학습 설정 (Training Config)

| 항목 | 설정 |
|------|------|
| 모델 | DeepLabV3+ (ResNet-50 backbone, COCO pretrained) |
| 입력 크기 | 896×896 |
| Batch Size | 6 |
| Epochs | 80 |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Loss | CrossEntropyLoss(ignore_index=255) + aux loss (0.4) |
| Augmentation | Resize, Pad, Flip(0.5), BrightnessContrast, ColorJitter, ShiftScaleRotate, GaussianBlur |
| Validation | Resize + PadOnly |
| AMP | Mixed Precision |
| Checkpoints | best_loss.pt (val_loss), best_miou.pt (mIoU) |

---

## 학습 및 평가

### 데이터 분할
```bash
python -m tools.make_splits \
  --img_dir data/images \
  --mask_dir masks_3 \
  --out_dir splits \
  --val_ratio 0.2 --seed 42
```

### 학습 실행
```bash
bash train.sh
```

### 평가 실행
```bash
bash eval.sh
# 또는
python -m src.eval_deeplabv3 \
  --img_dir data/images \
  --mask_dir masks_3 \
  --split_file splits/val.txt \
  --num_classes 4 \
  --weights exports/best.pt
```

---

## 평가 결과 (Validation)

| 체크포인트 | Pixel Accuracy | mIoU | mF1 |
|-------------|----------------|------|------|
| best_miou.pt | 0.8699 | 0.5852 | 0.7111 |

**Per-class 성능**

| 클래스 | 의미 | IoU | F1 |
|---------|--------|--------|--------|
| 0 | Background | 0.8554 | 0.9221 |
| 1 | 지형 | 0.7076 | 0.8288 |
| 2 | 건축물 | 0.5218 | 0.6858 |
| 3 | 음영 | 0.2561 | 0.4077 |

- 평가 대상: `splits/val.txt`에 포함된 이미지 전량(614장 중 validation split)

**요약**
- Pixel Accuracy 0.87, mIoU 0.59, mF1 0.71  
- 지형·건축물 클래스는 높은 정확도, 음영은 난이도 높은 클래스  
- 라벨 수정 후에도 안정적인 수렴과 신뢰도 확보

---

## 단일 이미지 추론 (fornextstep.py)

`fornextstep.py`는 단일 이미지를 입력받아  
모델 예측 결과와 시각화를 `fornextstep/<이미지이름>/` 경로에 저장합니다.

### 실행 예시
```bash
python fornextstep.py \
  --image_path data/images/img_00113.jpg \
  --yaml_path meta/classes_3.yaml \
  --weights_path exports/best.pt \
  --pred_save_dir fornextstep
```

### 결과 폴더 구조
```
fornextstep/
├─ img_00113/
│  ├─ original.jpg        # 원본 이미지
│  ├─ pred_colored.png    # 클래스별 색상 마스크
│  ├─ pred_overlay.jpg    # 예측 오버레이
│  ├─ gt_overlay.jpg      # (옵션) GT 오버레이
│  └─ sd_mask.png         # Inpaint용 이진 마스크
```

---

## 모델 가중치 (Weights)

학습된 DeepLabV3+ 모델 체크포인트는 아래에서 다운로드할 수 있습니다.

| 파일명 | 설명 | 다운로드 |
|--------|------|-----------|
| best_miou.pt | Validation mIoU 기준 최고 성능 | [Download](https://drive.google.com/uc?export=download&id=1yFWKSelvWWoRY1oOVMVBfKEs5DvcLFNS) |
| best_loss.pt | Validation loss 기준 최고 성능 | [Download](https://drive.google.com/uc?export=download&id=1vZfgGSxcRpjXMghgATDPC0CR3lMD5VXO) |

다운로드 후 다음 경로에 저장하세요:
```
runs/
├─ best_miou.pt
└─ best_loss.pt
```

---

## 기타 메모

- `.gitignore`에는 `data/`, `runs/`, `*.pt` 등 대용량 파일 제외  
- `ignore_index=255`를 학습/마스크 생성에서 일관 적용  
- 평가 및 추론은 `eval.sh`, `fornextstep.py`로 수행 가능  

Repository: https://github.com/Yedindin/nscaleup
