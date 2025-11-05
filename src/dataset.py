# -*- coding: utf-8 -*-
import cv2, numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
import torch

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, split_file, img_size, is_train):
        self.img_dir = Path(img_dir); self.mask_dir = Path(mask_dir)
        self.names = [x.strip() for x in Path(split_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.size = int(img_size)
        if is_train:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=self.size),
                A.PadIfNeeded(min_height=self.size, min_width=self.size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=self.size),
                A.PadIfNeeded(min_height=self.size, min_width=self.size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            ])

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        stem = Path(name).stem
        # img = cv2.imread(str(self.img_dir / name), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mask = cv2.imread(str(self.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        # if mask is None:
        #     raise FileNotFoundError(stem)

        # # Albumentations
        # out = self.tf(image=img, mask=mask)
        img = cv2.imread(str(self.img_dir / name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(stem)

        # 🔧 1) 크기 불일치 시, "이미지"를 "마스크" 크기에 맞춤
        if img.shape[:2] != mask.shape[:2]:
            H, W = mask.shape[:2]
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            # (마스크는 정수 라벨맵이므로 NEAREST만 써야 함)
            # 필요하면 반대로 mask를 이미지에 맞추려면:
            # mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Albumentations (image/mask 크기가 동일한 상태에서 적용)
        out = self.tf(image=img, mask=mask)
        img, mask = out["image"], out["mask"]

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img), torch.from_numpy(mask.astype(np.int64))
