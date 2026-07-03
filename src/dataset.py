"""
PyTorch Dataset for turf segmentation tiles.

Augmentation philosophy: with only 3 source scenes and an evaluation set
from an unknown, likely very different location, the dominant risk is the
model memorizing THIS dataset's color palette / sensor characteristics
rather than learning a transferable notion of "turf". So augmentation here
is weighted toward color/lighting jitter (simulating different cameras,
seasons, times of day) over purely geometric augmentation, though both are
included.
"""
import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size=512):
    return A.Compose([
        A.RandomCrop(img_size, img_size) if img_size < 512 else A.NoOp(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT,
            p=0.5,
        ),
        # Heavier color/lighting augmentation than usual — this is the main
        # generalization lever given the domain-shift risk to unseen imagery.
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.05, p=1.0),
        ], p=0.9),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(std_range=(0.04, 0.18), p=1.0),
            A.ImageCompression(quality_range=(60, 100), p=1.0),
        ], p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 40),
            hole_width_range=(16, 40),
            p=0.2,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class TurfDataset(Dataset):
    def __init__(self, manifest_path, split="train", transform=None):
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.entries = [m for m in manifest if m["split"] == split]
        self.transform = transform or (get_train_transform() if split == "train" else get_val_transform())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = cv2.cvtColor(cv2.imread(entry["image_path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=img, mask=mask)
        return augmented["image"], augmented["mask"].unsqueeze(0), entry["tile_id"]
