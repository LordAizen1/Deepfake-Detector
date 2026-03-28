import os
import random
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

FACE_DIR = "faces"      # output of preprocess.py
SEED     = 42
SPLIT    = {"train": 0.70, "val": 0.15, "test": 0.15}


# ─────────────────────────────────────────────
# VIDEO-LEVEL SPLIT
# (never let frames from the same video appear
#  in both train and test — avoids data leakage)
# ─────────────────────────────────────────────
def get_video_level_splits(face_dir: str, split: dict, seed: int = 42):
    """
    Returns three dicts: train_files, val_files, test_files.
    Each is a list of (image_path, label) tuples.
    Split is done per video folder, not per frame.
    """
    random.seed(seed)

    all_videos = []   # list of (video_folder_path, label)
    for label_name, label_idx in [("real", 0), ("fake", 1)]:
        label_dir = Path(face_dir) / label_name
        if not label_dir.exists():
            continue
        for video_dir in sorted(label_dir.iterdir()):
            if video_dir.is_dir():
                all_videos.append((video_dir, label_idx))

    random.shuffle(all_videos)
    n = len(all_videos)
    n_train = int(n * split["train"])
    n_val   = int(n * split["val"])

    train_vids = all_videos[:n_train]
    val_vids   = all_videos[n_train : n_train + n_val]
    test_vids  = all_videos[n_train + n_val :]

    def expand(video_list):
        files = []
        for video_dir, label in video_list:
            for img_path in sorted(video_dir.glob("*.png")):
                files.append((str(img_path), label))
        return files

    return expand(train_vids), expand(val_vids), expand(test_vids)


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class FaceForensicsDataset(Dataset):
    def __init__(self, file_list: list, transform=None):
        """
        file_list: list of (image_path, label) tuples
        label: 0 = real, 1 = fake
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    # Training: augment aggressively — the model must be robust
    # to compression, lighting, and orientation changes
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        # simulate JPEG compression artifacts (key for FF++ generalization)
        transforms.RandomApply([
            transforms.Lambda(lambda img: simulate_jpeg(img, quality=random.randint(50, 95)))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])

    # Val/Test: no augmentation, just normalize
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


def simulate_jpeg(pil_img, quality=75):
    """Simulate JPEG compression by encode→decode in memory."""
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ─────────────────────────────────────────────
# DATALOADERS — call this from your train script
# ─────────────────────────────────────────────
def get_dataloaders(face_dir=FACE_DIR, batch_size=32, num_workers=4):
    train_files, val_files, test_files = get_video_level_splits(face_dir, SPLIT, SEED)

    print(f"  Train frames : {len(train_files)}")
    print(f"  Val frames   : {len(val_files)}")
    print(f"  Test frames  : {len(test_files)}")

    train_tf, eval_tf = get_transforms()

    train_ds = FaceForensicsDataset(train_files, transform=train_tf)
    val_ds   = FaceForensicsDataset(val_files,   transform=eval_tf)
    test_ds  = FaceForensicsDataset(test_files,  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    imgs, labels = next(iter(train_loader))
    print(f"\nBatch shape : {imgs.shape}")    # [32, 3, 224, 224]
    print(f"Labels      : {labels[:8]}")      # mix of 0.0 and 1.0


# **Expected output after running `dataset.py`:**
# ```
#   Train frames : ~11200   (70% of videos × ~16 frames avg)
#   Val frames   : ~2400
#   Test frames  : ~2400

# Batch shape : torch.Size([32, 3, 224, 224])
# Labels      : tensor([1., 0., 1., 1., 0., 1., 0., 1.])