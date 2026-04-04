#!/usr/bin/env python3
"""
evaluate_generalization.py

Evaluates the trained model on unseen FF++ manipulation types to measure
cross-manipulation generalization. No retraining — inference only.

Unseen types: Face2Face, FaceSwap, FaceShifter, NeuralTextures
Real faces:   loaded from existing faces/real/ (already MTCNN-preprocessed)
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import timm
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
from facenet_pytorch import MTCNN
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT         = "checkpoints/best_model.pth"
FF_ROOT            = "o++_C23"
REAL_FACES_DIR     = "faces/real"
MANIP_TYPES        = ["Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRAME_INTERVAL     = 10      # same as preprocess.py
MAX_FRAMES_PER_VID = 10      # faces extracted per video
MIN_CONFIDENCE     = 0.90    # MTCNN detection threshold
MARGIN             = 0.20    # bounding box margin
FACE_SIZE          = 224
MAX_REAL_SAMPLES   = 2000    # cap on real face images used

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    model = timm.create_model("efficientnet_b4", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, 1))
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model

# ── Face extraction (mirrors preprocess.py exactly) ──────────────────────────
def extract_faces_from_video(video_path, mtcnn):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    faces = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or len(faces) >= MAX_FRAMES_PER_VID:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)

            boxes, probs = mtcnn.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                best_idx = int(np.argmax(probs))
                if probs[best_idx] >= MIN_CONFIDENCE:
                    x1, y1, x2, y2 = boxes[best_idx]
                    h, w  = frame.shape[:2]
                    bw    = x2 - x1
                    bh    = y2 - y1
                    mx    = int(bw * MARGIN)
                    my    = int(bh * MARGIN)
                    x1    = max(0, int(x1) - mx)
                    y1    = max(0, int(y1) - my)
                    x2    = min(w, int(x2) + mx)
                    y2    = min(h, int(y2) + my)
                    crop  = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(cv2.resize(crop, (FACE_SIZE, FACE_SIZE)), cv2.COLOR_BGR2RGB)
                        faces.append(Image.fromarray(crop_rgb))

        frame_idx += 1

    cap.release()
    return faces

# ── Batch inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, images, batch_size=32):
    probs = []
    for i in range(0, len(images), batch_size):
        batch   = images[i:i + batch_size]
        tensors = torch.stack([TRANSFORM(img) for img in batch]).to(DEVICE)
        logits  = model(tensors)
        p       = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
    return probs

# ── Evaluate one manipulation type (all fake) ─────────────────────────────────
def evaluate_fake(model, mtcnn, manip_type):
    from tqdm import tqdm

    video_dir = Path(FF_ROOT) / manip_type
    videos    = sorted(video_dir.glob("*.mp4"))

    probs = []
    for vpath in tqdm(videos, desc=f"  {manip_type}", ncols=80):
        faces = extract_faces_from_video(vpath, mtcnn)
        if faces:
            probs.extend(run_inference(model, faces))

    if not probs:
        return None

    labels = [1] * len(probs)
    preds  = [1 if p > 0.5 else 0 for p in probs]

    return {
        "samples":           len(probs),
        "accuracy":          round(accuracy_score(labels, preds), 4),
        "f1":                round(f1_score(labels, preds, zero_division=0), 4),
        "auc":               round(roc_auc_score(labels, probs), 4),
        "detected_fake_pct": round(sum(preds) / len(preds) * 100, 1),
    }

# ── Evaluate real faces (from existing preprocessed crops) ───────────────────
def evaluate_real(model):
    all_paths = list(Path(REAL_FACES_DIR).rglob("*.png"))
    random.shuffle(all_paths)
    all_paths = all_paths[:MAX_REAL_SAMPLES]

    images = [Image.open(p).convert("RGB") for p in all_paths]
    probs  = run_inference(model, images)
    labels = [0] * len(probs)
    preds  = [1 if p > 0.5 else 0 for p in probs]

    return {
        "samples":            len(probs),
        "accuracy":           round(accuracy_score(labels, preds), 4),
        "false_positive_pct": round(sum(preds) / len(preds) * 100, 1),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    print(f"Device: {DEVICE}\n")

    print("Loading model...")
    model = load_model()

    print("Loading MTCNN...")
    mtcnn = MTCNN(keep_all=False, device=DEVICE, min_face_size=40)

    results = {}

    print(f"\n{'=' * 68}")
    print(f"{'Manipulation Type':<22} {'Samples':>8} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'Detected%':>10}")
    print(f"{'=' * 68}")

    for manip in MANIP_TYPES:
        r = evaluate_fake(model, mtcnn, manip)
        if r:
            results[manip] = r
            print(f"{manip:<22} {r['samples']:>8} {r['accuracy']:>10.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['detected_fake_pct']:>9.1f}%")
        else:
            print(f"{manip:<22}   no faces extracted")

    print(f"\n  [Real] loading from faces/real/...", flush=True)
    real_r = evaluate_real(model)
    results["real_original"] = real_r
    print(f"{'Real (original)':<22} {real_r['samples']:>8} {real_r['accuracy']:>10.4f}{'':>8}{'':>8} FP rate: {real_r['false_positive_pct']}%")
    print(f"{'=' * 68}")

    out_path = "generalization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
