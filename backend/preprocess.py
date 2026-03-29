import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FF_ROOT         = "o++_C23"
FAKE_FOLDER     = "Deepfakes"
REAL_FOLDER     = "original"
OUTPUT_FACE_DIR = "faces"

FRAME_INTERVAL      = 10
FACE_SIZE           = 224
MARGIN              = 0.20
MIN_CONFIDENCE      = 0.90
MAX_FRAMES_PER_VID  = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
mtcnn = MTCNN(keep_all=False, device=device, min_face_size=40)

# ─────────────────────────────────────────────
# FACE DETECTION + CROP
# ─────────────────────────────────────────────
def detect_and_crop_face(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None

    best_idx = int(np.argmax(probs))
    if probs[best_idx] < MIN_CONFIDENCE:
        return None

    x1, y1, x2, y2 = boxes[best_idx]
    h, w = frame_bgr.shape[:2]

    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * MARGIN), int(bh * MARGIN)

    x1 = max(0, int(x1) - mx)
    y1 = max(0, int(y1) - my)
    x2 = min(w, int(x2) + mx)
    y2 = min(h, int(y2) + my)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE))

# ─────────────────────────────────────────────
# PROCESS VIDEO
# ─────────────────────────────────────────────
def process_video(video_path: str, label: str):
    video_name = Path(video_path).stem
    save_dir = Path(OUTPUT_FACE_DIR) / label / video_name

    # skip if already processed
    if save_dir.exists() and any(save_dir.glob("*.png")):
        return 0, 0

    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0

    frame_idx = 0
    saved = 0
    skipped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if saved >= MAX_FRAMES_PER_VID:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            face_crop = detect_and_crop_face(frame)
            if face_crop is not None:
                out_path = save_dir / f"frame_{frame_idx:05d}.png"
                cv2.imwrite(str(out_path), face_crop)
                saved += 1
            else:
                skipped += 1
        frame_idx += 1

    cap.release()
    return saved, skipped

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run(label_folder_pairs):
    total_saved = 0
    total_skipped = 0

    for label, folder in label_folder_pairs:
        video_dir = Path(FF_ROOT) / folder
        videos = list(video_dir.glob("*.mp4"))
        print(f"\n[{label.upper()}] {len(videos)} videos in '{folder}/'")

        for vp in tqdm(videos, desc=f"  {label}"):
            s, sk = process_video(str(vp), label)
            total_saved   += s
            total_skipped += sk

    print(f"\nDone. Faces saved: {total_saved}  |  Frames skipped (no face): {total_skipped}")
    print(f"  Output: {OUTPUT_FACE_DIR}/")

if __name__ == "__main__":
    run([
        ("fake", FAKE_FOLDER),
        ("real", REAL_FOLDER),
    ])
