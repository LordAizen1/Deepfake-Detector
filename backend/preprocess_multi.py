import sys
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
FF_ROOT            = "o++_C23"
OUTPUT_FACE_DIR    = "faces"

FRAME_INTERVAL     = 10
FACE_SIZE          = 224
MARGIN             = 0.20
MIN_CONFIDENCE     = 0.90
MAX_FRAMES_PER_VID = 20

MANIP_TYPES = {
    "face2face":     "Face2Face",
    "faceswap":      "FaceSwap",
    "faceshifter":   "FaceShifter",
    "neuraltextures":"NeuralTextures",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
mtcnn = MTCNN(keep_all=False, device=device, min_face_size=40)

# ─────────────────────────────────────────────
# FACE DETECTION + CROP  (mirrors preprocess.py)
# ─────────────────────────────────────────────
def detect_and_crop_face(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)

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
# PROCESS ONE VIDEO
# ─────────────────────────────────────────────
def process_video(video_path: str, save_dir: Path):
    if save_dir.exists() and any(save_dir.glob("*.png")):
        return 0, 0  # skip-if-exists

    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0

    frame_idx = 0
    saved     = 0
    skipped   = 0

    while True:
        ret, frame = cap.read()
        if not ret or saved >= MAX_FRAMES_PER_VID:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            face = detect_and_crop_face(frame)
            if face is not None:
                cv2.imwrite(str(save_dir / f"frame_{frame_idx:05d}.png"), face)
                saved += 1
            else:
                skipped += 1
        frame_idx += 1

    cap.release()
    return saved, skipped

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in MANIP_TYPES:
        print(f"Usage: python preprocess_multi.py <type>")
        print(f"Types: {', '.join(MANIP_TYPES.keys())}")
        sys.exit(1)

    manip_key    = sys.argv[1]                  # e.g. "face2face"
    folder_name  = MANIP_TYPES[manip_key]       # e.g. "Face2Face"
    output_label = f"fake_{manip_key}"          # e.g. "fake_face2face"

    video_dir  = Path(FF_ROOT) / folder_name
    output_dir = Path(OUTPUT_FACE_DIR) / output_label
    videos     = list(video_dir.glob("*.mp4"))

    print(f"\n[{folder_name}] {len(videos)} videos → {output_dir}/")

    total_saved   = 0
    total_skipped = 0

    for vp in tqdm(videos, desc=f"  {folder_name}"):
        video_name = Path(vp).stem
        save_dir   = output_dir / video_name
        s, sk      = process_video(str(vp), save_dir)
        total_saved   += s
        total_skipped += sk

    print(f"\nDone. Saved: {total_saved}  |  Skipped (no face): {total_skipped}")
    print(f"Output: {output_dir}/")
