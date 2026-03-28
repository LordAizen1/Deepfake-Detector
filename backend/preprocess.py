import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis


# ─────────────────────────────────────────────
# CONFIG — adjust these to your actual paths
# ─────────────────────────────────────────────
FF_ROOT         = "o++_C23"       # root of your dataset
FAKE_FOLDER     = "Deepfakes"                 # only Deepfakes for now
REAL_FOLDER     = "original"
OUTPUT_FACE_DIR = "faces"                     # final output — feed this to PyTorch

FRAME_INTERVAL      = 10       # every 10th frame
FACE_SIZE           = 224
MARGIN              = 0.20     # 20% margin around bbox
MIN_CONFIDENCE      = 0.90
MAX_FRAMES_PER_VID  = 20       # cap per video to keep dataset balanced

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)  # use CPU if no GPU
# ─────────────────────────────────────────────
# STEP 1 — extract frames + crop faces in one pass
# (avoids saving intermediate raw frames to disk)
# ─────────────────────────────────────────────
def process_video(video_path: str, label: str):
    """
    Extract every Nth frame, run RetinaFace, save 224x224 face crop.
    Returns (saved, skipped) counts.
    """
    video_name = Path(video_path).stem
    save_dir = Path(OUTPUT_FACE_DIR) / label / video_name
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

        # stop early if we've saved enough from this video
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


def detect_and_crop_face(img):
    faces = app.get(img)

    if len(faces) == 0:
        return None

    # pick highest confidence face
    best = max(faces, key=lambda f: f.det_score)
    if best.det_score < MIN_CONFIDENCE:
        return None

    x1, y1, x2, y2 = map(int, best.bbox)
    h, w = img.shape[:2]

    # margin
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * MARGIN), int(bh * MARGIN)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE))

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

    print(f"\n✓ Done. Faces saved: {total_saved}  |  Frames skipped (no face): {total_skipped}")
    print(f"  Output: {OUTPUT_FACE_DIR}/")


if __name__ == "__main__":
    run([
        ("fake", FAKE_FOLDER),
        ("real", REAL_FOLDER),
    ])