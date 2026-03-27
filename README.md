# Deepfake Detector

![Deepfake Detector UI](./public/image.png)

A web-based forensic tool for detecting AI-manipulated facial images using computer vision.

Built by **Team Vincenzo** — Computer Vision Course Project, IIIT-Delhi.

---

## Overview

This system analyzes a given face image and classifies it as **REAL** or **FAKE**, targeting manipulation types such as face-swap, expression-swap, and attribute changes.

**Planned extension:** Pixel-level localization of manipulated regions.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, Tailwind CSS |
| Backend | FastAPI (Python) |
| Model | Xception-based CNN |
| Datasets | DFFD, ForgeryNet, FaceForensics++ |

---

## Project Structure

```
deepfake-detector/
├── app/                  # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/
│   ├── ImageUpload.tsx   # Drag & drop upload
│   └── ResultDisplay.tsx # Verdict + confidence meter
├── backend/
│   ├── main.py           # FastAPI server
│   ├── model.py          # Model inference (swap in trained weights here)
│   └── requirements.txt
├── next.config.ts
└── tailwind.config.ts
```

---

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+
- NVIDIA GPU (RTX 3060 or better recommended)

### Frontend

```bash
npm install
npm run dev
```

Runs on `http://localhost:3000`

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Runs on `http://localhost:8000`

The Next.js app proxies `/api/*` → `http://localhost:8000/*` automatically.

---

## Model

The model stub lives in `backend/model.py`. To plug in your trained model:

1. Save your weights to `backend/weights/`
2. Update `DeepfakeDetector.__init__()` to load the model
3. Update `DeepfakeDetector.predict()` with real inference logic

---

## Datasets

| Dataset | Real Images | Fake Images |
|---------|------------|-------------|
| DFFD | 58,703 | 240,336 |
| ForgeryNet | 1,438,201 | 1,457,861 |
| FaceForensics++ | Video-based (frame extraction) | — |

---

## Evaluation Metrics

- **Detection:** Accuracy, AUC-ROC, F1-Score
- **Localization:** IoU, Pixel Accuracy, Dice Coefficient
- **Robustness:** Cross-dataset generalization, compression levels
- **Efficiency:** Inference time (ms), model size (MB)

---

## Team Vincenzo

| Name | ID |
|------|----|
| Balaiah Tarun | 2022132 |
| Md Kaif | 2022289 |
| Nishant Kumar Singh | 2022327 |
| Vaishvi Verma | 2022609 |
