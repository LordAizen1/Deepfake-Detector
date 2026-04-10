# Deepfake Detection System
### Findings & Documentation

**Authors:** Balaiah Tarun (2022132), Md Kaif (2022289), Nishant Kumar Singh (2022327), Vaishvi Verma (2022609)
**Team Vincenzo** | Computer Vision Course Project | IIIT-Delhi | March 2026

---

## 1. Project Overview

A web-based forensic tool that detects AI-manipulated (deepfake) facial images using an EfficientNet-B4 CNN, trained on the FaceForensics++ dataset. The system provides binary classification (REAL/FAKE) with confidence scores and Grad-CAM localization heatmaps.

### Tech Stack

| Component | Technology |
|---|---|
| Frontend | Next.js 15, Tailwind CSS |
| Backend | FastAPI (Python) |
| Model | EfficientNet-B4 (via `timm`) |
| Face Detection (Preprocessing) | MTCNN (facenet-pytorch) |
| Face Detection (Inference) | OpenCV Haar Cascade |
| Experiment Tracking | Weights & Biases (`W&B`) |
| Training Infrastructure | IIITD Precision Cluster (SLURM) |

---

## 2. Dataset

**FaceForensics++ (C23 compression)**
Source: Kaggle (`xdxd003/ff-c23`), Format: MP4 videos, Size: ~17.9 GB

| Category | Manipulation Type | Videos |
|---|---|---|
| Original (Real) | --- | 1,000 |
| Deepfakes | Face swap | 1,000 |
| Face2Face | Expression transfer | 1,000 |
| FaceSwap | Face swap | 1,000 |
| FaceShifter | Face swap | 1,000 |
| NeuralTextures | Neural rendering | 1,000 |
| DeepFakeDetection | Face swap | 1,000 |

*Only `Deepfakes/` and `original/` were used for training in the current iteration.*

---

## 3. Preprocessing Pipeline

The preprocessing pipeline (`backend/preprocess.py`) extracts face crops from raw videos:

1. Read MP4 videos using OpenCV
2. Extract every 10th frame (`FRAME_INTERVAL = 10`)
3. Run MTCNN face detection (PyTorch, GPU-accelerated) on each frame
4. Crop detected face with 20% margin around bounding box
5. Resize to 224 × 224 pixels
6. Save as PNG to `faces/real/` or `faces/fake/` (organized by video name)

| Parameter | Value |
|---|---|
| Frame interval | Every 10th frame |
| Face size | 224 × 224 |
| Bounding box margin | 20% |
| Min detection confidence | 0.90 |
| Max faces per video | 20 |

*Table: Preprocessing configuration.*

| Split | Frames |
|---|---|
| Train (70%) | 28,020 |
| Val (15%) | 6,017 |
| Test (15%) | 6,025 |
| **Total** | **40,062** |

*Table: Dataset split sizes.*

### Key Design Decision: Video-Level Split

The train/val/test split is performed at the **video level**, not the frame level. This prevents data leakage — frames from the same video never appear in both training and test sets.

---

## 4. Data Augmentation

Applied during training only (not validation/test):

| Augmentation | Parameters |
|---|---|
| Random horizontal flip | 50% probability |
| Color jitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 |
| Random rotation | up to 10 degrees |
| JPEG compression simulation | quality 50–95, 50% probability |
| ImageNet normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

*JPEG compression simulation is particularly important for FaceForensics++ generalization, as the dataset uses C23 (quality 23) compression.*

---

## 5. Model Architecture

**EfficientNet-B4** (pretrained on ImageNet, loaded via `timm`):
- Classifier head replaced with: `Dropout(0.4)` → `Linear(in_features, 1)`
- Single sigmoid output for binary classification
- Loss function: `BCEWithLogitsLoss`

### Training Strategy: Freeze → Unfreeze

| Phase | Epochs | What's Trained | Learning Rate |
|---|---|---|---|
| Head-only | 1–3 | Classifier head only | 1×10⁻³ (10× base) |
| Full fine-tune | 4–20 | All parameters | 1×10⁻⁴ with cosine annealing |

*Table: Two-phase training strategy.*

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Base learning rate | 1×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 20 |
| Scheduler | CosineAnnealingLR (η_min = 1×10⁻⁶) |
| Dropout | 0.4 |

---

## 6. Training Results

**Trained on:** NVIDIA A100-SXM4-40GB (IIITD Precision Cluster, gpu04)
**Training time:** ~1.5 hours (20 epochs)

### Training Curves

> *Training curves logged via Weights & Biases. Note the sharp improvement at epoch 3→4 when the backbone is unfrozen for full fine-tuning.*
>
![Training Loss](public/train_loss.png)
![Training Accuracy](public/train_accuracy.png)
![Val Loss](public/val_loss.png)
![Val Accuracy](public/val_accuracy.png)

### Final Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| Loss | 0.0110 | 0.1924 | 0.1207 |
| Accuracy | 99.5% | 95.9% | **96.1%** |
| F1-Score | 99.5% | 95.7% | **96.3%** |
| AUC-ROC | 99.99% | 99.3% | **99.5%** |

### Observations

- **Strong performance:** 96.1% test accuracy and 99.5% AUC on held-out test set.
- **Slight overfitting:** Train accuracy (99.5%) vs test (96.1%) shows a ~3.4% gap, which is expected and acceptable for this dataset size.
- **High AUC:** 99.5% AUC indicates excellent discriminative ability — the model ranks fake images higher than real ones almost perfectly.
- **Convergence:** Loss stabilized around epoch 15, with diminishing returns in later epochs.
- **Freeze→Unfreeze impact:** The sharp jump at epoch 3→4 validates the two-phase training strategy. Head-only training provides a warm start before full fine-tuning.

---

## 7. Inference Pipeline

The inference pipeline (`backend/model.py`) processes images as follows:

1. **Face validation:** OpenCV Haar Cascade checks for human face presence (rejects non-face images such as cats, landscapes, etc.)
2. **Preprocessing:** Resize to 224 × 224, normalize with ImageNet statistics
3. **Grad-CAM:** Hooks on the last convolutional layer (`conv_head`) capture feature maps and gradients to generate a localization heatmap
4. **Inference:** Forward pass through EfficientNet-B4, sigmoid on logit
5. **Output:** Label (REAL/FAKE), confidence score, individual probabilities, and Grad-CAM heatmap overlay

### Grad-CAM Localization

Grad-CAM (Gradient-weighted Class Activation Mapping) generates visual explanations for the model's predictions by:

- Computing gradients of the output logit with respect to the final convolutional layer's feature maps
- Global average pooling the gradients to obtain channel-wise importance weights
- Computing a weighted sum of the feature maps, followed by ReLU activation
- Upsampling the resulting heatmap and overlaying it on the original image

This highlights the facial regions the model considers most important for its REAL/FAKE decision, providing interpretability and approximate manipulation localization.

### Inference Speed

- CPU (laptop): ~100–200ms per image
- GPU: ~10–20ms per image

### Video Inference Pipeline

Video analysis was added as an extension to support temporal deepfake detection. The pipeline (`/predict_video` endpoint) works as follows:

1. Write video bytes to a temporary file on disk
2. Open with OpenCV `VideoCapture` and determine FPS
3. Sample every 10th frame (adjustable), up to a maximum of 20 face-containing frames
4. For each sampled frame, run Haar Cascade face detection with strict parameters (`minNeighbors=10`); skip frames with no face
5. Run EfficientNet-B4 inference on each valid frame
6. Track the *most decisive frame* — the one with fake probability furthest from 0.5 (highest confidence in either direction)
7. Aggregate: average `fake_prob` across all face frames; classify as FAKE if average > 0.5
8. Generate Grad-CAM on the most decisive frame
9. Return: label, confidence, per-frame results (frame index, timestamp, fake probability), frames analyzed count

A minimum of 3 face-containing frames is required to produce a result; fewer than 3 returns an error, preventing single false-positive face detections from yielding spurious predictions.

| Parameter | Value |
|---|---|
| Frame sampling interval | Every 10th frame |
| Max face frames analyzed | 20 |
| Min face frames required | 3 |
| Haar Cascade minNeighbors | 10 (stricter than image mode: 8) |
| Video size limit | 200 MB |

---

## 8. Limitations & Known Issues

### Model Limitations

1. **Cross-manipulation generalization gap (critical)** — the model achieves 96.1% on the FF++ `Deepfakes` test set but fails to generalize to other manipulation types within the same dataset. Quantitative evaluation on all four unseen FF++ manipulation types (10,000 samples each, using identical MTCNN preprocessing):

| Manipulation Type | Samples | Accuracy | F1 | Detected% |
|---|---|---|---|---|
| Face2Face | 10,000 | 4.35% | 0.083 | 4.3% |
| FaceSwap | 10,000 | 1.77% | 0.035 | 1.8% |
| FaceShifter | 10,000 | 9.59% | 0.175 | 9.6% |
| NeuralTextures | 10,000 | 6.67% | 0.125 | 6.7% |
| Real (original) | 2,000 | 98.45% | --- | FP: 1.6% |
| **Trained on: Deepfakes (test set)** | 6,025 | **96.1%** | **0.963** | --- |

   The real-face accuracy (98.45%) staying high confirms the model learned genuine signal — it is not predicting REAL for everything. The failure is specific to artifact type: the model learned face-swap blending artifacts from `Deepfakes/` that are absent in expression-transfer (Face2Face), neural rendering (NeuralTextures), and other face-swap variants (FaceSwap, FaceShifter) which use different pipelines.

2. **Only trained on Deepfakes manipulation type** — the model has only seen face-swap deepfakes from the `Deepfakes/` folder. It has not been trained on Face2Face, FaceSwap, FaceShifter, or NeuralTextures manipulations despite all being present in FF++.

3. **GAN and diffusion-generated faces not detected** — images from tools like StyleGAN (`thispersondoesnotexist.com`) or diffusion-based face generators operate differently from face-swap deepfakes and are not reliably detected.

4. **Compression sensitivity** — trained on C23 compression; performance may degrade on heavily compressed (social media) or uncompressed images.

5. **Single-face assumption** — the inference pipeline processes the entire image rather than individual face crops. Multi-face images may produce unreliable results.

### Infrastructure Findings

1. **ONNX Runtime GPU incompatible with Precision cluster** — InsightFace (`onnxruntime-gpu`) could not be made to work due to cuDNN version mismatches across nodes. Solved by switching to facenet-pytorch (MTCNN) which uses PyTorch CUDA directly.
2. **MIG partitions (gpu01 short queue) cause cuDNN init failures** — use full GPU allocations on medium/long queues instead.
3. **cuDNN version conflict** — PyTorch requires cuDNN 8, `onnxruntime-gpu` 1.18+ requires cuDNN 9. Installing both simultaneously breaks one or the other.

---

## 9. Extensions Implemented

The following improvements were made beyond the original proposal based on TA feedback:

1. **Video detection** *(implemented)* — the system now accepts video files (.mp4, .mov, .webm). Frames are sampled, face-validated, and individually classified; results are aggregated and a Grad-CAM heatmap from the most decisive frame is returned. See Section 7.2 for the full pipeline.

2. **Multi-media UI** *(implemented)* — the frontend accepts both images and videos via drag-and-drop. A video preview is shown during analysis, and per-frame count is displayed in results.

3. **Cross-manipulation generalization evaluation** *(implemented)* — the trained model was evaluated on all four unseen FF++ manipulation types with no retraining. Results are reported in Section 8.1 and demonstrate significant generalization failure, confirming the model's sensitivity to manipulation-specific artifact patterns.

4. **Adversarial robustness testing** *(implemented)* — FGSM (Fast Gradient Sign Method) adversarial attacks are integrated directly into the web interface. After obtaining a prediction on an image, the user can apply a perturbation of adjustable strength (ε) and observe whether the model's prediction changes. The perturbed image is visually indistinguishable from the original but may cause the model to flip its verdict, demonstrating the model's vulnerability to adversarial inputs.

### FGSM Adversarial Attack Implementation

Given an input image *x* with predicted label *y*, the FGSM perturbation is computed as:
where `y_target` is the opposite of the model's original prediction (maximising fooling), and ε controls the perturbation magnitude. The gradient is computed with respect to the input tensor, not the model weights. The `/adversarial` endpoint accepts ε ∈ [0.001, 0.1] and returns both the original and adversarial predictions along with the perturbed image for visual comparison.

---

## 10. Future Improvements

1. **Cross-dataset training** — include Celeb-DF v2, DFDC, or ForgeryNet to close the generalization gap identified in Section 8.1.
2. **Train on all FF++ manipulation types** — include Face2Face, FaceSwap, FaceShifter, NeuralTextures alongside the currently used Deepfakes subset.
3. **Adversarial training** — retrain with adversarially perturbed examples mixed into the training set to improve robustness against FGSM/PGD attacks.
4. **Enhanced localization** — train a segmentation head for pixel-level manipulation masks instead of relying on Grad-CAM approximations.
5. **Face cropping at inference** — use MTCNN to crop the face before running the classifier, matching the training pipeline more closely.
6. **Model compression** — distillation or pruning for faster inference on edge devices.
