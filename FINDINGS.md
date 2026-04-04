\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{enumitem}
\usepackage{float}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{listings}

\geometry{a4paper, margin=1in}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10},
}

\title{\textbf{Deepfake Detection System} \\ \large Findings \& Documentation}
\author{
    Balaiah Tarun (2022132) \and Md Kaif (2022289) \and
    Nishant Kumar Singh (2022327) \and Vaishvi Verma (2022609) \\[1em]
    \textbf{Team Vincenzo} \\
    Computer Vision Course Project \\
    IIIT-Delhi
}
\date{March 2026}

\begin{document}

\maketitle

% ============================================================
\section{Project Overview}

A web-based forensic tool that detects AI-manipulated (deepfake) facial images using an EfficientNet-B4 CNN, trained on the FaceForensics++ dataset. The system provides binary classification (TEXTSC{REAL}/\textsc{FAKE}) with confidence scores and Grad-CAM localization heatmaps.

\subsection{Tech Stack}

\begin{table}[H]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Component} & \textbf{Technology} \\
\midrule
Frontend & Next.js 15, Tailwind CSS \\
Backend & FastAPI (Python) \\
Model & EfficientNet-B4 (via \texttt{timm}) \\
Face Detection (Preprocessing) & MTCNN (facenet-pytorch) \\
Face Detection (Inference) & OpenCV Haar Cascade \\
Experiment Tracking & Weights \& Biases (W\&B) \\
Training Infrastructure & IIITD Precision Cluster (SLURM) \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{Dataset}

\textbf{FaceForensics++ (C23 compression)} \\
Source: Kaggle (\texttt{xdxd003/ff-c23}), Format: MP4 videos, Size: $\sim$17.9 GB

\begin{table}[H]
\centering
\begin{tabular}{@{}llc@{}}
\toprule
\textbf{Category} & \textbf{Manipulation Type} & \textbf{Videos} \\
\midrule
Original (Real) & --- & 1,000 \\
Deepfakes & Face swap & 1,000 \\
Face2Face & Expression transfer & 1,000 \\
FaceSwap & Face swap & 1,000 \\
FaceShifter & Face swap & 1,000 \\
NeuralTextures & Neural rendering & 1,000 \\
DeepFakeDetection & Face swap & 1,000 \\
\bottomrule
\end{tabular}
\caption{FaceForensics++ dataset composition. Only \texttt{Deepfakes/} and \texttt{original/} were used for training in the current iteration.}
\end{table}

% ============================================================
\section{Preprocessing Pipeline}

The preprocessing pipeline (\texttt{backend/preprocess.py}) extracts face crops from raw videos:

\begin{enumerate}
    \item Read MP4 videos using OpenCV
    \item Extract every 10th frame (\texttt{FRAME\_INTERVAL = 10})
    \item Run MTCNN face detection (PyTorch, GPU-accelerated) on each frame
    \item Crop detected face with 20\% margin around bounding box
    \item Resize to $224 \times 224$ pixels
    \item Save as PNG to \texttt{faces/real/} or \texttt{faces/fake/} (organized by video name)
\end{enumerate}

\begin{table}[H]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Frame interval & Every 10th frame \\
Face size & $224 \times 224$ \\
Bounding box margin & 20\% \\
Min detection confidence & 0.90 \\
Max faces per video & 20 \\
\bottomrule
\end{tabular}
\caption{Preprocessing configuration.}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Split} & \textbf{Frames} \\
\midrule
Train (70\%) & 28,020 \\
Val (15\%) & 6,017 \\
Test (15\%) & 6,025 \\
\midrule
\textbf{Total} & \textbf{40,062} \\
\bottomrule
\end{tabular}
\caption{Dataset split sizes.}
\end{table}

\subsection{Key Design Decision: Video-Level Split}
The train/val/test split is performed at the \textbf{video level}, not the frame level. This prevents data leakage --- frames from the same video never appear in both training and test sets.

% ============================================================
\section{Data Augmentation}

Applied during training only (not validation/test):

\begin{table}[H]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Augmentation} & \textbf{Parameters} \\
\midrule
Random horizontal flip & 50\% probability \\
Color jitter & brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 \\
Random rotation & up to 10 degrees \\
JPEG compression simulation & quality 50--95, 50\% probability \\
ImageNet normalization & mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] \\
\bottomrule
\end{tabular}
\caption{Training augmentations. JPEG compression simulation is particularly important for FaceForensics++ generalization, as the dataset uses C23 (quality 23) compression.}
\end{table}

% ============================================================
\section{Model Architecture}

\textbf{EfficientNet-B4} (pretrained on ImageNet, loaded via \texttt{timm}):
\begin{itemize}
    \item Classifier head replaced with: \texttt{Dropout(0.4)} $\rightarrow$ \texttt{Linear(in\_features, 1)}
    \item Single sigmoid output for binary classification
    \item Loss function: \texttt{BCEWithLogitsLoss}
\end{itemize}

\subsection{Training Strategy: Freeze $\rightarrow$ Unfreeze}

\begin{table}[H]
\centering
\begin{tabular}{@{}llll@{}}
\toprule
\textbf{Phase} & \textbf{Epochs} & \textbf{What's Trained} & \textbf{Learning Rate} \\
\midrule
Head-only & 1--3 & Classifier head only & $1 \times 10^{-3}$ (10$\times$ base) \\
Full fine-tune & 4--20 & All parameters & $1 \times 10^{-4}$ with cosine annealing \\
\bottomrule
\end{tabular}
\caption{Two-phase training strategy.}
\end{table}

\subsection{Hyperparameters}

\begin{table}[H]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Optimizer & AdamW \\
Base learning rate & $1 \times 10^{-4}$ \\
Weight decay & $1 \times 10^{-4}$ \\
Batch size & 32 \\
Epochs & 20 \\
Scheduler & CosineAnnealingLR ($\eta_{\min} = 1 \times 10^{-6}$) \\
Dropout & 0.4 \\
\bottomrule
\end{tabular}
\caption{Training hyperparameters.}
\end{table}

% ============================================================
\section{Training Results}

\textbf{Trained on:} NVIDIA A100-SXM4-40GB (IIITD Precision Cluster, gpu04) \\
\textbf{Training time:} $\sim$1.5 hours (20 epochs)

\subsection{Training Curves}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{public/train_loss.png}
    \caption{Training Loss}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{public/train_accuracy.png}
    \caption{Training Accuracy}
\end{subfigure}

\vspace{1em}

\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{public/val_loss.png}
    \caption{Validation Loss}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{public/val_accuracy.png}
    \caption{Validation Accuracy}
\end{subfigure}
\caption{Training curves logged via Weights \& Biases. Note the sharp improvement at epoch 3$\rightarrow$4 when the backbone is unfrozen for full fine-tuning.}
\end{figure}

\subsection{Final Metrics}

\begin{table}[H]
\centering
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Metric} & \textbf{Train} & \textbf{Validation} & \textbf{Test} \\
\midrule
Loss & 0.0110 & 0.1924 & 0.1207 \\
Accuracy & 99.5\% & 95.9\% & \textbf{96.1\%} \\
F1-Score & 99.5\% & 95.7\% & \textbf{96.3\%} \\
AUC-ROC & 99.99\% & 99.3\% & \textbf{99.5\%} \\
\bottomrule
\end{tabular}
\caption{Final evaluation metrics across splits.}
\end{table}

\subsection{Observations}

\begin{itemize}
    \item \textbf{Strong performance:} 96.1\% test accuracy and 99.5\% AUC on held-out test set.
    \item \textbf{Slight overfitting:} Train accuracy (99.5\%) vs test (96.1\%) shows a $\sim$3.4\% gap, which is expected and acceptable for this dataset size.
    \item \textbf{High AUC:} 99.5\% AUC indicates excellent discriminative ability --- the model ranks fake images higher than real ones almost perfectly.
    \item \textbf{Convergence:} Loss stabilized around epoch 15, with diminishing returns in later epochs.
    \item \textbf{Freeze$\rightarrow$Unfreeze impact:} The sharp jump at epoch 3$\rightarrow$4 validates the two-phase training strategy. Head-only training provides a warm start before full fine-tuning.
\end{itemize}

% ============================================================
\section{Inference Pipeline}

The inference pipeline (\texttt{backend/model.py}) processes images as follows:

\begin{enumerate}
    \item \textbf{Face validation:} OpenCV Haar Cascade checks for human face presence (rejects non-face images such as cats, landscapes, etc.)
    \item \textbf{Preprocessing:} Resize to $224 \times 224$, normalize with ImageNet statistics
    \item \textbf{Grad-CAM:} Hooks on the last convolutional layer (\texttt{conv\_head}) capture feature maps and gradients to generate a localization heatmap
    \item \textbf{Inference:} Forward pass through EfficientNet-B4, sigmoid on logit
    \item \textbf{Output:} Label (\textsc{REAL}/\textsc{FAKE}), confidence score, individual probabilities, and Grad-CAM heatmap overlay
\end{enumerate}

\subsection{Grad-CAM Localization}

Grad-CAM (Gradient-weighted Class Activation Mapping) generates visual explanations for the model's predictions by:
\begin{itemize}
    \item Computing gradients of the output logit with respect to the final convolutional layer's feature maps
    \item Global average pooling the gradients to obtain channel-wise importance weights
    \item Computing a weighted sum of the feature maps, followed by ReLU activation
    \item Upsampling the resulting heatmap and overlaying it on the original image
\end{itemize}

This highlights the facial regions the model considers most important for its \textsc{REAL}/\textsc{FAKE} decision, providing interpretability and approximate manipulation localization.

\subsection{Inference Speed}
\begin{itemize}
    \item CPU (laptop): $\sim$100--200ms per image
    \item GPU: $\sim$10--20ms per image
\end{itemize}

\subsection{Video Inference Pipeline}

Video analysis was added as an extension to support temporal deepfake detection. The pipeline (\texttt{/predict\_video} endpoint) works as follows:

\begin{enumerate}
    \item Write video bytes to a temporary file on disk
    \item Open with OpenCV \texttt{VideoCapture} and determine FPS
    \item Sample every 10th frame (adjustable), up to a maximum of 20 face-containing frames
    \item For each sampled frame, run Haar Cascade face detection with strict parameters (\texttt{minNeighbors=10}); skip frames with no face
    \item Run EfficientNet-B4 inference on each valid frame
    \item Track the \textit{most decisive frame} --- the one with fake probability furthest from 0.5 (highest confidence in either direction)
    \item Aggregate: average \texttt{fake\_prob} across all face frames; classify as \textsc{FAKE} if average $> 0.5$
    \item Generate Grad-CAM on the most decisive frame
    \item Return: label, confidence, per-frame results (frame index, timestamp, fake probability), frames analyzed count
\end{enumerate}

A minimum of 3 face-containing frames is required to produce a result; fewer than 3 returns an error, preventing single false-positive face detections from yielding spurious predictions.

\begin{table}[H]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Frame sampling interval & Every 10th frame \\
Max face frames analyzed & 20 \\
Min face frames required & 3 \\
Haar Cascade minNeighbors & 10 (stricter than image mode: 8) \\
Video size limit & 200 MB \\
\bottomrule
\end{tabular}
\caption{Video inference configuration.}
\end{table}

% ============================================================
\section{Limitations \& Known Issues}

\subsection{Model Limitations}

\begin{enumerate}
    \item \textbf{Cross-dataset generalization gap (critical)} --- the model achieves 96.1\% on the FF++ test set but fails to detect high-quality modern deepfakes. Qualitative testing on a well-known deepfake video (public, circa 2023--2024) produced a confidence of 97\% \textsc{REAL} --- a complete misclassification. This is a fundamental limitation of training on a single dataset: the model learns the artifact patterns specific to FF++ (early FaceSwap/DeepFaceLab compression artifacts from $\sim$2018--2019) and does not generalize to newer generation methods. This gap is documented extensively in the literature; Celeb-DF \cite{celeb-df} was specifically designed to expose it.

    \item \textbf{Only trained on Deepfakes manipulation type} --- the model has only seen face-swap deepfakes from the \texttt{Deepfakes/} folder. It has not been trained on Face2Face, FaceSwap, FaceShifter, or NeuralTextures manipulations despite all being present in FF++.

    \item \textbf{GAN and diffusion-generated faces not detected} --- images from tools like StyleGAN (\texttt{thispersondoesnotexist.com}) or diffusion-based face generators operate differently from face-swap deepfakes and are not reliably detected.

    \item \textbf{Compression sensitivity} --- trained on C23 compression; performance may degrade on heavily compressed (social media) or uncompressed images.

    \item \textbf{Single-face assumption} --- the inference pipeline processes the entire image rather than individual face crops. Multi-face images may produce unreliable results.
\end{enumerate}

\subsection{Infrastructure Findings}

\begin{enumerate}
    \item \textbf{ONNX Runtime GPU incompatible with Precision cluster} --- InsightFace (\texttt{onnxruntime-gpu}) could not be made to work due to cuDNN version mismatches across nodes. Solved by switching to facenet-pytorch (MTCNN) which uses PyTorch CUDA directly.
    \item \textbf{MIG partitions (gpu01 short queue) cause cuDNN init failures} --- use full GPU allocations on medium/long queues instead.
    \item \textbf{cuDNN version conflict} --- PyTorch requires cuDNN 8, \texttt{onnxruntime-gpu} 1.18+ requires cuDNN 9. Installing both simultaneously breaks one or the other.
\end{enumerate}

% ============================================================
\section{Extensions Implemented}

The following improvements were made beyond the original proposal based on TA feedback:

\begin{enumerate}
    \item \textbf{Video detection} (\textit{implemented}) --- the system now accepts video files (.mp4, .mov, .webm). Frames are sampled, face-validated, and individually classified; results are aggregated and a Grad-CAM heatmap from the most decisive frame is returned. See Section~7.2 for the full pipeline.
    \item \textbf{Multi-media UI} (\textit{implemented}) --- the frontend now accepts both images and videos via drag-and-drop. A video preview is shown during analysis, and per-frame count is displayed in results.
\end{enumerate}

% ============================================================
\section{Future Improvements}

\begin{enumerate}
    \item \textbf{Cross-dataset training} --- include Celeb-DF v2, DFDC, or ForgeryNet to close the generalization gap identified in Section~8.1. The current model's failure on modern deepfakes is directly attributable to training on a single-source, older-generation dataset.
    \item \textbf{Train on all FF++ manipulation types} --- include Face2Face, FaceSwap, FaceShifter, NeuralTextures alongside the currently used Deepfakes subset.
    \item \textbf{Adversarial robustness evaluation} --- apply FGSM/PGD perturbations to test images and measure how model confidence degrades; add optional adversarial mode to the UI.
    \item \textbf{Enhanced localization} --- train a segmentation head for pixel-level manipulation masks instead of relying on Grad-CAM approximations.
    \item \textbf{Face cropping at inference} --- use MTCNN to crop the face before running the classifier, matching the training pipeline more closely.
    \item \textbf{Model compression} --- distillation or pruning for faster inference on edge devices.
\end{enumerate}

% % ============================================================
% \section{Compute \& Cost Summary}

% \begin{table}[H]
% \centering
% \begin{tabular}{@{}lllllr@{}}
% \toprule
% \textbf{Task} & \textbf{Queue} & \textbf{Node} & \textbf{GPU} & \textbf{Time} & \textbf{Tokens} \\
% \midrule
% Preprocess (fake videos) & medium & gpu04 & A100 40GB & $\sim$40 min & 0.5 \\
% Preprocess (real, partial) & medium & gpu04 & A100 40GB & $\sim$20 min & 0.5 \\
% Preprocess (real, remaining) & medium & gpu04 & A100 40GB & $\sim$25 min & 0.5 \\
% Training (20 epochs) & medium & gpu04 & A100 40GB & $\sim$1.5 hr & 0.5 \\
% \midrule
% \multicolumn{5}{l}{\textbf{Total tokens (successful runs)}} & $\sim$2.0 \\
% \bottomrule
% \end{tabular}
% \caption{Compute cost summary on IIITD Precision Cluster.}
% \end{table}

\end{document}
