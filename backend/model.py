import os
import base64
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import io

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepfakeDetector:
    def __init__(self):
        model = timm.create_model("efficientnet_b4", pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1),
        )

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE)
        model.eval()
        self.model = model

        # Hook storage for Grad-CAM
        self._features = None
        self._gradients = None

        # Hook on last MBConv block (14×14 feature maps, better localization than conv_head's 7×7)
        self.model.blocks[-1].register_forward_hook(self._save_features)
        self.model.blocks[-1].register_full_backward_hook(self._save_gradients)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        print(f"Model loaded from {CHECKPOINT_PATH} (device: {DEVICE})")

    def _save_features(self, module, input, output):
        self._features = output

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def detect_face(self, image_bytes: bytes) -> bool:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
        return len(faces) > 0

    def _generate_gradcam(self, tensor: torch.Tensor, img_pil: Image.Image) -> str:
        """Generate Grad-CAM heatmap and return as base64 PNG."""
        tensor.requires_grad_(True)
        logit = self.model(tensor)

        self.model.zero_grad()
        logit.backward()

        # Grad-CAM: global avg pool gradients, weight feature maps
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self._features).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # Threshold: keep only top 15% of activations for tighter localization
        threshold = np.percentile(cam, 85)
        cam = np.where(cam >= threshold, cam - threshold, 0)

        # Normalize to 0-255
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = (cam * 255).astype(np.uint8)

        # Resize to match original image
        cam_resized = cv2.resize(cam, (224, 224))

        # Apply colormap
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

        # Overlay on original image
        orig = np.array(img_pil.resize((224, 224)))
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_bgr, 0.5, heatmap, 0.5, 0)

        # Encode to base64 PNG
        _, buf = cv2.imencode(".png", overlay)
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def predict(self, image_bytes: bytes) -> dict:
        if not self.detect_face(image_bytes):
            return {"error": "No human face detected in the image"}

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)

        # Run Grad-CAM (needs gradients, so no torch.no_grad)
        heatmap_b64 = self._generate_gradcam(tensor.clone(), img)

        # Run prediction (no grad needed)
        with torch.no_grad():
            logit = self.model(tensor)
            fake_prob = float(torch.sigmoid(logit).squeeze())
            real_prob = 1.0 - fake_prob
            label = "FAKE" if fake_prob > 0.5 else "REAL"

        return {
            "label": label,
            "confidence": round(max(fake_prob, real_prob), 4),
            "fake_prob": round(fake_prob, 4),
            "real_prob": round(real_prob, 4),
            "heatmap": heatmap_b64,
        }

    def predict_video(self, video_bytes: bytes, sample_every: int = 10, max_frames: int = 20) -> dict:
        """Sample frames from video, run per-frame prediction, aggregate results."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return {"error": "Could not open video file"}

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            frame_results = []
            best_frame_pil = None
            best_frame_tensor = None
            best_decisiveness = -1.0
            frame_idx = 0
            analyzed = 0

            while cap.isOpened() and analyzed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=10, minSize=(80, 80)
                    )

                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        margin = int(0.2 * max(w, h))
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(frame.shape[1], x + w + margin)
                        y2 = min(frame.shape[0], y + h + margin)
                        face_crop = frame[y1:y2, x1:x2]
                        img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        tensor = self.transform(img).unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            logit = self.model(tensor)
                            fake_prob = float(torch.sigmoid(logit).squeeze())

                        decisiveness = abs(fake_prob - 0.5)
                        if decisiveness > best_decisiveness:
                            best_decisiveness = decisiveness
                            best_frame_pil = img
                            best_frame_tensor = tensor.clone()

                        frame_results.append({
                            "frame": frame_idx,
                            "time_sec": round(frame_idx / fps, 2),
                            "fake_prob": round(fake_prob, 4),
                        })
                        analyzed += 1

                frame_idx += 1

            cap.release()
        finally:
            os.unlink(tmp_path)

        if len(frame_results) < 3:
            return {"error": "No human faces detected in the video"}

        avg_fake_prob = sum(r["fake_prob"] for r in frame_results) / len(frame_results)
        avg_real_prob = 1.0 - avg_fake_prob
        label = "FAKE" if avg_fake_prob > 0.5 else "REAL"

        heatmap_b64 = self._generate_gradcam(best_frame_tensor, best_frame_pil)

        return {
            "label": label,
            "confidence": round(max(avg_fake_prob, avg_real_prob), 4),
            "fake_prob": round(avg_fake_prob, 4),
            "real_prob": round(avg_real_prob, 4),
            "heatmap": heatmap_b64,
            "frames_analyzed": len(frame_results),
            "frame_results": frame_results,
        }

    def adversarial_attack(self, image_bytes: bytes, epsilon: float = 0.02) -> dict:
        """FGSM adversarial attack: add imperceptible noise to fool the model."""
        if not self.detect_face(image_bytes):
            return {"error": "No human face detected in the image"}

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)

        # Original prediction
        with torch.no_grad():
            logit = self.model(tensor)
            orig_fake_prob = float(torch.sigmoid(logit).squeeze())
            orig_label = "FAKE" if orig_fake_prob > 0.5 else "REAL"

        # FGSM: compute gradient of loss w.r.t. input
        tensor_adv = tensor.clone().requires_grad_(True)
        logit_adv = self.model(tensor_adv)

        # Use the opposite of the true label as target to maximise fooling
        target = torch.tensor([[0.0 if orig_fake_prob > 0.5 else 1.0]]).to(DEVICE)
        loss = F.binary_cross_entropy_with_logits(logit_adv, target)
        self.model.zero_grad()
        loss.backward()

        # Perturb input in the direction of the gradient sign
        perturbation = epsilon * tensor_adv.grad.sign()
        tensor_perturbed = (tensor_adv + perturbation).detach().clamp(
            min=(torch.tensor([0.485, 0.456, 0.406]).to(DEVICE) - torch.tensor([0.229, 0.224, 0.225]).to(DEVICE) * 3).min(),
            max=(torch.tensor([0.485, 0.456, 0.406]).to(DEVICE) + torch.tensor([0.229, 0.224, 0.225]).to(DEVICE) * 3).max(),
        )

        # Adversarial prediction
        with torch.no_grad():
            logit_final = self.model(tensor_perturbed)
            adv_fake_prob = float(torch.sigmoid(logit_final).squeeze())
            adv_label = "FAKE" if adv_fake_prob > 0.5 else "REAL"

        # Decode perturbed tensor back to image for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
        img_adv = (tensor_perturbed * std + mean).clamp(0, 1).squeeze(0)
        img_adv_np = (img_adv.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_adv_bgr = cv2.cvtColor(img_adv_np, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", img_adv_bgr)
        adv_image_b64 = f"data:image/png;base64,{base64.b64encode(buf).decode()}"

        return {
            "original_label":      orig_label,
            "original_fake_prob":  round(orig_fake_prob, 4),
            "original_real_prob":  round(1.0 - orig_fake_prob, 4),
            "adversarial_label":   adv_label,
            "adversarial_fake_prob": round(adv_fake_prob, 4),
            "adversarial_real_prob": round(1.0 - adv_fake_prob, 4),
            "adversarial_image":   adv_image_b64,
            "epsilon":             epsilon,
            "fooled":              orig_label != adv_label,
        }


detector = DeepfakeDetector()
