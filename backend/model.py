import os
import base64
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

        # Register hooks on the last conv layer (EfficientNet-B4: conv_head)
        self.model.conv_head.register_forward_hook(self._save_features)
        self.model.conv_head.register_full_backward_hook(self._save_gradients)

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

        # Normalize to 0-255
        cam = cam - cam.min()
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


detector = DeepfakeDetector()
