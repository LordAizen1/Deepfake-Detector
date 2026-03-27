"""
Model stub — replace with your actual trained model.
"""

import numpy as np
from PIL import Image
import io


class DeepfakeDetector:
    def __init__(self):
        # TODO: load your trained model here
        # e.g. self.model = torch.load("weights/best_model.pth")
        self.model = None
        print("Model stub loaded. Replace with actual model.")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        return np.array(img, dtype=np.float32) / 255.0

    def predict(self, image_bytes: bytes) -> dict:
        self.preprocess(image_bytes)

        # --- STUB: swap in real inference here ---
        fake_prob = float(np.random.uniform(0.0, 1.0))
        real_prob = 1.0 - fake_prob
        label = "FAKE" if fake_prob > 0.5 else "REAL"
        # -----------------------------------------

        return {
            "label": label,
            "confidence": round(max(fake_prob, real_prob), 4),
            "fake_prob": round(fake_prob, 4),
            "real_prob": round(real_prob, 4),
        }


detector = DeepfakeDetector()
