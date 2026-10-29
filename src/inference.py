import os

import numpy as np
import torch

from .net import GatedFusionNet
from .preprocessing import (ROOT, FaceDetector, NoFaceError, VideoReadError,
                            faces_from_video, to_inputs)

DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "gated_v2_fp16.pth")


class DeepfakeDetector:
    def __init__(self, weights=None, threshold=0.5, device="cpu"):
        path = weights or DEFAULT_WEIGHTS
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights not found: {path}. Run scripts/export_weights.py in Colab and copy "
                                    f"gated_v2_fp16.pth into the weights/ folder.")
        self.threshold = threshold
        self.device = torch.device(device)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
        self.model = GatedFusionNet()
        self.model.load_state_dict(sd, strict=True)            # any mismatch with training fails loudly
        self.model.eval().to(self.device)
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    @torch.no_grad()
    def predict_faces(self, faces):
        xs, xf = to_inputs(faces)
        logits, alpha = self.model(xs.to(self.device), xf.to(self.device))
        return torch.softmax(logits.float(), 1)[0, 1].item(), alpha.item()

    def _result(self, faces, n_detected):
        p, a = self.predict_faces(faces)
        fake = p >= self.threshold
        return {"status": "success", "prediction": "FAKE" if fake else "REAL",
                "confidence": p if fake else 1.0 - p, "fake_probability": p, "gate_alpha": a,
                "faces_detected": int(n_detected), "faces": faces}

    def predict_video(self, path):
        try:
            faces, n_detected = faces_from_video(path, self.detector)
        except (NoFaceError, VideoReadError) as e:
            return {"status": "error", "message": str(e)}
        return self._result(faces, n_detected)

    def predict_npz(self, path):
        """Pre-extracted training faces (8,224,224,3) - used to check train/inference parity."""
        faces = np.load(path)["frames"]
        return self._result(faces, len(faces))
