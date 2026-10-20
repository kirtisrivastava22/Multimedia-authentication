import os
from pathlib import Path

import numpy as np
import torch

from .models import FeatureGatedFusion, ResNet18Binary
from .preprocessing import (
    NUM_FRAMES,
    FaceCropper,
    frequency_tensor,
    sample_frames,
    spatial_tensor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", REPO_ROOT / "weights"))

SPATIAL_FILE = "spatial_resnet18.pth"
FREQUENCY_FILE = "frequency_resnet18.pth"
GATED_FILE = "gated_fusion_head.pth"
FACE_MODEL_FILE = "blaze_face_short_range.tflite"


def _state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


class DeepfakeDetector:
    """Frozen spatial + frequency ResNet18 backbones -> mean over frames -> gated fusion head."""

    def __init__(self, weights_dir=None, device=None, min_faces=4):
        weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.min_faces = min_faces

        for name in (SPATIAL_FILE, FREQUENCY_FILE, GATED_FILE):
            if not (weights_dir / name).exists():
                raise FileNotFoundError(
                    f"{weights_dir / name} is missing. See weights/README.md."
                )

        self.spatial = self._load_backbone(weights_dir / SPATIAL_FILE)
        self.frequency = self._load_backbone(weights_dir / FREQUENCY_FILE)

        self.head = FeatureGatedFusion().to(self.device)
        self.head.load_state_dict(_state_dict(weights_dir / GATED_FILE), strict=True)
        self.head.eval()

        self.cropper = FaceCropper(str(weights_dir / FACE_MODEL_FILE))

    def _load_backbone(self, path):
        model = ResNet18Binary()
        model.load_state_dict(_state_dict(path), strict=True)
        return model.backbone.to(self.device).eval()

    @torch.no_grad()
    def predict_faces(self, faces, threshold=0.5):
        """faces: list of (224,224,3) uint8 RGB face crops."""
        rgb = torch.stack([spatial_tensor(f) for f in faces]).to(self.device)
        frq = torch.stack([frequency_tensor(f) for f in faces]).to(self.device)

        s = torch.flatten(self.spatial(rgb), 1).mean(0, keepdim=True)
        q = torch.flatten(self.frequency(frq), 1).mean(0, keepdim=True)

        logits, alpha = self.head(s, q, return_gate=True)
        p_fake = torch.softmax(logits, dim=1)[0, 1].item()
        is_fake = p_fake >= threshold
        return {
            "status": "success",
            "prediction": "FAKE" if is_fake else "REAL",
            "confidence": p_fake if is_fake else 1.0 - p_fake,
            "fake_probability": p_fake,
            "gate_alpha": alpha.item(),
            "faces_detected": len(faces),
        }

    def predict_video(self, video_path, threshold=0.5):
        try:
            frames = sample_frames(video_path)
        except Exception as exc:  # unreadable / corrupt video
            return {"status": "error", "message": str(exc)}

        faces = [f for f in (self.cropper(fr) for fr in frames) if f is not None]
        if len(faces) < self.min_faces:
            return {
                "status": "error",
                "message": (
                    f"A face was found in only {len(faces)}/{NUM_FRAMES} sampled frames "
                    f"(need at least {self.min_faces}). Use a clip with a clearly visible face."
                ),
            }
        return self.predict_faces(faces, threshold)

    def predict_npz(self, npz_path, threshold=0.5):
        """Predict from a pre-extracted faces/*.npz file (used to sanity-check the video path)."""
        return self.predict_faces(list(np.load(npz_path)["frames"]), threshold)
