import os
import sys

import torch

# Allow running from anywhere inside the repo or standalone next to src/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models import FeatureGatedFusion, ResNet18Binary  # noqa: E402

PROJECT_DIR = "/content/drive/MyDrive/Multimedia_Authentication"
OUT_DIR = "weights_out"
os.makedirs(OUT_DIR, exist_ok=True)


def find(name):
    for d in (os.path.join(PROJECT_DIR, "models"), PROJECT_DIR):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def export(src_name, dst_name, model):
    ckpt = torch.load(find(src_name), map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)  # fail loudly on any mismatch
    out = os.path.join(OUT_DIR, dst_name)
    torch.save(model.state_dict(), out)
    print(f"{dst_name}: {os.path.getsize(out) / 1e6:.1f} MB")


export("spatial_resnet18_best.pth", "spatial_resnet18.pth", ResNet18Binary())
export("frequency_resnet18_best.pth", "frequency_resnet18.pth", ResNet18Binary())
export("gated_fusion_resnet18_auc_best.pth", "gated_fusion_head.pth", FeatureGatedFusion())
print("Done. Copy the files in", OUT_DIR, "into the repo's weights/ folder.")
