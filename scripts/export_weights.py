import os

import torch

PROJECT_DIR = "/content/drive/MyDrive/Multimedia_Authentication"
SRC = f"{PROJECT_DIR}/v2/models/gated_v2_best.pth"
OUT_DIR = f"{PROJECT_DIR}/v2/export"
os.makedirs(OUT_DIR, exist_ok=True)

ck = torch.load(SRC, map_location="cpu", weights_only=False)     # our own checkpoint
sd = ck["model_state_dict"]
out_sd = {k: (v.half() if v.is_floating_point() and "running_" not in k else v) for k, v in sd.items()}

groups = sorted({k.split(".")[0] for k in out_sd})
assert groups == ["classifier", "freq_bb", "gate", "spatial_bb"], f"unexpected layers: {groups}"
out = f"{OUT_DIR}/gated_v2_fp16.pth"
torch.save(out_sd, out)
print(f"saved {out}  ({os.path.getsize(out) / 1e6:.1f} MB) | trained epoch {ck['epoch']} | val AUC {float(ck['val_auc']):.4f}")
