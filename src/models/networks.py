"""Network definitions (identical to the ones used in the training notebooks)."""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet18Binary(nn.Module):
    """ResNet18 backbone (512-D) + linear REAL/FAKE classifier.

    Used for the spatial branch (RGB faces) and the frequency branch (DWT images).
    Only the backbone is reused at inference time.
    """

    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        return self.classifier(torch.flatten(self.backbone(x), 1))


class FeatureGatedFusion(nn.Module):
    """Adaptive gated fusion head.

    alpha in [0, 1] is predicted from the concatenated features:
        fused = alpha * spatial + (1 - alpha) * frequency
    Label convention everywhere in this project: REAL = 0, FAKE = 1.
    """

    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, spatial, frequency, return_gate=False):
        alpha = self.gate(torch.cat([spatial, frequency], dim=1))
        fused = alpha * spatial + (1 - alpha) * frequency
        logits = self.classifier(fused)
        return (logits, alpha) if return_gate else logits
