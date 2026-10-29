import torch
import torch.nn as nn
from torchvision.models import resnet18


def _backbone():
    r = resnet18(weights=None)                       # weights come from the exported checkpoint
    return nn.Sequential(*list(r.children())[:-1])   # -> [N, 512, 1, 1]


class GatedFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_bb = _backbone()
        self.freq_bb = _backbone()
        self.gate = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))

    @staticmethod
    def _feats(x, bb):                               # [B,T,3,H,W] -> [B,512] (mean over frames)
        B, T = x.shape[:2]
        return bb(x.flatten(0, 1)).flatten(1).view(B, T, 512).mean(1)

    def forward(self, xs, xf):
        s, f = self._feats(xs, self.spatial_bb), self._feats(xf, self.freq_bb)
        alpha = torch.sigmoid(self.gate(torch.cat([s, f], 1)))      # 1 = all spatial, 0 = all frequency
        return self.classifier(alpha * s + (1 - alpha) * f), alpha
