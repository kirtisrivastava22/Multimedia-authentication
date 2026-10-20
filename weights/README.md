# Weights

| File | What it is | Size |
|---|---|---|
| `spatial_resnet18.pth` | ResNet18 trained on RGB face crops (REAL=0, FAKE=1) | ~45 MB |
| `frequency_resnet18.pth` | ResNet18 trained on Haar-DWT high-frequency images | ~45 MB |
| `gated_fusion_head.pth` | Adaptive gate + classifier on the 512-D branch features | ~1 MB |
| `blaze_face_short_range.tflite` | MediaPipe face detector, downloaded automatically on first run | ~0.2 MB |

## Creating the files

The training checkpoints contain optimizer state and are too large for GitHub.
In Colab (Drive mounted), upload `scripts/export_weights.py` together with the `src/` folder and run:

```
!python export_weights.py
```

Then download the three files from `weights_out/` into this folder.

You can also keep the weights elsewhere and point to them with the `WEIGHTS_DIR` environment variable.
