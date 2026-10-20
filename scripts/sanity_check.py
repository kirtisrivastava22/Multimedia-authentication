"""Compare the video path against pre-extracted faces (.npz) for the same clip.

    python scripts/sanity_check.py path/to/clip.mp4 path/to/faces/clip.npz

The fake probabilities should be close. If they are not, preprocessing differs from training.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.inference import DeepfakeDetector  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)

detector = DeepfakeDetector()
print("video:", detector.predict_video(sys.argv[1]))
print("npz  :", detector.predict_npz(sys.argv[2]))
