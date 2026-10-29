import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.inference import DeepfakeDetector  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)

detector = DeepfakeDetector()
a, b = detector.predict_video(sys.argv[1]), detector.predict_npz(sys.argv[2])
for name, r in [("video", a), ("npz  ", b)]:
    r.pop("faces", None)
    print(name, r)
if a["status"] == "success":
    print(f"\n|P(fake) video - P(fake) npz| = {abs(a['fake_probability'] - b['fake_probability']):.4f}")
