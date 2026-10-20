
import os
import threading
import urllib.request

import cv2
import numpy as np
import pywt
from PIL import Image
from torchvision import transforms

NUM_FRAMES = 8
IMAGE_SIZE = (224, 224)
FACE_MARGIN = 0.20

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

imagenet_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ---------------------------------------------------------------- frames
def sample_frames(video_path, num_frames=NUM_FRAMES):
    """Uniformly sample frames; each WHOLE frame is resized to 224x224 (as in notebook 02)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Could not read the video length.")

    frames = []
    for idx in np.linspace(0, total - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, IMAGE_SIZE))
    cap.release()
    return frames


# ----------------------------------------------------------------- faces
class FaceCropper:
    """MediaPipe BlazeFace detector + 20% margin crop (as in notebook 03)."""

    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(FACE_MODEL_URL, model_path)

        self._detector = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                min_detection_confidence=0.5,
            )
        )
        self._lock = threading.Lock()  # Gradio calls handlers from worker threads

    def detect(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        with self._lock:
            result = self._detector.detect(
                self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
            )
        if not result.detections:
            return None

        h, w, _ = frame.shape
        best, best_area = None, 0
        for det in result.detections:
            b = det.bounding_box
            x, y = max(0, b.origin_x), max(0, b.origin_y)
            bw, bh = min(b.width, w - x), min(b.height, h - y)
            if bw * bh > best_area:
                best, best_area = (x, y, bw, bh), bw * bh
        return best

    @staticmethod
    def crop(frame, box, margin=FACE_MARGIN, size=IMAGE_SIZE):
        if box is None:
            return None
        x, y, w, h = box
        H, W, _ = frame.shape
        mx, my = int(w * margin), int(h * margin)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(W, x + w + mx), min(H, y + h + my)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return cv2.resize(face, size, interpolation=cv2.INTER_AREA)

    def __call__(self, frame):
        return self.crop(frame, self.detect(frame))


# ------------------------------------------------------------- frequency
def dwt_high_frequency(frame):
    """Haar DWT; signed mean over RGB for each of the LH / HL / HH sub-bands -> (112,112,3)."""
    frame = frame.astype(np.float32) / 255.0
    lh, hl, hh = [], [], []
    for c in range(3):
        _, (LH, HL, HH) = pywt.dwt2(frame[:, :, c], "haar")
        lh.append(LH)
        hl.append(HL)
        hh.append(HH)
    return np.stack(
        [np.mean(lh, axis=0), np.mean(hl, axis=0), np.mean(hh, axis=0)], axis=-1
    ).astype(np.float32)


def normalize_frequency_image(freq_img):
    """Per-image min-max to uint8 PIL image."""
    lo, hi = freq_img.min(), freq_img.max()
    freq_img = (freq_img - lo) / (hi - lo + 1e-8)
    return Image.fromarray((freq_img * 255).clip(0, 255).astype(np.uint8))


def frequency_tensor(face):
    """(224,224,3) uint8 face -> normalised frequency tensor (3,224,224)."""
    img = normalize_frequency_image(dwt_high_frequency(face)).resize(IMAGE_SIZE)
    return imagenet_transform(img)


def spatial_tensor(face):
    """(224,224,3) uint8 face -> normalised RGB tensor (3,224,224)."""
    return imagenet_transform(Image.fromarray(face))
