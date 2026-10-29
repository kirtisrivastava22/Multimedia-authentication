import os
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F

N_FRAMES = 8
FACE_SIZE = 224
CROP_SCALE = 1.4
DETECT_MAX_SIDE = 640
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FACE_MODEL = os.path.join(ROOT, "weights", "blaze_face_short_range.tflite")


class NoFaceError(RuntimeError):
    pass


class VideoReadError(RuntimeError):
    pass


def ensure_face_model(path=DEFAULT_FACE_MODEL):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, path)
        except Exception as e:
            raise RuntimeError(f"Face-detector model missing and download failed ({e}). "
                               f"Download {FACE_MODEL_URL} and save it as {path}") from e
    return path


class FaceDetector:
    """Largest face per frame, as (x, y, w, h) in full-resolution pixel coordinates."""

    def __init__(self, model_path=None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_face_model(model_path or DEFAULT_FACE_MODEL)),
            min_detection_confidence=0.5)
        self._det = vision.FaceDetector.create_from_options(opts)

    def box(self, frame_rgb):
        H, W = frame_rgb.shape[:2]
        s = min(1.0, DETECT_MAX_SIDE / max(H, W))
        small = cv2.resize(frame_rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame_rgb
        res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                              data=np.ascontiguousarray(small, dtype=np.uint8)))
        if not res.detections:
            return None
        b = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
        return (b.origin_x / s, b.origin_y / s, b.width / s, b.height / s)


def read_frames(path, n=N_FRAMES):
    """n evenly spaced RGB frames at the video's native resolution."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoReadError("Could not open the video file.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total >= n:
        for i in np.linspace(0, total - 1, n, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                frames = []
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) == n:
        return frames

    # fallback: container reports no / wrong length -> count frames, then decode sequentially keeping only n
    cap = cv2.VideoCapture(path)
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    if count < n:
        raise VideoReadError("The video is too short or could not be decoded.")
    want = set(np.linspace(0, count - 1, n, dtype=int).tolist())
    frames, cap = [], cv2.VideoCapture(path)
    for i in range(count):
        if i in want:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        elif not cap.grab():
            break
    cap.release()
    if len(frames) != n:
        raise VideoReadError("Could not decode enough frames from the video.")
    return frames


def square_crop(frame, box, scale=CROP_SCALE, out=FACE_SIZE):
    H, W = frame.shape[:2]
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = min(max(w, h) * scale, H, W)
    x1 = int(np.clip(cx - side / 2, 0, W - side))
    y1 = int(np.clip(cy - side / 2, 0, H - side))
    s = max(int(side), 2)
    return cv2.resize(frame[y1:y1 + s, x1:x1 + s], (out, out), interpolation=cv2.INTER_AREA)


def faces_from_frames(frames, detector):
    """-> (faces uint8 [n,224,224,3], number of frames where a face was really detected)."""
    boxes = [detector.box(f) for f in frames]
    good = [i for i, b in enumerate(boxes) if b is not None]
    if not good:
        raise NoFaceError("No face was detected in the sampled frames. Use a video with a clear, front-facing face.")
    for i, b in enumerate(boxes):                       # frames without a detection reuse the nearest detected box
        if b is None:
            boxes[i] = boxes[min(good, key=lambda g: abs(g - i))]
    faces = np.stack([square_crop(f, b) for f, b in zip(frames, boxes)]).astype(np.uint8)
    return faces, len(good)


def faces_from_video(path, detector):
    return faces_from_frames(read_frames(path), detector)

def _haar_highfreq(x):                                  # [N,3,H,W] -> [N,3,H/2,W/2]; equals pywt 'haar' (checked)
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    LH, HL, HH = (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2
    return torch.stack([LH.mean(1), HL.mean(1), HH.mean(1)], dim=1)


def to_inputs(faces_uint8):
    """faces uint8 [8,224,224,3] -> (spatial, frequency) float tensors, each [1,8,3,224,224]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.from_numpy(np.ascontiguousarray(faces_uint8)).permute(0, 3, 1, 2).float().div(255.0)
    xs = (x - mean) / std
    h = _haar_highfreq(x)
    mn, mx = h.amin(dim=(1, 2, 3), keepdim=True), h.amax(dim=(1, 2, 3), keepdim=True)
    h = (h - mn) / (mx - mn + 1e-8)
    h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
    xf = (h - mean) / std
    return xs.unsqueeze(0), xf.unsqueeze(0)
