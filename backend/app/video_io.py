from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SampledVideo:
    frames_rgb: np.ndarray  # [N,H,W,3], uint8
    timestamps: np.ndarray  # [N], seconds
    duration_sec: float
    native_fps: float


class VideoReadError(RuntimeError):
    pass


def sample_video(path: Path, sample_fps: float, frame_size: int, max_frames: int) -> SampledVideo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"Could not open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / native_fps if frame_count > 0 else 0.0
    step = max(1, int(round(native_fps / max(sample_fps, 0.1))))

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = resize_square(frame_rgb, frame_size)
            frames.append(frame_rgb)
            timestamps.append(idx / native_fps)
        idx += 1
    cap.release()

    if len(frames) < 2:
        raise VideoReadError("Need at least two sampled frames to analyze motion/state change.")

    if duration_sec <= 0.0 and timestamps:
        duration_sec = float(timestamps[-1])

    return SampledVideo(
        frames_rgb=np.stack(frames).astype(np.uint8),
        timestamps=np.asarray(timestamps, dtype=np.float32),
        duration_sec=float(duration_sec),
        native_fps=float(native_fps),
    )


def resize_square(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    scale = size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = max(0, (nh - size) // 2)
    x0 = max(0, (nw - size) // 2)
    crop = resized[y0 : y0 + size, x0 : x0 + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def motion_energy(frames_rgb: np.ndarray) -> np.ndarray:
    gray = np.asarray([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb], dtype=np.float32)
    diffs = np.abs(np.diff(gray, axis=0)) / 255.0
    return diffs.reshape(diffs.shape[0], -1).mean(axis=1)


def central_motion_bias(frames_rgb: np.ndarray) -> float:
    gray = np.asarray([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb], dtype=np.float32)
    diffs = np.abs(np.diff(gray, axis=0))
    if diffs.size == 0:
        return 0.0
    h, w = diffs.shape[1:]
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    center = diffs[:, y0:y1, x0:x1].mean() + 1e-6
    outer = (diffs.mean() + 1e-6)
    return float(center / outer)
