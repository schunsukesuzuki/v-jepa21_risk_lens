from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# This module provides utility functions for reading videos, sampling frames,
# resizing them into a fixed square format, and computing simple motion-based
# features. It is intended to be used as a preprocessing layer before passing
# frames to downstream video models such as V-JEPA, or as a lightweight fallback
# for classical motion-based analysis.


@dataclass
class SampledVideo:
    """
    Container for sampled video frames and their metadata.

    Attributes:
        frames_rgb:
            A NumPy array of sampled RGB frames with shape [N, H, W, 3].
            N is the number of sampled frames.
            H and W are the resized frame height and width.
            The array is stored as uint8.

        timestamps:
            A NumPy array with shape [N], where each value represents the
            timestamp in seconds for the corresponding sampled frame.

        duration_sec:
            The estimated duration of the original video in seconds.

        native_fps:
            The original FPS reported by the video file. If OpenCV cannot
            retrieve the FPS, the script falls back to 30.0 fps.
    """

    frames_rgb: np.ndarray  # [N,H,W,3], uint8
    timestamps: np.ndarray  # [N], seconds
    duration_sec: float
    native_fps: float


class VideoReadError(RuntimeError):
    """
    Custom error type for video reading and preprocessing failures.

    This allows callers to distinguish video I/O or sampling errors from other
    runtime errors. It is raised, for example, when the video cannot be opened
    or when fewer than two frames are sampled.
    """

    pass


def sample_video(path: Path, sample_fps: float, frame_size: int, max_frames: int) -> SampledVideo:
    """
    Read a video file, sample frames at approximately the requested FPS,
    resize each sampled frame into a square image, and return a SampledVideo.

    Args:
        path:
            Path to the input video file.

        sample_fps:
            Target sampling rate in frames per second. For example, if the
            source video is 30 fps and sample_fps is 3.0, this function will
            roughly keep one frame every 10 original frames.

        frame_size:
            Output size for each square frame. For example, frame_size=224
            produces frames with shape [224, 224, 3].

        max_frames:
            Maximum number of sampled frames to keep. This prevents long videos
            from producing too many frames and becoming expensive to process.

    Returns:
        A SampledVideo object containing:
            - sampled RGB frames
            - timestamps for each sampled frame
            - estimated video duration
            - native FPS

    Raises:
        VideoReadError:
            If the video cannot be opened or if fewer than two frames are
            sampled. At least two frames are required because downstream motion
            analysis relies on frame-to-frame differences.
    """

    # Open the video file with OpenCV.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"Could not open video: {path}")

    # Retrieve native FPS and total frame count.
    # If FPS cannot be read, fall back to 30.0 fps as a practical default.
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Estimate duration from frame count and FPS when possible.
    duration_sec = frame_count / native_fps if frame_count > 0 else 0.0

    # Compute the frame interval used for sampling.
    # max(sample_fps, 0.1) avoids division by zero or extremely small values.
    # The step is at least 1, meaning the function never skips all frames.
    step = max(1, int(round(native_fps / max(sample_fps, 0.1))))

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    idx = 0

    # Read frames sequentially until the video ends or max_frames is reached.
    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Keep only frames whose index matches the sampling interval.
        if idx % step == 0:
            # OpenCV reads frames in BGR order, so convert them to RGB.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Resize and center-crop the frame into a fixed square size.
            frame_rgb = resize_square(frame_rgb, frame_size)

            frames.append(frame_rgb)

            # Store the timestamp, in seconds, for this sampled frame.
            timestamps.append(idx / native_fps)
        idx += 1

    # Release the underlying video resource.
    cap.release()

    # At least two frames are required for frame-difference based analysis.
    if len(frames) < 2:
        raise VideoReadError("Need at least two sampled frames to analyze motion/state change.")

    # If the video metadata did not provide a valid duration, use the timestamp
    # of the last sampled frame as a fallback estimate.
    if duration_sec <= 0.0 and timestamps:
        duration_sec = float(timestamps[-1])

    return SampledVideo(
        frames_rgb=np.stack(frames).astype(np.uint8),
        timestamps=np.asarray(timestamps, dtype=np.float32),
        duration_sec=float(duration_sec),
        native_fps=float(native_fps),
    )


def resize_square(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    """
    Resize an RGB frame and center-crop it into a square of the requested size.

    The function first scales the image so that its shorter side becomes
    `size`, then crops the center region to obtain a [size, size, 3] image.

    This is a standard preprocessing step for CNN / ViT-style models, which
    usually expect fixed-size square inputs.

    Args:
        frame_rgb:
            Input RGB image as a NumPy array with shape [H, W, 3].

        size:
            Target height and width of the output square image.

    Returns:
        A square RGB image with shape [size, size, 3].
    """

    h, w = frame_rgb.shape[:2]

    # Scale the image so that the shorter side becomes exactly `size`.
    scale = size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    # Resize the image. INTER_AREA is generally suitable for downsampling.
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    # Compute top-left coordinates for a centered square crop.
    y0 = max(0, (nh - size) // 2)
    x0 = max(0, (nw - size) // 2)

    # Crop the centered square region.
    crop = resized[y0 : y0 + size, x0 : x0 + size]

    # As a safety fallback, resize again if rounding or edge cases produced
    # a crop that is not exactly [size, size].
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    return crop


def motion_energy(frames_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a simple frame-to-frame motion energy score.

    This function converts each RGB frame to grayscale, computes absolute
    differences between consecutive frames, normalizes the differences by 255,
    and then averages over all pixels.

    Args:
        frames_rgb:
            RGB frame sequence with shape [N, H, W, 3].

    Returns:
        A 1D NumPy array with shape [N - 1].
        Each value represents the average pixel-level change between two
        consecutive sampled frames.

    Interpretation:
        Larger values indicate stronger visual change between adjacent frames.
        This can reflect actual object motion, camera shake, lighting changes,
        shadows, or global scene changes. Therefore, this is a low-level motion
        indicator rather than a semantic anomaly detector.
    """

    # Convert all RGB frames to grayscale so motion can be measured as
    # brightness-level differences.
    gray = np.asarray([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb], dtype=np.float32)

    # Compute absolute differences between consecutive frames along the time axis.
    # The result has shape [N - 1, H, W].
    diffs = np.abs(np.diff(gray, axis=0)) / 255.0

    # Flatten each difference frame and average over pixels.
    # The output length is N - 1 because differences are defined between pairs.
    return diffs.reshape(diffs.shape[0], -1).mean(axis=1)


def central_motion_bias(frames_rgb: np.ndarray) -> float:
    """
    Estimate how strongly motion is concentrated near the center of the frame.

    The function computes frame-to-frame grayscale differences and compares
    the average motion inside the central 50% region of the image with the
    average motion over the whole image.

    Args:
        frames_rgb:
            RGB frame sequence with shape [N, H, W, 3].

    Returns:
        A float ratio:
            center_motion / whole_frame_motion

        A value around 1.0 means central motion is similar to the overall
        frame average.
        A value greater than 1.0 means motion is more concentrated in the
        center region.
        A value less than 1.0 means motion is relatively stronger outside
        the center region.

    Note:
        The variable name `outer` below is slightly misleading: it actually
        stores the whole-frame average motion, not only the outer region.
    """

    # Convert frames to grayscale.
    gray = np.asarray([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb], dtype=np.float32)

    # Compute absolute differences between consecutive frames.
    diffs = np.abs(np.diff(gray, axis=0))

    # If no differences exist, return 0.0.
    # This should normally happen only when fewer than two frames are provided.
    if diffs.size == 0:
        return 0.0

    h, w = diffs.shape[1:]

    # Define the central 50% region of the frame:
    # vertical range:   25% to 75%
    # horizontal range: 25% to 75%
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)

    # Average motion inside the center crop.
    # Add a small epsilon to avoid division-by-zero issues.
    center = diffs[:, y0:y1, x0:x1].mean() + 1e-6

    # Average motion over the whole frame.
    # Despite the name `outer`, this is not the outer-only region.
    outer = (diffs.mean() + 1e-6)

    # Return the ratio between central motion and whole-frame motion.
    return float(center / outer)
