from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .explainer import build_explanation
from .model_loader import ModelUnavailable, VJEPAModelAdapter, make_clips
from .schemas import AnalysisResult, DetectedEvent, TimelineSegment
from .domain_interpreter import build_geo_urban_insight, should_use_geo_urban_layer
from .video_io import central_motion_bias, motion_energy, sample_video


# Main video-analysis entry point.
#
# This function takes an uploaded video path and produces a complete
# AnalysisResult for the API/UI layer. It samples the video, obtains clip-level
# representations either from a loaded V-JEPA adapter or from an explicit
# classical demo fallback, computes representation deltas and motion energy,
# combines them into a risk curve, extracts salient temporal events, builds a
# timeline, estimates motion direction and central-motion concentration, and
# finally generates natural-language state and prediction summaries.
#
# In geo/urban mode, the same representation-change signal is reinterpreted as
# an animated urban value-map transition signal rather than as a physical
# contact/drop-risk signal. The implementation intentionally keeps model
# inference, temporal scoring, event extraction, domain interpretation, and
# diagnostics assembly inside one orchestration function.
def analyze_video(
    *,
    path: Path,
    original_filename: str,
    analysis_id: str,
    settings: Settings,
    model: VJEPAModelAdapter,
    demo_mode: bool = False,
    domain_mode: str = "auto",
) -> AnalysisResult:
    sampled = sample_video(path, settings.sample_fps, settings.frame_size, settings.max_frames)
    diagnostics: dict = {
        "native_fps": sampled.native_fps,
        "sample_fps": settings.sample_fps,
        "frame_size": settings.frame_size,
        "clip_size": settings.clip_size,
        "model_errors": model.errors,
        "discovered_model_files": model.discovered_files,
    }

    backend = model.backend
    model_loaded = model.loaded
    warnings: list[str] = []

    if model.loaded:
        embeddings, model_warnings = model.encode_clips(sampled.frames_rgb, settings.clip_size)
        warnings.extend(model_warnings)
        clip_times = clip_mid_timestamps(sampled.timestamps, settings.clip_size)
    else:
        if not (demo_mode and settings.allow_demo_fallback):
            raise ModelUnavailable(
                "No V-JEPA model was loaded. Put downloaded model files under ./models or ./models/vjepa2, "
                "or set ALLOW_DEMO_FALLBACK=true and pass demo_mode=true for a classical no-model demo. "
                f"Loader diagnostics: {model.errors}"
            )
        embeddings = classical_clip_features(sampled.frames_rgb, settings.clip_size)
        clip_times = clip_mid_timestamps(sampled.timestamps, settings.clip_size)
        backend = "classical-demo-explicit"
        model_loaded = False
        warnings.append("Explicit classical demo mode used; no V-JEPA embeddings were computed.")

    delta = embedding_delta(embeddings)
    frame_motion = motion_energy(sampled.frames_rgb)
    clip_motion = align_motion_to_clips(frame_motion, len(embeddings))
    risk_curve = score_risk(delta, clip_motion)
    risk_score = float(np.clip(risk_curve.max() if len(risk_curve) else 0.0, 0.0, 1.0))
    risk_level = "High" if risk_score >= 0.72 else "Medium" if risk_score >= 0.42 else "Low"

    change_indices = pick_events(risk_curve, clip_times)
    timeline = build_timeline(clip_times, risk_curve)
    events = build_events(change_indices, clip_times, risk_curve, delta, clip_motion)

    direction = estimate_motion_direction(sampled.frames_rgb)
    center_bias = central_motion_bias(sampled.frames_rgb)
    current_state, prediction = infer_state_and_prediction(
        risk_level=risk_level,
        direction=direction,
        center_bias=center_bias,
        event_count=len(events),
        motion=float(np.mean(frame_motion)) if len(frame_motion) else 0.0,
    )
    reason = build_explanation(
        risk_level=risk_level,
        risk_score=risk_score,
        backend=backend,
        model_loaded=model_loaded,
        events=events,
        center_bias=center_bias,
        direction=direction,
        llm_provider=settings.llm_provider,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
    )

    domain_insight = None
    if should_use_geo_urban_layer(original_filename, domain_mode):
        domain_insight = build_geo_urban_insight(
            filename=original_filename,
            frames_rgb=sampled.frames_rgb,
            risk_level=risk_level,
            risk_score=risk_score,
            event_count=len(events),
            backend=backend,
        )
        diagnostics["domain_mode"] = domain_mode
        diagnostics["domain_interpreter"] = "geo_urban_area_intelligence"
        diagnostics["domain_patch_version"] = "v12_event_time_safe"

        # Replace the physical contact-risk language with a geospatial / urban-value
        # interpretation. V-JEPA still supplies the representation-change signal,
        # but the explanation layer should describe map-state transitions, not
        # falling objects or hand-object contact.
        current_state = [
            "The uploaded video is treated as an animated urban value map rather than a physical interaction scene.",
            "V-JEPA detected representation changes across map frames; these are interpreted as candidate shifts in the visible value surface, hotspots, or corridor patterns.",
            f"{len(events)} temporal window(s) are salient enough to deserve analyst attention as urban-state transition candidates.",
        ]
        predicted_near_future_change = [
            "The next useful step is not physical accident prediction, but mapping these transition windows to named wards, corridors, station areas, or mesh cells.",
            "With a GeoJSON or mesh sidecar, the same representation-change trigger can be converted into hotspot emergence, value-gradient shift, or peripheral weakening labels.",
        ]

        # Extract a timestamp from a DetectedEvent-like object in a schema-tolerant way.
        #
        # Different iterations of the event schema have used names such as `time`,
        # `time_s`, `timestamp`, `center`, or `center_s`. This helper accepts either
        # object attributes or dictionary keys so that the geo/urban explanation layer
        # does not break when the event schema changes slightly.
        def _event_time(event: object) -> float | None:
            # DetectedEvent has used both `time` and `time_s` across iterations.
            # Keep this tolerant so UI/domain layers do not break when schema names change.
            for attr in ("time", "time_s", "timestamp", "center", "center_s"):
                value = getattr(event, attr, None)
                if isinstance(value, (int, float)):
                    return float(value)
            if isinstance(event, dict):
                for key in ("time", "time_s", "timestamp", "center", "center_s"):
                    value = event.get(key)
                    if isinstance(value, (int, float)):
                        return float(value)
            return None

        # Extract a display label from a DetectedEvent-like object in a schema-tolerant way.
        #
        # The helper accepts common label-like fields such as `label`, `type`, `title`,
        # or `name`. If none of them exists, it falls back to a generic transition-window
        # label so that explanation generation remains robust.
        def _event_label(event: object) -> str:
            for attr in ("label", "type", "title", "name"):
                value = getattr(event, attr, None)
                if isinstance(value, str) and value:
                    return value
            if isinstance(event, dict):
                for key in ("label", "type", "title", "name"):
                    value = event.get(key)
                    if isinstance(value, str) and value:
                        return value
            return "transition window"

        event_parts = []
        for event in events[:4]:
            t = _event_time(event)
            label = _event_label(event)
            if t is None:
                event_parts.append(label)
            else:
                event_parts.append(f"{t:.2f}s: {label}")
        event_summary = "; ".join(event_parts) if event_parts else "no strong transition window"

        explanation = (
            f"Urban value-map transition score is {risk_level} ({risk_score:.2f}). "
            "The analyzer compared consecutive V-JEPA clip representations and treated sharp deltas as candidate changes in the animated map's value surface. "
            f"The strongest windows were: {event_summary}. "
            "For this geo/urban-intelligence mode, these windows should be read as visual cues for geospatial regime shifts, hotspot emphasis, or corridor effects, "
            "then grounded with structured GIS layers such as ward polygons, mesh statistics, population dynamics, vacancy probability, station accessibility, and facility reachability."
        )

    diagnostics.update(
        {
            "warnings": warnings,
            "embedding_shape": list(embeddings.shape),
            "delta": delta.round(5).tolist(),
            "clip_motion": clip_motion.round(5).tolist(),
            "risk_curve": risk_curve.round(5).tolist(),
            "motion_direction": direction,
            "central_motion_bias": center_bias,
        }
    )

    return AnalysisResult(
        analysis_id=analysis_id,
        video_filename=original_filename,
        duration_sec=sampled.duration_sec,
        sampled_frames=len(sampled.frames_rgb),
        model_backend=backend,
        model_loaded=model_loaded,
        risk_level=risk_level,
        risk_score=round(risk_score, 3),
        current_state=current_state,
        predicted_near_future_change=prediction,
        reason=reason,
        timeline=timeline,
        detected_state_change=events,
        diagnostics=diagnostics,
        domain_insight=domain_insight,
    )


# Build explicit classical fallback features for each video clip.
#
# This function is used only when V-JEPA embeddings are unavailable and the
# caller has explicitly enabled demo fallback mode. It splits RGB frames into
# clips, computes simple 16-bin color histograms for each RGB channel, adds
# grayscale frame-difference statistics as a lightweight motion descriptor, then
# concatenates and L2-normalizes the resulting feature vector. The output is a
# clip-by-feature matrix that mimics the downstream shape of model embeddings,
# but it should not be interpreted as a semantic V-JEPA representation.
def classical_clip_features(frames_rgb: np.ndarray, clip_size: int) -> np.ndarray:
    clips = make_clips(frames_rgb, clip_size)
    feats = []
    for clip in clips:
        hist_parts = []
        for ch in range(3):
            hist = cv2.calcHist([clip.reshape(-1, 1, 3)], [ch], None, [16], [0, 256]).reshape(-1)
            hist = hist / (hist.sum() + 1e-6)
            hist_parts.append(hist)
        gray = np.asarray([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in clip], dtype=np.float32)
        diff_stats = np.abs(np.diff(gray, axis=0)).reshape(max(len(clip) - 1, 1), -1).mean(axis=1)
        stat = np.asarray([diff_stats.mean(), diff_stats.std(), diff_stats.max()], dtype=np.float32) / 255.0
        feat = np.concatenate([*hist_parts, stat])
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        feats.append(feat.astype(np.float32))
    return np.stack(feats)


# Compute the center timestamp of each clip.
#
# The model and fallback path operate on clips rather than individual frames, so
# the UI needs a representative timestamp for each clip-level embedding. This
# function uses a half-overlap stride (`clip_size // 2`) and records the timestamp
# at the center frame of each clip. For very short videos, it returns the middle
# timestamp of the available frames as a single clip timestamp.
def clip_mid_timestamps(timestamps: np.ndarray, clip_size: int) -> np.ndarray:
    n = len(timestamps)
    if n <= clip_size:
        return np.asarray([float(timestamps[n // 2])], dtype=np.float32)
    stride = max(1, clip_size // 2)
    mids = []
    for i in range(0, n - clip_size + 1, stride):
        mids.append(float(timestamps[i + clip_size // 2]))
    if (n - clip_size) % stride != 0:
        mids.append(float(timestamps[-clip_size // 2]))
    return np.asarray(mids, dtype=np.float32)


# Measure representation change between consecutive clip embeddings.
#
# The function computes the L2 norm of the difference between adjacent clip
# embeddings. The first clip has no previous representation, so its delta is set
# to zero. The resulting vector has the same length as the number of clips and is
# later combined with motion energy to form the risk curve.
def embedding_delta(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=np.float32)
    d = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    d = np.concatenate([[0.0], d]).astype(np.float32)
    return d


# Resample frame-level motion values onto the clip-level timeline.
#
# Motion energy is computed per frame or frame interval, while representation
# deltas are computed per clip. This function interpolates the frame-level motion
# sequence so that it has exactly `n_clips` values. The aligned motion vector can
# then be combined directly with the clip-level embedding-delta vector.
def align_motion_to_clips(frame_motion: np.ndarray, n_clips: int) -> np.ndarray:
    if n_clips <= 0:
        return np.asarray([], dtype=np.float32)
    if len(frame_motion) == 0:
        return np.zeros(n_clips, dtype=np.float32)
    xs = np.linspace(0, len(frame_motion) - 1, n_clips)
    return np.interp(xs, np.arange(len(frame_motion)), frame_motion).astype(np.float32)


# Convert a raw signal into a robust 0-to-1 score.
#
# Instead of mean and standard deviation, this function uses the median and MAD
# (median absolute deviation), which makes it less sensitive to extreme outliers.
# It then applies a sigmoid so that the normalized output remains bounded between
# 0 and 1 and can be safely mixed into the final risk curve.
def robust_norm(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    return 1.0 / (1.0 + np.exp(-z))


# Combine representation change and motion energy into a risk curve.
#
# The representation delta is treated as the primary signal and receives a 0.65
# weight. Motion energy is used as a secondary physical-dynamics signal with a
# 0.35 weight. The first clip is down-weighted because it has no previous clip for
# a meaningful representation-delta comparison. The final curve is clipped to the
# [0, 1] range and returned as float32.
def score_risk(delta: np.ndarray, motion: np.ndarray) -> np.ndarray:
    if len(delta) == 0:
        return np.asarray([], dtype=np.float32)
    rd = robust_norm(delta)
    rm = robust_norm(motion)
    curve = 0.65 * rd + 0.35 * rm
    # Suppress first clip because no previous representation exists.
    if len(curve):
        curve[0] *= 0.5
    return np.clip(curve, 0.0, 1.0).astype(np.float32)


# Select a small set of salient event indices from the risk curve.
#
# The threshold combines an absolute floor of 0.42 with an adaptive threshold
# based on the median and standard deviation of the current video's risk curve.
# Candidate clips above the threshold are sorted by score, then filtered so that
# selected events are separated by more than 0.6 seconds. This preserves temporal
# diversity and prevents the UI from showing many near-duplicate events around
# the same transition. At most five events are returned.
def pick_events(risk_curve: np.ndarray, times: np.ndarray) -> list[int]:
    if len(risk_curve) == 0:
        return []
    threshold = max(0.42, float(np.median(risk_curve) + 0.5 * np.std(risk_curve)))
    candidates = [i for i, s in enumerate(risk_curve) if s >= threshold]
    # Keep temporal diversity.
    picked = []
    for i in sorted(candidates, key=lambda j: risk_curve[j], reverse=True):
        if all(abs(float(times[i] - times[j])) > 0.6 for j in picked):
            picked.append(i)
        if len(picked) >= 5:
            break
    return sorted(picked)


# Convert clip-level scores into timeline segments for UI display.
#
# Each clip timestamp is treated as the center of a segment. The segment width is
# inferred from the distance between the first two clip timestamps, or defaults to
# half a second for a single timestamp. Scores are mapped to labels using the same
# threshold scheme as the overall risk level: `risk` for >= 0.72, `state change`
# for >= 0.42, and `normal` otherwise.
def build_timeline(times: np.ndarray, risk_curve: np.ndarray) -> list[TimelineSegment]:
    if len(times) == 0:
        return []
    segments = []
    for i, t in enumerate(times):
        score = float(risk_curve[i])
        label = "risk" if score >= 0.72 else "state change" if score >= 0.42 else "normal"
        if len(times) > 1:
            half = abs(float(times[1] - times[0])) / 2
        else:
            half = 0.5
        segments.append(TimelineSegment(start=max(0.0, float(t) - half), end=float(t) + half, label=label, score=round(score, 3)))
    return segments


# Build DetectedEvent objects from selected event indices.
#
# This function turns numeric event indices into structured event records with a
# timestamp, title, detail message, and rounded score. High-risk windows are
# described as possible object/contact instability. Events with unusually large
# representation deltas are described as motion discontinuities. Remaining
# selected candidates are labeled as generic state-change candidates.
def build_events(indices: list[int], times: np.ndarray, risk_curve: np.ndarray, delta: np.ndarray, motion: np.ndarray) -> list[DetectedEvent]:
    events = []
    for idx in indices:
        score = float(risk_curve[idx])
        if score >= 0.72:
            title = "risk window detected"
            detail = "Representation shift and motion energy are both elevated; near-future object/contact instability is plausible."
        elif float(delta[idx]) >= float(np.median(delta) + np.std(delta)):
            title = "motion discontinuity detected"
            detail = "Clip embedding changed faster than the surrounding temporal context."
        else:
            title = "state change candidate"
            detail = "Visual dynamics diverged from the local baseline."
        events.append(DetectedEvent(timestamp=round(float(times[idx]), 2), title=title, detail=detail, score=round(score, 3)))
    return events


# Estimate the dominant motion direction between the first and last frame.
#
# The function converts the first and last RGB frames to grayscale, computes dense
# Farneback optical flow, and uses the median horizontal and vertical flow as a
# robust estimate of global motion. Very small flow is treated as mostly
# stationary. Otherwise, the flow angle is mapped to one of rightward, downward,
# leftward, or upward. Because image coordinates use a downward-positive y-axis,
# positive vertical flow corresponds to `downward`.
def estimate_motion_direction(frames_rgb: np.ndarray) -> str:
    if len(frames_rgb) < 2:
        return "unclear"
    prev = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    nxt = cv2.cvtColor(frames_rgb[-1], cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    dx = float(np.median(flow[..., 0]))
    dy = float(np.median(flow[..., 1]))
    if abs(dx) + abs(dy) < 0.05:
        return "mostly stationary"
    angle = math.degrees(math.atan2(dy, dx))
    if -45 <= angle <= 45:
        return "rightward"
    if 45 < angle < 135:
        return "downward"
    if angle >= 135 or angle <= -135:
        return "leftward"
    return "upward"


# Generate rule-based current-state and near-future prediction messages.
#
# This helper converts numeric and categorical signals into short explanation
# bullets for the API response. Motion magnitude controls whether the scene is
# described as moving or stable. Central-motion bias explains whether movement is
# concentrated around the central interaction area. Event count reports whether
# representation-level state-change windows were found. The final prediction
# depends on the risk level: High suggests near-term instability, Medium suggests
# a monitored local transition, and Low suggests no immediate visible risk.
def infer_state_and_prediction(*, risk_level: str, direction: str, center_bias: float, event_count: int, motion: float) -> tuple[list[str], list[str]]:
    state = []
    prediction = []
    if motion > 0.025:
        state.append(f"A moving subject or object is visible; dominant motion is {direction}.")
    else:
        state.append("The scene is mostly stable, with limited frame-to-frame motion.")
    if center_bias > 1.15:
        state.append("Motion is concentrated near the central object/interaction region.")
    else:
        state.append("Motion is distributed broadly across the frame rather than isolated to one boundary.")
    if event_count:
        state.append(f"{event_count} temporal window(s) show a representation-level state change.")
    else:
        state.append("No strong state-change window was detected.")

    if risk_level == "High":
        prediction.append("The current interaction may become unstable within the next 1–2 seconds.")
        prediction.append("An object may be displaced, dropped, or contacted if the motion continues.")
    elif risk_level == "Medium":
        prediction.append("A local state transition is likely; continued contact or occlusion should be monitored.")
        prediction.append("The scene may remain safe, but the detected motion pattern is no longer baseline-normal.")
    else:
        prediction.append("No immediate near-future risk is apparent from the sampled clips.")
        prediction.append("The scene is likely to remain stable if the current motion pattern continues.")
    return state, prediction
