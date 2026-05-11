
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schemas import DomainInsight


@dataclass
class GeoVisualStats:
    warm_color_ratio: float
    cool_color_ratio: float
    map_like_score: float
    temporal_color_shift: float
    center_of_warmth_x: float
    center_of_warmth_y: float


def should_use_geo_urban_layer(filename: str, domain_mode: str) -> bool:
    mode = (domain_mode or "auto").strip().lower()
    if mode in {"off", "none", "default", "physical"}:
        return False
    if mode in {"geo", "geo_urban", "area_intelligence", "urban", "land_price", "real_estate"}:
        return True

    # Auto mode: use the domain layer for explicit map / geo / real-estate filenames.
    name = filename.lower()
    keywords = [
        "地価",
        "東京23区",
        "23区",
        "バリューマップ",
        "value map",
        "land price",
        "real estate",
        "map",
        "geo",
        "urban",
        "vacancy",
        "population",
        "area value",
    ]
    return any(k in name for k in keywords)


def compute_geo_visual_stats(frames_rgb: np.ndarray) -> GeoVisualStats:
    if len(frames_rgb) == 0:
        return GeoVisualStats(0, 0, 0, 0, 0.5, 0.5)

    # Downsample for stable, cheap visual statistics.
    sampled = frames_rgb[:: max(1, len(frames_rgb) // 12)]
    hsv_frames = []
    warm_ratios = []
    cool_ratios = []
    warm_centers = []

    for frame in sampled:
        small = cv2.resize(frame, (320, 220), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hsv_frames.append(hsv)

        h = hsv[..., 0].astype(np.float32) * 2.0  # OpenCV hue: 0-179 -> degrees
        s = hsv[..., 1].astype(np.float32) / 255.0
        v = hsv[..., 2].astype(np.float32) / 255.0

        # Exclude black background and text-dominated dark regions.
        valid = (v > 0.15) & (s > 0.20)

        # Typical heatmap warm colors: red/orange/yellow. Cool: cyan/blue.
        warm = valid & (((h >= 0) & (h <= 65)) | (h >= 330))
        cool = valid & ((h >= 170) & (h <= 260))

        denom = float(valid.sum()) + 1e-6
        warm_ratios.append(float(warm.sum()) / denom)
        cool_ratios.append(float(cool.sum()) / denom)

        if warm.any():
            ys, xs = np.where(warm)
            warm_centers.append((float(xs.mean()) / small.shape[1], float(ys.mean()) / small.shape[0]))

    # Color histogram changes across time approximate animation changes in a choropleth/value map.
    histograms = []
    for hsv in hsv_frames:
        mask = (hsv[..., 2] > 35).astype(np.uint8)
        hist = cv2.calcHist([hsv], [0, 1], mask, [24, 16], [0, 180, 0, 256]).reshape(-1)
        hist = hist / (hist.sum() + 1e-6)
        histograms.append(hist)
    if len(histograms) > 1:
        temporal_shift = float(np.mean([np.linalg.norm(histograms[i] - histograms[i - 1]) for i in range(1, len(histograms))]))
    else:
        temporal_shift = 0.0

    first = sampled[0]
    gray = cv2.cvtColor(cv2.resize(first, (320, 220), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    # Map-like frames often have dense boundaries, labels, and road/rail linework over a large region.
    edge_density = float((edges > 0).mean())
    colored_area = float(((np.asarray(warm_ratios).mean() if warm_ratios else 0) + (np.asarray(cool_ratios).mean() if cool_ratios else 0)))
    map_like_score = float(np.clip(0.55 * min(edge_density / 0.12, 1.0) + 0.45 * min(colored_area / 0.45, 1.0), 0, 1))

    if warm_centers:
        cx = float(np.mean([x for x, _ in warm_centers]))
        cy = float(np.mean([y for _, y in warm_centers]))
    else:
        cx, cy = 0.5, 0.5

    return GeoVisualStats(
        warm_color_ratio=float(np.mean(warm_ratios)) if warm_ratios else 0.0,
        cool_color_ratio=float(np.mean(cool_ratios)) if cool_ratios else 0.0,
        map_like_score=map_like_score,
        temporal_color_shift=temporal_shift,
        center_of_warmth_x=cx,
        center_of_warmth_y=cy,
    )


def build_geo_urban_insight(
    *,
    filename: str,
    frames_rgb: np.ndarray,
    risk_level: str,
    risk_score: float,
    event_count: int,
    backend: str,
) -> DomainInsight:
    stats = compute_geo_visual_stats(frames_rgb)
    lower = filename.lower()

    if "東京23区" in filename or "23区" in filename:
        area = "Tokyo 23 wards"
    elif "東京" in filename or "tokyo" in lower:
        area = "Tokyo area"
    else:
        area = "the target urban area"

    if "地価" in filename or "land price" in lower or "value" in lower:
        layer_name = "land-price / area-value map"
    else:
        layer_name = "geo-spatial value map"

    observations = [
        f"The uploaded video appears to be an animated {layer_name} for {area}.",
        "The visual structure is map-like: a colored spatial surface is overlaid with administrative / transport linework and labels.",
        f"The model found {event_count} temporal state-change window(s); for this domain, these should be read as map-value transition points rather than physical collision risks.",
        f"Warm-color concentration ratio is approximately {stats.warm_color_ratio:.2f}, while cool-color ratio is approximately {stats.cool_color_ratio:.2f}.",
    ]

    if stats.temporal_color_shift > 0.08:
        observations.append("The color distribution changes meaningfully over time, suggesting that the animation is showing time-indexed market or area-value transitions.")
    else:
        observations.append("The color distribution is relatively stable over the sampled frames; the video is closer to a narrated/animated map than a rapidly changing physical scene.")

    interpretation = [
        "For a land-price value map, representation-change windows can be treated as candidate periods where spatial value gradients, hotspots, or corridor effects become visually salient.",
        "The output should not be framed as object-risk prediction. It is better framed as urban-state transition detection: where and when the map’s value surface changes enough to deserve analyst attention.",
        "The high score means the visual representation changed sharply in the sampled clip sequence; in a geospatial demo, that is a cue for hotspot / regime-shift explanation rather than accident risk.",
    ]

    area_intelligence_angle = [
        "This is directly compatible with a population / vacancy / area-value simulator: the video layer can become a visual front-end for time-series geo features.",
        "Land price, population dynamics, vacancy probability, station accessibility, facility reachability, and land-use features can be fused into one explanation layer.",
        "The MVP can answer: which wards or corridors look structurally strong, which peripheral areas show weak value persistence, and which locations require more granular block-level investigation.",
        "The useful product angle is not simply video understanding; it is converting animated geo-spatial evidence into decision-support text for EBPM, real-estate screening, and municipal planning.",
    ]

    recommended_next_steps = [
        "Add a metadata sidecar for the video: year range, color legend, unit, source, and target geography.",
        "Attach a GeoJSON / mesh / ward polygon layer so detected visual-change windows can be mapped to named areas rather than generic frame regions.",
        "Replace the generic risk label with domain labels such as hotspot emergence, value-gradient shift, corridor effect, or peripheral weakening.",
        "Use the V-JEPA embedding delta as a trigger, then use structured geo data to explain the socioeconomic meaning of the visual transition.",
    ]

    caveat = (
        "This domain layer uses visual representation changes and filename/context hints. "
        "It does not yet OCR the legend or georeference pixels to exact wards. "
        "For production, connect the video to structured GIS layers and source tables."
    )

    return DomainInsight(
        domain="geo_urban_area_intelligence",
        title=f"Geo / Urban Intelligence interpretation for {area}",
        observations=observations,
        interpretation=interpretation,
        area_intelligence_angle=area_intelligence_angle,
        recommended_next_steps=recommended_next_steps,
        caveat=caveat,
    )
