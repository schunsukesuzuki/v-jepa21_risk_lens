"""Create a small synthetic hand/object-risk-like video for smoke testing.

Run inside backend container or local Python with opencv-python installed:
python scripts/make_synthetic_video.py data/uploads/synthetic_warehouse_risk.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def main(out_path: str = "data/uploads/synthetic_warehouse_risk.mp4") -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps, frames = 640, 360, 24, 168
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        img = np.full((height, width, 3), 245, dtype=np.uint8)
        # shelf
        cv2.rectangle(img, (400, 80), (560, 300), (190, 190, 190), -1)
        cv2.rectangle(img, (395, 75), (565, 305), (80, 80, 80), 3)
        # object that becomes unstable after contact
        wobble = 0
        if i > 110:
            wobble = int(10 * np.sin(i * 0.45))
        obj_x, obj_y = 452 + wobble, 170 + max(0, i - 120) // 3
        cv2.rectangle(img, (obj_x, obj_y), (obj_x + 55, obj_y + 70), (40, 120, 220), -1)
        # approaching hand/arm
        hand_x = min(430, 60 + i * 3)
        hand_y = 195
        cv2.rectangle(img, (0, 210), (hand_x, 242), (70, 120, 180), -1)
        cv2.circle(img, (hand_x, hand_y), 24, (70, 120, 180), -1)
        if 92 < i < 116:
            cv2.putText(img, "contact", (270, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
        if i >= 116:
            cv2.putText(img, "unstable", (270, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
        writer.write(img)
    writer.release()
    print(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/uploads/synthetic_warehouse_risk.mp4")
