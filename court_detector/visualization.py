"""Small, reusable visualizations for court geometry and player positions."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .court_reference import CourtReference
    from .homography import project_points
    from .player_detector import PlayerDetection
except ImportError:  # pragma: no cover
    from court_reference import CourtReference
    from homography import project_points
    from player_detector import PlayerDetection


REFERENCE = CourtReference()


def draw_court_overlay(
    frame: np.ndarray,
    matrix: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (255, 80, 220),
) -> np.ndarray:
    if matrix is None:
        return frame
    lines = np.asarray(REFERENCE.get_important_lines(), dtype=np.float32).reshape(-1, 2, 2)
    projected = project_points(lines.reshape(-1, 2), matrix).reshape(-1, 2, 2)
    for line in projected:
        if np.all(np.isfinite(line)):
            cv2.line(frame, tuple(np.rint(line[0]).astype(int)), tuple(np.rint(line[1]).astype(int)), color, 1, cv2.LINE_AA)
    return frame


def draw_minimap(
    frame: np.ndarray,
    detections: list[PlayerDetection],
    *,
    origin: tuple[int, int] = (20, 390),
    size: tuple[int, int] = (155, 310),
) -> np.ndarray:
    """Draw a compact top-down court with the selected player positions."""
    x0, y0 = origin
    width, height = size
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (20, 70, 25), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    scale_x = width / REFERENCE.court_total_width
    scale_y = height / REFERENCE.court_total_height

    def map_point(point) -> tuple[int, int]:
        return int(x0 + point[0] * scale_x), int(y0 + point[1] * scale_y)

    lines = np.asarray(REFERENCE.get_important_lines(), dtype=np.float32).reshape(-1, 2, 2)
    for start, end in lines:
        cv2.line(frame, map_point(start), map_point(end), (235, 235, 235), 1, cv2.LINE_AA)
    colors = {"near": (40, 220, 40), "far": (20, 170, 255)}
    for detection in detections:
        if detection.court_position is not None:
            cv2.circle(frame, map_point(detection.court_position), 5, colors[detection.side], -1)
    return frame
