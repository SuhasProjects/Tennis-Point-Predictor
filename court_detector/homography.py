"""Robust court homography estimation and validation utilities.

Matrices in this module map canonical court coordinates to image coordinates.
Use :func:`invert_homography` for image-to-court projection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from .court_reference import CourtReference
except ImportError:  # pragma: no cover - direct script compatibility
    from court_reference import CourtReference


court_ref = CourtReference()
refer_kps = np.asarray(court_ref.key_points, dtype=np.float32).reshape(-1, 1, 2)

court_conf_ind: dict[int, list[int]] = {}
for configuration_id, configuration in court_ref.court_conf.items():
    court_conf_ind[configuration_id] = [
        court_ref.key_points.index(point) for point in configuration
    ]


@dataclass(frozen=True)
class HomographyEstimate:
    """A validated reference-to-image homography and fit diagnostics."""

    matrix: np.ndarray
    reprojection_error: float
    inlier_count: int
    observed_count: int


def _as_point_array(points: Sequence[Sequence[float | None]]) -> np.ndarray:
    result = np.full((len(points), 2), np.nan, dtype=np.float32)
    for index, point in enumerate(points):
        if point is None or len(point) < 2 or point[0] is None or point[1] is None:
            continue
        result[index] = (float(point[0]), float(point[1]))
    return result


def normalize_homography(matrix: np.ndarray) -> np.ndarray | None:
    """Return a finite, scale-normalized 3x3 matrix or ``None``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    scale = matrix[2, 2]
    if abs(scale) < 1e-10:
        return None
    matrix = matrix / scale
    if abs(np.linalg.det(matrix)) < 1e-10:
        return None
    return matrix


def invert_homography(matrix: np.ndarray | None) -> np.ndarray | None:
    """Safely invert a homography, returning ``None`` when it is singular."""
    if matrix is None:
        return None
    normalized = normalize_homography(matrix)
    if normalized is None:
        return None
    try:
        inverse = np.linalg.inv(normalized)
    except np.linalg.LinAlgError:
        return None
    return normalize_homography(inverse)


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project an ``(N, 2)`` array through a homography."""
    values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(values, matrix).reshape(-1, 2)


def is_plausible_homography(
    matrix: np.ndarray,
    image_size: tuple[int, int] = (1280, 720),
) -> bool:
    """Reject singular, mirrored, tiny, or wildly off-screen court projections."""
    matrix = normalize_homography(matrix)
    if matrix is None:
        return False

    width, height = image_size
    projected = project_points(np.asarray(court_ref.border_points, dtype=np.float32), matrix)
    if not np.all(np.isfinite(projected)):
        return False

    # border_points are top-left, top-right, bottom-right, bottom-left.
    polygon = projected.astype(np.float32)
    area = abs(float(cv2.contourArea(polygon)))
    image_area = float(width * height)
    if area < image_area * 0.035 or area > image_area * 1.8:
        return False
    if not cv2.isContourConvex(polygon):
        return False

    top_width = np.linalg.norm(projected[1] - projected[0])
    bottom_width = np.linalg.norm(projected[2] - projected[3])
    if min(top_width, bottom_width) < width * 0.08:
        return False
    if bottom_width < top_width * 0.65 or bottom_width > top_width * 5.0:
        return False

    top_y = float((projected[0, 1] + projected[1, 1]) / 2)
    bottom_y = float((projected[2, 1] + projected[3, 1]) / 2)
    if bottom_y - top_y < height * 0.18:
        return False

    margin_x, margin_y = width * 0.75, height * 0.75
    return not (
        projected[:, 0].max() < -margin_x
        or projected[:, 0].min() > width + margin_x
        or projected[:, 1].max() < -margin_y
        or projected[:, 1].min() > height + margin_y
    )


def estimate_homography(
    points: Sequence[Sequence[float | None]],
    image_size: tuple[int, int] = (1280, 720),
    max_reprojection_error: float = 35.0,
    min_inliers: int = 8,
) -> HomographyEstimate | None:
    """Estimate the best validated court-to-image homography.

    The network occasionally emits one shifted landmark. We retain its proven
    four-point configurations, score each against all visible landmarks, and
    additionally try a RANSAC all-point fit.
    """
    observed = _as_point_array(points)
    count = min(len(observed), len(refer_kps))
    observed = observed[:count]
    canonical = refer_kps.reshape(-1, 2)[:count]
    visible = np.all(np.isfinite(observed), axis=1)
    observed_count = int(visible.sum())
    if observed_count < 4:
        return None

    candidates: list[np.ndarray] = []
    for indices in court_conf_ind.values():
        indices_array = np.asarray(indices, dtype=np.int32)
        if indices_array.max(initial=0) >= count or not visible[indices_array].all():
            continue
        candidates.append(
            cv2.getPerspectiveTransform(
                canonical[indices_array].astype(np.float32),
                observed[indices_array].astype(np.float32),
            )
        )

    if observed_count >= 5:
        ransac_matrix, _ = cv2.findHomography(
            canonical[visible], observed[visible], cv2.RANSAC, 10.0
        )
        if ransac_matrix is not None:
            candidates.append(ransac_matrix)

    best: HomographyEstimate | None = None
    for candidate in candidates:
        matrix = normalize_homography(candidate)
        if matrix is None or not is_plausible_homography(matrix, image_size):
            continue
        projected = project_points(canonical[visible], matrix)
        errors = np.linalg.norm(projected - observed[visible], axis=1)
        inliers = errors <= max_reprojection_error
        inlier_count = int(inliers.sum())
        if inlier_count < min_inliers:
            continue
        error = float(np.median(errors[inliers]))
        estimate = HomographyEstimate(matrix, error, inlier_count, observed_count)
        if best is None or (estimate.inlier_count, -estimate.reprojection_error) > (
            best.inlier_count,
            -best.reprojection_error,
        ):
            best = estimate
    return best


def get_trans_matrix(points: Sequence[Sequence[float | None]]) -> np.ndarray | None:
    """Backward-compatible wrapper returning only the matrix."""
    estimate = estimate_homography(points)
    return None if estimate is None else estimate.matrix
