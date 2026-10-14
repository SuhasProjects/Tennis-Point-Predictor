"""Court landmark inference with validated, temporally stable homography state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .homography import estimate_homography, is_plausible_homography, project_points, refer_kps
    from .postprocess import postprocess, refine_kps
    from .tracknet import BallTrackerNet
except ImportError:  # pragma: no cover - direct script compatibility
    from homography import estimate_homography, is_plausible_homography, project_points, refer_kps
    from postprocess import postprocess, refine_kps
    from tracknet import BallTrackerNet


@dataclass(frozen=True)
class CourtDetection:
    keypoints: list[dict[str, int]]
    matrix: np.ndarray | None
    reprojection_error: float | None
    inlier_count: int
    detected: bool
    stale_frames: int
    scene_cut: bool = False


class CourtDetector:
    def __init__(
        self,
        weights_path: str | Path,
        *,
        smoothing_factor: float = 0.2,
        max_stale_frames: int = 30,
    ) -> None:
        """Load the court model once and initialize temporal tracking state."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BallTrackerNet(out_channels=15).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        self.smoothing_factor = float(smoothing_factor)
        self.max_stale_frames = int(max_stale_frames)
        self._smoothed_points: np.ndarray | None = None
        self._last_matrix: np.ndarray | None = None
        self._last_detection: CourtDetection | None = None
        self._stale_frames = 0
        self._last_histogram: np.ndarray | None = None
        self._last_thumbnail: np.ndarray | None = None

    def detect(self, image_path, use_refine_kps=True, use_homography=True):
        """Process an image from a file path (legacy tuple API)."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        return self.detect_frame(image, use_refine_kps, use_homography)

    def _infer_points(
        self, image: np.ndarray, use_refine_kps: bool
    ) -> list[tuple[float | None, float | None]]:
        """Run the heatmap model and return its 14 raw image landmarks."""
        img = cv2.resize(image, (640, 360))
        tensor = torch.from_numpy(
            np.moveaxis(img.astype(np.float32) / 255.0, 2, 0)
        ).unsqueeze(0)

        with torch.inference_mode():
            output = self.model(tensor.to(self.device))[0]
            prediction = torch.sigmoid(output).cpu().numpy()

        points: list[tuple[float | None, float | None]] = []
        for keypoint_index in range(14):
            heatmap = (prediction[keypoint_index] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if (
                use_refine_kps
                and keypoint_index not in (8, 9, 12)
                and x_pred is not None
                and y_pred is not None
            ):
                refined = refine_kps(image, int(y_pred), int(x_pred))
                if refined is not None:
                    x_pred, y_pred = refined
            points.append((x_pred, y_pred))
        return points

    @staticmethod
    def _keypoint_dicts(points: np.ndarray) -> list[dict[str, int]]:
        return [
            {"id": index, "x": round(float(point[0])), "y": round(float(point[1]))}
            for index, point in enumerate(points)
            if np.all(np.isfinite(point))
        ]

    def detect_frame(self, image, use_refine_kps=False, use_homography=True):
        """Process one frame without temporal state (legacy tuple API)."""
        resized = cv2.resize(image, (1280, 720))
        points = self._infer_points(resized, use_refine_kps)
        estimate = (
            estimate_homography(points, image_size=(1280, 720))
            if use_homography
            else None
        )
        matrix = None if estimate is None else estimate.matrix
        if matrix is not None:
            display_points = project_points(refer_kps.reshape(-1, 2), matrix)
        else:
            display_points = np.asarray(
                [
                    [np.nan if value is None else value for value in point]
                    for point in points
                ],
                dtype=np.float32,
            )
        return resized, self._keypoint_dicts(display_points), matrix

    def reset_tracking(self) -> None:
        """Clear temporal geometry state, for example after seeking a video."""
        self._smoothed_points = None
        self._last_matrix = None
        self._last_detection = None
        self._stale_frames = 0
        self._last_histogram = None
        self._last_thumbnail = None

    def _is_scene_cut(self, image: np.ndarray) -> bool:
        """Detect broadcast cuts cheaply so an old court is never reused."""
        thumbnail = cv2.resize(image, (160, 90)).astype(np.float32)
        hsv = cv2.cvtColor(thumbnail.astype(np.uint8), cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        scene_cut = False
        if self._last_histogram is not None:
            distance = cv2.compareHist(
                self._last_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA
            )
            scene_cut = distance > 0.55
        if self._last_thumbnail is not None:
            pixel_change = float(np.mean(np.abs(thumbnail - self._last_thumbnail)) / 255.0)
            scene_cut = scene_cut or pixel_change > 0.12
        self._last_histogram = histogram
        self._last_thumbnail = thumbnail
        return scene_cut

    def track_frame(
        self,
        image: np.ndarray,
        frame_index: int,
        *,
        detect_interval: int = 5,
        use_refine_kps: bool = False,
    ) -> tuple[np.ndarray, CourtDetection]:
        """Return stable geometry while safely reusing recent valid fits."""
        resized = cv2.resize(image, (1280, 720))
        interval = max(1, int(detect_interval))
        scene_cut = self._is_scene_cut(resized)
        if scene_cut:
            self._smoothed_points = None
            self._last_matrix = None
            self._last_detection = None
            self._stale_frames = 0
        should_detect = scene_cut or self._last_detection is None or frame_index % interval == 0
        if not should_detect and self._last_detection is not None:
            cached = CourtDetection(
                self._last_detection.keypoints,
                self._last_detection.matrix,
                self._last_detection.reprojection_error,
                self._last_detection.inlier_count,
                False,
                self._stale_frames,
                False,
            )
            return resized, cached

        raw_points = self._infer_points(resized, use_refine_kps)
        estimate = estimate_homography(raw_points, image_size=(1280, 720))
        if estimate is None:
            self._stale_frames += interval
            if self._last_matrix is None or self._stale_frames > self.max_stale_frames:
                detection = CourtDetection(
                    [], None, None, 0, False, self._stale_frames, scene_cut
                )
            else:
                projected = project_points(refer_kps.reshape(-1, 2), self._last_matrix)
                detection = CourtDetection(
                    self._keypoint_dicts(projected),
                    self._last_matrix.copy(),
                    self._last_detection.reprojection_error if self._last_detection else None,
                    self._last_detection.inlier_count if self._last_detection else 0,
                    False,
                    self._stale_frames,
                    scene_cut,
                )
            self._last_detection = detection
            return resized, detection

        detected_points = project_points(refer_kps.reshape(-1, 2), estimate.matrix)
        reset_for_cut = False
        if self._smoothed_points is not None:
            displacement = np.median(
                np.linalg.norm(detected_points - self._smoothed_points, axis=1)
            )
            reset_for_cut = displacement > 0.12 * np.hypot(1280, 720)

        if self._smoothed_points is None or reset_for_cut:
            self._smoothed_points = detected_points
        else:
            alpha = self.smoothing_factor
            self._smoothed_points = (
                (1.0 - alpha) * self._smoothed_points + alpha * detected_points
            )

        smoothed_matrix, _ = cv2.findHomography(
            refer_kps.reshape(-1, 2), self._smoothed_points, method=0
        )
        if smoothed_matrix is None or not is_plausible_homography(
            smoothed_matrix, (1280, 720)
        ):
            smoothed_matrix = estimate.matrix
            self._smoothed_points = detected_points

        self._last_matrix = smoothed_matrix
        self._stale_frames = 0
        projected = project_points(refer_kps.reshape(-1, 2), smoothed_matrix)
        detection = CourtDetection(
            self._keypoint_dicts(projected),
            smoothed_matrix.copy(),
            estimate.reprojection_error,
            estimate.inlier_count,
            True,
            0,
            scene_cut,
        )
        self._last_detection = detection
        return resized, detection
