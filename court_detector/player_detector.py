"""Pose inference and stable selection of the two active tennis players."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from .court_reference import CourtReference
    from .homography import invert_homography, project_points
except ImportError:  # pragma: no cover - direct script compatibility
    from court_reference import CourtReference
    from homography import invert_homography, project_points


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov8s-pose.pt"
PlayerSide = Literal["far", "near"]


@dataclass(frozen=True)
class PlayerDetection:
    side: PlayerSide
    bbox: np.ndarray
    confidence: float
    keypoints: np.ndarray
    foot_image: np.ndarray
    court_position: np.ndarray | None
    selection_score: float

    def as_record(self) -> dict:
        """Convert NumPy values to types accepted by JSON/database drivers."""
        return {
            "player_id": self.side,
            "bbox": self.bbox.astype(float).tolist(),
            "confidence": float(self.confidence),
            "keypoints": self.keypoints.astype(float).tolist(),
            "foot_image": self.foot_image.astype(float).tolist(),
            "court_position": (
                None
                if self.court_position is None
                else self.court_position.astype(float).tolist()
            ),
            "selection_score": float(self.selection_score),
        }


@dataclass
class _TrackState:
    position: np.ndarray
    velocity: np.ndarray
    missed_frames: int = 0


class PlayerPoseDetector:
    """Detect people, project their feet, and track one player per court half."""

    KEYPOINT_NAMES = (
        "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear", "L-Should",
        "R-Should", "L-Elbow", "R-Elbow", "L-Wrist", "R-Wrist",
        "L-Hip", "R-Hip", "L-Knee", "R-Knee", "L-Ankle", "R-Ankle",
    )
    SKELETON = (
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    )

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        conf: float = 0.10,
        imgsz: int = 1280,
        keypoint_conf: float = 0.25,
        max_track_gap: int = 15,
        allow_without_homography: bool = False,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.keypoint_conf = float(keypoint_conf)
        self.max_track_gap = int(max_track_gap)
        self.allow_without_homography = bool(allow_without_homography)
        self.reference = CourtReference()
        self.net_y = float(self.reference.net[0][1])
        self.center_x = (
            self.reference.left_court_line[0][0]
            + self.reference.right_court_line[0][0]
        ) / 2.0

        # Approximately 2.5 m side runoff and 6.5 m baseline runoff in the
        # canonical scale.  The lateral limit still covers normal defensive
        # movement, while excluding line judges and ball people beside the
        # doubles alley on wide broadcast shots.
        self.ref_x_min = self.reference.left_court_line[0][0] - 250
        self.ref_x_max = self.reference.right_court_line[0][0] + 250
        self.ref_y_min = self.reference.baseline_top[0][1] - 650
        self.ref_y_max = self.reference.baseline_bottom[0][1] + 650
        self._tracks: dict[PlayerSide, _TrackState] = {}

    def reset_tracking(self) -> None:
        self._tracks.clear()

    def is_in_playing_area(
        self, foot_x: float, foot_y: float, inv_matrix: np.ndarray | None
    ) -> bool:
        """Backward-compatible projected playing-area check."""
        if inv_matrix is None:
            return True
        point = project_points(np.asarray([[foot_x, foot_y]], np.float32), inv_matrix)[0]
        return bool(
            self.ref_x_min <= point[0] <= self.ref_x_max
            and self.ref_y_min <= point[1] <= self.ref_y_max
        )

    def _foot_position(self, keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        ankles = keypoints[[15, 16]]
        visible = (
            (ankles[:, 0] > 0)
            & (ankles[:, 1] > 0)
            & (ankles[:, 2] >= self.keypoint_conf)
        )
        if visible.any():
            return ankles[visible, :2].mean(axis=0).astype(np.float32)
        return np.asarray([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float32)

    def _location_score(self, side: PlayerSide, point: np.ndarray) -> float:
        x_score = max(0.0, 1.0 - abs(float(point[0]) - self.center_x) / 900.0)
        baseline_y = (
            float(self.reference.baseline_top[0][1])
            if side == "far"
            else float(self.reference.baseline_bottom[0][1])
        )
        y_score = max(0.0, 1.0 - abs(float(point[1]) - baseline_y) / 1500.0)
        return 0.64 * x_score + 0.36 * y_score

    def _score_candidate(
        self,
        side: PlayerSide,
        confidence: float,
        court_position: np.ndarray | None,
        image_position: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> float:
        if court_position is not None:
            location_score = self._location_score(side, court_position)
            state = self._tracks.get(side)
            if state is None:
                continuity = location_score
            else:
                predicted = state.position + state.velocity
                distance = float(np.linalg.norm(court_position - predicted))
                continuity = max(0.0, 1.0 - distance / 1100.0)
            return 0.42 * confidence + 0.33 * location_score + 0.25 * continuity

        # Homography unavailable: use conservative top/bottom and center priors.
        height, width = frame_shape[:2]
        x_score = max(0.0, 1.0 - abs(float(image_position[0]) - width / 2) / (width / 2))
        target_y = height * (0.32 if side == "far" else 0.78)
        y_score = max(0.0, 1.0 - abs(float(image_position[1]) - target_y) / height)
        return 0.5 * confidence + 0.3 * x_score + 0.2 * y_score

    def _update_tracks(self, selected: list[PlayerDetection]) -> None:
        selected_by_side = {detection.side: detection for detection in selected}
        for side in ("far", "near"):
            detection = selected_by_side.get(side)
            state = self._tracks.get(side)
            if detection is None or detection.court_position is None:
                if state is not None:
                    state.missed_frames += 1
                    if state.missed_frames > self.max_track_gap:
                        del self._tracks[side]
                continue
            if state is None:
                self._tracks[side] = _TrackState(
                    detection.court_position.copy(), np.zeros(2, dtype=np.float32)
                )
            else:
                displacement = detection.court_position - state.position
                state.velocity = 0.65 * state.velocity + 0.35 * displacement
                # Pose models occasionally return a torso-only box for a tiny
                # far player.  Its bbox bottom is then hundreds of canonical
                # units away from the real feet.  Keep the observation in the
                # output, but do not let one such frame move the tracking gate
                # far enough to reject the correct detection on the next
                # frame.
                state.position = (
                    0.5 * state.position + 0.5 * detection.court_position
                ).astype(np.float32)
                state.missed_frames = 0

    @staticmethod
    def _result_arrays(result, offset: tuple[int, int] = (0, 0)) -> list[tuple]:
        """Convert an Ultralytics result to image-space NumPy arrays."""
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
        xy = result.keypoints.xy.cpu().numpy().astype(np.float32)
        if result.keypoints.conf is None:
            keypoint_confidences = np.where(np.any(xy > 0, axis=2), 1.0, 0.0)
        else:
            keypoint_confidences = result.keypoints.conf.cpu().numpy().astype(np.float32)

        offset_array = np.asarray(offset, dtype=np.float32)
        boxes[:, [0, 2]] += offset_array[0]
        boxes[:, [1, 3]] += offset_array[1]
        visible = np.all(xy > 0, axis=2)
        xy[visible] += offset_array
        keypoints = np.concatenate([xy, keypoint_confidences[..., None]], axis=2)
        return list(zip(boxes, confidences, keypoints))

    def _append_candidates(
        self,
        raw_detections: list[tuple],
        inverse: np.ndarray | None,
        frame: np.ndarray,
        by_side: dict[PlayerSide, list[PlayerDetection]],
    ) -> None:
        for bbox, confidence, person_keypoints in raw_detections:
            foot = self._foot_position(person_keypoints, bbox)
            court_position = None
            if inverse is not None:
                court_position = project_points(foot.reshape(1, 2), inverse)[0]
                if not (
                    self.ref_x_min <= court_position[0] <= self.ref_x_max
                    and self.ref_y_min <= court_position[1] <= self.ref_y_max
                ):
                    continue
                side: PlayerSide = "far" if court_position[1] < self.net_y else "near"
                state = self._tracks.get(side)
                if state is not None:
                    predicted = state.position + state.velocity
                    track_is_peripheral = self._location_score(
                        side, state.position
                    ) < 0.45
                    candidate_is_central = self._location_score(
                        side, court_position
                    ) >= 0.60
                    if (
                        # Leave room for a small homography refresh jump.  The
                        # track state is smoothed below, so a single noisy pose
                        # cannot drag this wider gate across the court.
                        np.linalg.norm(court_position - predicted) > 375.0
                        and not (track_is_peripheral and candidate_is_central)
                    ):
                        continue
            else:
                side = "far" if foot[1] < frame.shape[0] * 0.53 else "near"
            score = self._score_candidate(
                side, float(confidence), court_position, foot, frame.shape
            )
            by_side[side].append(
                PlayerDetection(
                    side=side, bbox=bbox, confidence=float(confidence),
                    keypoints=person_keypoints, foot_image=foot,
                    court_position=court_position, selection_score=score,
                )
            )

    def _far_court_crop(
        self, frame: np.ndarray, matrix_trans: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int]] | None:
        canonical = np.asarray(
            [
                [self.ref_x_min, self.ref_y_min],
                [self.ref_x_max, self.ref_y_min],
                [self.ref_x_max, self.net_y],
                [self.ref_x_min, self.net_y],
            ],
            dtype=np.float32,
        )
        polygon = project_points(canonical, matrix_trans)
        if not np.all(np.isfinite(polygon)):
            return None
        height, width = frame.shape[:2]
        padding = 30
        x1 = max(0, int(np.floor(polygon[:, 0].min())) - padding)
        y1 = max(0, int(np.floor(polygon[:, 1].min())) - padding)
        x2 = min(width, int(np.ceil(polygon[:, 0].max())) + padding)
        y2 = min(height, int(np.ceil(polygon[:, 1].max())) + padding)
        if x2 - x1 < 128 or y2 - y1 < 96:
            return None
        return frame[y1:y2, x1:x2], (x1, y1)

    def detect_players(
        self, frame: np.ndarray, matrix_trans: np.ndarray | None = None
    ) -> list[PlayerDetection]:
        """Return at most one stable active-player detection per court half."""
        inverse = invert_homography(matrix_trans)
        if inverse is None and not self.allow_without_homography:
            self._update_tracks([])
            return []
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[0],
            max_det=24,
            verbose=False,
        )
        by_side: dict[PlayerSide, list[PlayerDetection]] = {"far": [], "near": []}
        self._append_candidates(self._result_arrays(results[0]), inverse, frame, by_side)

        # Tiny far players can vanish when a low-resolution broadcast is
        # letterboxed as a full frame. A court-guided crop gives that half a
        # second, higher-effective-resolution pass only when it is needed.
        far_location_scores = [
            self._location_score("far", candidate.court_position)
            for candidate in by_side["far"]
            if candidate.court_position is not None
        ]
        far_is_peripheral = bool(far_location_scores) and max(
            far_location_scores
        ) < 0.45
        if (not by_side["far"] or far_is_peripheral) and matrix_trans is not None:
            crop_data = self._far_court_crop(frame, matrix_trans)
            if crop_data is not None:
                crop, offset = crop_data
                crop_result = self.model(
                    crop, imgsz=self.imgsz, conf=min(self.conf, 0.05),
                    classes=[0], max_det=24, verbose=False,
                )[0]
                self._append_candidates(
                    self._result_arrays(crop_result, offset), inverse, frame, by_side
                )
                crop_location_scores = [
                    self._location_score("far", candidate.court_position)
                    for candidate in by_side["far"]
                    if candidate.court_position is not None
                ]
                still_needs_detail = not crop_location_scores or max(
                    crop_location_scores
                ) < 0.45
                if still_needs_detail and self.imgsz < 1600:
                    detailed_result = self.model(
                        crop, imgsz=1600, conf=min(self.conf, 0.05),
                        classes=[0], max_det=24, verbose=False,
                    )[0]
                    self._append_candidates(
                        self._result_arrays(detailed_result, offset),
                        inverse,
                        frame,
                        by_side,
                    )

        selected: list[PlayerDetection] = []
        for side, values in by_side.items():
            if not values:
                continue
            state = self._tracks.get(side)
            if state is not None and self._location_score(side, state.position) < 0.45:
                central = [
                    candidate
                    for candidate in values
                    if candidate.court_position is not None
                    and self._location_score(side, candidate.court_position) >= 0.60
                ]
                if central:
                    values = central
            selected.append(max(values, key=lambda item: item.selection_score))
        self._update_tracks(selected)
        return sorted(selected, key=lambda item: item.side)

    def annotate(self, frame: np.ndarray, detections: list[PlayerDetection]) -> np.ndarray:
        colors = {"near": (40, 220, 40), "far": (20, 170, 255)}
        for detection in detections:
            color = colors[detection.side]
            points = detection.keypoints
            visible = (
                (points[:, 0] > 0)
                & (points[:, 1] > 0)
                & (points[:, 2] >= self.keypoint_conf)
            )
            for start, end in self.SKELETON:
                if visible[start] and visible[end]:
                    cv2.line(
                        frame,
                        tuple(np.rint(points[start, :2]).astype(int)),
                        tuple(np.rint(points[end, :2]).astype(int)),
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            for point in points[visible, :2]:
                cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)

            x1, y1, x2, y2 = np.rint(detection.bbox).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.side} {detection.confidence:.2f}"
            if detection.court_position is not None:
                label += f" ({detection.court_position[0]:.0f},{detection.court_position[1]:.0f})"
            cv2.putText(
                frame, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 1, cv2.LINE_AA,
            )
        return frame
