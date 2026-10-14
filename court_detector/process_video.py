"""Reusable video processor built on the stateful court/player detectors."""

from __future__ import annotations

from pathlib import Path

import cv2

try:
    from .visualization import draw_court_overlay, draw_minimap
except ImportError:  # pragma: no cover
    from visualization import draw_court_overlay, draw_minimap


class CourtVideoProcessor:
    def __init__(self, court_detector, player_detector) -> None:
        self.court_detector = court_detector
        self.player_detector = player_detector

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        detect_interval: int = 5,
        smoothing_factor: float = 0.2,
    ) -> None:
        """Annotate a video using the same stable geometry as the CLI pipeline."""
        self.court_detector.smoothing_factor = float(smoothing_factor)
        self.court_detector.reset_tracking()
        self.player_detector.reset_tracking()
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video at {output_path}")

        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame, court = self.court_detector.track_frame(
                    frame, frame_index, detect_interval=detect_interval
                )
                if court.scene_cut:
                    self.player_detector.reset_tracking()
                detections = self.player_detector.detect_players(frame, court.matrix)
                draw_court_overlay(frame, court.matrix)
                self.player_detector.annotate(frame, detections)
                if court.matrix is not None:
                    draw_minimap(frame, detections)
                writer.write(frame)
                frame_index += 1
        finally:
            cap.release()
            writer.release()
