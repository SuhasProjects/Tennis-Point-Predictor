from pathlib import Path

import cv2
import pytest

from court_detector import CourtDetector, PlayerPoseDetector

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_supplied_images_select_one_player_per_half():
    court_detector = CourtDetector(ROOT / "models" / "model_tennis_court_det.pt")
    player_detector = PlayerPoseDetector(ROOT / "models" / "yolov8s-pose.pt", conf=0.10)
    for image_path in sorted((ROOT / "data" / "Images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        frame, court = court_detector.track_frame(image, 0, detect_interval=1)
        detections = player_detector.detect_players(frame, court.matrix)
        assert court.matrix is not None
        assert court.inlier_count >= 12
        assert {detection.side for detection in detections} == {"near", "far"}
        court_detector.reset_tracking()
        player_detector.reset_tracking()
