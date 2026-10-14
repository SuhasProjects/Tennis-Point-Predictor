import numpy as np

from court_detector.player_detector import (
    PlayerDetection,
    PlayerPoseDetector,
    _TrackState,
)


def _detection(position: tuple[float, float]) -> PlayerDetection:
    point = np.asarray(position, dtype=np.float32)
    return PlayerDetection(
        side="far",
        bbox=np.zeros(4, dtype=np.float32),
        confidence=0.5,
        keypoints=np.zeros((17, 3), dtype=np.float32),
        foot_image=np.zeros(2, dtype=np.float32),
        court_position=point,
        selection_score=0.5,
    )


def test_track_update_smooths_single_bad_foot_observation() -> None:
    detector = PlayerPoseDetector.__new__(PlayerPoseDetector)
    detector.max_track_gap = 15
    detector._tracks = {
        "far": _TrackState(
            position=np.asarray([1000.0, 1500.0], dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
        )
    }

    detector._update_tracks([_detection((1000.0, 1200.0))])

    state = detector._tracks["far"]
    np.testing.assert_allclose(state.position, [1000.0, 1350.0])
    np.testing.assert_allclose(state.velocity, [0.0, -105.0])
    assert np.linalg.norm(np.asarray([1000.0, 1520.0]) - (state.position + state.velocity)) < 375

