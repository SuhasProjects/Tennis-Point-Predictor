import numpy as np

from court_detector.homography import (
    estimate_homography,
    invert_homography,
    is_plausible_homography,
    project_points,
    refer_kps,
)

KNOWN_MATRIX = np.asarray(
    [[0.49, -0.11, 245.0], [0.002, 0.041, 240.0], [0.000002, -0.00018, 1.0]],
    dtype=np.float64,
)


def test_estimate_homography_is_robust_to_missing_and_outlier_points():
    canonical = refer_kps.reshape(-1, 2)
    observed = project_points(canonical, KNOWN_MATRIX)
    observed[3] += (120, -90)  # deliberately wrong landmark
    points = [tuple(point) for point in observed]
    points[10] = (None, None)
    estimate = estimate_homography(points)
    assert estimate is not None
    assert estimate.inlier_count >= 11
    expected = project_points(canonical, KNOWN_MATRIX)
    actual = project_points(canonical, estimate.matrix)
    errors = np.linalg.norm(expected - actual, axis=1)
    assert np.median(errors) < 2.0


def test_inverse_round_trip():
    inverse = invert_homography(KNOWN_MATRIX)
    assert inverse is not None
    points = np.asarray([[286, 561], [832, 1748], [1379, 2935]], np.float32)
    image = project_points(points, KNOWN_MATRIX)
    restored = project_points(image, inverse)
    np.testing.assert_allclose(restored, points, atol=0.01)


def test_rejects_degenerate_matrix():
    assert not is_plausible_homography(np.zeros((3, 3)))
    assert invert_homography(np.zeros((3, 3))) is None
    assert estimate_homography([(None, None)] * 14) is None


def test_rejects_underconstrained_partial_court_fit():
    observed = project_points(refer_kps.reshape(-1, 2), KNOWN_MATRIX)
    points = [(None, None)] * 14
    for index in (0, 1, 2, 3, 4):
        points[index] = tuple(observed[index])
    assert estimate_homography(points) is None
