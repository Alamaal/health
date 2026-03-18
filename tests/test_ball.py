"""
Unit tests for sports.common.ball module.
"""

import numpy as np
import pytest
import supervision as sv

from sports.common.ball import BallAnnotator, BallSmoother, BallTracker


def _make_detections(xyxy: list) -> sv.Detections:
    """Helper to create sv.Detections from a list of bounding boxes."""
    if len(xyxy) == 0:
        return sv.Detections.empty()
    boxes = np.array(xyxy, dtype=np.float32)
    return sv.Detections(xyxy=boxes)


class TestBallAnnotator:
    def test_annotate_returns_frame(self):
        annotator = BallAnnotator(radius=10, buffer_size=5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = _make_detections([[40, 40, 60, 60]])
        result = annotator.annotate(frame, detections)
        assert result.shape == frame.shape

    def test_annotate_empty_detections(self):
        annotator = BallAnnotator(radius=10, buffer_size=5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = sv.Detections.empty()
        result = annotator.annotate(frame, detections)
        assert result.shape == frame.shape

    def test_interpolate_radius_single_element(self):
        annotator = BallAnnotator(radius=10)
        assert annotator.interpolate_radius(0, 1) == 10

    def test_interpolate_radius_range(self):
        annotator = BallAnnotator(radius=10)
        r_start = annotator.interpolate_radius(0, 5)
        r_end = annotator.interpolate_radius(4, 5)
        assert r_start <= r_end
        assert r_end == 10


class TestBallTracker:
    def test_update_no_detections_returns_empty(self):
        tracker = BallTracker(buffer_size=10)
        detections = sv.Detections.empty()
        result = tracker.update(detections)
        assert len(result) == 0

    def test_update_single_detection_returns_it(self):
        tracker = BallTracker(buffer_size=10)
        detections = _make_detections([[45, 45, 55, 55]])
        result = tracker.update(detections)
        assert len(result) == 1

    def test_update_multiple_detections_returns_one(self):
        tracker = BallTracker(buffer_size=10)
        # Seed the buffer first
        tracker.update(_make_detections([[45, 45, 55, 55]]))
        tracker.update(_make_detections([[46, 46, 56, 56]]))

        # Now present two candidates; tracker should pick the closest to centroid
        detections = _make_detections([
            [45, 45, 55, 55],   # close to centroid
            [90, 90, 100, 100],  # far from centroid
        ])
        result = tracker.update(detections)
        assert len(result) == 1
        # The returned box should be the one close to centroid
        center_x = (result.xyxy[0][0] + result.xyxy[0][2]) / 2
        assert center_x < 60


class TestBallSmoother:
    """Tests for the BallSmoother online jitter-removal filter."""

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            BallSmoother(window=0)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError):
            BallSmoother(window=-1)

    def test_negative_noise_floor_raises(self):
        with pytest.raises(ValueError):
            BallSmoother(noise_floor_px=-1.0)

    def test_zero_max_velocity_raises(self):
        with pytest.raises(ValueError):
            BallSmoother(max_velocity_px_per_frame=0.0)

    def test_negative_max_velocity_raises(self):
        with pytest.raises(ValueError):
            BallSmoother(max_velocity_px_per_frame=-5.0)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def test_update_none_before_any_detection_returns_none(self):
        """Returns None when no position has ever been accepted."""
        smoother = BallSmoother()
        assert smoother.update(None) is None

    def test_first_valid_position_is_accepted(self):
        """The very first position is always accepted (no previous reference)."""
        smoother = BallSmoother(noise_floor_px=5.0)
        result = smoother.update(np.array([100.0, 200.0]))
        assert result is not None
        np.testing.assert_array_almost_equal(result, [100.0, 200.0])

    # ------------------------------------------------------------------
    # Moving-average smoothing
    # ------------------------------------------------------------------

    def test_moving_average_converges(self):
        """Smoothed position is the mean of accepted positions in the window."""
        smoother = BallSmoother(window=3, noise_floor_px=0.0, max_velocity_px_per_frame=1e9)
        positions = [
            np.array([0.0, 0.0]),
            np.array([6.0, 0.0]),
            np.array([12.0, 0.0]),
        ]
        for pos in positions:
            result = smoother.update(pos)
        # After 3 updates, result should be mean of all three.
        expected_x = (0.0 + 6.0 + 12.0) / 3.0
        assert abs(result[0] - expected_x) < 1e-6

    def test_window_slides(self):
        """Oldest position is dropped when the window is full."""
        smoother = BallSmoother(window=2, noise_floor_px=0.0, max_velocity_px_per_frame=1e9)
        smoother.update(np.array([0.0, 0.0]))
        smoother.update(np.array([10.0, 0.0]))
        result = smoother.update(np.array([20.0, 0.0]))
        # Window contains [10, 20]; mean = 15.
        assert abs(result[0] - 15.0) < 1e-6

    # ------------------------------------------------------------------
    # Velocity clamping
    # ------------------------------------------------------------------

    def test_velocity_clamp_discards_large_jump(self):
        """Positions beyond max_velocity_px_per_frame are rejected."""
        smoother = BallSmoother(
            window=3, noise_floor_px=0.0, max_velocity_px_per_frame=10.0
        )
        first = smoother.update(np.array([100.0, 100.0]))
        # Jump of 1000 px — far beyond the 10 px/frame limit.
        result = smoother.update(np.array([1100.0, 100.0]))
        # Should hold the previous smoothed position.
        np.testing.assert_array_almost_equal(result, first)

    def test_velocity_clamp_accepts_small_jump(self):
        """Positions within max_velocity_px_per_frame are accepted."""
        smoother = BallSmoother(
            window=1, noise_floor_px=0.0, max_velocity_px_per_frame=50.0
        )
        smoother.update(np.array([100.0, 100.0]))
        result = smoother.update(np.array([130.0, 100.0]))  # 30 px < 50 px limit
        # With window=1, result equals the accepted position itself.
        np.testing.assert_array_almost_equal(result, [130.0, 100.0])

    # ------------------------------------------------------------------
    # Noise floor
    # ------------------------------------------------------------------

    def test_noise_floor_rejects_tiny_movement(self):
        """Sub-floor movements are ignored and the previous position is held."""
        smoother = BallSmoother(
            window=1, noise_floor_px=5.0, max_velocity_px_per_frame=1e9
        )
        smoother.update(np.array([100.0, 100.0]))
        result = smoother.update(np.array([100.0, 102.0]))  # 2 px < 5 px floor
        # Position should be held at [100, 100].
        np.testing.assert_array_almost_equal(result, [100.0, 100.0])

    def test_noise_floor_accepts_sufficient_movement(self):
        """Movements above the noise floor are accepted."""
        smoother = BallSmoother(
            window=1, noise_floor_px=5.0, max_velocity_px_per_frame=1e9
        )
        smoother.update(np.array([100.0, 100.0]))
        result = smoother.update(np.array([100.0, 110.0]))  # 10 px > 5 px floor
        np.testing.assert_array_almost_equal(result, [100.0, 110.0])

    # ------------------------------------------------------------------
    # None input after valid detection
    # ------------------------------------------------------------------

    def test_none_after_detection_returns_last_smoothed(self):
        """When the ball is not detected, the last smoothed position is returned."""
        smoother = BallSmoother(window=3, noise_floor_px=0.0, max_velocity_px_per_frame=1e9)
        smoother.update(np.array([50.0, 80.0]))
        smoother.update(np.array([55.0, 82.0]))
        last = smoother.update(np.array([60.0, 84.0]))
        result = smoother.update(None)  # No detection this frame.
        np.testing.assert_array_almost_equal(result, last)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_state(self):
        """After reset(), the smoother behaves as if freshly constructed."""
        smoother = BallSmoother()
        smoother.update(np.array([100.0, 100.0]))
        smoother.reset()
        assert smoother.update(None) is None
        result = smoother.update(np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])
