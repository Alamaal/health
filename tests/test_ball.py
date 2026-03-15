"""
Unit tests for sports.common.ball module.
"""

import numpy as np
import pytest
import supervision as sv

from sports.common.ball import BallAnnotator, BallTracker


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
