"""
Unit tests for sports.annotators.soccer module.

These tests validate the drawing utilities including the bug fix for
draw_paths_on_pitch() where the return statement was incorrectly placed
inside the for-loop, causing only the first path to be drawn.
"""

import numpy as np
import pytest

from sports.annotators.soccer import (
    draw_pitch,
    draw_paths_on_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
)
from sports.configs.soccer import SoccerPitchConfiguration
import supervision as sv


@pytest.fixture
def config():
    return SoccerPitchConfiguration()


class TestDrawPitch:
    def test_returns_ndarray(self, config):
        pitch = draw_pitch(config)
        assert isinstance(pitch, np.ndarray)

    def test_default_shape(self, config):
        pitch = draw_pitch(config, scale=0.1, padding=50)
        expected_h = int(config.width * 0.1) + 2 * 50
        expected_w = int(config.length * 0.1) + 2 * 50
        assert pitch.shape == (expected_h, expected_w, 3)

    def test_background_color(self, config):
        color = sv.Color(0, 128, 0)
        pitch = draw_pitch(config, background_color=color)
        # corners should be close to the background color (BGR)
        assert pitch[5, 5, 1] > 100  # green channel dominant


class TestDrawPathsOnPitch:
    """Tests for draw_paths_on_pitch – including the critical return-inside-loop fix."""

    def test_single_path_drawn(self, config):
        path = np.array([[0, 0], [1000, 1000], [2000, 500]], dtype=float)
        pitch = draw_paths_on_pitch(config=config, paths=[path])
        assert isinstance(pitch, np.ndarray)

    def test_multiple_paths_all_drawn(self, config):
        """
        Regression test for the bug where `return pitch` was inside the loop,
        causing only the first path to be drawn.

        We compare two pitches:
        - one drawn with path_a alone
        - one drawn with both path_a and path_b

        If the second path is actually rendered, the two images must differ.
        """
        base = draw_pitch(config=config)

        path_a = np.array([[0, 0], [500, 0]], dtype=float)
        path_b = np.array([[6000, 3500], [7000, 3500]], dtype=float)

        # pitch with only path_a
        pitch_one = draw_paths_on_pitch(
            config=config, paths=[path_a], pitch=base.copy()
        )
        # pitch with both paths
        pitch_two = draw_paths_on_pitch(
            config=config, paths=[path_a, path_b], pitch=base.copy()
        )

        # The second path (path_b) changes pixels; the two images must differ.
        assert not np.array_equal(pitch_one, pitch_two), (
            "draw_paths_on_pitch returned early after first path "
            "(return-inside-loop bug not fixed)"
        )

    def test_path_too_short_skipped(self, config):
        """A single-point path should be silently skipped."""
        path = np.array([[1000, 1000]], dtype=float)
        pitch = draw_paths_on_pitch(config=config, paths=[path])
        assert isinstance(pitch, np.ndarray)

    def test_empty_paths_list(self, config):
        pitch = draw_paths_on_pitch(config=config, paths=[])
        assert isinstance(pitch, np.ndarray)

    def test_existing_pitch_reused(self, config):
        """When a pitch is passed in, it should be modified and returned."""
        existing = draw_pitch(config=config)
        path = np.array([[0, 0], [500, 500]], dtype=float)
        result = draw_paths_on_pitch(config=config, paths=[path], pitch=existing)
        assert result is existing


class TestDrawPointsOnPitch:
    def test_returns_ndarray(self, config):
        xy = np.array([[1000, 1000], [5000, 3500]], dtype=float)
        pitch = draw_points_on_pitch(config=config, xy=xy)
        assert isinstance(pitch, np.ndarray)

    def test_empty_points(self, config):
        xy = np.empty((0, 2), dtype=float)
        pitch = draw_points_on_pitch(config=config, xy=xy)
        assert isinstance(pitch, np.ndarray)


class TestDrawPitchVoronoiDiagram:
    def test_returns_ndarray(self, config):
        team1 = np.array([[1000, 3500], [3000, 1500]], dtype=float)
        team2 = np.array([[9000, 3500], [7000, 1500]], dtype=float)
        result = draw_pitch_voronoi_diagram(
            config=config, team_1_xy=team1, team_2_xy=team2
        )
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3
