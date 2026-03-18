"""
Unit tests for sports.common.stats module.

Covers:
- TeamVoteBuffer: temporal team-assignment stabilisation
- PassDetector: pass quality filters (speed, distance, debounce, ID switch guard),
  pass counting, and last-pass event tracking
- PossessionTracker: proximity-weighted possession counting
- PassEvent: dataclass record of a single pass
- draw_stats_overlay / draw_pass_label: video overlay helpers
- compute_stable_homography / transform_points_homography: pitch heatmap helpers
- ball_progression: net ball progression from X-track data
- defensive_leakage: opponent receptions inside penalty box
- vertical_lane_density: player/ball density across vertical lanes
- possession_by_thirds: possession share per pitch third
- team_compactness: team bounding-box area
- space_creation: attacker–defence centroid distance
- expected_threat: static xT value for ball position
- pitch_width_utilization: percentage activity per vertical lane
- _arr_summary: descriptive statistics helper
- build_match_report: comprehensive JSON match report
"""

import json

import numpy as np
import pytest

from sports.common.stats import (
    PassDetector,
    PassEvent,
    PossessionTracker,
    TeamVoteBuffer,
    _arr_summary,
    ball_progression,
    build_match_report,
    compute_stable_homography,
    defensive_leakage,
    draw_pass_label,
    draw_stats_overlay,
    expected_threat,
    perspective_owner_dist,
    pitch_width_utilization,
    possession_by_thirds,
    space_creation,
    team_compactness,
    transform_points_homography,
    vertical_lane_density,
)


# ---------------------------------------------------------------------------
# TeamVoteBuffer
# ---------------------------------------------------------------------------

class TestTeamVoteBuffer:
    """Team vote buffer stabilises noisy per-frame classifier predictions."""

    def test_returns_raw_below_min_votes(self):
        """Before min_votes accumulate, the raw prediction is returned."""
        buf = TeamVoteBuffer(buffer_size=64, min_votes=8)
        result = buf.update(canonical_id=1, raw_team_id=1)
        assert result == 1  # only 1 vote → return raw

    def test_majority_wins_after_min_votes(self):
        """After min_votes frames, the majority team is returned."""
        buf = TeamVoteBuffer(buffer_size=64, min_votes=4)
        # 3 votes for team 0 (wrong), then 5 votes for team 1 (correct)
        for _ in range(3):
            buf.update(1, 0)
        for _ in range(5):
            buf.update(1, 1)
        result = buf.update(1, 1)
        assert result == 1  # majority is team 1

    def test_buffer_size_limits_history(self):
        """Old votes are dropped when the buffer is full."""
        buf = TeamVoteBuffer(buffer_size=4, min_votes=4)
        # Fill with team 0 first
        for _ in range(4):
            buf.update(1, 0)
        # Now push 4 team-1 votes — old team-0 votes are overwritten
        for _ in range(4):
            buf.update(1, 1)
        result = buf.update(1, 1)
        assert result == 1  # only team 1 in the window

    def test_invalid_team_not_stored(self):
        """Invalid team IDs (-1) are not counted in the buffer."""
        buf = TeamVoteBuffer(buffer_size=64, min_votes=3)
        buf.update(1, -1)
        buf.update(1, -1)
        # Still below min_votes because -1 was not stored
        result = buf.update(1, -1)
        assert result == -1  # raw returned when no valid votes

    def test_get_stable_team_returns_none_before_min(self):
        """get_stable_team returns None before min_votes are accumulated."""
        buf = TeamVoteBuffer(buffer_size=64, min_votes=5)
        buf.update(2, 0)
        buf.update(2, 0)
        assert buf.get_stable_team(2) is None

    def test_get_stable_team_returns_correct_after_min(self):
        """get_stable_team returns the majority team after min_votes."""
        buf = TeamVoteBuffer(buffer_size=64, min_votes=3)
        for _ in range(3):
            buf.update(3, 1)
        assert buf.get_stable_team(3) == 1

    def test_separate_canonical_ids_are_independent(self):
        """Different canonical IDs maintain independent vote histories."""
        buf = TeamVoteBuffer(buffer_size=64, min_votes=3)
        for _ in range(5):
            buf.update(10, 0)
        for _ in range(5):
            buf.update(20, 1)
        assert buf.get_stable_team(10) == 0
        assert buf.get_stable_team(20) == 1


# ---------------------------------------------------------------------------
# PassDetector
# ---------------------------------------------------------------------------

class TestPassDetector:
    """Pass detector correctly applies all quality filters."""

    def _make_detector(self, **kwargs):
        defaults = dict(
            min_ball_speed_px_per_sec=300.0,
            pass_debounce_sec=1.35,
            pass_min_dist_px=85.0,
            id_switch_guard_px=65.0,
            fps=25.0,
        )
        defaults.update(kwargs)
        return PassDetector(**defaults)

    def _valid_pass_kwargs(self):
        return dict(
            prev_canonical_id=1,
            prev_team_id=0,
            new_canonical_id=2,
            new_team_id=0,
            prev_player_xy=np.array([100.0, 200.0]),
            new_player_xy=np.array([300.0, 200.0]),
            ball_speed_px_per_sec=500.0,
            ball_displacement_px=200.0,
            time_sec=5.0,
        )

    # --- Basic valid pass ---

    def test_valid_pass_accepted(self):
        d = self._make_detector()
        assert d.check(**self._valid_pass_kwargs()) is True

    # --- Same-team same-player: not a pass ---

    def test_same_player_not_a_pass(self):
        d = self._make_detector()
        kwargs = self._valid_pass_kwargs()
        kwargs["new_canonical_id"] = kwargs["prev_canonical_id"]
        assert d.check(**kwargs) is False

    # --- Cross-team transfer: not a pass (turnover) ---

    def test_cross_team_not_a_pass(self):
        d = self._make_detector()
        kwargs = self._valid_pass_kwargs()
        kwargs["new_team_id"] = 1  # opponent receives
        assert d.check(**kwargs) is False

    # --- Ball speed too low ---

    def test_slow_ball_rejected(self):
        d = self._make_detector()
        kwargs = self._valid_pass_kwargs()
        kwargs["ball_speed_px_per_sec"] = 100.0  # below 300 threshold
        assert d.check(**kwargs) is False

    # --- Ball displacement too short ---

    def test_short_displacement_rejected(self):
        d = self._make_detector()
        kwargs = self._valid_pass_kwargs()
        kwargs["ball_displacement_px"] = 30.0  # below 85 threshold
        assert d.check(**kwargs) is False

    # --- ID switch guard ---

    def test_id_switch_guard_rejects_close_players(self):
        """Two players less than id_switch_guard_px apart = same physical player."""
        d = self._make_detector()
        kwargs = self._valid_pass_kwargs()
        kwargs["prev_player_xy"] = np.array([100.0, 200.0])
        kwargs["new_player_xy"] = np.array([100.0 + 30.0, 200.0])  # 30 px apart < 65
        assert d.check(**kwargs) is False

    def test_id_switch_guard_accepts_far_players(self):
        """Players further than id_switch_guard_px apart should not be blocked."""
        d = self._make_detector()
        kwargs = self._valid_pass_kwargs()
        kwargs["prev_player_xy"] = np.array([100.0, 200.0])
        kwargs["new_player_xy"] = np.array([300.0, 200.0])  # 200 px apart > 65
        assert d.check(**kwargs) is True

    # --- Debounce ---

    def test_debounce_rejects_too_soon(self):
        d = self._make_detector(pass_debounce_sec=1.35)
        kwargs = self._valid_pass_kwargs()
        # First pass at t=5.0
        d.check(**kwargs)
        # Second pass at t=5.5 (only 0.5s later < 1.35s debounce)
        kwargs2 = dict(kwargs)
        kwargs2["prev_canonical_id"] = 2
        kwargs2["new_canonical_id"] = 3
        kwargs2["time_sec"] = 5.5
        assert d.check(**kwargs2) is False

    def test_debounce_accepts_after_window(self):
        d = self._make_detector(pass_debounce_sec=1.35)
        kwargs = self._valid_pass_kwargs()
        d.check(**kwargs)
        # Second pass at t=7.0 (2.0s later > 1.35s debounce)
        kwargs2 = dict(kwargs)
        kwargs2["prev_canonical_id"] = 2
        kwargs2["new_canonical_id"] = 3
        kwargs2["time_sec"] = 7.0
        assert d.check(**kwargs2) is True

    def test_debounce_independent_per_team(self):
        """Debounce is tracked separately for each team."""
        d = self._make_detector(pass_debounce_sec=1.35)
        # Team 0 passes at t=5.0
        d.check(**self._valid_pass_kwargs())
        # Team 1 passes at t=5.3 — should NOT be blocked by team 0's debounce
        kwargs1 = self._valid_pass_kwargs()
        kwargs1["prev_team_id"] = 1
        kwargs1["new_team_id"] = 1
        kwargs1["prev_canonical_id"] = 10
        kwargs1["new_canonical_id"] = 11
        kwargs1["time_sec"] = 5.3
        assert d.check(**kwargs1) is True

    # --- None player positions ---

    def test_none_positions_skip_id_guard(self):
        """When positions are None, the ID-switch guard is not applied."""
        d = self._make_detector()
        kwargs = self._valid_pass_kwargs()
        kwargs["prev_player_xy"] = None
        kwargs["new_player_xy"] = None
        # Still accepted (other filters pass)
        assert d.check(**kwargs) is True

    # --- None canonical IDs ---

    def test_none_canonical_id_rejected(self):
        d = self._make_detector()
        kwargs = self._valid_pass_kwargs()
        kwargs["prev_canonical_id"] = None
        assert d.check(**kwargs) is False

    # --- reset ---

    def test_reset_clears_debounce(self):
        d = self._make_detector(pass_debounce_sec=1.35)
        kwargs = self._valid_pass_kwargs()
        d.check(**kwargs)  # t=5.0
        d.reset()
        # After reset, same team can pass immediately
        kwargs2 = dict(kwargs)
        kwargs2["prev_canonical_id"] = 2
        kwargs2["new_canonical_id"] = 3
        kwargs2["time_sec"] = 5.5
        assert d.check(**kwargs2) is True

    # --- Sequential passes counted correctly ---

    def test_sequential_passes_counted(self):
        """Each well-spaced pass by the same team is counted once."""
        d = self._make_detector(pass_debounce_sec=1.0)
        base_kwargs = self._valid_pass_kwargs()
        accepted = 0
        for i in range(5):
            kwargs = dict(base_kwargs)
            kwargs["prev_canonical_id"] = i * 2
            kwargs["new_canonical_id"] = i * 2 + 1
            kwargs["time_sec"] = float(i * 2)
            if d.check(**kwargs):
                accepted += 1
        assert accepted == 5


# ---------------------------------------------------------------------------
# PossessionTracker
# ---------------------------------------------------------------------------

class TestPossessionTracker:
    """Proximity-weighted possession counting."""

    def test_weight_at_zero_distance(self):
        """Ball exactly at player's feet → weight = 1.0."""
        pt = PossessionTracker(max_owner_dist_px=80.0)
        w = pt.update(0, np.array([100.0, 100.0]), np.array([100.0, 100.0]))
        assert abs(w - 1.0) < 1e-9

    def test_weight_at_max_distance(self):
        """Ball at maximum ownership distance → weight = 0.5."""
        pt = PossessionTracker(max_owner_dist_px=80.0)
        w = pt.update(0, np.array([100.0, 100.0]), np.array([180.0, 100.0]))
        assert abs(w - 0.5) < 1e-9

    def test_weight_in_between(self):
        """Weight is interpolated linearly between 0.5 and 1.0."""
        pt = PossessionTracker(max_owner_dist_px=80.0)
        # Distance = 40 px → weight = 1.0 - 0.5*(40/80) = 0.75
        w = pt.update(0, np.array([100.0, 100.0]), np.array([140.0, 100.0]))
        assert abs(w - 0.75) < 1e-9

    def test_accumulation_across_frames(self):
        """Weighted frames accumulate correctly over multiple updates."""
        pt = PossessionTracker(max_owner_dist_px=80.0)
        # 2 frames at weight=1.0 and 1 frame at weight=0.5
        pt.update(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))    # w=1.0
        pt.update(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))    # w=1.0
        pt.update(0, np.array([0.0, 0.0]), np.array([80.0, 0.0]))   # w=0.5
        assert abs(pt.weighted_frames(0) - 2.5) < 1e-9

    def test_teams_are_independent(self):
        """Possession of team 0 and team 1 are tracked separately."""
        pt = PossessionTracker(max_owner_dist_px=80.0)
        pt.update(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))    # team 0 w=1.0
        pt.update(1, np.array([0.0, 0.0]), np.array([40.0, 0.0]))   # team 1 w=0.75
        assert abs(pt.weighted_frames(0) - 1.0) < 1e-9
        assert abs(pt.weighted_frames(1) - 0.75) < 1e-9

    def test_possession_pct_total_frames(self):
        """possession_pct divides by total_frames correctly."""
        pt = PossessionTracker(max_owner_dist_px=80.0)
        # 2 full-weight frames out of 4 total → 50%
        pt.update(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        pt.update(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        pct = pt.possession_pct(0, total_frames=4)
        assert abs(pct - 50.0) < 1e-9

    def test_possession_pct_zero_total_frames(self):
        """possession_pct returns 0 when total_frames=0 (no division by zero)."""
        pt = PossessionTracker()
        pt.update(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        assert pt.possession_pct(0, total_frames=0) == 0.0

    def test_possession_pct_normalized_sums_to_100(self):
        """possession_pct_normalized sums to exactly 100% across two teams."""
        pt = PossessionTracker(max_owner_dist_px=80.0)
        origin = np.array([0.0, 0.0])
        for _ in range(3):
            pt.update(0, origin, origin)
        for _ in range(7):
            pt.update(1, origin, origin)
        total = pt.possession_pct_normalized(0) + pt.possession_pct_normalized(1)
        assert abs(total - 100.0) < 1e-9

    def test_possession_pct_normalized_proportions(self):
        """possession_pct_normalized reflects the correct share per team."""
        pt = PossessionTracker(max_owner_dist_px=80.0)
        origin = np.array([0.0, 0.0])
        for _ in range(1):
            pt.update(0, origin, origin)
        for _ in range(3):
            pt.update(1, origin, origin)
        assert abs(pt.possession_pct_normalized(0) - 25.0) < 1e-9
        assert abs(pt.possession_pct_normalized(1) - 75.0) < 1e-9

    def test_possession_pct_normalized_no_data(self):
        """possession_pct_normalized returns 0.0 when nothing has been recorded."""
        pt = PossessionTracker()
        assert pt.possession_pct_normalized(0) == 0.0

    def test_reset_clears_data(self):
        """reset() zeroes all accumulated possession."""
        pt = PossessionTracker()
        pt.update(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        pt.reset()
        assert pt.weighted_frames(0) == 0.0
        assert pt.possession_pct(0, total_frames=10) == 0.0



# ---------------------------------------------------------------------------
# perspective_owner_dist
# ---------------------------------------------------------------------------

class TestPerspectiveOwnerDist:
    """perspective_owner_dist scales ownership radius with vertical position."""

    def test_top_of_frame_returns_far_scale(self):
        """Player at y=0 (top) should return far_scale * base."""
        result = perspective_owner_dist(player_y=0, frame_h=1080, base_dist_px=120,
                                       near_scale=1.4, far_scale=0.6)
        assert abs(result - 120 * 0.6) < 1e-9

    def test_bottom_of_frame_returns_near_scale(self):
        """Player at y=frame_h (bottom) should return near_scale * base."""
        result = perspective_owner_dist(player_y=1080, frame_h=1080, base_dist_px=120,
                                       near_scale=1.4, far_scale=0.6)
        assert abs(result - 120 * 1.4) < 1e-9

    def test_mid_frame_is_average(self):
        """Player at mid-height should return average of near and far scales."""
        result = perspective_owner_dist(player_y=540, frame_h=1080, base_dist_px=100,
                                       near_scale=1.4, far_scale=0.6)
        expected = 100 * (0.6 + (1.4 - 0.6) * 0.5)
        assert abs(result - expected) < 1e-9

    def test_zero_frame_height_returns_base(self):
        """Zero frame height should not raise and should return the base distance."""
        result = perspective_owner_dist(player_y=500, frame_h=0, base_dist_px=120)
        assert result == 120.0

    def test_out_of_bounds_y_is_clamped(self):
        """y > frame_h is clamped to the near-side maximum."""
        result_high = perspective_owner_dist(player_y=2000, frame_h=1080, base_dist_px=100,
                                             near_scale=1.4, far_scale=0.6)
        result_low = perspective_owner_dist(player_y=-100, frame_h=1080, base_dist_px=100,
                                            near_scale=1.4, far_scale=0.6)
        assert abs(result_high - 100 * 1.4) < 1e-9
        assert abs(result_low - 100 * 0.6) < 1e-9


# ---------------------------------------------------------------------------
# PassEvent
# ---------------------------------------------------------------------------

class TestPassEvent:
    """PassEvent dataclass stores pass details."""

    def test_fields(self):
        e = PassEvent(team_id=0, time_sec=5.5, from_player_id=1, to_player_id=2)
        assert e.team_id == 0
        assert e.time_sec == 5.5
        assert e.from_player_id == 1
        assert e.to_player_id == 2


# ---------------------------------------------------------------------------
# PassDetector pass counts & last event
# ---------------------------------------------------------------------------

class TestPassDetectorCounts:
    """Pass counting and last_pass_event property."""

    def _make_detector(self, **kwargs):
        defaults = dict(
            min_ball_speed_px_per_sec=300.0,
            pass_debounce_sec=1.35,
            pass_min_dist_px=85.0,
            id_switch_guard_px=65.0,
            fps=25.0,
        )
        defaults.update(kwargs)
        return PassDetector(**defaults)

    def _valid_pass_kwargs(self, time_sec=5.0):
        return dict(
            prev_canonical_id=1,
            prev_team_id=0,
            new_canonical_id=2,
            new_team_id=0,
            prev_player_xy=np.array([100.0, 200.0]),
            new_player_xy=np.array([300.0, 200.0]),
            ball_speed_px_per_sec=500.0,
            ball_displacement_px=200.0,
            time_sec=time_sec,
        )

    def test_pass_count_increments(self):
        """get_pass_count increments when a pass is accepted."""
        d = self._make_detector()
        assert d.get_pass_count(0) == 0
        d.check(**self._valid_pass_kwargs())
        assert d.get_pass_count(0) == 1

    def test_pass_count_per_team(self):
        """Pass counts are independent per team."""
        d = self._make_detector(pass_debounce_sec=0.5)
        d.check(**self._valid_pass_kwargs(time_sec=0.0))
        kw1 = self._valid_pass_kwargs(time_sec=2.0)
        kw1["prev_team_id"] = 1
        kw1["new_team_id"] = 1
        kw1["prev_canonical_id"] = 10
        kw1["new_canonical_id"] = 11
        d.check(**kw1)
        assert d.get_pass_count(0) == 1
        assert d.get_pass_count(1) == 1

    def test_last_pass_event_set(self):
        """last_pass_event is populated after a successful check."""
        d = self._make_detector()
        assert d.last_pass_event is None
        d.check(**self._valid_pass_kwargs())
        ev = d.last_pass_event
        assert ev is not None
        assert ev.team_id == 0
        assert ev.time_sec == 5.0
        assert ev.from_player_id == 1
        assert ev.to_player_id == 2

    def test_last_pass_event_not_set_on_reject(self):
        """last_pass_event stays None when a pass is rejected."""
        d = self._make_detector()
        kw = self._valid_pass_kwargs()
        kw["ball_speed_px_per_sec"] = 10.0  # too slow
        d.check(**kw)
        assert d.last_pass_event is None

    def test_reset_clears_counts_and_event(self):
        """reset() clears pass counts and last_pass_event."""
        d = self._make_detector()
        d.check(**self._valid_pass_kwargs())
        assert d.get_pass_count(0) == 1
        assert d.last_pass_event is not None
        d.reset()
        assert d.get_pass_count(0) == 0
        assert d.last_pass_event is None


# ---------------------------------------------------------------------------
# draw_stats_overlay & draw_pass_label
# ---------------------------------------------------------------------------

class TestOverlayHelpers:
    """Video overlay drawing helpers."""

    def _dummy_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_stats_overlay_returns_frame(self):
        """draw_stats_overlay returns an ndarray of the same shape."""
        frame = self._dummy_frame()
        d = PassDetector()
        pt = PossessionTracker()
        result = draw_stats_overlay(frame, d, pt, frame_index=100)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_draw_stats_overlay_writes_pixels(self):
        """draw_stats_overlay modifies pixels (not a no-op)."""
        frame = self._dummy_frame()
        d = PassDetector()
        pt = PossessionTracker()
        draw_stats_overlay(frame, d, pt, frame_index=100)
        assert frame.sum() > 0

    def test_draw_pass_label_returns_frame(self):
        """draw_pass_label returns an ndarray of the same shape."""
        frame = self._dummy_frame()
        ev = PassEvent(team_id=0, time_sec=3.5, from_player_id=1, to_player_id=2)
        result = draw_pass_label(frame, ev, np.array([320.0, 240.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_draw_pass_label_writes_pixels(self):
        """draw_pass_label modifies pixels."""
        frame = self._dummy_frame()
        ev = PassEvent(team_id=0, time_sec=3.5, from_player_id=1, to_player_id=2)
        draw_pass_label(frame, ev, np.array([320.0, 240.0]))
        assert frame.sum() > 0


# ---------------------------------------------------------------------------
# compute_stable_homography & transform_points_homography
# ---------------------------------------------------------------------------

class TestPitchHomographyHelpers:
    """Tests for the pitch heatmap homography helpers."""

    # A minimal set of four coplanar source ↔ target point pairs that span a
    # region large enough for findHomography to succeed.
    _SRC = np.array([
        [100.0, 100.0],
        [900.0, 100.0],
        [900.0, 600.0],
        [100.0, 600.0],
    ], dtype=np.float32)

    _DST = np.array([
        [0.0, 0.0],
        [105.0, 0.0],
        [105.0, 68.0],
        [0.0, 68.0],
    ], dtype=np.float32)

    def test_homography_from_single_frame(self):
        """compute_stable_homography returns a (3,3) matrix for one frame."""
        H = compute_stable_homography([self._SRC], self._DST)
        assert H is not None
        assert H.shape == (3, 3)

    def test_homography_from_multiple_frames(self):
        """compute_stable_homography pools keypoints from multiple frames."""
        H = compute_stable_homography([self._SRC, self._SRC], self._DST)
        assert H is not None
        assert H.shape == (3, 3)

    def test_homography_none_when_too_few_points(self):
        """Returns None when fewer than 4 valid source points are available."""
        src_few = np.array([[100.0, 100.0], [900.0, 100.0]], dtype=np.float32)
        H = compute_stable_homography([src_few], self._DST[:2])
        assert H is None

    def test_zero_keypoints_filtered_out(self):
        """Points at (0, 0) or (1, 1) are treated as undetected and skipped."""
        src_with_zeros = np.array([
            [0.0, 0.0],   # undetected — filtered out
            [1.0, 1.0],   # undetected — filtered out
            [100.0, 100.0],
            [900.0, 100.0],
            [900.0, 600.0],
            [100.0, 600.0],
        ], dtype=np.float32)
        dst_with_padding = np.vstack([
            np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            self._DST,
        ])
        H = compute_stable_homography([src_with_zeros], dst_with_padding)
        assert H is not None

    def test_transform_matches_expected(self):
        """transform_points_homography maps source corners to target corners."""
        H = compute_stable_homography([self._SRC], self._DST)
        assert H is not None
        transformed = transform_points_homography(self._SRC, H)
        assert transformed.shape == (4, 2)
        np.testing.assert_allclose(transformed, self._DST, atol=1.0)

    def test_transform_empty_input(self):
        """transform_points_homography handles an empty input array gracefully."""
        H = compute_stable_homography([self._SRC], self._DST)
        assert H is not None
        empty = np.empty((0, 2), dtype=np.float32)
        result = transform_points_homography(empty, H)
        assert result.shape == (0, 2)

    def test_homography_empty_keypoints_list(self):
        """compute_stable_homography returns None for an empty keypoints list."""
        assert compute_stable_homography([], self._DST) is None

    def test_homography_ignores_bad_frame_shape(self):
        """Frames with wrong array shapes are skipped without raising."""
        bad_frame = np.array([1, 2, 3], dtype=np.float32)  # 1-D, wrong shape
        H = compute_stable_homography([bad_frame, self._SRC], self._DST)
        assert H is not None  # falls back to the valid frame


# ---------------------------------------------------------------------------
# ball_progression
# ---------------------------------------------------------------------------

class TestBallProgression:
    """ball_progression computes forward/backward/net metres from X track."""

    def test_all_forward(self):
        """Monotonically increasing X → all metres are forward."""
        result = ball_progression(np.array([0.0, 10.0, 20.0, 35.0]))
        assert abs(result["forward_m"] - 35.0) < 1e-9
        assert result["backward_m"] == 0.0
        assert abs(result["net_m"] - 35.0) < 1e-9

    def test_all_backward(self):
        """Monotonically decreasing X → all metres are backward."""
        result = ball_progression(np.array([35.0, 20.0, 10.0, 0.0]))
        assert result["forward_m"] == 0.0
        assert abs(result["backward_m"] - 35.0) < 1e-9
        assert abs(result["net_m"] - (-35.0)) < 1e-9

    def test_mixed_movement(self):
        """Mixed movement correctly splits forward and backward metres."""
        # +20, -5, +10 → forward=30, backward=5, net=25
        result = ball_progression(np.array([0.0, 20.0, 15.0, 25.0]))
        assert abs(result["forward_m"] - 30.0) < 1e-9
        assert abs(result["backward_m"] - 5.0) < 1e-9
        assert abs(result["net_m"] - 25.0) < 1e-9

    def test_reverse_play_direction(self):
        """play_direction=-1 flips which movement is forward."""
        # Ball moves from 35 → 0 (decreasing X) in the attacking direction
        result = ball_progression(np.array([35.0, 20.0, 10.0]), play_direction=-1)
        assert abs(result["forward_m"] - 25.0) < 1e-9
        assert result["backward_m"] == 0.0

    def test_single_point_returns_zeros(self):
        """A single position has no movement; all values are 0."""
        result = ball_progression(np.array([50.0]))
        assert result == {"forward_m": 0.0, "backward_m": 0.0, "net_m": 0.0}

    def test_empty_returns_zeros(self):
        """Empty array returns zeros."""
        result = ball_progression(np.array([]))
        assert result == {"forward_m": 0.0, "backward_m": 0.0, "net_m": 0.0}

    def test_stationary_ball_returns_zeros(self):
        """Ball at the same position every frame returns all zeros."""
        result = ball_progression(np.full(10, 52.5))
        assert result == {"forward_m": 0.0, "backward_m": 0.0, "net_m": 0.0}


# ---------------------------------------------------------------------------
# defensive_leakage
# ---------------------------------------------------------------------------

class TestDefensiveLeakage:
    """defensive_leakage counts opponent receptions inside the penalty box."""

    def _make_arrays(self, bx, by, team):
        return (
            np.array(bx, dtype=float),
            np.array(by, dtype=float),
            np.array(team, dtype=int),
        )

    def test_opponent_in_box(self):
        """Frames where opponent (team 1) has ball inside the box are counted."""
        bx, by, team = self._make_arrays(
            [85.0, 90.0, 95.0],
            [25.0, 34.0, 43.0],
            [1, 1, 0],
        )
        # box: x in [80,105], y in [13.85,54.15]
        count = defensive_leakage(bx, by, team, 1, (80.0, 105.0), (13.85, 54.15))
        assert count == 2

    def test_opponent_outside_box(self):
        """Ball outside the box is not counted even if opponent has possession."""
        bx, by, team = self._make_arrays([50.0, 60.0], [34.0, 34.0], [1, 1])
        count = defensive_leakage(bx, by, team, 1, (80.0, 105.0), (13.85, 54.15))
        assert count == 0

    def test_own_team_in_box_not_counted(self):
        """Own-team possession inside the box is not leakage."""
        bx, by, team = self._make_arrays([90.0], [34.0], [0])
        count = defensive_leakage(bx, by, team, 1, (80.0, 105.0), (13.85, 54.15))
        assert count == 0

    def test_empty_arrays_return_zero(self):
        bx, by, team = self._make_arrays([], [], [])
        count = defensive_leakage(bx, by, team, 1, (80.0, 105.0), (13.85, 54.15))
        assert count == 0


# ---------------------------------------------------------------------------
# vertical_lane_density
# ---------------------------------------------------------------------------

class TestVerticalLaneDensity:
    """vertical_lane_density splits a dimension into N equal lanes."""

    def test_equal_distribution(self):
        """Positions uniformly spread across 5 lanes → equal counts."""
        # 5 lanes of 10 m each (50 m total), 2 positions per lane
        positions = np.array([5.0, 5.0, 15.0, 15.0, 25.0, 25.0, 35.0, 35.0, 45.0, 45.0])
        counts = vertical_lane_density(positions, pitch_dim_m=50.0, n_lanes=5)
        np.testing.assert_array_equal(counts, [2, 2, 2, 2, 2])

    def test_boundary_clamped_to_last_lane(self):
        """A position exactly at pitch_dim_m falls in the last lane."""
        counts = vertical_lane_density(np.array([68.0]), pitch_dim_m=68.0, n_lanes=5)
        assert counts[4] == 1
        assert counts[:4].sum() == 0

    def test_output_shape(self):
        """Returns an array of length n_lanes."""
        counts = vertical_lane_density(np.arange(20, dtype=float), pitch_dim_m=68.0, n_lanes=5)
        assert counts.shape == (5,)

    def test_total_count_matches_input(self):
        """Sum of all lane counts equals number of input positions."""
        positions = np.random.uniform(0, 68, 100)
        counts = vertical_lane_density(positions, pitch_dim_m=68.0, n_lanes=5)
        assert counts.sum() == 100

    def test_single_lane(self):
        """n_lanes=1 puts everything in a single bucket."""
        counts = vertical_lane_density(np.array([10.0, 30.0, 60.0]), 68.0, n_lanes=1)
        assert counts.shape == (1,)
        assert counts[0] == 3


# ---------------------------------------------------------------------------
# possession_by_thirds
# ---------------------------------------------------------------------------

class TestPossessionByThirds:
    """possession_by_thirds calculates possession % per pitch third."""

    def test_all_in_defensive_third(self):
        """Ball always in defensive third for the team → 100% defensive."""
        bx = np.array([10.0, 15.0, 20.0])
        team = np.array([0, 0, 0])
        d, m, a = possession_by_thirds(bx, pitch_length_m=105.0, team_possession=team,
                                       team_id=0)
        assert abs(d - 100.0) < 1e-9
        assert m == 0.0
        assert a == 0.0

    def test_all_in_attacking_third(self):
        """Ball always in attacking third → 100% attacking."""
        bx = np.array([80.0, 90.0, 100.0])
        team = np.array([0, 0, 0])
        d, m, a = possession_by_thirds(bx, pitch_length_m=105.0, team_possession=team,
                                       team_id=0)
        assert d == 0.0
        assert m == 0.0
        assert abs(a - 100.0) < 1e-9

    def test_even_distribution(self):
        """One frame per third → ~33.3% each."""
        bx = np.array([17.5, 52.5, 87.5])
        team = np.array([0, 0, 0])
        d, m, a = possession_by_thirds(bx, pitch_length_m=105.0, team_possession=team,
                                       team_id=0)
        assert abs(d - 100.0 / 3) < 1e-6
        assert abs(m - 100.0 / 3) < 1e-6
        assert abs(a - 100.0 / 3) < 1e-6

    def test_only_own_team_frames_counted(self):
        """Frames belonging to the other team are excluded."""
        bx = np.array([10.0, 80.0, 50.0])
        team = np.array([0, 1, 0])
        # Frames 0 and 2 belong to team 0: x=10 (defensive), x=50 (middle)
        d, m, a = possession_by_thirds(bx, pitch_length_m=105.0, team_possession=team,
                                       team_id=0)
        assert abs(d - 50.0) < 1e-9
        assert abs(m - 50.0) < 1e-9
        assert a == 0.0

    def test_no_possession_returns_zeros(self):
        """Team has no possession frames → (0, 0, 0)."""
        bx = np.array([50.0, 50.0])
        team = np.array([1, 1])
        result = possession_by_thirds(bx, 105.0, team, team_id=0)
        assert result == (0.0, 0.0, 0.0)

    def test_percentages_sum_to_100(self):
        """The three percentages always sum to 100 when team has possession."""
        bx = np.random.uniform(0, 105, 50)
        team = np.zeros(50, dtype=int)
        d, m, a = possession_by_thirds(bx, 105.0, team, team_id=0)
        assert abs(d + m + a - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# team_compactness
# ---------------------------------------------------------------------------

class TestTeamCompactness:
    """team_compactness returns the bounding-box area of player positions."""

    def test_simple_rectangle(self):
        """Four corners of a known rectangle → correct area."""
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
        assert abs(team_compactness(pts) - 50.0) < 1e-9

    def test_single_player_returns_zero(self):
        """A single player has zero bounding-box area."""
        pts = np.array([[50.0, 34.0]])
        assert team_compactness(pts) == 0.0

    def test_two_players_on_a_line(self):
        """Two players on the same Y → height=0 → area=0."""
        pts = np.array([[10.0, 34.0], [20.0, 34.0]])
        assert team_compactness(pts) == 0.0

    def test_empty_positions(self):
        """Empty array returns 0."""
        assert team_compactness(np.empty((0, 2))) == 0.0

    def test_wrong_shape_returns_zero(self):
        """1-D or (N,3) arrays return 0 without raising."""
        assert team_compactness(np.array([1.0, 2.0, 3.0])) == 0.0
        assert team_compactness(np.ones((5, 3))) == 0.0

    def test_larger_spread_gives_larger_area(self):
        """More spread-out players produce a larger area."""
        compact = np.array([[48.0, 32.0], [52.0, 36.0], [50.0, 34.0], [49.0, 33.0]])
        spread = np.array([[20.0, 10.0], [85.0, 10.0], [85.0, 58.0], [20.0, 58.0]])
        assert team_compactness(spread) > team_compactness(compact)


# ---------------------------------------------------------------------------
# space_creation
# ---------------------------------------------------------------------------

class TestSpaceCreation:
    """space_creation measures attacker–defensive-centroid distance."""

    def test_known_distance(self):
        """Attacker at (80,34) with all defenders at (50,34) → 30 m."""
        attacker = np.array([80.0, 34.0])
        defenders = np.array([[50.0, 34.0], [50.0, 34.0], [50.0, 34.0]])
        assert abs(space_creation(attacker, defenders) - 30.0) < 1e-9

    def test_attacker_at_centroid_is_zero(self):
        """Attacker at the centroid → 0 distance."""
        defenders = np.array([[40.0, 20.0], [60.0, 20.0], [50.0, 40.0]])
        centroid = defenders.mean(axis=0)
        assert abs(space_creation(centroid, defenders)) < 1e-9

    def test_empty_defenders_returns_zero(self):
        """No defenders → returns 0."""
        assert space_creation(np.array([80.0, 34.0]), np.empty((0, 2))) == 0.0

    def test_single_defender(self):
        """Single defender: distance equals Euclidean dist to that defender."""
        attacker = np.array([3.0, 4.0])
        defenders = np.array([[0.0, 0.0]])
        assert abs(space_creation(attacker, defenders) - 5.0) < 1e-9

    def test_wrong_shape_returns_zero(self):
        """Defender array with wrong shape returns 0 without raising."""
        assert space_creation(np.array([50.0, 34.0]), np.ones((4,))) == 0.0


# ---------------------------------------------------------------------------
# expected_threat
# ---------------------------------------------------------------------------

class TestExpectedThreat:
    """expected_threat returns xT values that increase towards the opponent goal."""

    def test_attacking_zone_higher_than_defensive(self):
        """xT near the opponent goal is greater than xT near own goal."""
        xt_att = expected_threat(100.0, 34.0, 105.0, 68.0)
        xt_def = expected_threat(5.0, 34.0, 105.0, 68.0)
        assert xt_att > xt_def

    def test_centre_higher_than_wide(self):
        """Central positions in the attacking third have higher xT than wide."""
        xt_centre = expected_threat(90.0, 34.0, 105.0, 68.0)
        xt_wide = expected_threat(90.0, 5.0, 105.0, 68.0)
        assert xt_centre > xt_wide

    def test_out_of_bounds_returns_zero(self):
        """Positions outside the pitch boundaries return 0."""
        assert expected_threat(-1.0, 34.0, 105.0, 68.0) == 0.0
        assert expected_threat(110.0, 34.0, 105.0, 68.0) == 0.0
        assert expected_threat(52.5, -5.0, 105.0, 68.0) == 0.0
        assert expected_threat(52.5, 80.0, 105.0, 68.0) == 0.0

    def test_values_in_unit_range(self):
        """All xT values lie in [0, 1]."""
        xs = np.linspace(0, 105, 20)
        ys = np.linspace(0, 68, 12)
        for x in xs:
            for y in ys:
                v = expected_threat(x, y, 105.0, 68.0)
                assert 0.0 <= v <= 1.0, f"xT={v} at ({x},{y}) is outside [0,1]"

    def test_custom_grid_respected(self):
        """A custom 2×2 xT grid overrides the default grid."""
        custom = np.array([[0.1, 0.9], [0.2, 0.8]])
        # Top-left zone: row 0, col 0 → 0.1  (y=0.1 → row=0, x=0.1 → col=0)
        v = expected_threat(0.1, 0.1, 2.0, 2.0, xt_grid=custom)
        assert abs(v - 0.1) < 1e-9
        # Top-right zone: row 0, col 1 → 0.9  (x=1.9 → col=1)
        v2 = expected_threat(1.9, 0.1, 2.0, 2.0, xt_grid=custom)
        assert abs(v2 - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# pitch_width_utilization
# ---------------------------------------------------------------------------

class TestPitchWidthUtilization:
    """pitch_width_utilization returns percentage activity per vertical lane."""

    def test_sums_to_100(self):
        """Percentages across all lanes sum to exactly 100."""
        positions = np.random.uniform(0, 68, 50)
        pct = pitch_width_utilization(positions, pitch_width_m=68.0, n_lanes=5)
        assert abs(pct.sum() - 100.0) < 1e-9

    def test_all_in_one_lane(self):
        """All positions in one lane → that lane is 100%, others 0%."""
        positions = np.array([5.0, 6.0, 7.0])  # all in lane 0 of [0,68)/5
        pct = pitch_width_utilization(positions, pitch_width_m=68.0, n_lanes=5)
        assert abs(pct[0] - 100.0) < 1e-9
        assert pct[1:].sum() == 0.0

    def test_output_shape(self):
        """Returns array of length n_lanes."""
        pct = pitch_width_utilization(np.arange(20, dtype=float), 68.0, n_lanes=5)
        assert pct.shape == (5,)

    def test_empty_positions_returns_zeros(self):
        """Empty positions → all-zero percentages."""
        pct = pitch_width_utilization(np.array([]), 68.0, n_lanes=5)
        np.testing.assert_array_equal(pct, np.zeros(5))

    def test_equal_distribution(self):
        """Perfectly uniform distribution → each lane gets 20%."""
        # 5 lanes × 2 positions each
        positions = np.array([6.0, 6.0, 20.0, 20.0, 34.0, 34.0, 48.0, 48.0, 62.0, 62.0])
        pct = pitch_width_utilization(positions, pitch_width_m=68.0, n_lanes=5)
        np.testing.assert_allclose(pct, [20.0, 20.0, 20.0, 20.0, 20.0], atol=1e-9)


# ---------------------------------------------------------------------------
# _arr_summary helper
# ---------------------------------------------------------------------------

class TestArrSummary:
    """_arr_summary returns correct descriptive stats and is JSON-safe."""

    def test_empty_array_returns_none_fields(self):
        result = _arr_summary(np.array([]), 4)
        assert result["count"] == 0
        for key in ("mean", "std", "min", "max", "median"):
            assert result[key] is None

    def test_single_value(self):
        result = _arr_summary(np.array([7.0]), 4)
        assert result["count"] == 1
        assert result["mean"] == 7.0
        assert result["std"] == 0.0
        assert result["min"] == 7.0
        assert result["max"] == 7.0
        assert result["median"] == 7.0

    def test_known_values(self):
        """[1, 2, 3, 4, 5] → mean=3, std=√2≈1.4142, min=1, max=5, median=3."""
        result = _arr_summary(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 6)
        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert abs(result["std"] - round(float(np.std([1, 2, 3, 4, 5], ddof=0)), 6)) < 1e-9
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["median"] == 3.0

    def test_rounding_respected(self):
        """Values are rounded to the requested decimal places."""
        result = _arr_summary(np.array([1.0 / 3.0, 2.0 / 3.0]), 2)
        assert result["mean"] == round(0.5, 2)
        assert isinstance(result["mean"], float)

    def test_all_values_are_native_python_types(self):
        """Output must be JSON-serializable (no numpy scalars)."""
        result = _arr_summary(np.array([1.0, 2.0, 3.0]), 4)
        assert isinstance(result["count"], int)
        for key in ("mean", "std", "min", "max", "median"):
            assert isinstance(result[key], float)

    def test_population_std(self):
        """std uses ddof=0 (population formula)."""
        arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = _arr_summary(arr, 10)
        expected_std = float(np.std(arr, ddof=0))
        assert abs(result["std"] - round(expected_std, 10)) < 1e-9


# ---------------------------------------------------------------------------
# build_match_report – structure and basic correctness
# ---------------------------------------------------------------------------

def _make_report(n: int = 300, seed: int = 0, fps: float = 25.0, **kwargs) -> dict:
    """Helper: generate synthetic data and call build_match_report."""
    rng = np.random.default_rng(seed)
    bx = rng.uniform(0, 105, n)
    by = rng.uniform(0, 68, n)
    poss = rng.choice([-1, 0, 1], size=n, p=[0.1, 0.45, 0.45])
    return build_match_report(bx, by, poss, fps=fps, **kwargs)


class TestBuildMatchReportStructure:
    """Verify the top-level keys and nested structure of the report."""

    _TOP_KEYS = {
        "meta",
        "possession",
        "passes",
        "ball_progression",
        "ball_position_stats",
        "possession_by_thirds",
        "defensive_leakage",
        "pitch_width_utilization",
        "expected_threat",
        "team_compactness",
        "team_separation",
    }

    def test_all_top_level_keys_present(self):
        report = _make_report()
        assert set(report.keys()) == self._TOP_KEYS

    def test_meta_fields(self):
        report = _make_report(n=250, fps=30.0)
        meta = report["meta"]
        assert meta["total_frames"] == 250
        assert meta["fps"] == 30.0
        assert meta["pitch_length_m"] == 105.0
        assert meta["pitch_width_m"] == 68.0
        assert meta["n_lanes"] == 5
        assert len(meta["lane_labels"]) == 5
        assert "decimal_places" in meta

    def test_meta_duration_correct(self):
        report = _make_report(n=250, fps=25.0)
        assert report["meta"]["duration_sec"] == round(250 / 25.0, 4)

    def test_custom_team_labels(self):
        report = _make_report(team_labels=("AlMereikh", "AlHilal"))
        poss = report["possession"]
        assert "AlMereikh" in poss
        assert "AlHilal" in poss
        prog = report["ball_progression"]
        assert "AlMereikh" in prog
        assert "AlHilal" in prog

    def test_possession_has_contested_key(self):
        assert "contested" in _make_report()["possession"]

    def test_passes_section_has_team_keys(self):
        report = _make_report(passes_team0=12, passes_team1=18,
                              team_labels=("Home", "Away"))
        assert report["passes"]["Home"] == 12
        assert report["passes"]["Away"] == 18

    def test_ball_progression_has_overall_key(self):
        assert "overall" in _make_report()["ball_progression"]

    def test_thirds_has_three_sub_keys(self):
        report = _make_report()
        for team_block in report["possession_by_thirds"].values():
            assert "defensive_third" in team_block
            assert "middle_third" in team_block
            assert "attacking_third" in team_block

    def test_compactness_none_without_positions(self):
        report = _make_report()
        assert report["team_compactness"]["team_0"] is None
        assert report["team_compactness"]["team_1"] is None

    def test_separation_none_without_positions(self):
        assert _make_report()["team_separation"] is None

    def test_xt_section_structure(self):
        xt = _make_report()["expected_threat"]
        assert "ball" in xt
        assert "ball_during_team_0" in xt
        assert "ball_during_team_1" in xt

    def test_xt_has_total_xt(self):
        xt = _make_report()["expected_threat"]
        assert "total_xt" in xt["ball_during_team_0"]
        assert "total_xt" in xt["ball_during_team_1"]


class TestBuildMatchReportAccuracy:
    """Verify numerical correctness of key stats."""

    def _fixed_report(self):
        bx = np.array([10.0, 20.0, 30.0, 40.0, 60.0, 80.0])
        by = np.array([34.0, 34.0, 34.0, 34.0, 34.0, 34.0])
        poss = np.array([0, 0, 0, 1, 1, 1])
        return build_match_report(
            bx, by, poss,
            fps=25.0,
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            team_labels=("home", "away"),
            play_directions=(1, -1),
            decimal_places=6,
        )

    def test_possession_frame_counts(self):
        r = self._fixed_report()
        assert r["possession"]["home"]["frames"] == 3
        assert r["possession"]["away"]["frames"] == 3

    def test_possession_pct_of_total(self):
        r = self._fixed_report()
        assert abs(r["possession"]["home"]["pct_of_total"] - 50.0) < 1e-9
        assert abs(r["possession"]["away"]["pct_of_total"] - 50.0) < 1e-9

    def test_possession_pct_of_owned(self):
        r = self._fixed_report()
        assert abs(r["possession"]["home"]["pct_of_owned"] - 50.0) < 1e-9

    def test_ball_progression_during_possession(self):
        """Team 0 holds possession at frames 0,1,2: x=[10,20,30,40].
        delta[0]=10, delta[1]=10, delta[2]=10 are all attributed to team 0
        (poss_for_delta[0..2] == 0) → forward_m = 30."""
        r = self._fixed_report()
        home_prog = r["ball_progression"]["home"]
        assert abs(home_prog["forward_m"] - 30.0) < 1e-9
        assert home_prog["backward_m"] == 0.0
        assert abs(home_prog["net_m"] - 30.0) < 1e-9

    def test_ball_progression_opponent_direction(self):
        """Team 1 (play_dir=-1): ball x=[40,60,80] → delta=[20,20] × -1 = -20 each → backward."""
        r = self._fixed_report()
        away_prog = r["ball_progression"]["away"]
        # delta[3]=20, delta[4]=20 (poss at frame 3 and 4 is 1)
        # play_dir=-1 → d = -20 each → backward_m=40
        assert abs(away_prog["backward_m"] - 40.0) < 1e-9
        assert away_prog["forward_m"] == 0.0

    def test_possession_by_thirds_all_defensive(self):
        """Ball at x=[10,20,30] (all in defensive third for 105m pitch)."""
        r = self._fixed_report()
        # third = 105/3 = 35m; x<35 → defensive
        home = r["possession_by_thirds"]["home"]
        assert abs(home["defensive_third"]["pct"] - 100.0) < 1e-9
        assert abs(home["middle_third"]["pct"]) < 1e-9
        assert abs(home["attacking_third"]["pct"]) < 1e-9

    def test_thirds_pct_sum_to_100(self):
        r = _make_report(n=300)
        for team_block in r["possession_by_thirds"].values():
            total = (
                team_block["defensive_third"]["pct"]
                + team_block["middle_third"]["pct"]
                + team_block["attacking_third"]["pct"]
            )
            assert abs(total - 100.0) < 1e-6 or total == 0.0

    def test_duration_sec_uses_fps(self):
        r = self._fixed_report()
        assert abs(r["meta"]["duration_sec"] - 6 / 25.0) < 1e-9

    def test_decimal_places_respected(self):
        bx = np.array([52.5, 52.5])
        by = np.array([34.0, 34.0])
        poss = np.array([0, 0])
        r = build_match_report(bx, by, poss, decimal_places=2)
        # Ball progression forward_m should be rounded to 2 decimal places
        val = r["ball_progression"]["overall"]["forward_m"]
        assert val == round(val, 2)

    def test_total_xt_is_sum_of_per_frame_xt(self):
        """total_xt must equal sum of per-frame xT values for that team."""
        bx = np.array([90.0, 92.0, 94.0, 80.0, 85.0])
        by = np.array([34.0, 34.0, 34.0, 34.0, 34.0])
        poss = np.array([0, 0, 0, 1, 1])
        r = build_match_report(bx, by, poss, decimal_places=8)
        expected_t0 = round(sum(
            expected_threat(float(x), float(y), 105.0, 68.0)
            for x, y in zip([90.0, 92.0, 94.0], [34.0, 34.0, 34.0])
        ), 8)
        assert abs(r["expected_threat"]["ball_during_team_0"]["total_xt"] - expected_t0) < 1e-6


class TestBuildMatchReportJsonSerializable:
    """The full report must survive json.dumps without errors."""

    def test_basic_report_serializable(self):
        json.dumps(_make_report())

    def test_report_with_positions_serializable(self):
        rng = np.random.default_rng(1)
        n = 100
        bx = rng.uniform(0, 105, n)
        by = rng.uniform(0, 68, n)
        poss = rng.choice([0, 1], size=n)
        t0_pos = [rng.uniform(0, 105, (5, 2)) for _ in range(n)]
        t1_pos = [rng.uniform(0, 105, (5, 2)) for _ in range(n)]
        r = build_match_report(
            bx, by, poss,
            team0_positions_per_frame=t0_pos,
            team1_positions_per_frame=t1_pos,
        )
        json.dumps(r)

    def test_edge_case_all_contested_serializable(self):
        bx = np.array([52.5, 52.5, 52.5])
        by = np.array([34.0, 34.0, 34.0])
        poss = np.array([-1, -1, -1])
        json.dumps(build_match_report(bx, by, poss))

    def test_single_frame_serializable(self):
        json.dumps(build_match_report(
            np.array([52.5]), np.array([34.0]), np.array([0])
        ))

    def test_empty_arrays_serializable(self):
        json.dumps(build_match_report(
            np.array([]), np.array([]), np.array([])
        ))


class TestBuildMatchReportWithPositions:
    """Tests for compactness and separation sections when positions are supplied."""

    def _report_with_positions(self, n: int = 20):
        rng = np.random.default_rng(42)
        bx = rng.uniform(0, 105, n)
        by = rng.uniform(0, 68, n)
        poss = rng.choice([0, 1], size=n)
        t0_pos = [rng.uniform([20, 10], [50, 58], (6, 2)) for _ in range(n)]
        t1_pos = [rng.uniform([55, 10], [85, 58], (6, 2)) for _ in range(n)]
        return build_match_report(
            bx, by, poss,
            team0_positions_per_frame=t0_pos,
            team1_positions_per_frame=t1_pos,
        )

    def test_compactness_not_none(self):
        r = self._report_with_positions()
        assert r["team_compactness"]["team_0"] is not None
        assert r["team_compactness"]["team_1"] is not None

    def test_compactness_structure(self):
        r = self._report_with_positions()
        block = r["team_compactness"]["team_0"]
        for key in ("count", "mean", "std", "min", "max", "median"):
            assert key in block

    def test_compactness_count_matches_frames(self):
        n = 20
        r = self._report_with_positions(n=n)
        assert r["team_compactness"]["team_0"]["count"] == n

    def test_separation_not_none(self):
        assert self._report_with_positions()["team_separation"] is not None

    def test_separation_structure(self):
        sep = self._report_with_positions()["team_separation"]
        for key in ("count", "mean", "std", "min", "max", "median"):
            assert key in sep

    def test_separation_positive(self):
        """Centroid distance must be non-negative."""
        sep = self._report_with_positions()["team_separation"]
        assert sep["min"] >= 0.0
        assert sep["mean"] >= 0.0

    def test_player_lane_keys_present(self):
        r = self._report_with_positions()
        wu = r["pitch_width_utilization"]
        assert "team_0_players" in wu
        assert "team_1_players" in wu

    def test_player_lane_pcts_sum_to_100(self):
        r = self._report_with_positions()
        for key in ("team_0_players", "team_1_players"):
            total_pct = sum(
                lane["pct"] for lane in r["pitch_width_utilization"][key].values()
            )
            # Tolerance: rounding to decimal_places=4 across 5 lanes can
            # introduce up to 5 × 0.00005 = 0.00025 accumulated error.
            assert abs(total_pct - 100.0) < 1e-3


class TestBuildMatchReportEdgeCases:
    """Edge cases: empty data, all-contested, custom parameters."""

    def test_empty_returns_correct_structure(self):
        r = build_match_report(np.array([]), np.array([]), np.array([]))
        assert r["meta"]["total_frames"] == 0
        assert r["meta"]["duration_sec"] == 0.0

    def test_all_contested_possession_is_zero(self):
        bx = np.array([52.5, 52.5])
        by = np.array([34.0, 34.0])
        poss = np.array([-1, -1])
        r = build_match_report(bx, by, poss)
        assert r["possession"]["team_0"]["frames"] == 0
        assert r["possession"]["team_1"]["frames"] == 0

    def test_custom_n_lanes(self):
        r = _make_report(n_lanes=3)
        assert r["meta"]["n_lanes"] == 3
        assert len(r["meta"]["lane_labels"]) == 3
        assert len(r["pitch_width_utilization"]["ball"]) == 3

    def test_custom_play_direction_flips_progression(self):
        """Increasing X with play_direction=-1 is backward movement."""
        bx = np.array([10.0, 20.0, 30.0])
        by = np.array([34.0, 34.0, 34.0])
        poss = np.array([0, 0, 0])
        r = build_match_report(bx, by, poss, play_directions=(-1, 1))
        prog = r["ball_progression"]["team_0"]
        assert prog["backward_m"] == 20.0
        assert prog["forward_m"] == 0.0

    def test_possession_pct_of_owned_sums_to_100(self):
        bx = np.ones(10) * 52.5
        by = np.ones(10) * 34.0
        poss = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        r = build_match_report(bx, by, poss)
        t0 = r["possession"]["team_0"]["pct_of_owned"]
        t1 = r["possession"]["team_1"]["pct_of_owned"]
        assert abs(t0 + t1 - 100.0) < 1e-6

    def test_defensive_leakage_counts_correctly(self):
        """Opponent (team 1) has ball 2 frames inside team 0's penalty box."""
        bx = np.array([5.0, 10.0, 50.0])    # first two in left penalty area
        by = np.array([34.0, 34.0, 34.0])
        poss = np.array([1, 1, 0])           # team 1 has ball in first two frames
        r = build_match_report(bx, by, poss, pitch_length_m=105.0, pitch_width_m=68.0,
                               penalty_area_length_m=16.5, penalty_area_width_m=40.32)
        leak = r["defensive_leakage"]["team_0_defensive_box"]
        assert leak["frames_opponent_in_box"] == 2

    def test_xT_grid_float64_precision(self):
        """xT values from the default grid must be float64."""
        import sports.common.stats as st
        assert st._XT_GRID_16x12.dtype == np.float64
