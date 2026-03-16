"""
Unit tests for sports.common.stats module.

Covers:
- TeamVoteBuffer: temporal team-assignment stabilisation
- PassDetector: pass quality filters (speed, distance, debounce, ID switch guard),
  pass counting, and last-pass event tracking
- PossessionTracker: proximity-weighted possession counting
- PassEvent: dataclass record of a single pass
- draw_stats_overlay / draw_pass_label: video overlay helpers
"""

import numpy as np
import pytest

from sports.common.stats import (
    PassDetector,
    PassEvent,
    PossessionTracker,
    TeamVoteBuffer,
    draw_pass_label,
    draw_stats_overlay,
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

    def test_reset_clears_data(self):
        """reset() zeroes all accumulated possession."""
        pt = PossessionTracker()
        pt.update(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        pt.reset()
        assert pt.weighted_frames(0) == 0.0
        assert pt.possession_pct(0, total_frames=10) == 0.0


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
