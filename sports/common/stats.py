"""
Match statistics helpers for soccer video analysis.

Provides pass detection with quality filters and possession tracking with
proximity weighting.  These helpers are used by the analysis notebook but are
kept here so they can be unit-tested without running an actual video.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Default thresholds (all can be overridden at construction time)
# ---------------------------------------------------------------------------

# Pass quality filters
DEFAULT_MIN_BALL_SPEED_PX_PER_SEC: float = 300.0   # filter dribble noise (note 2)
DEFAULT_PASS_DEBOUNCE_SEC: float = 1.35             # ignore back-to-back passes (note 12)
DEFAULT_PASS_MIN_DIST_PX: float = 85.0              # ignore short passes (note 13)
DEFAULT_ID_SWITCH_GUARD_PX: float = 65.0            # same-team change < this = same player (note 11)

# Possession proximity weighting (note 1)
DEFAULT_MIN_OWNER_DIST_PX: float = 80.0

# Team temporal-confirmation buffer (notes 0, 8)
DEFAULT_TEAM_VOTE_BUFFER_SIZE: int = 64
DEFAULT_TEAM_VOTE_MIN_FRAMES: int = 8


# ---------------------------------------------------------------------------
# TeamVoteBuffer
# ---------------------------------------------------------------------------

class TeamVoteBuffer:
    """
    Accumulates per-player team-classification votes across frames.

    Because ball-tracking reassigns ByteTrack IDs frequently, this class
    should be keyed by the *canonical* stable IDs produced by
    :class:`sports.common.team.PlayerReIdentifier`.  Over time the majority
    vote converges to the correct team even when individual frame predictions
    are noisy.

    Args:
        buffer_size (int): Maximum history length per player.  Older votes
            are discarded once the buffer is full.  Defaults to 64.
        min_votes (int): Minimum accumulated votes before the buffer result
            is trusted.  Below this threshold the raw per-frame prediction is
            returned unchanged.  Defaults to 8.
    """

    def __init__(
        self,
        buffer_size: int = DEFAULT_TEAM_VOTE_BUFFER_SIZE,
        min_votes: int = DEFAULT_TEAM_VOTE_MIN_FRAMES,
    ) -> None:
        self._buffer_size = max(1, buffer_size)
        self._min_votes = max(1, min_votes)
        # canonical_id → deque[int]  (only 0/1 values pushed)
        self._buffers: Dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, canonical_id: int, raw_team_id: int) -> int:
        """
        Record *raw_team_id* for *canonical_id* and return the stabilised
        team assignment.

        Args:
            canonical_id (int): Stable player ID (from PlayerReIdentifier).
            raw_team_id (int): Team prediction from the classifier for this
                frame.  Values other than 0 or 1 are accepted but only 0/1
                are stored in the buffer.

        Returns:
            int: Stabilised team ID (0 or 1) when the buffer has accumulated
                at least *min_votes* valid votes, otherwise *raw_team_id*.
                Returns -1 if *raw_team_id* is not in (0, 1) and the buffer
                has fewer than *min_votes* entries.
        """
        if raw_team_id in (0, 1):
            buf = self._buffers.setdefault(
                canonical_id, deque(maxlen=self._buffer_size)
            )
            buf.append(raw_team_id)

        buf = self._buffers.get(canonical_id)
        if buf is not None and len(buf) >= self._min_votes:
            c = Counter(buf)
            return c.most_common(1)[0][0]

        return raw_team_id if raw_team_id in (0, 1) else -1

    def get_stable_team(self, canonical_id: int) -> Optional[int]:
        """
        Return the current stable team for *canonical_id*, or *None* if not
        enough votes have been accumulated yet.
        """
        buf = self._buffers.get(canonical_id)
        if buf is None or len(buf) < self._min_votes:
            return None
        return Counter(buf).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# PassEvent
# ---------------------------------------------------------------------------

@dataclass
class PassEvent:
    """Immutable record of a single detected pass."""
    team_id: int
    time_sec: float
    from_player_id: int
    to_player_id: int


# ---------------------------------------------------------------------------
# PassDetector
# ---------------------------------------------------------------------------

class PassDetector:
    """
    Detects completed passes from possession-owner transitions.

    A transition from *prev_owner* to *new_owner* is counted as a pass only
    when all of the following quality filters pass:

    * Both owners belong to the same team and are different players.
    * Ball speed at the moment of transfer ≥ *min_ball_speed_px_per_sec*
      (filters dribble noise – note 2).
    * Ball displacement since the previous owner took possession ≥
      *pass_min_dist_px* (note 13).
    * The two players are more than *id_switch_guard_px* apart, preventing
      ByteTrack ID-reassignment artefacts from being counted as passes
      (note 11).
    * At least *pass_debounce_sec* seconds have elapsed since the last
      recorded pass for this team (note 12).

    Args:
        min_ball_speed_px_per_sec (float): Ball speed threshold in pixels/sec.
        pass_debounce_sec (float): Minimum seconds between consecutive same-
            team passes.
        pass_min_dist_px (float): Minimum ball travel distance (px) for a
            valid pass.
        id_switch_guard_px (float): If the two "different" players are closer
            than this distance, treat the event as a tracker-ID switch (same
            player) and ignore it.
        fps (float): Frames per second of the source video.  Used to convert
            frame counts to seconds.
    """

    def __init__(
        self,
        min_ball_speed_px_per_sec: float = DEFAULT_MIN_BALL_SPEED_PX_PER_SEC,
        pass_debounce_sec: float = DEFAULT_PASS_DEBOUNCE_SEC,
        pass_min_dist_px: float = DEFAULT_PASS_MIN_DIST_PX,
        id_switch_guard_px: float = DEFAULT_ID_SWITCH_GUARD_PX,
        fps: float = 25.0,
    ) -> None:
        self._min_speed = float(min_ball_speed_px_per_sec)
        self._debounce = float(pass_debounce_sec)
        self._min_dist = float(pass_min_dist_px)
        self._id_guard = float(id_switch_guard_px)
        self._fps = float(fps)

        # team_id → timestamp of last accepted pass
        self._last_pass_time: Dict[int, float] = {}
        # team_id → accumulated pass count
        self._pass_counts: Dict[int, int] = {}
        # Most recent pass event, or None
        self._last_pass_event: Optional["PassEvent"] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_pass_event(self) -> Optional["PassEvent"]:
        """Return the most recent :class:`PassEvent`, or *None*."""
        return self._last_pass_event

    def get_pass_count(self, team_id: int) -> int:
        """Return the total number of passes recorded for *team_id*."""
        return self._pass_counts.get(team_id, 0)

    def check(
        self,
        prev_canonical_id: Optional[int],
        prev_team_id: int,
        new_canonical_id: Optional[int],
        new_team_id: int,
        prev_player_xy: Optional[np.ndarray],
        new_player_xy: Optional[np.ndarray],
        ball_speed_px_per_sec: float,
        ball_displacement_px: float,
        time_sec: float,
    ) -> bool:
        """
        Return *True* if this owner transition qualifies as a completed pass.

        When a pass is accepted, the per-team pass count is incremented and the
        :attr:`last_pass_event` property is updated with the event details.

        Args:
            prev_canonical_id: Stable canonical ID of the previous ball owner,
                or *None* if unknown.
            prev_team_id: Team of the previous owner.
            new_canonical_id: Stable canonical ID of the new ball owner, or
                *None* if unknown.
            new_team_id: Team of the new owner.
            prev_player_xy: Pixel position of the previous owner at the time
                of the transition, or *None*.
            new_player_xy: Pixel position of the new owner, or *None*.
            ball_speed_px_per_sec: Instantaneous ball speed (pixels/sec).
            ball_displacement_px: Total ball displacement since the previous
                owner first received the ball (pixels).
            time_sec: Timestamp of the transition (seconds from video start).

        Returns:
            bool: True if a pass is detected and recorded; False otherwise.
        """
        # Must be a same-team transfer between two known, *different* players.
        if prev_team_id != new_team_id:
            return False
        if prev_canonical_id is None or new_canonical_id is None:
            return False
        if prev_canonical_id == new_canonical_id:
            return False

        # ID-switch guard: players too close → same physical player (note 11).
        if prev_player_xy is not None and new_player_xy is not None:
            dist = float(np.linalg.norm(
                np.asarray(prev_player_xy) - np.asarray(new_player_xy)
            ))
            if dist < self._id_guard:
                return False

        # Ball speed filter (note 2).
        if ball_speed_px_per_sec < self._min_speed:
            return False

        # Minimum ball displacement filter (note 13).
        if ball_displacement_px < self._min_dist:
            return False

        # Debounce filter (note 12).
        last_time = self._last_pass_time.get(prev_team_id, -self._debounce)
        if (time_sec - last_time) < self._debounce:
            return False

        # All checks passed — record the pass.
        self._last_pass_time[prev_team_id] = time_sec
        self._pass_counts[prev_team_id] = self._pass_counts.get(prev_team_id, 0) + 1
        self._last_pass_event = PassEvent(
            team_id=prev_team_id,
            time_sec=time_sec,
            from_player_id=prev_canonical_id,
            to_player_id=new_canonical_id,
        )
        return True

    def reset(self) -> None:
        """Reset all internal state (debounce timers, pass counts, last event)."""
        self._last_pass_time.clear()
        self._pass_counts.clear()
        self._last_pass_event = None


# ---------------------------------------------------------------------------
# PossessionTracker
# ---------------------------------------------------------------------------

class PossessionTracker:
    """
    Tracks ball possession per team with proximity-weighted frame counts.

    Instead of counting each possession frame as exactly 1, it weights each
    frame by how close the ball is to the owning player (note 1):

    .. code-block:: text

        weight = 1.0 - 0.5 * (distance / max_owner_dist_px)

    giving 1.0 when the ball is exactly at the player's feet and 0.5 at the
    edge of the ownership radius.

    Args:
        max_owner_dist_px (float): Maximum ball-to-player distance for
            possession to be assigned.
    """

    def __init__(
        self, max_owner_dist_px: float = DEFAULT_MIN_OWNER_DIST_PX
    ) -> None:
        self._max_dist = float(max_owner_dist_px)
        self._possession: Dict[int, float] = {}  # team_id → weighted frames

    def update(
        self,
        team_id: int,
        ball_xy: np.ndarray,
        player_xy: np.ndarray,
    ) -> float:
        """
        Record one possession frame for *team_id* and return the weight used.

        Args:
            team_id (int): Team that currently has possession (0 or 1).
            ball_xy (np.ndarray): Ball centre coordinates ``[x, y]``.
            player_xy (np.ndarray): Owning player's bottom-centre
                coordinates ``[x, y]``.

        Returns:
            float: The weight applied (in the range [0.5, 1.0]).
        """
        dist = float(np.linalg.norm(
            np.asarray(ball_xy) - np.asarray(player_xy)
        ))
        dist = min(dist, self._max_dist)
        weight = 1.0 - 0.5 * (dist / self._max_dist)
        self._possession[team_id] = self._possession.get(team_id, 0.0) + weight
        return weight

    def weighted_frames(self, team_id: int) -> float:
        """Return the accumulated weighted possession frames for *team_id*."""
        return self._possession.get(team_id, 0.0)

    def possession_pct(self, team_id: int, total_frames: int) -> float:
        """
        Return possession as a percentage of *total_frames*.

        Args:
            team_id (int): Team to query.
            total_frames (int): Total frames in the video (denominator).

        Returns:
            float: Percentage in [0, 100].
        """
        if total_frames <= 0:
            return 0.0
        return 100.0 * self._possession.get(team_id, 0.0) / float(total_frames)

    def reset(self) -> None:
        """Clear all accumulated possession data."""
        self._possession.clear()


# ---------------------------------------------------------------------------
# Video overlay helpers
# ---------------------------------------------------------------------------

def _format_time(sec: float) -> str:
    """Format seconds as ``MM:SS``."""
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def draw_stats_overlay(
    frame: np.ndarray,
    pass_detector: PassDetector,
    possession_tracker: PossessionTracker,
    frame_index: int,
    team_labels: Tuple[str, str] = ("Team A", "Team B"),
) -> np.ndarray:
    """
    Draw a semi-transparent statistics panel on the top-left of *frame*.

    Shows per-team pass counts and possession percentages.

    Args:
        frame: Video frame (BGR, will be modified in-place).
        pass_detector: Current pass detector with accumulated counts.
        possession_tracker: Current possession tracker.
        frame_index: Current frame number (used as total_frames denominator).
        team_labels: Display names for team 0 and team 1.

    Returns:
        The annotated frame.
    """
    import cv2  # local import to keep module testable without OpenCV

    total = max(frame_index, 1)
    poss_0 = possession_tracker.possession_pct(0, total)
    poss_1 = possession_tracker.possession_pct(1, total)
    passes_0 = pass_detector.get_pass_count(0)
    passes_1 = pass_detector.get_pass_count(1)

    lines = [
        f"{team_labels[0]}  Passes: {passes_0}  Poss: {poss_0:.1f}%",
        f"{team_labels[1]}  Passes: {passes_1}  Poss: {poss_1:.1f}%",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    pad = 10
    line_h = 30

    box_w = max(cv2.getTextSize(l, font, scale, thickness)[0][0] for l in lines) + 2 * pad
    box_h = line_h * len(lines) + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, text in enumerate(lines):
        y = pad + (i + 1) * line_h
        cv2.putText(frame, text, (pad, y), font, scale, (255, 255, 255), thickness)

    return frame


def draw_pass_label(
    frame: np.ndarray,
    pass_event: PassEvent,
    ball_xy: np.ndarray,
) -> np.ndarray:
    """
    Draw a "PASS" label near the ball at the exact moment a pass is detected.

    Args:
        frame: Video frame (BGR, modified in-place).
        pass_event: The pass event with timing information.
        ball_xy: Ball centre ``[x, y]`` in pixel coordinates.

    Returns:
        The annotated frame.
    """
    import cv2

    ts = _format_time(pass_event.time_sec)
    label = f"PASS {ts}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    bx, by = int(ball_xy[0]), int(ball_xy[1])

    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx = max(bx - tw // 2, 0)
    ty = max(by - 15, th + 5)

    cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
    cv2.putText(frame, label, (tx, ty), font, scale, (0, 255, 255), thickness)
    return frame
