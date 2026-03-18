"""
Match statistics helpers for soccer video analysis.

Provides pass detection with quality filters, possession tracking with
proximity weighting, and pitch heatmap generation utilities.  These helpers
are used by the analysis notebook but are kept here so they can be
unit-tested without running an actual video.
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
DEFAULT_PASS_DEBOUNCE_SEC: float = 1.0              # ignore back-to-back passes (note 12)
DEFAULT_PASS_MIN_DIST_PX: float = 60.0              # ignore short passes (note 13)
DEFAULT_ID_SWITCH_GUARD_PX: float = 150.0           # same-team change < this = same player (note 11)

# Possession proximity weighting (note 1)
DEFAULT_MIN_OWNER_DIST_PX: float = 140.0

# Team temporal-confirmation buffer (notes 0, 8)
DEFAULT_TEAM_VOTE_BUFFER_SIZE: int = 250
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

@dataclass(frozen=True)
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

    def possession_pct_normalized(self, team_id: int) -> float:
        """
        Return possession as a percentage of total *known* possession frames
        (i.e. frames where any team had the ball), so that the values for all
        teams sum to exactly 100 % — matching the TV-style display.

        Args:
            team_id (int): Team to query.

        Returns:
            float: Percentage in [0, 100].  Returns 0.0 when no possession has
            been recorded yet.
        """
        total = sum(self._possession.values())
        if total <= 0:
            return 0.0
        return 100.0 * self._possession.get(team_id, 0.0) / total

    def reset(self) -> None:
        """Clear all accumulated possession data."""
        self._possession.clear()


# ---------------------------------------------------------------------------
# Perspective-aware ownership distance helper
# ---------------------------------------------------------------------------

def perspective_owner_dist(
    player_y: float,
    frame_h: float,
    base_dist_px: float,
    near_scale: float = 1.4,
    far_scale: float = 0.6,
) -> float:
    """
    Return a perspective-corrected ball-ownership distance threshold for a
    player at vertical position *player_y* in a frame of height *frame_h*.

    For medium-high angle camera footage (typical stadium/TV broadcast), players
    near the bottom of the frame (close to camera) appear larger and should be
    granted a wider ownership radius.  Players near the top of the frame (far
    side of the pitch) appear smaller and need a tighter radius.

    The threshold scales linearly between *far_scale* × *base_dist_px* at
    ``player_y == 0`` and *near_scale* × *base_dist_px* at
    ``player_y == frame_h``.

    Args:
        player_y (float): Player's Y coordinate in the frame (pixels from top).
        frame_h (float): Full frame height in pixels.  Must be > 0.
        base_dist_px (float): The nominal ownership distance at mid-height.
        near_scale (float): Multiplier applied when the player is at the very
            bottom of the frame (closest to camera).  Defaults to 1.4.
        far_scale (float): Multiplier applied when the player is at the very
            top of the frame (farthest from camera).  Defaults to 0.6.

    Returns:
        float: Adjusted ownership distance in pixels.
    """
    if frame_h <= 0:
        return float(base_dist_px)
    t = float(player_y) / float(frame_h)
    t = max(0.0, min(1.0, t))
    scale = far_scale + (near_scale - far_scale) * t
    return float(base_dist_px) * scale


# ---------------------------------------------------------------------------
# Pitch heatmap helpers
# ---------------------------------------------------------------------------

def compute_stable_homography(
    keypoints_list: List[np.ndarray],
    target_vertices: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute a robust homography matrix from a collection of per-frame keypoint
    detections and their corresponding pitch target vertices.

    All valid (non-zero) source keypoints from all provided frames are pooled
    together and passed to ``cv2.findHomography`` with RANSAC so that outlier
    detections are automatically rejected.

    Args:
        keypoints_list: List of ``(N, 2)`` arrays of source pixel coordinates,
            one array per sampled frame.  Entries where either coordinate is
            ≤ 1 are treated as undetected (the pitch detection model marks
            missing keypoints at ``(0, 0)`` or similar near-zero values) and
            are excluded from the computation.
        target_vertices: ``(N, 2)`` array of the corresponding pitch
            coordinates (same ordering as the keypoint class labels).

    Returns:
        A ``(3, 3)`` homography matrix, or *None* if fewer than four valid
        point correspondences are available across all frames.
    """
    import cv2  # local import – keeps module importable without OpenCV

    target_vertices = np.asarray(target_vertices, dtype=np.float32)
    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []

    for kp_xy in keypoints_list:
        kp_xy = np.asarray(kp_xy, dtype=np.float32)
        if kp_xy.ndim != 2 or kp_xy.shape[1] != 2:
            continue
        n = min(len(kp_xy), len(target_vertices))
        # The pitch detection model marks undetected keypoints at (0, 0) or
        # very small pixel values (≤ 1).  Require both coordinates to be
        # strictly greater than 1 so that these placeholder values are excluded.
        valid = (kp_xy[:n, 0] > 1) & (kp_xy[:n, 1] > 1)
        if not np.any(valid):
            continue
        src_pts.append(kp_xy[:n][valid])
        dst_pts.append(target_vertices[:n][valid])

    if not src_pts:
        return None

    all_src = np.concatenate(src_pts, axis=0)
    all_dst = np.concatenate(dst_pts, axis=0)

    if len(all_src) < 4:
        return None

    H, _ = cv2.findHomography(all_src, all_dst, cv2.RANSAC, 5.0)
    return H  # may be None if RANSAC failed


def transform_points_homography(
    points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Apply a homography matrix *H* to an ``(N, 2)`` array of pixel coordinates
    and return the transformed ``(N, 2)`` array in pitch space.

    Args:
        points: ``(N, 2)`` float array of source pixel coordinates.
        H: ``(3, 3)`` homography matrix returned by
            :func:`compute_stable_homography` or ``cv2.findHomography``.

    Returns:
        ``(N, 2)`` float array of transformed coordinates.  If *points* is
        empty the input is returned unchanged.
    """
    import cv2

    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points
    reshaped = points.reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(reshaped, H)
    return transformed.reshape(-1, 2).astype(np.float32)


def generate_pitch_heatmap(
    pitch_x: np.ndarray,
    pitch_y: np.ndarray,
    pitch_length_m: float,
    pitch_width_m: float,
    title: str = "",
    figsize: Tuple[int, int] = (13, 8),
    cmap: str = "hot",
    alpha: float = 0.65,
    levels: int = 100,
    thresh: float = 0.05,
    pitch_color: str = "#22312b",
    line_color: str = "#c7d5cc",
) -> "plt.Figure":  # type: ignore[name-defined]
    """
    Draw a KDE heatmap on a top-down soccer pitch using ``mplsoccer`` and
    ``seaborn``.

    Args:
        pitch_x: 1-D array of player X positions in pitch metres (along the
            length axis).
        pitch_y: 1-D array of player Y positions in pitch metres (along the
            width axis).
        pitch_length_m: Pitch length in metres (used for ``mplsoccer`` and
            for clipping out-of-bounds points).
        pitch_width_m: Pitch width in metres.
        title: Figure title.  Empty string → no title.
        figsize: ``(width, height)`` in inches for the figure.
        cmap: Matplotlib colour-map name for the KDE fill.
        alpha: Opacity of the KDE layer.
        levels: Number of contour levels in the KDE.
        thresh: KDE threshold below which density is not drawn.
        pitch_color: Background colour of the pitch.
        line_color: Colour of pitch markings.

    Returns:
        ``matplotlib.figure.Figure`` – caller is responsible for
        ``plt.savefig`` / ``plt.show`` / ``plt.close``.
    """
    try:
        from mplsoccer import Pitch  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mplsoccer and seaborn are required for heatmap generation. "
            "Install them with: pip install mplsoccer seaborn"
        ) from exc

    pitch_x = np.asarray(pitch_x, dtype=float)
    pitch_y = np.asarray(pitch_y, dtype=float)

    # Clip to valid pitch area
    mask = (
        (pitch_x >= 0) & (pitch_x <= pitch_length_m) &
        (pitch_y >= 0) & (pitch_y <= pitch_width_m)
    )
    pitch_x = pitch_x[mask]
    pitch_y = pitch_y[mask]

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=pitch_length_m,
        pitch_width=pitch_width_m,
        pitch_color=pitch_color,
        line_color=line_color,
    )
    fig, ax = pitch.draw(figsize=figsize)

    if len(pitch_x) >= 2:
        sns.kdeplot(
            x=pitch_x,
            y=pitch_y,
            fill=True,
            thresh=thresh,
            levels=levels,
            cmap=cmap,
            alpha=alpha,
            ax=ax,
        )

    if title:
        ax.set_title(title, color="white", fontsize=16, pad=10)

    return fig


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
    poss_0 = possession_tracker.possession_pct_normalized(0)
    poss_1 = possession_tracker.possession_pct_normalized(1)
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


# ---------------------------------------------------------------------------
# Pitch analytics helpers
# ---------------------------------------------------------------------------

def ball_progression(
    pitch_x: np.ndarray,
    play_direction: int = 1,
) -> dict:
    """
    Calculate net ball progression from a sequence of ball X positions.

    Uses frame-to-frame differences of the ball's X coordinate to separate
    forward movement (in the play direction) from backward movement.

    Args:
        pitch_x: 1-D array of ball X positions in pitch metres along the
            length axis.
        play_direction: ``+1`` if increasing X is the attacking direction,
            ``-1`` if decreasing X is the attacking direction.  Defaults to
            ``+1``.

    Returns:
        A dict with keys:

        - ``forward_m`` (float): Total metres moved in the attacking direction.
        - ``backward_m`` (float): Total metres moved against the attacking
          direction (always positive).
        - ``net_m`` (float): Net progression (``forward_m - backward_m``).
    """
    arr = np.asarray(pitch_x, dtype=float).ravel()
    if len(arr) < 2:
        return {"forward_m": 0.0, "backward_m": 0.0, "net_m": 0.0}

    delta = np.diff(arr) * float(play_direction)
    forward_m = float(delta[delta > 0].sum())
    backward_m = float(np.abs(delta[delta < 0]).sum())
    return {
        "forward_m": forward_m,
        "backward_m": backward_m,
        "net_m": forward_m - backward_m,
    }


def defensive_leakage(
    ball_pitch_x: np.ndarray,
    ball_pitch_y: np.ndarray,
    possession_team: np.ndarray,
    opponent_team_id: int,
    box_x_range: Tuple[float, float],
    box_y_range: Tuple[float, float],
) -> int:
    """
    Count frames in which the opponent has possession inside the penalty box.

    Args:
        ball_pitch_x: 1-D array of ball X positions in pitch metres.
        ball_pitch_y: 1-D array of ball Y positions in pitch metres.
        possession_team: 1-D integer array of the team ID with possession at
            each frame.  Use ``-1`` for contested / unknown frames.
        opponent_team_id: The team ID of the opponent.
        box_x_range: ``(x_min, x_max)`` metres defining the penalty box along
            the length axis.
        box_y_range: ``(y_min, y_max)`` metres defining the penalty box along
            the width axis.

    Returns:
        Number of frames where the opponent has the ball inside the box.
    """
    bx = np.asarray(ball_pitch_x, dtype=float).ravel()
    by = np.asarray(ball_pitch_y, dtype=float).ravel()
    team = np.asarray(possession_team, dtype=int).ravel()

    n = min(len(bx), len(by), len(team))
    bx, by, team = bx[:n], by[:n], team[:n]

    x_min, x_max = float(box_x_range[0]), float(box_x_range[1])
    y_min, y_max = float(box_y_range[0]), float(box_y_range[1])

    in_box = (bx >= x_min) & (bx <= x_max) & (by >= y_min) & (by <= y_max)
    is_opponent = team == opponent_team_id
    return int(np.sum(in_box & is_opponent))


def vertical_lane_density(
    positions: np.ndarray,
    pitch_dim_m: float,
    n_lanes: int = 5,
) -> np.ndarray:
    """
    Count how many positions fall in each of *n_lanes* equal vertical lanes.

    Vertical lanes divide the *width* dimension of the pitch (Y axis).  Pass
    ball Y coordinates to analyse ball utilisation; pass player Y coordinates
    to analyse player distribution.

    Args:
        positions: 1-D array of Y (or X) positions in pitch metres.
        pitch_dim_m: Total extent of the dimension in metres (e.g. pitch width
            68 m).
        n_lanes: Number of equal lanes to divide the dimension into.  Defaults
            to 5.

    Returns:
        Integer ``ndarray`` of shape ``(n_lanes,)`` with the count in each
        lane, from the lowest value to the highest.
    """
    positions = np.asarray(positions, dtype=float).ravel()
    n_lanes = max(1, int(n_lanes))
    lane_width = float(pitch_dim_m) / n_lanes
    counts = np.zeros(n_lanes, dtype=int)
    for pos in positions:
        idx = int(pos / lane_width)
        idx = max(0, min(n_lanes - 1, idx))
        counts[idx] += 1
    return counts


def possession_by_thirds(
    ball_pitch_x: np.ndarray,
    pitch_length_m: float,
    team_possession: np.ndarray,
    team_id: int,
) -> Tuple[float, float, float]:
    """
    Calculate what percentage of a team's possession occurs in each pitch third.

    The pitch is divided into three equal thirds along the length axis (X).
    Only frames attributed to *team_id* are considered.

    Args:
        ball_pitch_x: 1-D array of ball X positions in pitch metres.
        pitch_length_m: Full pitch length in metres.
        team_possession: 1-D integer array of the possessing team ID per
            frame.  Use ``-1`` for contested frames.
        team_id: The team whose possession thirds should be reported.

    Returns:
        A tuple ``(defensive_pct, middle_pct, attacking_pct)`` where each
        value is the percentage of the team's possession frames located in that
        third.  Returns ``(0.0, 0.0, 0.0)`` if the team has no possession
        frames.
    """
    bx = np.asarray(ball_pitch_x, dtype=float).ravel()
    team = np.asarray(team_possession, dtype=int).ravel()

    n = min(len(bx), len(team))
    bx, team = bx[:n], team[:n]

    owned = bx[team == team_id]
    total = len(owned)
    if total == 0:
        return (0.0, 0.0, 0.0)

    third = float(pitch_length_m) / 3.0
    def_pct = float(np.sum(owned < third)) / total * 100.0
    mid_pct = float(np.sum((owned >= third) & (owned < 2 * third))) / total * 100.0
    att_pct = float(np.sum(owned >= 2 * third)) / total * 100.0
    return (def_pct, mid_pct, att_pct)


def team_compactness(
    player_positions: np.ndarray,
) -> float:
    """
    Compute the area of the axis-aligned bounding rectangle around all players.

    Args:
        player_positions: ``(N, 2)`` array of ``[x, y]`` positions in pitch
            metres.  Fewer than 2 points returns 0.0.

    Returns:
        Bounding-box area in square metres.  Zero if fewer than two players
        are provided.
    """
    pts = np.asarray(player_positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return 0.0
    width = float(pts[:, 0].max() - pts[:, 0].min())
    height = float(pts[:, 1].max() - pts[:, 1].min())
    return width * height


def space_creation(
    attacker_xy: np.ndarray,
    defender_positions: np.ndarray,
) -> float:
    """
    Measure the distance between an attacker and the defensive centroid.

    A larger distance indicates the attacker has created more space between
    themselves and the opponent's defensive block.

    Args:
        attacker_xy: ``[x, y]`` position of the attacker in pitch metres.
        defender_positions: ``(N, 2)`` array of ``[x, y]`` positions of the
            defending team in pitch metres.  At least one defender is required.

    Returns:
        Euclidean distance in metres between the attacker and the mean
        position of the defenders.  Returns 0.0 if *defender_positions* is
        empty.
    """
    axy = np.asarray(attacker_xy, dtype=float).ravel()[:2]
    defs = np.asarray(defender_positions, dtype=float)
    if defs.ndim != 2 or defs.shape[1] != 2 or len(defs) == 0:
        return 0.0
    centroid = defs.mean(axis=0)
    return float(np.linalg.norm(axy - centroid))


def _build_xt_grid() -> np.ndarray:
    """Build the default 16×12 static xT grid.

    Values are generated from a geometric approximation of expected threat
    based on Karun Singh's framework:
    ``P(shot|zone) × P(goal|shot,zone)`` using distance and angle to goal.
    Column 0 = defensive end, column 15 = attacking end.
    """
    cols = np.arange(16)[np.newaxis, :]   # shape (1, 16)
    rows = np.arange(12)[:, np.newaxis]   # shape (12, 1)
    gl, gw = 105.0, 68.0                  # pitch dimensions used for grid
    x = (cols + 0.5) * gl / 16
    y = (rows + 0.5) * gw / 12
    dist = np.sqrt((gl - x) ** 2 + (gw / 2 - y) ** 2)
    # Angle subtended by goal (7.32 m wide) from position (x, y)
    d_sq = (gl - x) ** 2 + (gw / 2 - y) ** 2
    angle = np.where(
        d_sq < 1e-6,
        0.0,
        np.arctan2(7.32 * (gl - x), d_sq - (7.32 / 2) ** 2).clip(0, np.pi),
    )
    # P(shot) × P(goal|shot) approximation
    raw = 1.0 / (1.0 + np.exp(0.16 * dist - 2.6)) * (angle / np.pi) ** 0.5
    return np.clip(raw, 0.0, 1.0).astype(np.float64)


_XT_GRID_16x12 = _build_xt_grid()


def expected_threat(
    pitch_x: float,
    pitch_y: float,
    pitch_length_m: float,
    pitch_width_m: float,
    xt_grid: Optional[np.ndarray] = None,
) -> float:
    """
    Return the expected-threat (xT) value for a ball position.

    Uses a static 16-column × 12-row threat grid where column 0 corresponds
    to the defensive end and column 15 to the attacking end.

    Args:
        pitch_x: Ball X position in pitch metres along the length axis.
        pitch_y: Ball Y position in pitch metres along the width axis.
        pitch_length_m: Full pitch length in metres.
        pitch_width_m: Full pitch width in metres.
        xt_grid: Optional ``(12, 16)`` float array overriding the default
            static grid.  Rows correspond to the width axis, columns to the
            length axis.

    Returns:
        xT value in ``[0, 1]`` for the given position.  Returns ``0.0`` for
        positions outside the pitch boundaries.
    """
    grid = _XT_GRID_16x12 if xt_grid is None else np.asarray(xt_grid, dtype=float)
    n_rows, n_cols = grid.shape

    x = float(pitch_x)
    y = float(pitch_y)
    if x < 0 or x > pitch_length_m or y < 0 or y > pitch_width_m:
        return 0.0

    col = int(np.clip(x / pitch_length_m * n_cols, 0, n_cols - 1))
    row = int(np.clip(y / pitch_width_m * n_rows, 0, n_rows - 1))
    return float(grid[row, col])


def pitch_width_utilization(
    positions: np.ndarray,
    pitch_width_m: float,
    n_lanes: int = 5,
) -> np.ndarray:
    """
    Calculate the percentage of activity in each vertical lane.

    Identical to :func:`vertical_lane_density` but returns percentages rather
    than raw counts.  Useful for reporting how well a team (or the ball)
    utilises the full width of the pitch.

    Lane order: ``[right-wing, right-half, central, left-half, left-wing]``
    when the Y axis runs from right touchline (0) to left touchline
    (*pitch_width_m*).

    Args:
        positions: 1-D array of Y positions in pitch metres.
        pitch_width_m: Pitch width in metres (e.g. 68 m).
        n_lanes: Number of equal lanes.  Defaults to 5.

    Returns:
        Float ``ndarray`` of shape ``(n_lanes,)`` with the percentage of
        positions falling in each lane.  Sums to 100 when there is at least
        one position; returns all-zeros for empty input.
    """
    counts = vertical_lane_density(positions, pitch_width_m, n_lanes)
    total = counts.sum()
    if total == 0:
        return np.zeros(n_lanes, dtype=float)
    return counts.astype(float) / total * 100.0


# ---------------------------------------------------------------------------
# JSON match report
# ---------------------------------------------------------------------------

_LANE_LABELS_5: Tuple[str, ...] = (
    "right_wing",
    "right_half",
    "central",
    "left_half",
    "left_wing",
)


def _r(val: float, ndigits: int) -> float:
    """Round *val* to *ndigits* decimal places; return a native Python float."""
    return round(float(val), ndigits)


def _arr_summary(arr: np.ndarray, ndigits: int) -> dict:
    """
    Return descriptive statistics for a 1-D numeric array as a JSON-safe dict.

    Keys: ``count`` (int), ``mean``, ``std``, ``min``, ``max``, ``median``
    (float rounded to *ndigits*).  ``None`` is used for each stat when *arr*
    is empty.  Standard deviation uses the population formula (``ddof=0``).
    """
    a = np.asarray(arr, dtype=float).ravel()
    if len(a) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
        }
    return {
        "count": int(len(a)),
        "mean": _r(np.mean(a), ndigits),
        "std": _r(np.std(a, ddof=0), ndigits),
        "min": _r(np.min(a), ndigits),
        "max": _r(np.max(a), ndigits),
        "median": _r(np.median(a), ndigits),
    }


def build_match_report(
    ball_pitch_x: np.ndarray,
    ball_pitch_y: np.ndarray,
    possession_team: np.ndarray,
    fps: float = 25.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    team_labels: Tuple[str, str] = ("team_0", "team_1"),
    play_directions: Tuple[int, int] = (1, -1),
    passes_team0: int = 0,
    passes_team1: int = 0,
    team0_positions_per_frame: Optional[List[np.ndarray]] = None,
    team1_positions_per_frame: Optional[List[np.ndarray]] = None,
    penalty_area_length_m: float = 16.5,
    penalty_area_width_m: float = 40.32,
    n_lanes: int = 5,
    xt_grid: Optional[np.ndarray] = None,
    decimal_places: int = 4,
) -> dict:
    """
    Build a comprehensive match statistics report as a JSON-serializable dict.

    All individual analytics functions (:func:`ball_progression`,
    :func:`defensive_leakage`, :func:`possession_by_thirds`,
    :func:`team_compactness`, :func:`space_creation`,
    :func:`expected_threat`, :func:`pitch_width_utilization`) are called
    internally and their results combined into one rich, nested dict.

    All floating-point values are rounded to *decimal_places* decimal places
    (default 4) for high-precision output.  Internal calculations always use
    float64.

    Args:
        ball_pitch_x: 1-D array of ball X positions in pitch metres.
        ball_pitch_y: 1-D array of ball Y positions in pitch metres.
        possession_team: 1-D int array; team ID owning the ball each frame
            (``0`` or ``1``), or ``-1`` for contested / unknown frames.
        fps: Video frame rate in frames per second.  Used to convert frame
            counts to durations.  Defaults to 25.0.
        pitch_length_m: Pitch length in metres (X axis).  Defaults to 105.0.
        pitch_width_m: Pitch width in metres (Y axis).  Defaults to 68.0.
        team_labels: Display names for team 0 and team 1.
        play_directions: ``(+1, -1)`` means team 0 attacks toward increasing X
            and team 1 toward decreasing X.
        passes_team0: Accumulated pass count for team 0 (from
            :class:`PassDetector`).
        passes_team1: Accumulated pass count for team 1.
        team0_positions_per_frame: Optional list of ``(N_players, 2)`` arrays,
            one per frame, giving team 0 player X/Y positions in pitch metres.
            Required for compactness and centroid-separation stats.
        team1_positions_per_frame: Same for team 1.
        penalty_area_length_m: Length of the penalty area from the goal line
            in metres.  Defaults to 16.5 (standard FIFA).
        penalty_area_width_m: Width of the penalty area in metres.  Defaults
            to 40.32 (standard FIFA).
        n_lanes: Number of equal vertical lanes for width-utilisation stats.
            Defaults to 5.
        xt_grid: Optional ``(n_rows, n_cols)`` float array overriding the
            default xT grid inside :func:`expected_threat`.
        decimal_places: Rounding precision for all floating-point output
            values.  Defaults to 4.

    Returns:
        A nested dict that is fully JSON-serializable (all values are native
        Python scalars, lists, or dicts — no NumPy objects).  Top-level keys:

        - ``"meta"`` — pitch/video metadata and report parameters.
        - ``"possession"`` — frame counts, percentages (of total and of owned
          frames), and durations in seconds.
        - ``"passes"`` — per-team pass counts.
        - ``"ball_progression"`` — overall and per-team forward / backward /
          net metres, computed from frame-to-frame ball deltas attributed to
          the possessing team.
        - ``"ball_position_stats"`` — descriptive statistics of ball X and Y
          for all frames and split by possessing team.
        - ``"possession_by_thirds"`` — per-team possession split across
          defensive / middle / attacking thirds with percentage and frame count.
        - ``"defensive_leakage"`` — frames and duration the opponent held the
          ball inside each team's penalty area.
        - ``"pitch_width_utilization"`` — percentage and count of ball/player
          activity in each vertical lane.
        - ``"expected_threat"`` — descriptive statistics and total accumulated
          xT for all frames and per possessing team.
        - ``"team_compactness"`` — bounding-box area statistics per team
          (``None`` when positions are not provided).
        - ``"team_separation"`` — centroid-to-centroid distance statistics
          across all frames (``None`` when positions are not provided).
    """
    dp = int(decimal_places)

    # ── align input arrays ──────────────────────────────────────────────────
    bx = np.asarray(ball_pitch_x, dtype=float).ravel()
    by = np.asarray(ball_pitch_y, dtype=float).ravel()
    poss = np.asarray(possession_team, dtype=int).ravel()
    n = min(len(bx), len(by), len(poss))
    bx, by, poss = bx[:n], by[:n], poss[:n]

    lbl0, lbl1 = str(team_labels[0]), str(team_labels[1])
    dir0, dir1 = int(play_directions[0]), int(play_directions[1])
    fps = float(fps)

    if n_lanes == 5:
        lane_labels: List[str] = list(_LANE_LABELS_5)
    else:
        lane_labels = [f"lane_{i + 1}" for i in range(n_lanes)]

    # ── meta ────────────────────────────────────────────────────────────────
    meta: dict = {
        "pitch_length_m": _r(pitch_length_m, dp),
        "pitch_width_m": _r(pitch_width_m, dp),
        "total_frames": n,
        "fps": _r(fps, dp),
        "duration_sec": _r(n / fps, dp) if fps > 0 else None,
        "team_labels": {"0": lbl0, "1": lbl1},
        "n_lanes": n_lanes,
        "lane_labels": lane_labels,
        "penalty_area_length_m": _r(penalty_area_length_m, dp),
        "penalty_area_width_m": _r(penalty_area_width_m, dp),
        "decimal_places": dp,
    }

    # ── possession ──────────────────────────────────────────────────────────
    t0_frames = int(np.sum(poss == 0))
    t1_frames = int(np.sum(poss == 1))
    contested_frames = int(np.sum(poss == -1))
    owned_frames = t0_frames + t1_frames

    def _poss_block(frames: int) -> dict:
        pct_total = _r(frames / n * 100, dp) if n > 0 else 0.0
        pct_owned = _r(frames / owned_frames * 100, dp) if owned_frames > 0 else 0.0
        duration = _r(frames / fps, dp) if fps > 0 else None
        return {
            "frames": frames,
            "pct_of_total": pct_total,
            "pct_of_owned": pct_owned,
            "duration_sec": duration,
        }

    possession_section: dict = {
        lbl0: _poss_block(t0_frames),
        lbl1: _poss_block(t1_frames),
        "contested": {
            "frames": contested_frames,
            "pct_of_total": _r(contested_frames / n * 100, dp) if n > 0 else 0.0,
            "duration_sec": _r(contested_frames / fps, dp) if fps > 0 else None,
        },
    }

    # ── passes ──────────────────────────────────────────────────────────────
    passes_section: dict = {
        lbl0: int(passes_team0),
        lbl1: int(passes_team1),
    }

    # ── ball progression ────────────────────────────────────────────────────
    # Frame-to-frame deltas attributed to the possessing team at frame i.
    if n > 1:
        delta_x_raw = np.diff(bx)
        poss_for_delta = poss[:n - 1]
    else:
        delta_x_raw = np.array([], dtype=float)
        poss_for_delta = np.array([], dtype=int)

    def _progression_block(delta: np.ndarray, play_dir: int) -> dict:
        d = delta * float(play_dir)
        fwd = float(d[d > 0].sum()) if len(d) > 0 else 0.0
        bwd = float(np.abs(d[d < 0]).sum()) if len(d) > 0 else 0.0
        return {
            "forward_m": _r(fwd, dp),
            "backward_m": _r(bwd, dp),
            "net_m": _r(fwd - bwd, dp),
        }

    ball_progression_section: dict = {
        "overall": _progression_block(delta_x_raw, dir0),
        lbl0: _progression_block(delta_x_raw[poss_for_delta == 0], dir0),
        lbl1: _progression_block(delta_x_raw[poss_for_delta == 1], dir1),
    }

    # ── ball position statistics ────────────────────────────────────────────
    ball_position_section: dict = {
        "x": _arr_summary(bx, dp),
        "y": _arr_summary(by, dp),
        "x_during_" + lbl0: _arr_summary(bx[poss == 0], dp),
        "x_during_" + lbl1: _arr_summary(bx[poss == 1], dp),
        "y_during_" + lbl0: _arr_summary(by[poss == 0], dp),
        "y_during_" + lbl1: _arr_summary(by[poss == 1], dp),
    }

    # ── possession by thirds ────────────────────────────────────────────────
    def _thirds_block(team_id: int) -> dict:
        d, m, a = possession_by_thirds(bx, pitch_length_m, poss, team_id)
        team_frames = int(np.sum(poss == team_id))
        frames_def = int(round(d / 100.0 * team_frames)) if team_frames > 0 else 0
        frames_mid = int(round(m / 100.0 * team_frames)) if team_frames > 0 else 0
        frames_att = max(0, team_frames - frames_def - frames_mid)
        return {
            "defensive_third": {"pct": _r(d, dp), "frames": frames_def},
            "middle_third": {"pct": _r(m, dp), "frames": frames_mid},
            "attacking_third": {"pct": _r(a, dp), "frames": frames_att},
        }

    thirds_section: dict = {
        lbl0: _thirds_block(0),
        lbl1: _thirds_block(1),
    }

    # ── defensive leakage ───────────────────────────────────────────────────
    y_box_min = (pitch_width_m - penalty_area_width_m) / 2.0
    y_box_max = (pitch_width_m + penalty_area_width_m) / 2.0
    if dir0 > 0:
        # team 0 attacks right → defends left (x = 0)
        t0_def_box_x: Tuple[float, float] = (0.0, penalty_area_length_m)
        t1_def_box_x: Tuple[float, float] = (
            pitch_length_m - penalty_area_length_m,
            pitch_length_m,
        )
    else:
        t0_def_box_x = (pitch_length_m - penalty_area_length_m, pitch_length_m)
        t1_def_box_x = (0.0, penalty_area_length_m)
    box_y: Tuple[float, float] = (y_box_min, y_box_max)

    def _leakage_block(box_x: Tuple[float, float], opponent_id: int) -> dict:
        frames = defensive_leakage(bx, by, poss, opponent_id, box_x, box_y)
        return {
            "frames_opponent_in_box": frames,
            "pct_of_total": _r(frames / n * 100, dp) if n > 0 else 0.0,
            "duration_sec": _r(frames / fps, dp) if fps > 0 else None,
        }

    leakage_section: dict = {
        lbl0 + "_defensive_box": _leakage_block(t0_def_box_x, 1),
        lbl1 + "_defensive_box": _leakage_block(t1_def_box_x, 0),
    }

    # ── pitch width utilization ─────────────────────────────────────────────
    def _lane_block(y_positions: np.ndarray) -> dict:
        pct = pitch_width_utilization(y_positions, pitch_width_m, n_lanes)
        cnts = vertical_lane_density(y_positions, pitch_width_m, n_lanes)
        return {
            label: {"pct": _r(float(pct[i]), dp), "count": int(cnts[i])}
            for i, label in enumerate(lane_labels)
        }

    width_section: dict = {
        "ball": _lane_block(by),
        "ball_during_" + lbl0: _lane_block(by[poss == 0]),
        "ball_during_" + lbl1: _lane_block(by[poss == 1]),
    }
    if team0_positions_per_frame is not None:
        chunks = [
            np.asarray(pts, dtype=float)[:, 1]
            for pts in team0_positions_per_frame
            if (
                np.asarray(pts).ndim == 2
                and np.asarray(pts).shape[1] >= 2
                and len(pts) > 0
            )
        ]
        all_t0_y = np.concatenate(chunks) if chunks else np.array([], dtype=float)
        width_section[lbl0 + "_players"] = _lane_block(all_t0_y)
    if team1_positions_per_frame is not None:
        chunks = [
            np.asarray(pts, dtype=float)[:, 1]
            for pts in team1_positions_per_frame
            if (
                np.asarray(pts).ndim == 2
                and np.asarray(pts).shape[1] >= 2
                and len(pts) > 0
            )
        ]
        all_t1_y = np.concatenate(chunks) if chunks else np.array([], dtype=float)
        width_section[lbl1 + "_players"] = _lane_block(all_t1_y)

    # ── expected threat (xT) ────────────────────────────────────────────────
    xt_vals = np.array(
        [
            expected_threat(float(x), float(y), pitch_length_m, pitch_width_m, xt_grid)
            for x, y in zip(bx, by)
        ],
        dtype=float,
    )
    xt_t0 = xt_vals[poss == 0]
    xt_t1 = xt_vals[poss == 1]
    xt_all_summary = _arr_summary(xt_vals, dp)
    xt_t0_summary = _arr_summary(xt_t0, dp)
    xt_t1_summary = _arr_summary(xt_t1, dp)
    xt_t0_summary["total_xt"] = _r(float(xt_t0.sum()), dp)
    xt_t1_summary["total_xt"] = _r(float(xt_t1.sum()), dp)
    xt_section: dict = {
        "ball": xt_all_summary,
        "ball_during_" + lbl0: xt_t0_summary,
        "ball_during_" + lbl1: xt_t1_summary,
    }

    # ── team compactness ────────────────────────────────────────────────────
    def _compact_stats(positions_per_frame: Optional[List[np.ndarray]]) -> Optional[dict]:
        if positions_per_frame is None:
            return None
        areas = []
        for pts in positions_per_frame:
            a = np.asarray(pts, dtype=float)
            if a.ndim == 2 and a.shape[1] == 2 and len(a) >= 2:
                areas.append(team_compactness(a))
        return _arr_summary(np.array(areas, dtype=float), dp)

    compactness_section: dict = {
        lbl0: _compact_stats(team0_positions_per_frame),
        lbl1: _compact_stats(team1_positions_per_frame),
    }

    # ── team separation (centroid-to-centroid distance) ─────────────────────
    def _separation_stats(
        t0_pos: Optional[List[np.ndarray]],
        t1_pos: Optional[List[np.ndarray]],
    ) -> Optional[dict]:
        if t0_pos is None or t1_pos is None:
            return None
        distances = []
        for pts0, pts1 in zip(t0_pos, t1_pos):
            a0 = np.asarray(pts0, dtype=float)
            a1 = np.asarray(pts1, dtype=float)
            if (
                a0.ndim == 2 and a0.shape[1] == 2 and len(a0) > 0
                and a1.ndim == 2 and a1.shape[1] == 2 and len(a1) > 0
            ):
                distances.append(float(np.linalg.norm(a0.mean(axis=0) - a1.mean(axis=0))))
        return _arr_summary(np.array(distances, dtype=float), dp)

    separation_section = _separation_stats(
        team0_positions_per_frame, team1_positions_per_frame
    )

    # ── assemble report ─────────────────────────────────────────────────────
    return {
        "meta": meta,
        "possession": possession_section,
        "passes": passes_section,
        "ball_progression": ball_progression_section,
        "ball_position_stats": ball_position_section,
        "possession_by_thirds": thirds_section,
        "defensive_leakage": leakage_section,
        "pitch_width_utilization": width_section,
        "expected_threat": xt_section,
        "team_compactness": compactness_section,
        "team_separation": separation_section,
    }
