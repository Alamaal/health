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
