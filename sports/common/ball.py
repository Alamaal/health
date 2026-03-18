from collections import deque
from typing import Optional

import cv2
import numpy as np
import supervision as sv


class BallSmoother:
    """
    Online filter that removes camera-jitter artifacts from ball positions.

    Two complementary filters are applied each time a new position is
    supplied via :meth:`update`:

    1. **Noise floor** – frame-to-frame displacements whose Euclidean
       magnitude is *smaller* than *noise_floor_px* are treated as
       sub-pixel jitter and the position is held at the previous smoothed
       value.  This prevents "phantom" metre-accumulation caused by
       single-pixel oscillations over thousands of frames.

    2. **Velocity clamping** – frame-to-frame displacements *larger* than
       *max_velocity_px_per_frame* are treated as tracking-noise spikes
       (e.g. a rogue detection on the opposite side of the pitch) and are
       discarded.  The previous smoothed position is returned unchanged.

    3. **Moving average** – accepted positions are stored in a rolling
       window of length *window*.  The output is the mean of all positions
       currently in the window, which attenuates residual high-frequency
       jitter without introducing the lag of a heavier filter.

    Args:
        window (int): Length of the rolling-average window. Defaults to 5.
        noise_floor_px (float): Minimum displacement (pixels) for a
            frame-to-frame movement to be accepted.  Movements strictly
            below this value are ignored.  Defaults to 5.0.
        max_velocity_px_per_frame (float): Maximum displacement (pixels)
            between consecutive frames.  Any larger jump is treated as a
            tracking error and discarded.  At 30 fps and a typical
            1920 × 1080 pitch mapping, ~60 px/frame corresponds to
            roughly 150 km/h.  Defaults to 60.0.

    Example::

        smoother = BallSmoother(window=5, noise_floor_px=5.0)
        for frame in video:
            raw_xy = detect_ball(frame)          # may be None
            smooth_xy = smoother.update(raw_xy)  # None until first detection
    """

    def __init__(
        self,
        window: int = 5,
        noise_floor_px: float = 5.0,
        max_velocity_px_per_frame: float = 60.0,
    ) -> None:
        if int(window) < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")
        if float(noise_floor_px) < 0.0:
            raise ValueError(f"noise_floor_px must be >= 0, got {noise_floor_px!r}")
        if float(max_velocity_px_per_frame) <= 0.0:
            raise ValueError(
                f"max_velocity_px_per_frame must be > 0, got {max_velocity_px_per_frame!r}"
            )
        self._window = int(window)
        self._noise_floor = float(noise_floor_px)
        self._max_velocity = float(max_velocity_px_per_frame)
        self._history: deque = deque(maxlen=self._window)
        self._last_smoothed: Optional[np.ndarray] = None

    def update(self, xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Accept a raw ball position and return the smoothed position.

        Args:
            xy (Optional[np.ndarray]): Raw ball centre ``[x, y]`` in pixel
                coordinates, or *None* if the ball was not detected this
                frame.

        Returns:
            Optional[np.ndarray]: Smoothed ``[x, y]`` position, or *None*
                if no valid position has been accepted yet.
        """
        if xy is None:
            return self._last_smoothed

        xy = np.asarray(xy, dtype=float).ravel()[:2]

        if self._last_smoothed is not None:
            displacement = float(np.linalg.norm(xy - self._last_smoothed))

            # Velocity clamping: discard impossibly fast movements.
            if displacement > self._max_velocity:
                return self._last_smoothed

            # Noise floor: ignore sub-pixel jitter.
            if displacement < self._noise_floor:
                return self._last_smoothed

        self._history.append(xy)
        smoothed = np.mean(self._history, axis=0)
        self._last_smoothed = smoothed
        return smoothed

    def reset(self) -> None:
        """Clear the internal history (e.g. after a scene cut)."""
        self._history.clear()
        self._last_smoothed = None


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    A class used to track a soccer ball's position across video frames.

    The BallTracker class maintains a buffer of recent ball positions and uses this
    buffer to predict the ball's position in the current frame by selecting the
    detection closest to the average position (centroid) of the recent positions.

    Attributes:
        buffer (collections.deque): A deque buffer to store recent ball positions.
    """
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with new detections and returns the detection closest to the
        centroid of recent positions.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The detection closest to the centroid of recent positions.
            If there are no detections, returns the input detections.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]
