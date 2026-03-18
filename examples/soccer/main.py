import argparse
from enum import Enum
from typing import Dict, Iterator, List, Optional
import warnings

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator, BallSmoother
from sports.common.team import (
    TeamClassifier,
    PlayerReIdentifier,
    resolve_players_team_with_cache as _resolve_players_team_with_cache,
)
from sports.common.stats import (
    PassDetector,
    PossessionTracker,
    TeamVoteBuffer,
    draw_pass_label,
    draw_stats_overlay,
)
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

# Maximum ball-to-player distance (px) for possession assignment.
MAX_OWNER_DIST_PX: float = 80.0
# Duration (seconds) to keep the "PASS" label visible on screen.
PASS_LABEL_DISPLAY_SEC: float = 1.0


def _validate_model_path(path: str) -> None:
    """
    Validate that a model file exists at the given path.

    Args:
        path (str): Absolute path to the model file.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run './setup.sh' inside examples/soccer/ to download the required models."
        )


COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def _is_cuda_device(device: str) -> bool:
    """Return True if the given device string refers to a CUDA GPU."""
    return str(device).lower().startswith('cuda')


def _safe_tracker_id(value) -> Optional[int]:
    """
    Safely convert a tracker ID value to int, returning None for NaN or invalid values.

    Args:
        value: The tracker ID value to convert.

    Returns:
        Optional[int]: Integer tracker ID, or None if conversion is not possible.
    """
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


# _resolve_players_team_with_cache is imported from sports.common.team as
# resolve_players_team_with_cache and re-exported here under the private name
# so all existing call sites inside this module continue to work unchanged.


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_players = players_xy[players_team_id == 0]
    team_1_players = players_xy[players_team_id == 1]

    if len(team_0_players) == 0 or len(team_1_players) == 0:
        warnings.warn(
            f"Cannot resolve goalkeeper teams: team 0 has {len(team_0_players)} "
            f"player(s), team 1 has {len(team_1_players)} player(s). "
            "Assigning all goalkeepers to team 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros(len(goalkeepers_xy), dtype=int)

    team_0_centroid = team_0_players.mean(axis=0)
    team_1_centroid = team_1_players.mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    """
    Render a bird's-eye radar view by projecting player positions onto a pitch diagram.

    Uses homography computed from detected pitch keypoints to transform each player's
    bottom-centre pixel position to pitch coordinates, then draws those points on a
    synthetic pitch image.

    Args:
        detections (sv.Detections): Merged player/goalkeeper/referee detections.
        keypoints (sv.KeyPoints): Pitch keypoint detections from the pitch model.
        color_lookup (np.ndarray): Integer array mapping each detection to a colour
            index (0 = team 1, 1 = team 2, 2 = referee, 3 = other).

    Returns:
        np.ndarray: Bird's-eye pitch image with player positions annotated.
            Returns a blank pitch if fewer than 4 keypoints are detected (homography
            requires at least 4 point correspondences).
    """
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    num_keypoints = int(np.sum(mask))
    if num_keypoints < 4:
        warnings.warn(
            f"Insufficient pitch keypoints detected for homography "
            f"(need 4+, got {num_keypoints}). Returning blank pitch.",
            RuntimeWarning,
            stacklevel=2,
        )
        return draw_pitch(config=CONFIG)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    _validate_model_path(PITCH_DETECTION_MODEL_PATH)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    _validate_model_path(PLAYER_DETECTION_MODEL_PATH)
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    _validate_model_path(BALL_DETECTION_MODEL_PATH)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    _validate_model_path(PLAYER_DETECTION_MODEL_PATH)
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    minimum_consecutive_frames=3
)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(
    source_video_path: str, device: str, stride: int = STRIDE
) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        stride (int, optional): Frame stride used when collecting player crops for
            team classifier training. Defaults to STRIDE.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    _validate_model_path(PLAYER_DETECTION_MODEL_PATH)
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    if not crops:
        raise RuntimeError(
            "No player crops collected during team-classification pass. "
            "Ensure the video contains players visible to the detection model."
        )

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=3,
    )
    track_team_cache: Dict[int, int] = {}
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = _resolve_players_team_with_cache(
            team_classifier, crops, players.tracker_id, track_team_cache)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str, stride: int = STRIDE) -> Iterator[np.ndarray]:
    """
    Run the full radar pipeline on a video and yield annotated frames.

    Combines player detection, team classification, player tracking, pitch
    detection, **ball tracking**, **pass detection**, and **possession tracking**
    to produce frames with player annotations, a bird's-eye radar overlay, live
    statistics (pass counts and possession percentages), and a ``PASS`` label
    rendered at the exact moment each pass is detected.

    A pass is registered whenever the ball transitions from one player to a
    *different* player on the **same** team and the transfer passes quality
    filters (speed, displacement, debounce, ID-switch guard).

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda').
        stride (int, optional): Frame stride used when collecting player crops for
            team classifier training. Defaults to STRIDE.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames with radar overlay,
            pass labels and statistics.
    """
    _validate_model_path(PLAYER_DETECTION_MODEL_PATH)
    _validate_model_path(PITCH_DETECTION_MODEL_PATH)
    _validate_model_path(BALL_DETECTION_MODEL_PATH)
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    # --- First pass: collect crops for team classifier ---
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    if not crops:
        raise RuntimeError(
            "No player crops collected during radar crop-collection pass. "
            "Ensure the video contains players visible to the detection model."
        )

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # --- Second pass: per-frame analysis ---
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps or 25.0

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=3,
    )

    # Ball detection slicer
    def _ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=_ball_callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    ball_tracker = BallTracker(buffer_size=20)
    ball_smoother = BallSmoother(window=5, noise_floor_px=5.0, max_velocity_px_per_frame=60.0)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    reid = PlayerReIdentifier(
        max_frames_lost=900,
        position_tolerance_px=150,
        min_consecutive_frames=10,
    )
    vote_buffer = TeamVoteBuffer(buffer_size=250, min_votes=8)
    pass_detector = PassDetector(fps=fps)
    possession_tracker = PossessionTracker(max_owner_dist_px=MAX_OWNER_DIST_PX)

    track_team_cache: Dict[int, int] = {}

    # Per-frame state for pass detection
    prev_owner_canonical: Optional[int] = None
    prev_owner_team: int = -1
    prev_owner_xy: Optional[np.ndarray] = None
    prev_ball_xy: Optional[np.ndarray] = None
    ball_origin_xy: Optional[np.ndarray] = None  # ball pos when current owner first got it
    last_pass_display_until: float = -1.0  # timestamp until which to show PASS label
    last_pass_ball_xy: Optional[np.ndarray] = None  # ball pos for PASS label

    for frame_idx, frame in enumerate(frame_generator):
        time_sec = frame_idx / fps

        # --- Pitch keypoints ---
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # --- Player detection & tracking ---
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = _resolve_players_team_with_cache(
            team_classifier, crops, players.tracker_id, track_team_cache)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # --- Ball detection & tracking ---
        ball_dets = ball_slicer(frame).with_nms(threshold=0.1)
        ball_dets = ball_tracker.update(ball_dets)
        ball_xy: Optional[np.ndarray] = None
        if len(ball_dets) > 0:
            raw_ball_xy = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
            ball_xy = ball_smoother.update(raw_ball_xy)
        else:
            ball_xy = ball_smoother.update(None)

        # --- Stable IDs (re-identification) & team vote buffering ---
        all_field_players = sv.Detections.merge([players, goalkeepers])
        all_field_team_ids = np.concatenate([players_team_id, goalkeepers_team_id]) \
            if len(goalkeepers) > 0 else players_team_id.copy()
        all_field_xy = all_field_players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) \
            if len(all_field_players) > 0 else np.empty((0, 2))

        # Extract SigLIP embeddings for players/goalkeepers that are new to the
        # re-identifier gallery.  Embeddings are only needed for brand-new
        # tracker IDs (i.e. first appearance or re-appearance after loss) so
        # that we can compare against gallery entries for appearance-based
        # re-identification.  Already-tracked players reuse stored embeddings.
        all_field_crops = get_crops(frame, all_field_players) \
            if len(all_field_players) > 0 else []
        new_embed_indices = []
        for i in range(len(all_field_players)):
            raw_tid = (
                all_field_players.tracker_id[i]
                if all_field_players.tracker_id is not None else None
            )
            tid_int = _safe_tracker_id(raw_tid)
            if tid_int is not None and not reid.has_tracker_id(tid_int):
                new_embed_indices.append(i)
        field_embeddings: List[Optional[np.ndarray]] = [None] * len(all_field_players)
        if new_embed_indices and all_field_crops:
            new_crops_for_reid = [all_field_crops[i] for i in new_embed_indices]
            reid_embs = team_classifier.extract_features(
                new_crops_for_reid, verbose=False
            )
            for j, idx in enumerate(new_embed_indices):
                field_embeddings[idx] = reid_embs[j]

        canonical_ids = []
        stable_team_ids = []
        for i in range(len(all_field_players)):
            tid = _safe_tracker_id(
                all_field_players.tracker_id[i]
                if all_field_players.tracker_id is not None else None
            )
            if tid is None:
                canonical_ids.append(None)
                stable_team_ids.append(int(all_field_team_ids[i]))
                continue
            cid = reid.get_stable_id(
                tid, all_field_xy[i],
                team_id=int(all_field_team_ids[i]),
                embedding=field_embeddings[i],
            )
            canonical_ids.append(cid)
            stable_team_ids.append(vote_buffer.update(cid, int(all_field_team_ids[i])))
        reid.end_frame()

        # --- Possession & pass detection ---
        owner_idx: Optional[int] = None
        if ball_xy is not None and len(all_field_xy) > 0:
            dists = np.linalg.norm(all_field_xy - ball_xy, axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] <= MAX_OWNER_DIST_PX:
                owner_idx = nearest

        if owner_idx is not None and ball_xy is not None:
            cur_team = stable_team_ids[owner_idx]
            cur_canonical = canonical_ids[owner_idx]
            cur_xy = all_field_xy[owner_idx]

            if cur_team in (0, 1):
                possession_tracker.update(cur_team, ball_xy, cur_xy)

            # Pass detection: ownership changed?
            if cur_canonical != prev_owner_canonical and prev_owner_canonical is not None:
                # Compute ball speed & displacement
                ball_speed = 0.0
                ball_disp = 0.0
                if prev_ball_xy is not None:
                    dt = 1.0 / fps
                    ball_speed = float(np.linalg.norm(ball_xy - prev_ball_xy)) / dt
                if ball_origin_xy is not None:
                    ball_disp = float(np.linalg.norm(ball_xy - ball_origin_xy))

                is_pass = pass_detector.check(
                    prev_canonical_id=prev_owner_canonical,
                    prev_team_id=prev_owner_team,
                    new_canonical_id=cur_canonical,
                    new_team_id=cur_team,
                    prev_player_xy=prev_owner_xy,
                    new_player_xy=cur_xy,
                    ball_speed_px_per_sec=ball_speed,
                    ball_displacement_px=ball_disp,
                    time_sec=time_sec,
                )
                if is_pass:
                    last_pass_display_until = time_sec + PASS_LABEL_DISPLAY_SEC
                    last_pass_ball_xy = ball_xy.copy()

                # New owner takes over
                ball_origin_xy = ball_xy.copy()

            prev_owner_canonical = cur_canonical
            prev_owner_team = cur_team
            prev_owner_xy = cur_xy.copy()

        if ball_xy is not None:
            prev_ball_xy = ball_xy.copy()

        # --- Merge detections for annotation ---
        merged_detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in merged_detections.tracker_id]

        # --- Annotate frame ---
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, merged_detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, merged_detections, labels,
            custom_color_lookup=color_lookup)
        annotated_frame = ball_annotator.annotate(annotated_frame, ball_dets)

        # --- PASS label ---
        if time_sec <= last_pass_display_until and last_pass_ball_xy is not None:
            pe = pass_detector.last_pass_event
            if pe is not None:
                annotated_frame = draw_pass_label(annotated_frame, pe, last_pass_ball_xy)

        # --- Stats overlay ---
        annotated_frame = draw_stats_overlay(
            annotated_frame, pass_detector, possession_tracker, frame_idx + 1)

        # --- Radar overlay ---
        h, w, _ = frame.shape
        radar = render_radar(merged_detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    stride: int = STRIDE,
    display: bool = False,
) -> None:
    """
    Run soccer video analysis in the specified mode and write the output video.

    Args:
        source_video_path (str): Path to the input video file.
        target_video_path (str): Path where the annotated output video will be saved.
        device (str): Compute device to use (e.g., 'cpu', 'cuda', 'mps').
        mode (Mode): Analysis mode to execute.
        stride (int, optional): Frame stride for crop-collection phases.
            Defaults to STRIDE.
        display (bool, optional): When True, show each annotated frame in an
            OpenCV window in real time. Press 'q' to quit early.
            Defaults to False (headless / server-friendly processing).
    """
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device, stride=stride)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device, stride=stride)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            if display:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Soccer AI: real-time player detection, tracking, team '
                    'classification and radar visualization.'
    )
    parser.add_argument(
        '--source_video_path', type=str, required=True,
        help='Path to the input video file.')
    parser.add_argument(
        '--target_video_path', type=str, required=True,
        help='Path where the annotated output video will be saved.')
    parser.add_argument(
        '--device', type=str, default='cpu',
        help="Compute device: 'cpu', 'cuda', or 'mps'. Defaults to 'cpu'.")
    parser.add_argument(
        '--mode', type=Mode, default=Mode.PLAYER_DETECTION,
        choices=list(Mode),
        help=f"Analysis mode. Choices: {[m.value for m in Mode]}. "
             f"Defaults to PLAYER_DETECTION.")
    parser.add_argument(
        '--stride', type=int, default=STRIDE,
        help=f'Frame stride for crop-collection phases (TEAM_CLASSIFICATION, RADAR). '
             f'Defaults to {STRIDE}.')
    parser.add_argument(
        '--display', action='store_true', default=False,
        help='Show annotated frames in a live OpenCV window during processing. '
             'Press q to quit early. Disabled by default (headless mode).')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        stride=args.stride,
        display=args.display,
    )
