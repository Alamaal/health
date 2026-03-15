import argparse
from enum import Enum
from typing import Dict, Iterator, List, Optional
import warnings

import os
import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
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
PLAYER_IMGSZ = 640
BALL_IMGSZ = 640
CONFIG = SoccerPitchConfiguration()


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


def _resolve_players_team_with_cache(
    team_classifier: TeamClassifier,
    crops: List[np.ndarray],
    tracker_ids,
    team_cache: Dict[int, int],
) -> np.ndarray:
    """
    Predict team IDs for player crops, using a cache keyed by tracker ID to avoid
    redundant inference on already-classified players.

    Args:
        team_classifier (TeamClassifier): Fitted team classifier.
        crops (List[np.ndarray]): Cropped images of detected players.
        tracker_ids: Array of tracker IDs corresponding to each crop (may contain NaN).
        team_cache (Dict[int, int]): Mutable cache mapping tracker ID to team ID.

    Returns:
        np.ndarray: Integer array of team IDs (0 or 1) for each crop.
    """
    if len(crops) == 0:
        return np.array([], dtype=int)

    team_ids = np.full(len(crops), -1, dtype=int)
    to_classify_idx = []
    to_classify_crops = []
    to_classify_tids = []

    for i in range(len(crops)):
        tid = _safe_tracker_id(tracker_ids[i]) if tracker_ids is not None else None
        if tid is not None and tid in team_cache:
            team_ids[i] = int(team_cache[tid])
        else:
            to_classify_idx.append(i)
            to_classify_crops.append(crops[i])
            to_classify_tids.append(tid)

    if len(to_classify_crops) > 0:
        predicted = team_classifier.predict(to_classify_crops).astype(int)
        for idx, pred_team, tid in zip(to_classify_idx, predicted, to_classify_tids):
            team_ids[idx] = int(pred_team)
            if tid is not None and team_ids[idx] in (0, 1):
                team_cache[tid] = int(team_ids[idx])

    return team_ids


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
    use_half = _is_cuda_device(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, half=use_half, verbose=False)[0]
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
    use_half = _is_cuda_device(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=PLAYER_IMGSZ, half=use_half, verbose=False)[0]
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
    use_half = _is_cuda_device(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    for frame in frame_generator:
        result = ball_detection_model(frame, imgsz=BALL_IMGSZ, half=use_half, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result).with_nms(threshold=0.1)
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
    use_half = _is_cuda_device(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=PLAYER_IMGSZ, half=use_half, verbose=False)[0]
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
    use_half = _is_cuda_device(device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=PLAYER_IMGSZ, half=use_half, verbose=False)[0]
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
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    track_team_cache: Dict[int, int] = {}
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=PLAYER_IMGSZ, half=use_half, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = _resolve_players_team_with_cache(
            team_classifier=team_classifier,
            crops=crops,
            tracker_ids=players.tracker_id,
            team_cache=track_team_cache,
        )

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

    Combines player detection, team classification, player tracking, and pitch
    detection to produce frames with both player annotations and a bird's-eye radar
    overlay showing team positions on a miniature pitch diagram.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda').
        stride (int, optional): Frame stride used when collecting player crops for
            team classifier training. Defaults to STRIDE.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames with radar overlay.
    """
    _validate_model_path(PLAYER_DETECTION_MODEL_PATH)
    _validate_model_path(PITCH_DETECTION_MODEL_PATH)
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    use_half = _is_cuda_device(device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=PLAYER_IMGSZ, half=use_half, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    if not crops:
        raise RuntimeError(
            "No player crops collected during radar crop-collection pass. "
            "Ensure the video contains players visible to the detection model."
        )

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    track_team_cache: Dict[int, int] = {}
    for frame in frame_generator:
        result = pitch_detection_model(frame, half=use_half, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=PLAYER_IMGSZ, half=use_half, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = _resolve_players_team_with_cache(
            team_classifier=team_classifier,
            crops=crops,
            tracker_ids=players.tracker_id,
            team_cache=track_team_cache,
        )

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
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
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
