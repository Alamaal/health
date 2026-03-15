from typing import Dict, Generator, Iterable, List, Optional, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        current_batch.append(element)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch

class TeamClassifier:
    def __init__(
        self,
        device: str = 'cpu',
        batch_size: int = 32,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the TeamClassifier with device and batch size.

        Args:
            device (str): The device to run the model on ('cpu' or 'cuda').
            batch_size (int): The batch size for processing images.
            model_path (Optional[str]): Path or HuggingFace identifier for the
                SigLIP vision model.  Defaults to ``SIGLIP_MODEL_PATH``.
        """
        _model_path = model_path or SIGLIP_MODEL_PATH
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(_model_path).to(device)

        if "cuda" in device:
            self.features_model = self.features_model.half()

        self.processor = AutoProcessor.from_pretrained(_model_path)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2, n_init=10)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array. Returns an empty array
                with shape ``(0,)`` when *crops* is empty.
        """
        if len(crops) == 0:
            return np.array([])

        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = list(create_batches(crops, self.batch_size))
        data = []

        with torch.inference_mode():
            for batch in tqdm(batches, desc='Turbo Embedding extraction'):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

                if "cuda" in self.device:
                    inputs = {k: v.half() if v.dtype == torch.float else v for k, v in inputs.items()}

                # التصحيح هنا: نستخدم الموديل مباشرة ثم نأخذ الـ pooler_output
                outputs = self.features_model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Raises:
            ValueError: If *crops* is empty. At least one crop is required to fit
                the classifier.
        """
        if len(crops) == 0:
            raise ValueError(
                "TeamClassifier.fit() requires at least one crop. "
                "No player crops were collected — check that the video contains "
                "detectable players and that the player detection model is correct."
            )
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


class PlayerReIdentifier:
    """
    Provides stable player IDs across an entire match video.

    ByteTrack (and similar trackers) assigns a brand-new integer ID whenever
    a player re-enters the frame after an occlusion or camera cut.  This class
    wraps around the raw tracker IDs and returns a *canonical* (stable) ID by
    searching a gallery of recently-lost players.  Matching is done using the
    player's pitch position and (optionally) their team assignment, so no
    additional GPU inference is required beyond what the tracker already does.

    Typical usage inside a per-frame loop::

        reid = PlayerReIdentifier(max_frames_lost=90, position_tolerance_px=120)

        # After sv.ByteTrack.update_with_detections(detections):
        for i, (tid, xy) in enumerate(zip(tracker_ids, positions_xy)):
            stable_id = reid.get_stable_id(tid, xy, team_id=team_ids[i])
            # Use stable_id instead of tid for all statistics / event logging.

        reid.end_frame()   # Age gallery; call once per processed frame.
    """

    def __init__(
        self,
        max_frames_lost: int = 90,
        position_tolerance_px: float = 120.0,
    ) -> None:
        """
        Initialise the re-identifier.

        Args:
            max_frames_lost (int): Number of processed frames a lost track is
                retained in the gallery before being discarded.  At 30 fps with
                a stride of 3 this equals ~9 s of video.  Defaults to 90.
            position_tolerance_px (float): Maximum Euclidean distance (in source
                image pixels) between a newly-created track and a gallery entry
                for them to be considered the same player.  Defaults to 120.
        """
        self._max_frames_lost = int(max_frames_lost)
        self._position_tolerance = float(position_tolerance_px)

        # gallery: canonical_tid → {'xy': ndarray, 'team_id': int|None, 'age': int}
        self._gallery: Dict[int, dict] = {}
        # raw tracker_id → canonical_id  (set on first sight of tracker_id)
        self._id_map: Dict[int, int] = {}
        # raw tracker IDs seen in the *current* frame (cleared in end_frame)
        self._active_tids: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stable_id(
        self,
        tracker_id: int,
        xy: np.ndarray,
        team_id: Optional[int] = None,
    ) -> int:
        """
        Return a stable canonical player ID for *tracker_id*.

        If *tracker_id* has been seen in a previous frame its canonical ID is
        returned unchanged and the gallery entry is refreshed.

        If *tracker_id* is brand-new the gallery of recently-lost tracks is
        searched for a position match of the same team.  When a match is found
        the original ID is re-used; otherwise *tracker_id* itself becomes the
        canonical ID (and is added to the gallery).

        Args:
            tracker_id (int): Raw tracker ID from ByteTrack.
            xy (np.ndarray): Bottom-centre pixel coordinate ``[x, y]``.
            team_id (Optional[int]): Team assignment (0 or 1) of the player,
                or *None* if not yet classified.

        Returns:
            int: Stable canonical player ID.
        """
        tracker_id = int(tracker_id)
        self._active_tids.add(tracker_id)
        xy = np.asarray(xy, dtype=float)

        if tracker_id in self._id_map:
            canonical = self._id_map[tracker_id]
            # Refresh gallery with latest position and team info.
            self._gallery[canonical] = {
                "xy": xy,
                "team_id": team_id if team_id is not None
                           else self._gallery.get(canonical, {}).get("team_id"),
                "age": 0,
            }
            return canonical

        # Brand-new tracker ID — search gallery for a position match.
        canonical = self._find_match(xy, team_id)
        self._id_map[tracker_id] = canonical
        self._gallery[canonical] = {"xy": xy, "team_id": team_id, "age": 0}
        return canonical

    def end_frame(self, active_tracker_ids: Optional[set] = None) -> None:
        """
        Age the gallery and remove entries that have been lost too long.

        Call **once per processed frame** after all :meth:`get_stable_id` calls
        for that frame have been made.

        Args:
            active_tracker_ids (Optional[set]): Set of raw tracker IDs that were
                active in the current frame.  When *None*, the IDs collected via
                :meth:`get_stable_id` during the current frame are used.
        """
        current_raw = (
            active_tracker_ids
            if active_tracker_ids is not None
            else self._active_tids
        )
        canonical_active = {
            self._id_map.get(tid, tid) for tid in current_raw
        }

        to_remove = []
        for canon_tid in list(self._gallery.keys()):
            if canon_tid in canonical_active:
                self._gallery[canon_tid]["age"] = 0
            else:
                self._gallery[canon_tid]["age"] += 1
                if self._gallery[canon_tid]["age"] >= self._max_frames_lost:
                    to_remove.append(canon_tid)

        for canon_tid in to_remove:
            del self._gallery[canon_tid]

        self._active_tids = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_match(
        self, xy: np.ndarray, team_id: Optional[int]
    ) -> int:
        """
        Search the gallery for the closest lost track within tolerance.

        Only lost tracks (``age > 0``) are considered.  When *team_id* is
        provided, only gallery entries with the same team (or unknown team) are
        considered.

        Returns the canonical ID of the best match, or a new unique ID derived
        from the largest existing canonical ID + 1 when no match is found.
        """
        best_tid: Optional[int] = None
        best_dist = float("inf")

        for canon_tid, info in self._gallery.items():
            # Only try to re-use genuinely *lost* tracks.
            if info["age"] == 0:
                continue

            # Skip tracks from a different team.
            entry_team = info.get("team_id")
            if (
                team_id is not None
                and entry_team is not None
                and int(entry_team) != int(team_id)
            ):
                continue

            dist = float(np.linalg.norm(xy - info["xy"]))
            if dist < best_dist and dist <= self._position_tolerance:
                best_dist = dist
                best_tid = canon_tid

        if best_tid is not None:
            return best_tid

        # No suitable gallery match — allocate a truly new canonical ID.
        # We use max(existing canonical IDs, raw active IDs) + 1 so that the
        # canonical ID never collides with a live ByteTrack ID.
        all_ids = set(self._gallery.keys()) | set(self._id_map.values())
        if all_ids:
            return max(all_ids) + 1
        return 0
