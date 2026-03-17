from collections import Counter
from typing import Dict, Generator, Iterable, List, Optional, TypeVar

import numpy as np
import supervision as sv
from tqdm import tqdm

try:
    import torch
    import umap
    from sklearn.cluster import KMeans
    from transformers import AutoProcessor, SiglipVisionModel
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

V = TypeVar("V")
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
# Small value added to denominators to prevent division by zero.
_EPS: float = 1e-9

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

        Raises:
            ImportError: If ``torch``, ``umap-learn``, ``scikit-learn``, or
                ``transformers`` are not installed.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "TeamClassifier requires torch, umap-learn, scikit-learn, and "
                "transformers. Install them with: "
                "pip install torch umap-learn scikit-learn transformers"
            )
        _model_path = model_path or SIGLIP_MODEL_PATH
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(_model_path).to(device)

        if "cuda" in device:
            self.features_model = self.features_model.half()

        self.processor = AutoProcessor.from_pretrained(_model_path)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2, n_init=10)

    def extract_features(
        self, crops: List[np.ndarray], verbose: bool = True
    ) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.
            verbose (bool): Whether to display a tqdm progress bar during
                extraction.  Set to ``False`` when calling per-frame to keep
                console output clean.  Defaults to ``True``.

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
            for batch in tqdm(batches, desc='Extracting player embeddings',
                              disable=not verbose):
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
        data = self.extract_features(crops, verbose=False)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

    def predict_with_confidence(
        self, crops: List[np.ndarray]
    ) -> tuple:
        """
        Predict team IDs and return a per-sample confidence score.

        Confidence is derived from the KMeans centroid distances.  For each
        sample the score is::

            confidence_i = d_other / (d_winner + d_other)

        where *d_winner* is the Euclidean distance to the assigned cluster
        centre and *d_other* is the distance to the other cluster centre.

        This maps naturally to the intuitive scale:

        * ``1.0`` — player is exactly on the winning centroid (perfectly
          certain).
        * ``0.5`` — player is equidistant from both centroids (maximally
          ambiguous).

        Args:
            crops (List[np.ndarray]): List of player crop images (BGR).

        Returns:
            tuple: A ``(team_ids, confidences)`` pair where

                * ``team_ids`` is an int ``np.ndarray`` of shape ``(N,)``
                  containing 0 or 1 for each crop.
                * ``confidences`` is a float ``np.ndarray`` of shape ``(N,)``
                  with values in ``[0.5, 1.0]``.

                Both arrays are empty (shape ``(0,)``) when *crops* is empty.
        """
        if len(crops) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        data = self.extract_features(crops, verbose=False)
        projections = self.reducer.transform(data)
        # distances shape: (N, 2) — distance to each of the 2 cluster centres
        distances = self.cluster_model.transform(projections)
        team_ids = np.argmin(distances, axis=1).astype(int)

        d_winner = distances[np.arange(len(team_ids)), team_ids]
        d_other = distances[np.arange(len(team_ids)), 1 - team_ids]
        confidences = d_other / (d_winner + d_other + _EPS)

        return team_ids, confidences


class PlayerReIdentifier:
    """
    Provides stable player IDs across an entire match video.

    ByteTrack (and similar trackers) assigns a brand-new integer ID whenever
    a player re-enters the frame after an occlusion or camera cut.  This class
    wraps around the raw tracker IDs and returns a *canonical* (stable) ID by
    searching a gallery of recently-lost players.

    Matching combines two complementary signals:

    1. **Visual embedding similarity** – cosine similarity between SigLIP
       embeddings of the player crop, when *embedding* vectors are supplied via
       :meth:`get_stable_id`.  This is the preferred method because it is
       robust to position changes caused by camera pans or player sprints.
    2. **Spatial proximity** – Euclidean distance between pitch positions
       (bottom-centre pixel coordinates).  Used as an automatic fallback when
       no embedding is available for either the new track or the gallery entry.

    A simple *Gallery System* is maintained internally: each canonical player ID
    maps to their last-known position, team, visual embedding, and the number of
    frames they have been absent.  Gallery entries older than *max_frames_lost*
    frames are pruned.  The visual embedding is kept as a running average over
    all frames the player has been observed, so that it captures a stable
    appearance representation.

    Typical usage inside a per-frame loop::

        reid = PlayerReIdentifier(max_frames_lost=900, position_tolerance_px=120)

        # After sv.ByteTrack.update_with_detections(detections):
        for i, (tid, xy) in enumerate(zip(tracker_ids, positions_xy)):
            stable_id = reid.get_stable_id(
                tid, xy, team_id=team_ids[i], embedding=embeddings[i]
            )
            # Use stable_id instead of tid for all statistics / event logging.

        reid.end_frame()   # Age gallery; call once per processed frame.
    """

    def __init__(
        self,
        max_frames_lost: int = 900,
        position_tolerance_px: float = 120.0,
        embedding_similarity_threshold: float = 0.85,
    ) -> None:
        """
        Initialise the re-identifier.

        Args:
            max_frames_lost (int): Number of processed frames a lost track is
                retained in the gallery before being discarded.  At 30 fps this
                equals ~30 s of video.  Defaults to 900.
            position_tolerance_px (float): Maximum Euclidean distance (in source
                image pixels) between a newly-created track and a gallery entry
                for them to be considered the same player (spatial fallback).
                Defaults to 120.
            embedding_similarity_threshold (float): Minimum cosine similarity
                between two player visual embeddings for a re-identification
                match to be accepted.  Only applied when embeddings are
                available for both the new track and the gallery entry.
                Defaults to 0.85.
        """
        self._max_frames_lost = int(max_frames_lost)
        self._position_tolerance = float(position_tolerance_px)
        self._embedding_similarity_threshold = float(embedding_similarity_threshold)

        # gallery: canonical_tid → {
        #   'xy': ndarray, 'team_id': int|None, 'age': int,
        #   'embedding': ndarray|None
        # }
        self._gallery: Dict[int, dict] = {}
        # raw tracker_id → canonical_id  (set on first sight of tracker_id)
        self._id_map: Dict[int, int] = {}
        # raw tracker IDs seen in the *current* frame (cleared in end_frame)
        self._active_tids: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_tracker_id(self, tracker_id: int) -> bool:
        """
        Return *True* if *tracker_id* has already been seen in a previous frame.

        This can be used by callers to decide whether expensive embedding
        extraction is needed — embeddings are only required for brand-new
        tracker IDs that have not yet been assigned a canonical ID.

        Args:
            tracker_id (int): Raw tracker ID from ByteTrack.

        Returns:
            bool: ``True`` if the tracker ID has a canonical mapping.
        """
        return int(tracker_id) in self._id_map

    def get_stable_id(
        self,
        tracker_id: int,
        xy: np.ndarray,
        team_id: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> int:
        """
        Return a stable canonical player ID for *tracker_id*.

        If *tracker_id* has been seen in a previous frame its canonical ID is
        returned unchanged and the gallery entry (position, team, and
        embedding) is refreshed.

        If *tracker_id* is brand-new the gallery of recently-lost tracks is
        searched for a match.  When *embedding* is provided, cosine similarity
        against stored gallery embeddings is the primary matching criterion
        (within the same team).  If no embedding match is found, Euclidean
        distance between pitch positions is used as a fallback.  When a match
        is found the original canonical ID is re-used; otherwise *tracker_id*
        itself becomes the canonical ID.

        Args:
            tracker_id (int): Raw tracker ID from ByteTrack.
            xy (np.ndarray): Bottom-centre pixel coordinate ``[x, y]``.
            team_id (Optional[int]): Team assignment (0 or 1) of the player,
                or *None* if not yet classified.
            embedding (Optional[np.ndarray]): Visual feature vector for the
                player crop (e.g. from SigLIP ``pooler_output``).  When
                provided for a brand-new tracker ID, it is compared against
                gallery embeddings to enable appearance-based re-identification.
                When provided for a known tracker ID, the gallery embedding is
                updated via a running average.

        Returns:
            int: Stable canonical player ID.
        """
        tracker_id = int(tracker_id)
        self._active_tids.add(tracker_id)
        xy = np.asarray(xy, dtype=float)

        if tracker_id in self._id_map:
            canonical = self._id_map[tracker_id]
            entry = self._gallery.get(canonical, {})
            # Refresh gallery with latest position, team and embedding.
            self._gallery[canonical] = {
                "xy": xy,
                "team_id": team_id if team_id is not None
                           else entry.get("team_id"),
                "age": 0,
                "embedding": self._update_embedding(
                    entry.get("embedding"), embedding
                ),
            }
            return canonical

        # Brand-new tracker ID — search gallery for a position or embedding match.
        canonical = self._find_match(xy, team_id, embedding)
        self._id_map[tracker_id] = canonical
        self._gallery[canonical] = {
            "xy": xy, "team_id": team_id, "age": 0, "embedding": embedding,
        }
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
        self,
        xy: np.ndarray,
        team_id: Optional[int],
        embedding: Optional[np.ndarray] = None,
    ) -> int:
        """
        Search the gallery for the closest lost track within tolerance.

        Only lost tracks (``age > 0``) are considered.  When *team_id* is
        provided, only gallery entries with the same team (or unknown team) are
        searched.

        Matching priority:
        1. **Embedding similarity** – when both the query *embedding* and the
           gallery entry have a stored embedding, cosine similarity is used.
           The entry with the highest similarity above
           ``_embedding_similarity_threshold`` is preferred.
        2. **Spatial proximity** – used as a fallback when no embedding-based
           match is found (e.g. the gallery entry pre-dates embedding support,
           or embeddings were not provided).

        Returns the canonical ID of the best match, or a new unique ID when
        no match is found.
        """
        best_emb_tid: Optional[int] = None
        best_emb_sim: float = -1.0
        best_pos_tid: Optional[int] = None
        best_pos_dist: float = float("inf")

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

            # --- Embedding-based matching (preferred) ---
            gallery_emb = info.get("embedding")
            if embedding is not None and gallery_emb is not None:
                sim = self._cosine_similarity(embedding, gallery_emb)
                if sim >= self._embedding_similarity_threshold and sim > best_emb_sim:
                    best_emb_sim = sim
                    best_emb_tid = canon_tid

            # --- Spatial fallback ---
            dist = float(np.linalg.norm(xy - info["xy"]))
            if dist < best_pos_dist and dist <= self._position_tolerance:
                best_pos_dist = dist
                best_pos_tid = canon_tid

        # Prefer embedding match over spatial match.
        if best_emb_tid is not None:
            return best_emb_tid
        if best_pos_tid is not None:
            return best_pos_tid

        # No suitable gallery match — allocate a truly new canonical ID.
        # We use max(existing canonical IDs, raw active IDs) + 1 so that the
        # canonical ID never collides with a live ByteTrack ID.
        all_ids = set(self._gallery.keys()) | set(self._id_map.values())
        if all_ids:
            return max(all_ids) + 1
        return 0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1-D (or higher-D) vectors.

        Args:
            a (np.ndarray): First feature vector.
            b (np.ndarray): Second feature vector.

        Returns:
            float: Cosine similarity in ``[-1, 1]``.  Returns ``0.0`` when
                either vector has near-zero norm.
        """
        a = a.flatten()
        b = b.flatten()
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < _EPS or norm_b < _EPS:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _update_embedding(
        existing: Optional[np.ndarray],
        new: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Update a stored gallery embedding using an exponential moving average.

        An EMA with ``alpha=0.15`` (15 % new, 85 % old) lets the gallery
        representation converge over many re-appearances rather than
        oscillating between the last two observations.

        Args:
            existing (Optional[np.ndarray]): The currently stored embedding,
                or *None* if no embedding has been stored yet.
            new (Optional[np.ndarray]): The new embedding to incorporate,
                or *None* if no new embedding is available.

        Returns:
            Optional[np.ndarray]: Updated embedding, or *None* if both inputs
                are *None*.
        """
        if new is None:
            return existing
        if existing is None:
            return new.copy()
        # EMA: weight recent updates lightly so the gallery stabilises over time.
        _EMBED_EMA_ALPHA = 0.15
        return existing * (1.0 - _EMBED_EMA_ALPHA) + new * _EMBED_EMA_ALPHA


# ---------------------------------------------------------------------------
# Team-cache helper utilities
# ---------------------------------------------------------------------------

def _safe_tracker_id(value) -> Optional[int]:
    """
    Safely convert a tracker ID value to int, returning None for NaN or invalid.

    Args:
        value: The tracker ID value to convert (may be float NaN from ByteTrack).

    Returns:
        Optional[int]: Integer tracker ID, or *None* if the value is NaN,
        *None*, or otherwise unconvertible.
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


def majority_vote_team_reassignment(
    player_team_history: Dict[int, List[int]],
    majority_threshold: float = 0.8,
) -> Dict[int, int]:
    """
    Post-process player team assignments using majority-vote reinforcement.

    After tracking is complete, some canonical player IDs may have been
    misclassified into the wrong team on individual frames due to ID
    fragmentation or colour-model bias.  This function collects the full
    per-frame team-classification history for each canonical ID and returns
    a *corrected* mapping: any player whose dominant team accounts for at
    least *majority_threshold* fraction of their recorded votes is included
    in the result with that team.  Players whose history does not reach the
    threshold are omitted, meaning callers should keep their original
    assignment for those players.

    Typical usage::

        # Accumulate raw per-frame predictions during tracking:
        history: Dict[int, List[int]] = defaultdict(list)
        for frame in frames:
            for canonical_id, raw_team in zip(canonical_ids, raw_teams):
                history[canonical_id].append(raw_team)

        # Post-process after the full video has been processed:
        corrected = majority_vote_team_reassignment(history)

        # Re-map events (only players in corrected dict are changed):
        for event in events:
            if event["player_id"] in corrected:
                event["team"] = corrected[event["player_id"]]

    Args:
        player_team_history (Dict[int, List[int]]): Mapping from canonical
            player ID to a list of per-frame raw team-ID predictions (each
            element must be 0 or 1).  Players with an empty history are
            omitted from the result.
        majority_threshold (float): Minimum fraction of votes that the
            dominant team must receive for the player to be included in the
            corrected result.  Must be in ``(0.5, 1.0]``.  Defaults to 0.8
            (80 %).  Players whose dominant team receives fewer than this
            fraction of votes are omitted — callers keep their original
            assignment.

    Returns:
        Dict[int, int]: Mapping from canonical player ID to corrected team
        ID (0 or 1).  Only players whose dominant team reaches
        *majority_threshold* are included.
    """
    majority_threshold = float(majority_threshold)
    result: Dict[int, int] = {}
    for canonical_id, history in player_team_history.items():
        valid = [t for t in history if t in (0, 1)]
        if not valid:
            continue
        total = len(valid)
        counts = Counter(valid)
        best_team, best_count = counts.most_common(1)[0]
        if best_count / total >= majority_threshold:
            result[canonical_id] = best_team
    return result


def resolve_players_team_with_cache(
    team_classifier: "TeamClassifier",
    crops: List[np.ndarray],
    tracker_ids,
    team_cache: Dict[int, int],
    confidence_threshold: float = 0.65,
) -> np.ndarray:
    """
    Predict team IDs for player crops, using a per-tracker-ID cache to skip
    redundant inference on players already classified with high confidence.

    A prediction is only cached when its confidence score (from
    :meth:`TeamClassifier.predict_with_confidence`) is at or above
    *confidence_threshold*.  Low-confidence predictions are still *returned*
    but not written to the cache, so the player is re-evaluated on the next
    frame.  This prevents a single uncertain crop (e.g. motion blur, heavy
    occlusion) from permanently locking in the wrong team label and ensures
    the downstream :class:`~sports.common.stats.TeamVoteBuffer` always
    receives the best available prediction.

    Args:
        team_classifier (TeamClassifier): Fitted team classifier.
        crops (List[np.ndarray]): Cropped player images (BGR, arbitrary size).
        tracker_ids: Array-like of tracker IDs for each crop (may contain NaN
            values from ByteTrack when a detection has no assigned ID).
        team_cache (Dict[int, int]): Mutable mapping from tracker ID to team
            ID.  High-confidence predictions are written here; low-confidence
            predictions are not.
        confidence_threshold (float): Minimum confidence in ``[0.5, 1.0]``
            required to cache a prediction.  Defaults to 0.65.

    Returns:
        np.ndarray: Integer array of shape ``(N,)`` containing team IDs
        (0 or 1) for each crop.
    """
    if len(crops) == 0:
        return np.array([], dtype=int)

    team_ids = np.full(len(crops), -1, dtype=int)
    to_classify_idx: List[int] = []
    to_classify_crops: List[np.ndarray] = []
    to_classify_tids: List[Optional[int]] = []

    for i in range(len(crops)):
        tid = _safe_tracker_id(tracker_ids[i]) if tracker_ids is not None else None
        if tid is not None and tid in team_cache:
            team_ids[i] = int(team_cache[tid])
        else:
            to_classify_idx.append(i)
            to_classify_crops.append(crops[i])
            to_classify_tids.append(tid)

    if to_classify_crops:
        predicted, confidences = team_classifier.predict_with_confidence(
            to_classify_crops
        )
        predicted = predicted.astype(int)
        for idx, pred_team, conf, tid in zip(
            to_classify_idx, predicted, confidences, to_classify_tids
        ):
            team_ids[idx] = int(pred_team)
            # Only cache predictions we are confident about; uncertain crops
            # will be re-classified next frame.
            if tid is not None and team_ids[idx] in (0, 1) and conf >= confidence_threshold:
                team_cache[tid] = int(team_ids[idx])

    return team_ids
