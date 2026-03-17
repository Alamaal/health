"""
Unit tests for team classification accuracy improvements.

Covers:
- TeamClassifier.predict_with_confidence():
  - Returns (team_ids, confidences) pair
  - Confidence formula: d_other / (d_winner + d_other) ∈ [0.5, 1.0]
  - Edge case: empty crops returns empty arrays
  - Confidence is 1.0 when a point sits exactly on a cluster centroid
  - Confidence is 0.5 when a point is equidistant from both centroids

- resolve_players_team_with_cache() confidence gating:
  - High-confidence predictions ARE cached
  - Low-confidence predictions are NOT cached (re-evaluated each frame)
  - Already-cached predictions bypass the classifier
  - Returned team IDs are correct regardless of whether they are cached
  - Empty crops returns empty array

The tests for TeamClassifier use a lightweight stub (no real model loading) that
replaces only the learned components (UMAP reducer + KMeans cluster_model) with
simple numpy mocks so the mathematical logic can be exercised without GPU/torch.
"""

import ast
import os

import numpy as np
import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..")
_TEAM_PY = os.path.join(_REPO, "sports", "common", "team.py")
_MAIN_PY = os.path.join(_REPO, "examples", "soccer", "main.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_team_module():
    """Import sports.common.team; skip if heavy deps are unavailable."""
    try:
        import sports.common.team as team_mod
        return team_mod
    except ImportError:
        pytest.skip("sports.common.team unavailable (missing heavy deps)")


def _import_resolve_fn():
    """
    Import resolve_players_team_with_cache from sports.common.team.

    This function lives in sports.common.team (pure numpy, no pipeline deps)
    so it can be imported without ultralytics / cv2 / supervision.
    """
    try:
        from sports.common.team import resolve_players_team_with_cache
        return resolve_players_team_with_cache
    except ImportError:
        return None


class _MockReducer:
    """Identity reducer: passes projections through unchanged."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class _MockKMeans:
    """
    KMeans stub with two centroids that can be set explicitly.

    `transform` returns Euclidean distances to each centroid, exactly like
    `sklearn.cluster.KMeans.transform`.
    """

    def __init__(self, centroids: np.ndarray) -> None:
        # shape: (2, D)
        self.cluster_centers_ = np.asarray(centroids, dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return distances from each sample in X to each centroid. Shape (N, 2)."""
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2
        )
        return dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(self.transform(X), axis=1)


def _make_stub_classifier(centroids: np.ndarray):
    """
    Build a TeamClassifier stub with a MockKMeans and identity reducer.

    The SigLIP model is not loaded; `extract_features` is patched to return
    the crops as-is (they are expected to already be numpy arrays of the right
    shape when passed in tests).
    """
    team_mod = _import_team_module()

    class _StubClassifier(team_mod.TeamClassifier):
        def __init__(self, centroids):
            # Skip model loading entirely
            self.device = "cpu"
            self.batch_size = 32
            self.reducer = _MockReducer()
            self.cluster_model = _MockKMeans(centroids)

        def extract_features(self, crops, verbose=False):  # noqa: D102
            # In tests crops are already float arrays of shape (D,) — stack them.
            if len(crops) == 0:
                return np.array([])
            return np.stack([np.asarray(c, dtype=float) for c in crops])

    return _StubClassifier(centroids)


# ---------------------------------------------------------------------------
# Source-level checks (AST)
# ---------------------------------------------------------------------------

class TestPredictWithConfidenceSourceLevel:
    """Verify predict_with_confidence exists and returns a tuple."""

    def test_method_exists_in_source(self):
        """predict_with_confidence must be defined in TeamClassifier."""
        with open(_TEAM_PY) as f:
            source = f.read()
        tree = ast.parse(source)
        method_names = [
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        ]
        assert "predict_with_confidence" in method_names, (
            "predict_with_confidence() not found in team.py"
        )

    def test_confidence_formula_present(self):
        """The d_other / (d_winner + d_other) formula must appear in the source."""
        with open(_TEAM_PY) as f:
            source = f.read()
        assert "d_winner" in source and "d_other" in source, (
            "Confidence formula variables d_winner / d_other not found in team.py"
        )

    def test_confidence_threshold_in_resolve_cache(self):
        """resolve_players_team_with_cache must accept a confidence_threshold param."""
        with open(_TEAM_PY) as f:
            source = f.read()
        assert "confidence_threshold" in source, (
            "confidence_threshold parameter not found in resolve_players_team_with_cache"
        )

    def test_predict_with_confidence_called_in_resolve_cache(self):
        """resolve_players_team_with_cache must call predict_with_confidence."""
        with open(_TEAM_PY) as f:
            source = f.read()
        tree = ast.parse(source)
        resolve_fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "resolve_players_team_with_cache"
        )
        lines = source.splitlines()
        fn_text = "\n".join(lines[resolve_fn.lineno - 1: resolve_fn.end_lineno])
        assert "predict_with_confidence" in fn_text, (
            "resolve_players_team_with_cache must call predict_with_confidence"
        )

    def test_confidence_gates_caching_in_source(self):
        """The caching write must be guarded by a confidence comparison."""
        with open(_TEAM_PY) as f:
            source = f.read()
        tree = ast.parse(source)
        resolve_fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "resolve_players_team_with_cache"
        )
        lines = source.splitlines()
        fn_text = "\n".join(lines[resolve_fn.lineno - 1: resolve_fn.end_lineno])
        # The caching line (team_cache[tid] = ...) must be inside an if-block
        # that checks confidence against the threshold.
        assert "conf" in fn_text and "confidence_threshold" in fn_text, (
            "Confidence gating not found in resolve_players_team_with_cache"
        )

    def test_main_imports_resolve_from_team(self):
        """main.py must import resolve_players_team_with_cache from sports.common.team."""
        with open(_MAIN_PY) as f:
            source = f.read()
        assert "resolve_players_team_with_cache" in source, (
            "main.py does not reference resolve_players_team_with_cache"
        )


# ---------------------------------------------------------------------------
# predict_with_confidence — mathematical correctness
# ---------------------------------------------------------------------------

class TestPredictWithConfidenceMath:
    """
    Tests use the stub classifier (no real model) to verify the confidence
    formula independently of SigLIP / UMAP.
    """

    def test_empty_crops_returns_empty_arrays(self):
        """predict_with_confidence([]) must return two empty arrays."""
        clf = _make_stub_classifier(np.array([[0.0, 0.0], [10.0, 0.0]]))
        team_ids, confidences = clf.predict_with_confidence([])
        assert isinstance(team_ids, np.ndarray) and team_ids.size == 0
        assert isinstance(confidences, np.ndarray) and confidences.size == 0

    def test_team_ids_match_predict(self):
        """team_ids from predict_with_confidence must equal predict() output."""
        centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
        clf = _make_stub_classifier(centroids)
        crops = [np.array([1.0, 0.0]), np.array([9.0, 0.0]), np.array([0.0, 0.0])]
        team_ids, _ = clf.predict_with_confidence(crops)
        expected = clf.predict(crops).astype(int)
        np.testing.assert_array_equal(team_ids, expected)

    def test_confidence_at_centroid_is_one(self):
        """A point exactly on cluster centroid 0 should have confidence=1.0."""
        centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
        clf = _make_stub_classifier(centroids)
        # Crop exactly at centroid 0 (distance to centroid 0 = 0, to centroid 1 = 10)
        crops = [np.array([0.0, 0.0])]
        _, confidences = clf.predict_with_confidence(crops)
        assert abs(confidences[0] - 1.0) < 1e-6, (
            f"Expected confidence=1.0 at centroid, got {confidences[0]}"
        )

    def test_confidence_equidistant_is_half(self):
        """A point equidistant from both centroids should have confidence≈0.5."""
        centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
        clf = _make_stub_classifier(centroids)
        # Midpoint between the two centroids
        crops = [np.array([5.0, 0.0])]
        _, confidences = clf.predict_with_confidence(crops)
        assert abs(confidences[0] - 0.5) < 1e-6, (
            f"Expected confidence=0.5 at midpoint, got {confidences[0]}"
        )

    def test_confidence_in_valid_range(self):
        """Confidence values must always be in [0.5, 1.0]."""
        centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
        clf = _make_stub_classifier(centroids)
        # Sample points at various positions along the axis
        crops = [np.array([float(x), 0.0]) for x in range(0, 11)]
        _, confidences = clf.predict_with_confidence(crops)
        assert np.all(confidences >= 0.5 - 1e-9), "Confidence below 0.5 detected"
        assert np.all(confidences <= 1.0 + 1e-9), "Confidence above 1.0 detected"

    def test_confidence_higher_for_clear_membership(self):
        """A point closer to its assigned centroid must have higher confidence
        than a point near the decision boundary."""
        centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
        clf = _make_stub_classifier(centroids)
        # Clear member of cluster 0 (distance 1 vs 9)
        crops_clear = [np.array([1.0, 0.0])]
        # Borderline member of cluster 0 (distance 4 vs 6)
        crops_border = [np.array([4.0, 0.0])]
        _, conf_clear = clf.predict_with_confidence(crops_clear)
        _, conf_border = clf.predict_with_confidence(crops_border)
        assert conf_clear[0] > conf_border[0], (
            "Clear member should have higher confidence than borderline member"
        )

    def test_returns_tuple_of_two_arrays(self):
        """predict_with_confidence must return exactly a (team_ids, confidences) tuple."""
        clf = _make_stub_classifier(np.array([[0.0], [5.0]]))
        result = clf.predict_with_confidence([np.array([1.0]), np.array([4.0])])
        assert len(result) == 2, "Expected a 2-tuple"
        team_ids, confidences = result
        assert np.issubdtype(team_ids.dtype, np.integer), \
            f"team_ids dtype should be int-like, got {team_ids.dtype}"
        assert np.issubdtype(confidences.dtype, np.floating), \
            f"confidences dtype should be float, got {confidences.dtype}"
        assert team_ids.shape == confidences.shape


# ---------------------------------------------------------------------------
# _resolve_players_team_with_cache — confidence-gated caching
# ---------------------------------------------------------------------------

class TestResolvePlayersTeamWithCacheCachingBehaviour:
    """
    Tests for _resolve_players_team_with_cache() confidence-gated caching.

    The classifier is replaced by a lightweight mock (no SigLIP/torch required)
    so the caching logic can be exercised in isolation.
    """

    @pytest.fixture(scope="class")
    def resolve_fn(self):
        """Import _resolve_players_team_with_cache via stub-module injection."""
        fn = _import_resolve_fn()
        if fn is None:
            pytest.skip(
                "Cannot import _resolve_players_team_with_cache "
                "(heavy pipeline deps unavailable)"
            )
        return fn

    class _HighConfClassifier:
        """Always predicts team 0 with confidence 0.9."""

        def predict_with_confidence(self, crops):
            n = len(crops)
            return np.zeros(n, dtype=int), np.full(n, 0.9)

        def predict(self, crops):
            return np.zeros(len(crops), dtype=int)

    class _LowConfClassifier:
        """Always predicts team 1 with confidence 0.55 (below default 0.65)."""

        def predict_with_confidence(self, crops):
            n = len(crops)
            return np.ones(n, dtype=int), np.full(n, 0.55)

        def predict(self, crops):
            return np.ones(len(crops), dtype=int)

    def test_high_confidence_prediction_is_cached(self, resolve_fn):
        """A prediction with confidence ≥ threshold must be stored in team_cache."""
        cache = {}
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)]
        tracker_ids = np.array([42])
        resolve_fn(self._HighConfClassifier(), crops, tracker_ids, cache)
        assert 42 in cache, "High-confidence prediction should be cached"
        assert cache[42] == 0

    def test_low_confidence_prediction_is_not_cached(self, resolve_fn):
        """A prediction with confidence < threshold must NOT be stored in team_cache."""
        cache = {}
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)]
        tracker_ids = np.array([42])
        resolve_fn(self._LowConfClassifier(), crops, tracker_ids, cache)
        assert 42 not in cache, "Low-confidence prediction must not be cached"

    def test_low_confidence_prediction_still_returns_team_id(self, resolve_fn):
        """Even when confidence is low the best-guess team ID is returned."""
        cache = {}
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)]
        tracker_ids = np.array([42])
        team_ids = resolve_fn(self._LowConfClassifier(), crops, tracker_ids, cache)
        assert team_ids[0] == 1, "Best-guess team ID should still be returned"

    def test_cached_player_bypasses_classifier(self, resolve_fn):
        """If tracker_id is already in team_cache the classifier is not called."""
        call_count = {"n": 0}

        class _CountingClassifier:
            def predict_with_confidence(self, crops):
                call_count["n"] += len(crops)
                return np.zeros(len(crops), dtype=int), np.ones(len(crops))

        cache = {42: 0}   # Pre-populated cache
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)]
        tracker_ids = np.array([42])
        team_ids = resolve_fn(_CountingClassifier(), crops, tracker_ids, cache)
        assert call_count["n"] == 0, "Classifier should not be called for cached player"
        assert team_ids[0] == 0

    def test_custom_confidence_threshold(self, resolve_fn):
        """A prediction exactly at the threshold must be cached."""

        class _ExactThreshClassifier:
            def predict_with_confidence(self, crops):
                n = len(crops)
                return np.zeros(n, dtype=int), np.full(n, 0.80)

        cache = {}
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)]
        tracker_ids = np.array([7])
        resolve_fn(
            _ExactThreshClassifier(), crops, tracker_ids, cache,
            confidence_threshold=0.80,
        )
        assert 7 in cache, "Prediction at exactly the threshold should be cached"

    def test_below_custom_threshold_not_cached(self, resolve_fn):
        """A prediction just below a custom threshold must not be cached."""

        class _JustBelowThreshClassifier:
            def predict_with_confidence(self, crops):
                n = len(crops)
                return np.zeros(n, dtype=int), np.full(n, 0.79)

        cache = {}
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)]
        tracker_ids = np.array([7])
        resolve_fn(
            _JustBelowThreshClassifier(), crops, tracker_ids, cache,
            confidence_threshold=0.80,
        )
        assert 7 not in cache, "Prediction below threshold must not be cached"

    def test_empty_crops_returns_empty(self, resolve_fn):
        """Empty crops list must return an empty int array."""
        result = resolve_fn(
            self._HighConfClassifier(), [], np.array([]), {}
        )
        assert isinstance(result, np.ndarray) and result.size == 0

    def test_none_tracker_id_not_cached(self, resolve_fn):
        """Players without a valid tracker ID (NaN) must not pollute the cache."""
        cache = {}
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)]
        tracker_ids = np.array([float("nan")])
        resolve_fn(self._HighConfClassifier(), crops, tracker_ids, cache)
        assert len(cache) == 0, "NaN tracker ID must not add an entry to the cache"

    def test_multiple_players_independent_caching(self, resolve_fn):
        """High-conf and low-conf players in the same frame are handled independently."""
        # Each call returns the next confidence value from the instance list.
        class _MixedClassifier:
            def __init__(self, confs):
                self._confs = list(confs)

            def predict_with_confidence(self, crops):
                n = len(crops)
                return np.zeros(n, dtype=int), np.full(n, self._confs.pop(0))

        cache = {}
        tracker_ids_list = [np.array([10]), np.array([20]), np.array([30])]
        crops = [np.zeros((10, 10, 3), dtype=np.uint8)]

        clf = _MixedClassifier([0.9, 0.55, 0.8])
        for tid_arr in tracker_ids_list:
            resolve_fn(clf, crops, tid_arr, cache)

        # tracker 10 (conf 0.9 ≥ 0.65) → cached
        assert 10 in cache
        # tracker 20 (conf 0.55 < 0.65) → NOT cached
        assert 20 not in cache
        # tracker 30 (conf 0.8 ≥ 0.65) → cached
        assert 30 in cache

