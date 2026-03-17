"""
Unit tests for the PlayerReIdentifier class in sports.common.team.

Covers:
- Default parameter values (max_frames_lost=900, embedding_similarity_threshold=0.85)
- Gallery-based spatial re-identification (position fallback)
- Embedding-based re-identification (cosine similarity matching)
- Embedding preferred over spatial match
- has_tracker_id public method
- _cosine_similarity static helper
- _update_embedding static helper
- Gallery ageing and pruning
- team_id filtering in gallery search
- Running-average gallery embedding update for known tracks
"""

import numpy as np
import pytest

from sports.common.team import PlayerReIdentifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reid(**kwargs) -> PlayerReIdentifier:
    defaults = dict(
        max_frames_lost=300,
        position_tolerance_px=120.0,
        embedding_similarity_threshold=0.85,
    )
    defaults.update(kwargs)
    return PlayerReIdentifier(**defaults)


def _unit_vec(dim: int, index: int) -> np.ndarray:
    """Return a unit vector with 1.0 at *index* and 0 elsewhere."""
    v = np.zeros(dim, dtype=float)
    v[index] = 1.0
    return v


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_max_frames_lost(self):
        """Default max_frames_lost should be 900."""
        reid = PlayerReIdentifier()
        assert reid._max_frames_lost == 900

    def test_default_embedding_similarity_threshold(self):
        """Default embedding_similarity_threshold should be 0.85."""
        reid = PlayerReIdentifier()
        assert abs(reid._embedding_similarity_threshold - 0.85) < 1e-9

    def test_default_position_tolerance(self):
        """Default position_tolerance_px should be 120."""
        reid = PlayerReIdentifier()
        assert abs(reid._position_tolerance - 120.0) < 1e-9


# ---------------------------------------------------------------------------
# has_tracker_id
# ---------------------------------------------------------------------------

class TestHasTrackerId:
    def test_unknown_tracker_id_returns_false(self):
        reid = _make_reid()
        assert reid.has_tracker_id(99) is False

    def test_known_tracker_id_returns_true(self):
        reid = _make_reid()
        reid.get_stable_id(1, np.array([100.0, 200.0]))
        assert reid.has_tracker_id(1) is True

    def test_different_id_still_unknown(self):
        reid = _make_reid()
        reid.get_stable_id(1, np.array([100.0, 200.0]))
        assert reid.has_tracker_id(2) is False


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(PlayerReIdentifier._cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(PlayerReIdentifier._cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0])
        assert abs(PlayerReIdentifier._cosine_similarity(v, -v) + 1.0) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = np.array([1.0, 0.0])
        b = np.zeros(2)
        assert PlayerReIdentifier._cosine_similarity(a, b) == 0.0

    def test_flattens_multidim_input(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        assert abs(PlayerReIdentifier._cosine_similarity(a, b) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _update_embedding
# ---------------------------------------------------------------------------

class TestUpdateEmbedding:
    def test_none_existing_returns_copy_of_new(self):
        new = np.array([1.0, 2.0])
        result = PlayerReIdentifier._update_embedding(None, new)
        assert result is not None
        np.testing.assert_array_almost_equal(result, new)
        assert result is not new  # should be a copy

    def test_none_new_returns_existing(self):
        existing = np.array([3.0, 4.0])
        result = PlayerReIdentifier._update_embedding(existing, None)
        assert result is existing

    def test_both_none_returns_none(self):
        assert PlayerReIdentifier._update_embedding(None, None) is None

    def test_running_average(self):
        existing = np.array([0.0, 0.0])
        new = np.array([2.0, 4.0])
        result = PlayerReIdentifier._update_embedding(existing, new)
        # EMA: 0.85 * existing + 0.15 * new
        expected = 0.85 * existing + 0.15 * new
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# Spatial re-identification (no embeddings)
# ---------------------------------------------------------------------------

class TestSpatialReId:
    def test_same_tracker_id_returns_same_canonical(self):
        reid = _make_reid()
        cid1 = reid.get_stable_id(1, np.array([100.0, 200.0]))
        cid2 = reid.get_stable_id(1, np.array([105.0, 202.0]))
        assert cid1 == cid2

    def test_different_tracker_ids_same_position_after_loss(self):
        """A re-appearing player at nearly the same position gets the same canonical ID."""
        reid = _make_reid(position_tolerance_px=50.0)
        cid1 = reid.get_stable_id(1, np.array([100.0, 200.0]))
        # Frame 1 active → age stays 0; frame 2 player absent → age increments to 1.
        reid.end_frame()
        reid.end_frame()  # Player 1 is now lost (age=1).

        # New tracker ID at same position → should map to same canonical ID.
        cid2 = reid.get_stable_id(2, np.array([100.0, 200.0]))
        assert cid2 == cid1

    def test_different_tracker_id_far_away_gets_new_canonical(self):
        """A new tracker ID that is far from all gallery entries gets a new ID."""
        reid = _make_reid(position_tolerance_px=50.0)
        cid1 = reid.get_stable_id(1, np.array([100.0, 200.0]))
        reid.end_frame()
        reid.end_frame()  # Player 1 is lost (age=1).

        # New player far away → new canonical ID.
        cid2 = reid.get_stable_id(2, np.array([600.0, 400.0]))
        assert cid2 != cid1

    def test_team_filter_prevents_cross_team_match(self):
        """Gallery entries from a different team should not be reused."""
        reid = _make_reid(position_tolerance_px=500.0)
        cid1 = reid.get_stable_id(1, np.array([100.0, 200.0]), team_id=0)
        reid.end_frame()
        reid.end_frame()  # Player 1 is lost (age=1).

        # Same position but different team → should NOT match.
        cid2 = reid.get_stable_id(2, np.array([100.0, 200.0]), team_id=1)
        assert cid2 != cid1

    def test_team_filter_allows_same_team_match(self):
        """Gallery entries from the same team can be reused."""
        reid = _make_reid(position_tolerance_px=500.0)
        cid1 = reid.get_stable_id(1, np.array([100.0, 200.0]), team_id=0)
        reid.end_frame()
        reid.end_frame()  # Player 1 is lost (age=1).
        cid2 = reid.get_stable_id(2, np.array([100.0, 200.0]), team_id=0)
        assert cid2 == cid1

    def test_active_track_not_matched(self):
        """Active (age=0) gallery entries should not be matched against."""
        reid = _make_reid(position_tolerance_px=500.0)
        cid1 = reid.get_stable_id(1, np.array([100.0, 200.0]))
        # Tracker 1 is still active (age=0 after get_stable_id).
        # A brand-new tracker ID at the same position should NOT steal tracker 1's canonical.
        cid2 = reid.get_stable_id(2, np.array([100.0, 200.0]))
        assert cid2 != cid1


# ---------------------------------------------------------------------------
# Embedding-based re-identification
# ---------------------------------------------------------------------------

class TestEmbeddingReId:
    def test_high_similarity_embedding_matches_lost_track(self):
        """A re-appearing player with a very similar embedding reuses the canonical ID."""
        reid = _make_reid(embedding_similarity_threshold=0.9, position_tolerance_px=10.0)
        emb = np.array([1.0, 0.0, 0.0])
        cid1 = reid.get_stable_id(
            1, np.array([100.0, 100.0]), team_id=0, embedding=emb
        )
        reid.end_frame()
        reid.end_frame()  # Player 1 is lost (age=1).

        # New tracker at completely different position but very similar embedding.
        emb2 = np.array([0.999, 0.001, 0.0])  # cos_sim ≈ 0.9999 > 0.9
        emb2 /= np.linalg.norm(emb2)
        cid2 = reid.get_stable_id(
            2, np.array([900.0, 800.0]), team_id=0, embedding=emb2
        )
        assert cid2 == cid1

    def test_low_similarity_embedding_does_not_match(self):
        """A player with a dissimilar embedding does NOT reuse the canonical ID."""
        # Use position_tolerance_px=5.0 so the new tracker (placed 50px away) is outside
        # spatial tolerance, ensuring only embedding similarity is tested.
        reid = _make_reid(embedding_similarity_threshold=0.9, position_tolerance_px=5.0)
        emb_a = _unit_vec(4, 0)  # [1, 0, 0, 0]
        cid1 = reid.get_stable_id(1, np.array([100.0, 100.0]), team_id=0, embedding=emb_a)
        reid.end_frame()
        reid.end_frame()  # Player 1 is lost (age=1).

        # Different position (outside tolerance) AND orthogonal embedding (cos_sim=0 < 0.9)
        emb_b = _unit_vec(4, 1)  # [0, 1, 0, 0] — orthogonal, cos_sim = 0 < 0.9
        cid2 = reid.get_stable_id(2, np.array([200.0, 200.0]), team_id=0, embedding=emb_b)
        assert cid2 != cid1

    def test_embedding_preferred_over_spatial(self):
        """When embedding similarity is high, embedding match is chosen over spatial."""
        reid = _make_reid(
            embedding_similarity_threshold=0.9,
            position_tolerance_px=500.0,
        )
        emb_player5 = _unit_vec(4, 0)  # canonical embedding for player 5
        emb_player7 = _unit_vec(4, 1)  # canonical embedding for player 7

        cid5 = reid.get_stable_id(1, np.array([100.0, 100.0]), team_id=0, embedding=emb_player5)
        cid7 = reid.get_stable_id(2, np.array([200.0, 100.0]), team_id=0, embedding=emb_player7)
        reid.end_frame()
        reid.end_frame()  # Both lost (age=1).

        # New tracker near player5's last position but with player7's embedding.
        # Embedding should win → gets player7's ID.
        cid_new = reid.get_stable_id(
            3, np.array([105.0, 100.0]), team_id=0, embedding=emb_player7
        )
        assert cid_new == cid7

    def test_no_gallery_embedding_falls_back_to_spatial(self):
        """When gallery has no embedding, spatial matching is used as fallback."""
        reid = _make_reid(
            embedding_similarity_threshold=0.9, position_tolerance_px=200.0
        )
        # Gallery entry without embedding.
        cid1 = reid.get_stable_id(1, np.array([100.0, 100.0]), team_id=0, embedding=None)
        reid.end_frame()
        reid.end_frame()  # Player 1 is lost (age=1).

        # Query with embedding but gallery has none → spatial fallback used.
        query_emb = _unit_vec(4, 0)
        cid2 = reid.get_stable_id(
            2, np.array([110.0, 100.0]), team_id=0, embedding=query_emb
        )
        assert cid2 == cid1

    def test_embedding_team_filter_respected(self):
        """High-similarity embedding from a different team should not match."""
        reid = _make_reid(embedding_similarity_threshold=0.5, position_tolerance_px=10.0)
        emb = _unit_vec(4, 0)
        cid0 = reid.get_stable_id(1, np.array([100.0, 100.0]), team_id=0, embedding=emb)
        reid.end_frame()
        reid.end_frame()  # Player 1 is lost (age=1).

        # Same embedding but team_id=1 → should not match team 0's gallery entry.
        cid1 = reid.get_stable_id(2, np.array([900.0, 900.0]), team_id=1, embedding=emb)
        assert cid1 != cid0


# ---------------------------------------------------------------------------
# Gallery ageing and pruning
# ---------------------------------------------------------------------------

class TestGalleryAgeing:
    def test_gallery_pruned_after_max_frames_lost(self):
        """A gallery entry is removed after max_frames_lost frames without the player."""
        reid = _make_reid(max_frames_lost=3, position_tolerance_px=500.0)
        cid1 = reid.get_stable_id(1, np.array([100.0, 100.0]))
        # end_frame #1: player was active this frame → age=0
        reid.end_frame()
        # end_frame #2, #3, #4: player absent → age=1, 2, 3 → pruned on #4
        reid.end_frame()
        reid.end_frame()
        reid.end_frame()

        # After pruning, a new player at the same position gets a fresh ID.
        cid2 = reid.get_stable_id(2, np.array([100.0, 100.0]))
        assert cid2 != cid1

    def test_gallery_not_pruned_before_max_frames_lost(self):
        """A gallery entry is still available before max_frames_lost is reached."""
        reid = _make_reid(max_frames_lost=5, position_tolerance_px=500.0)
        cid1 = reid.get_stable_id(1, np.array([100.0, 100.0]))
        # end_frame #1: age=0 (was active); end_frame #2: age=1 (absent)
        reid.end_frame()
        reid.end_frame()

        cid2 = reid.get_stable_id(2, np.array([100.0, 100.0]))
        assert cid2 == cid1

    def test_gallery_age_reset_when_player_returns(self):
        """Age is reset to 0 when the same canonical ID becomes active again."""
        reid = _make_reid(max_frames_lost=3, position_tolerance_px=500.0)
        cid1 = reid.get_stable_id(1, np.array([100.0, 100.0]))
        reid.end_frame()  # active → age=0
        reid.end_frame()  # absent → age=1
        reid.end_frame()  # absent → age=2

        # Player re-appears before being pruned.
        reid.get_stable_id(1, np.array([100.0, 100.0]))
        reid.end_frame()  # active again → age=0
        reid.end_frame()  # absent → age=1
        reid.end_frame()  # absent → age=2 — NOT removed yet

        assert cid1 in reid._gallery


# ---------------------------------------------------------------------------
# Gallery embedding update for known tracks (running average)
# ---------------------------------------------------------------------------

class TestGalleryEmbeddingUpdate:
    def test_embedding_updated_on_revisit(self):
        """Embedding in gallery is updated with a running average on revisit."""
        reid = _make_reid()
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])

        cid = reid.get_stable_id(1, np.array([0.0, 0.0]), embedding=emb1)
        reid.end_frame()
        # Revisit same tracker with new embedding.
        reid.get_stable_id(1, np.array([0.0, 0.0]), embedding=emb2)
        reid.end_frame()

        stored = reid._gallery[cid]["embedding"]
        # EMA: 0.85 * emb1 + 0.15 * emb2
        expected = 0.85 * emb1 + 0.15 * emb2
        np.testing.assert_array_almost_equal(stored, expected)

    def test_embedding_preserved_when_update_is_none(self):
        """Gallery embedding is preserved when no new embedding is provided on revisit."""
        reid = _make_reid()
        emb = np.array([1.0, 2.0])
        cid = reid.get_stable_id(1, np.array([0.0, 0.0]), embedding=emb)
        reid.end_frame()
        reid.get_stable_id(1, np.array([0.0, 0.0]), embedding=None)
        reid.end_frame()

        stored = reid._gallery[cid]["embedding"]
        np.testing.assert_array_almost_equal(stored, emb)
