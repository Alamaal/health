"""
Unit tests for sports.common.view module.
"""

import numpy as np
import pytest

from sports.common.view import ViewTransformer


@pytest.fixture
def square_to_square():
    """Simple transform: unit square to a 2× scaled square."""
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    return ViewTransformer(source=source, target=target)


class TestViewTransformerInit:
    def test_valid_construction(self):
        source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
        vt = ViewTransformer(source=source, target=target)
        assert vt.m is not None

    def test_shape_mismatch_raises(self):
        source = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
        target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
        with pytest.raises(ValueError, match="same shape"):
            ViewTransformer(source=source, target=target)

    def test_non_2d_points_raises(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        target = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float32)
        with pytest.raises(ValueError, match="2D coordinates"):
            ViewTransformer(source=source, target=target)


class TestTransformPoints:
    def test_empty_points(self, square_to_square):
        pts = np.empty((0, 2), dtype=np.float32)
        result = square_to_square.transform_points(pts)
        assert result.shape == (0, 2)

    def test_corner_transform(self, square_to_square):
        pts = np.array([[0.5, 0.5]], dtype=np.float32)
        result = square_to_square.transform_points(pts)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [1.0, 1.0], atol=1e-4)

    def test_non_2d_points_raises(self, square_to_square):
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        with pytest.raises(ValueError, match="2D coordinates"):
            square_to_square.transform_points(pts)


class TestTransformImage:
    def test_transform_image_shape(self, square_to_square):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = square_to_square.transform_image(img, resolution_wh=(200, 200))
        assert result.shape == (200, 200, 3)

    def test_invalid_image_dims_raises(self, square_to_square):
        img = np.zeros((100, 100, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="grayscale or color"):
            square_to_square.transform_image(img, resolution_wh=(100, 100))
