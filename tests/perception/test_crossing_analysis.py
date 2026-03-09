"""Tests for crossing over/under analysis module."""

import cv2
import numpy as np
import pytest

from src.perception.crossing_analysis import (
    CrossingInfo,
    _circular_mean_hue,
    _hsv_color_distance,
    _pair_branches_by_color,
    analyze_crossing_over_under,
)
from src.perception.keypoint_detection import Keypoint


# ── helper tests ─────────────────────────────────────────────────────


class TestCircularMeanHue:
    """Tests for circular mean hue computation."""

    def test_simple_average(self):
        hues = np.array([30.0, 40.0, 50.0])
        result = _circular_mean_hue(hues)
        assert pytest.approx(result, abs=1.0) == 40.0

    def test_wrap_around_zero(self):
        """Hues near 0 and 180 should average near 0/180, not 90."""
        hues = np.array([5.0, 175.0])
        result = _circular_mean_hue(hues)
        # Should wrap around to near 0 or 180
        assert result < 10.0 or result > 170.0

    def test_wrap_around_values(self):
        """170 and 10 should average near 0/180."""
        hues = np.array([170.0, 10.0])
        result = _circular_mean_hue(hues)
        assert result < 15.0 or result > 165.0

    def test_empty_array(self):
        assert _circular_mean_hue(np.array([])) == 0.0

    def test_single_value(self):
        result = _circular_mean_hue(np.array([90.0]))
        assert pytest.approx(result, abs=0.1) == 90.0


class TestHsvColorDistance:
    """Tests for HSV color distance."""

    def test_identical_colors(self):
        c = (60.0, 200.0, 200.0)
        assert _hsv_color_distance(c, c) == 0.0

    def test_symmetric(self):
        c1 = (30.0, 100.0, 150.0)
        c2 = (90.0, 200.0, 100.0)
        assert _hsv_color_distance(c1, c2) == pytest.approx(
            _hsv_color_distance(c2, c1)
        )

    def test_hue_wrap(self):
        """Distance between hue 5 and 175 should be small (10/180)."""
        c1 = (5.0, 200.0, 200.0)
        c2 = (175.0, 200.0, 200.0)
        dist = _hsv_color_distance(c1, c2, hue_weight=1.0,
                                   saturation_weight=0.0,
                                   value_weight=0.0)
        assert dist < 0.06  # 10/180 ≈ 0.056

    def test_different_colors(self):
        c1 = (0.0, 255.0, 255.0)  # Red
        c2 = (60.0, 255.0, 255.0)  # Green-yellow
        dist = _hsv_color_distance(c1, c2)
        assert dist > 0.1


class TestPairBranchesByColor:
    """Tests for branch pairing by color similarity."""

    def test_obvious_pairs(self):
        """Red/Red + Blue/Blue pairing is obvious."""
        colors = [
            (0.0, 255.0, 255.0),    # red  idx 0
            (120.0, 255.0, 255.0),   # blue idx 1
            (2.0, 250.0, 250.0),     # red  idx 2
            (118.0, 250.0, 250.0),   # blue idx 3
        ]
        pair_a, pair_b = _pair_branches_by_color(colors)
        # One pair should be the reds (0,2), the other blues (1,3)
        pairs_set = {frozenset(pair_a), frozenset(pair_b)}
        assert frozenset({0, 2}) in pairs_set
        assert frozenset({1, 3}) in pairs_set

    def test_green_pink_pairs(self):
        """Green/Green + Pink/Pink."""
        colors = [
            (60.0, 200.0, 200.0),   # green idx 0
            (150.0, 200.0, 200.0),  # pink  idx 1
            (62.0, 195.0, 195.0),   # green idx 2
            (148.0, 195.0, 195.0),  # pink  idx 3
        ]
        pair_a, pair_b = _pair_branches_by_color(colors)
        pairs_set = {frozenset(pair_a), frozenset(pair_b)}
        assert frozenset({0, 2}) in pairs_set
        assert frozenset({1, 3}) in pairs_set


# ── integration test ─────────────────────────────────────────────────


def _make_crossing_image(
    size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Keypoint]]:
    """Create a synthetic image with two colored lines crossing.

    Returns BGR image, skeleton, rope mask, and crossing keypoints.
    The top (over) line is drawn second so its color is visible at
    the center.
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    skeleton = np.zeros((size, size), dtype=np.uint8)

    cx, cy = size // 2, size // 2

    # Under strand: green, drawn first (diagonal \)
    green_bgr = (0, 255, 0)
    cv2.line(image, (10, 10), (90, 90), green_bgr, 3)
    cv2.line(mask, (10, 10), (90, 90), 255, 3)
    cv2.line(skeleton, (10, 10), (90, 90), 255, 1)

    # Over strand: blue, drawn second (diagonal /)
    blue_bgr = (255, 0, 0)
    cv2.line(image, (90, 10), (10, 90), blue_bgr, 3)
    cv2.line(mask, (90, 10), (10, 90), 255, 3)
    cv2.line(skeleton, (90, 10), (10, 90), 255, 1)

    crossing = Keypoint(
        position=(float(cx), float(cy)),
        keypoint_type="crossing",
        confidence=0.9,
    )

    return image, skeleton, mask, [crossing]


class TestAnalyzeCrossingSynthetic:
    """Integration test with a synthetic crossing image."""

    def test_returns_crossing_info(self):
        image, skeleton, mask, crossings = _make_crossing_image()
        results = analyze_crossing_over_under(
            image, skeleton, mask, crossings
        )
        # May or may not detect depending on skeleton junction quality,
        # but should not crash
        assert isinstance(results, list)
        for ci in results:
            assert isinstance(ci, CrossingInfo)
            assert 0.0 <= ci.confidence <= 1.0

    def test_none_inputs(self):
        """Gracefully handle None inputs."""
        assert analyze_crossing_over_under(None, None, None, []) == []

    def test_empty_crossings(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert analyze_crossing_over_under(
            image, skeleton, mask, []
        ) == []


class TestFewerThan4Branches:
    """Crossing with fewer than 4 branches should be skipped."""

    def test_straight_line_no_crossing(self):
        """A straight line has no 4-branch junction."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        skeleton = np.zeros((50, 50), dtype=np.uint8)

        # Single horizontal line
        cv2.line(image, (5, 25), (45, 25), (0, 200, 200), 2)
        cv2.line(mask, (5, 25), (45, 25), 255, 2)
        cv2.line(skeleton, (5, 25), (45, 25), 255, 1)

        crossing = Keypoint(
            position=(25.0, 25.0),
            keypoint_type="crossing",
            confidence=0.5,
        )

        results = analyze_crossing_over_under(
            image, skeleton, mask, [crossing]
        )
        # Should return empty — no 4-branch junction at that location
        assert results == []
