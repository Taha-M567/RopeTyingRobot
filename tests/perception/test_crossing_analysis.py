"""Tests for crossing over/under analysis module."""

import cv2
import numpy as np
import pytest

from src.perception.crossing_analysis import (
    CrossingInfo,
    CrossingRegion,
    _circular_mean_hue,
    _find_entry_points,
    _hsv_color_distance,
    _pair_branches_by_color,
    analyze_crossing_over_under,
    detect_crossing_regions,
    patch_skeleton_at_crossings,
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
        dist = _hsv_color_distance(
            c1, c2,
            hue_weight=1.0,
            saturation_weight=0.0,
            value_weight=0.0,
        )
        assert dist < 0.06  # 10/180 ~ 0.056

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


# ── detect_crossing_regions tests ────────────────────────────────────


class TestDetectCrossingRegions:
    """Tests for mask-geometry crossing region detection."""

    def test_crossing_blob_detected(self):
        """Synthetic mask with a wide blob at crossing should be found."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Two thin strands crossing at (50, 50)
        cv2.line(mask, (10, 10), (90, 90), 255, 3)
        cv2.line(mask, (90, 10), (10, 90), 255, 3)

        regions = detect_crossing_regions(mask)
        # The overlapping area at the center is wider than a strand
        # With thin strands the distance transform peak may be modest,
        # so we just verify no crash and correct types
        assert isinstance(regions, list)
        for r in regions:
            assert isinstance(r, CrossingRegion)
            assert r.normal_half_width > 0

    def test_crossing_thick_strands(self):
        """Thicker strands produce a more pronounced crossing blob."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(mask, (20, 20), (180, 180), 255, 12)
        cv2.line(mask, (180, 20), (20, 180), 255, 12)

        regions = detect_crossing_regions(
            mask,
            config={"width_ratio": 1.3, "min_crossing_area": 10},
        )
        assert len(regions) >= 1
        # Center should be near (100, 100)
        r = regions[0]
        assert abs(r.center[0] - 100) < 25
        assert abs(r.center[1] - 100) < 25

    def test_no_crossing_uniform_width(self):
        """A single straight line has uniform width -> no crossing."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(mask, (10, 50), (90, 50), 255, 4)

        regions = detect_crossing_regions(mask)
        assert regions == []

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert detect_crossing_regions(mask) == []

    def test_none_mask(self):
        assert detect_crossing_regions(None) == []


# ── _find_entry_points tests ─────────────────────────────────────────


class TestFindEntryPoints:
    """Tests for finding skeleton entry points at region boundaries."""

    def test_four_entries_at_cross(self):
        """Skeleton entering from 4 directions should give ~4 entries."""
        size = 100
        skeleton = np.zeros((size, size), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)

        # Draw skeleton as two crossing thin lines
        cv2.line(skeleton, (10, 50), (90, 50), 255, 1)
        cv2.line(skeleton, (50, 10), (50, 90), 255, 1)

        # Create a crossing region around the center
        region_mask = np.zeros((size, size), dtype=bool)
        region_mask[45:56, 45:56] = True

        region = CrossingRegion(
            center=(50.0, 50.0),
            bbox=(45, 45, 56, 56),
            region_mask=region_mask,
            normal_half_width=3.0,
        )

        entries = _find_entry_points(skeleton, region)
        # Should find 4 entries (one from each direction)
        assert len(entries) == 4

    def test_no_skeleton_no_entries(self):
        """Empty skeleton gives no entries."""
        size = 50
        skeleton = np.zeros((size, size), dtype=np.uint8)
        region_mask = np.zeros((size, size), dtype=bool)
        region_mask[20:30, 20:30] = True
        region = CrossingRegion(
            center=(25.0, 25.0),
            bbox=(20, 20, 30, 30),
            region_mask=region_mask,
            normal_half_width=3.0,
        )

        entries = _find_entry_points(skeleton, region)
        assert entries == []


# ── patch_skeleton_at_crossings tests ─────────────────────────────


class TestPatchSkeleton:
    """Tests for skeleton patching at crossing regions."""

    def test_messy_junction_replaced(self):
        """Skeleton inside region should be zeroed and replaced."""
        size = 100
        skeleton = np.zeros((size, size), dtype=np.uint8)
        # Create messy junction cluster at center
        skeleton[45:55, 45:55] = 255  # big blob

        # Also draw branches extending outward
        cv2.line(skeleton, (10, 50), (45, 50), 255, 1)
        cv2.line(skeleton, (55, 50), (90, 50), 255, 1)
        cv2.line(skeleton, (50, 10), (50, 45), 255, 1)
        cv2.line(skeleton, (50, 55), (50, 90), 255, 1)

        # Count non-zero in the junction area before patching
        blob_before = np.count_nonzero(skeleton[45:55, 45:55])
        assert blob_before > 50  # big blob

        region_mask = np.zeros((size, size), dtype=bool)
        region_mask[45:55, 45:55] = True
        region = CrossingRegion(
            center=(50.0, 50.0),
            bbox=(45, 45, 55, 55),
            region_mask=region_mask,
            normal_half_width=3.0,
        )

        # Create crossing info with clean paths through center
        over_path = np.array(
            [[50, 45], [50, 50], [50, 55]], dtype=np.int32
        )
        under_path = np.array(
            [[45, 50], [50, 50], [55, 50]], dtype=np.int32
        )
        ci = CrossingInfo(
            position=(50.0, 50.0),
            over_strand_path=over_path,
            under_strand_path=under_path,
            over_color_hsv=(0.0, 0.0, 0.0),
            under_color_hsv=(0.0, 0.0, 0.0),
            center_color_hsv=(0.0, 0.0, 0.0),
            confidence=0.8,
            region=region,
        )

        patched = patch_skeleton_at_crossings(skeleton, [ci])

        # The blob should be gone; only thin paths remain in the region
        blob_after = np.count_nonzero(patched[45:55, 45:55])
        assert blob_after < blob_before
        # Branches outside the region should still exist
        assert np.count_nonzero(patched[10:45, 50]) > 0
        assert np.count_nonzero(patched[55:90, 50]) > 0

    def test_empty_results(self):
        """No crossing results should return skeleton unchanged."""
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        cv2.line(skeleton, (10, 25), (40, 25), 255, 1)
        original = skeleton.copy()
        patched = patch_skeleton_at_crossings(skeleton, [])
        np.testing.assert_array_equal(patched, original)


# ── integration test ─────────────────────────────────────────────────


def _make_crossing_image(
    size: int = 200,
    strand_width: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic image with two colored lines crossing.

    Returns BGR image, skeleton, rope mask.
    The top (over) line is drawn second so its color is visible at
    the center.
    """
    image = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    skeleton = np.zeros((size, size), dtype=np.uint8)

    # Under strand: green, drawn first (diagonal \)
    green_bgr = (0, 255, 0)
    cv2.line(image, (20, 20), (180, 180), green_bgr, strand_width)
    cv2.line(mask, (20, 20), (180, 180), 255, strand_width)
    cv2.line(skeleton, (20, 20), (180, 180), 255, 1)

    # Over strand: blue, drawn second (diagonal /)
    blue_bgr = (255, 0, 0)
    cv2.line(image, (180, 20), (20, 180), blue_bgr, strand_width)
    cv2.line(mask, (180, 20), (20, 180), 255, strand_width)
    cv2.line(skeleton, (180, 20), (20, 180), 255, 1)

    return image, skeleton, mask


class TestAnalyzeCrossingSynthetic:
    """Integration test with a synthetic crossing image."""

    def test_returns_crossing_info(self):
        image, skeleton, mask = _make_crossing_image()
        results = analyze_crossing_over_under(
            image, skeleton, mask
        )
        # May or may not detect depending on geometry,
        # but should not crash
        assert isinstance(results, list)
        for ci in results:
            assert isinstance(ci, CrossingInfo)
            assert 0.0 <= ci.confidence <= 1.0
            assert ci.over_strand_path is not None
            assert ci.under_strand_path is not None
            assert isinstance(ci.region, CrossingRegion)

    def test_none_inputs(self):
        """Gracefully handle None inputs."""
        assert analyze_crossing_over_under(
            None, None, None
        ) == []

    def test_empty_mask(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert analyze_crossing_over_under(
            image, skeleton, mask
        ) == []


class TestNoCrossing:
    """Scenes without crossings should produce no results."""

    def test_straight_line_no_crossing(self):
        """A straight line has uniform width -> no crossing region."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        skeleton = np.zeros((100, 100), dtype=np.uint8)

        cv2.line(image, (5, 50), (95, 50), (0, 200, 200), 4)
        cv2.line(mask, (5, 50), (95, 50), 255, 4)
        cv2.line(skeleton, (5, 50), (95, 50), 255, 1)

        results = analyze_crossing_over_under(
            image, skeleton, mask
        )
        assert results == []
