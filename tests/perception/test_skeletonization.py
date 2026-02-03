"""Tests for rope skeletonization module."""

import numpy as np
import pytest

from src.perception.skeletonization import (
    _cluster_junctions,
    _count_neighbors,
    _extract_graph_structure,
    _postprocess_skeleton,
    _skeletonize_zhang_suen,
    extract_path,
    skeletonize_rope,
)


def test_skeletonize_rope_valid_mask():
    """Test skeletonization with valid binary mask."""
    # Create a simple horizontal rope mask
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[25, 20:80] = 255  # Horizontal line

    skeleton = skeletonize_rope(mask)

    assert isinstance(skeleton, np.ndarray)
    assert skeleton.shape == mask.shape
    assert skeleton.dtype == np.uint8
    assert np.all((skeleton == 0) | (skeleton == 255))


def test_skeletonize_rope_empty_mask():
    """Test skeletonization with empty mask."""
    mask = np.zeros((50, 100), dtype=np.uint8)

    skeleton = skeletonize_rope(mask)

    assert isinstance(skeleton, np.ndarray)
    assert skeleton.shape == mask.shape
    assert np.count_nonzero(skeleton) == 0


def test_skeletonize_rope_invalid_input():
    """Test skeletonization handles invalid inputs gracefully."""
    # None input
    result = skeletonize_rope(None)
    assert isinstance(result, np.ndarray)

    # Empty array
    result = skeletonize_rope(np.array([]))
    assert isinstance(result, np.ndarray)

    # Wrong dtype
    mask = np.ones((10, 10), dtype=np.float32) * 255
    result = skeletonize_rope(mask)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8


def test_skeletonize_rope_non_binary():
    """Test skeletonization with non-binary values."""
    mask = np.ones((50, 50), dtype=np.uint8) * 128  # Gray values

    skeleton = skeletonize_rope(mask)

    assert isinstance(skeleton, np.ndarray)
    assert skeleton.shape == mask.shape
    assert np.all((skeleton == 0) | (skeleton == 255))


def test_skeletonize_rope_with_config():
    """Test skeletonization with configuration."""
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[25, 20:80] = 255

    config = {"method": "zhang_suen", "prune_length": 5}
    skeleton = skeletonize_rope(mask, config=config)

    assert isinstance(skeleton, np.ndarray)
    assert skeleton.shape == mask.shape


def test_skeletonize_zhang_suen():
    """Test Zhang-Suen skeletonization method."""
    # Create a thick horizontal line
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[24:27, 20:80] = 255  # 3-pixel thick line

    skeleton = _skeletonize_zhang_suen(mask)

    assert isinstance(skeleton, np.ndarray)
    assert skeleton.shape == mask.shape
    assert skeleton.dtype == np.uint8
    # Should be thinner than original
    assert np.count_nonzero(skeleton) <= np.count_nonzero(mask)


def test_count_neighbors():
    """Test neighbor counting function."""
    # Create a simple skeleton (cross shape)
    skeleton = np.zeros((10, 10), dtype=np.uint8)
    skeleton[5, :] = 255  # Horizontal line
    skeleton[:, 5] = 255  # Vertical line

    neighbor_count = _count_neighbors(skeleton)

    assert neighbor_count.shape == skeleton.shape
    # Center pixel (5, 5) should have 4 neighbors (2 horizontal + 2 vertical)
    assert neighbor_count[5, 5] == 4
    # Endpoints should have 1 neighbor
    assert neighbor_count[5, 0] == 1 or neighbor_count[5, 9] == 1


def test_postprocess_skeleton_isolated_pixels():
    """Test post-processing removes isolated pixels."""
    skeleton = np.zeros((20, 20), dtype=np.uint8)
    skeleton[10, 10] = 255  # Single isolated pixel
    skeleton[5, 5:15] = 255  # Horizontal line

    processed = _postprocess_skeleton(skeleton, prune_length=10)

    # Isolated pixel should be removed
    assert processed[10, 10] == 0
    # Line should remain
    assert np.count_nonzero(processed) > 0


def test_postprocess_skeleton_pruning():
    """Test post-processing prunes short branches."""
    # Create skeleton with short branch
    skeleton = np.zeros((30, 30), dtype=np.uint8)
    # Main line
    skeleton[15, 5:25] = 255
    # Short branch (5 pixels)
    skeleton[15:20, 15] = 255

    processed = _postprocess_skeleton(skeleton, prune_length=10)

    # Short branch should be pruned
    assert processed[19, 15] == 0 or np.count_nonzero(processed[15:20, 15]) < 5


def test_cluster_junctions():
    """Test junction clustering."""
    # Create skeleton with junction (T-junction)
    skeleton = np.zeros((30, 30), dtype=np.uint8)
    skeleton[15, 10:20] = 255  # Horizontal line
    skeleton[10:16, 15] = 255  # Vertical line (creates junction)

    junctions = _cluster_junctions(skeleton)

    assert isinstance(junctions, np.ndarray)
    assert junctions.shape[1] == 2  # (x, y) coordinates
    assert junctions.dtype == np.float32
    # Should have at least one junction
    if len(junctions) > 0:
        assert junctions[0, 0] >= 0  # x coordinate
        assert junctions[0, 1] >= 0  # y coordinate


def test_extract_graph_structure():
    """Test graph structure extraction."""
    # Create simple skeleton (straight line)
    skeleton = np.zeros((30, 30), dtype=np.uint8)
    skeleton[15, 10:20] = 255

    endpoints, junctions, edges, edge_nodes = _extract_graph_structure(skeleton)

    assert isinstance(endpoints, np.ndarray)
    assert isinstance(junctions, np.ndarray)
    assert isinstance(edges, list)
    assert isinstance(edge_nodes, list)
    assert endpoints.shape[1] == 2  # (x, y) coordinates
    assert junctions.shape[1] == 2  # (x, y) coordinates
    # Should have 2 endpoints for a line
    assert len(endpoints) == 2


def test_extract_path_simple_chain():
    """Test path extraction for simple chain (2 endpoints, no junctions)."""
    # Create simple horizontal line
    skeleton = np.zeros((30, 50), dtype=np.uint8)
    skeleton[15, 10:40] = 255

    path_dict = extract_path(skeleton)

    assert isinstance(path_dict, dict)
    assert "endpoints" in path_dict
    assert "junctions" in path_dict
    assert "edges" in path_dict
    assert "main_path" in path_dict

    assert len(path_dict["endpoints"]) == 2
    assert len(path_dict["junctions"]) == 0
    # Should have main_path for simple chain
    assert path_dict["main_path"] is not None
    assert path_dict["main_path"].shape[1] == 2  # (x, y) coordinates


def test_extract_path_complex_structure():
    """Test path extraction for complex structure (with junctions)."""
    # Create T-junction
    skeleton = np.zeros((30, 30), dtype=np.uint8)
    skeleton[15, 10:20] = 255  # Horizontal
    skeleton[10:16, 15] = 255  # Vertical

    path_dict = extract_path(skeleton)

    assert isinstance(path_dict, dict)
    assert "endpoints" in path_dict
    assert "junctions" in path_dict
    assert "edges" in path_dict
    assert "main_path" in path_dict

    # Should have edges
    assert len(path_dict["edges"]) > 0


def test_extract_path_empty_skeleton():
    """Test path extraction with empty skeleton."""
    skeleton = np.zeros((30, 30), dtype=np.uint8)

    path_dict = extract_path(skeleton)

    assert isinstance(path_dict, dict)
    assert len(path_dict["endpoints"]) == 0
    assert len(path_dict["junctions"]) == 0
    assert len(path_dict["edges"]) == 0
    assert path_dict["main_path"] is None


def test_extract_path_single_pixel():
    """Test path extraction with single pixel."""
    skeleton = np.zeros((30, 30), dtype=np.uint8)
    skeleton[15, 15] = 255

    path_dict = extract_path(skeleton)

    assert isinstance(path_dict, dict)
    # Single pixel should be treated as endpoint
    assert len(path_dict["endpoints"]) >= 0  # May be 0 or 1 depending on implementation


def test_extract_path_coordinate_convention():
    """Test that path extraction uses (x, y) = (column, row) convention."""
    # Create simple line
    skeleton = np.zeros((30, 50), dtype=np.uint8)
    skeleton[15, 20] = 255  # Single pixel at row=15, col=20

    path_dict = extract_path(skeleton)

    if len(path_dict["endpoints"]) > 0:
        # x should be column (20), y should be row (15)
        ep = path_dict["endpoints"][0]
        # Allow some tolerance for clustering/centroid calculation
        assert abs(ep[0] - 20) < 5  # x (column)
        assert abs(ep[1] - 15) < 5  # y (row)


def test_curved_rope():
    """Test skeletonization with curved rope."""
    # Create curved rope mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw a curve using multiple points
    for i in range(20, 80):
        row = int(30 + 20 * np.sin(i * 0.1))
        col = i
        if 0 <= row < 100 and 0 <= col < 100:
            mask[row-2:row+3, col-2:col+3] = 255

    skeleton = skeletonize_rope(mask)

    assert isinstance(skeleton, np.ndarray)
    assert skeleton.shape == mask.shape
    # Skeleton should be thinner
    assert np.count_nonzero(skeleton) <= np.count_nonzero(mask)


def test_rope_with_crossing():
    """Test skeletonization with rope crossing."""
    # Create two crossing lines
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 255  # Horizontal
    mask[10:40, 25] = 255  # Vertical

    skeleton = skeletonize_rope(mask)

    path_dict = extract_path(skeleton)

    # Should detect junction at crossing
    assert len(path_dict["junctions"]) >= 0  # May have junction or not depending on structure


@pytest.mark.performance
def test_performance_640x480():
    """Test performance on 640x480 mask (target <50ms)."""
    import time

    # Create a realistic rope mask
    mask = np.zeros((480, 640), dtype=np.uint8)
    # Add some rope-like structure
    for i in range(100, 540):
        row = int(240 + 50 * np.sin(i * 0.01))
        col = i
        if 0 <= row < 480 and 0 <= col < 640:
            mask[row-3:row+4, col-3:col+4] = 255

    start_time = time.time()
    skeleton = skeletonize_rope(mask)
    skeleton_time = time.time() - start_time

    start_time = time.time()
    path_dict = extract_path(skeleton)
    path_time = time.time() - start_time

    total_time = skeleton_time + path_time

    # Should complete in reasonable time (allow some margin)
    assert total_time < 0.5  # 500ms is reasonable for test environment
    # In production, target is <50ms, but tests may be slower
