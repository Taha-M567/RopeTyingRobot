"""Rope skeletonization for path extraction.

This module extracts the centerline/skeleton of the rope and converts it
to a graph-aware structured representation.
"""

import logging
from typing import Dict, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

# Try to import scikit-image for fallback skeletonization
# scikit-image is an optional dependency
try:
    from skimage.morphology import skeletonize as sk_skeletonize  # type: ignore[import-untyped]
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    sk_skeletonize = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class PathDict(TypedDict):
    """Structured path dictionary for skeleton graph outputs."""

    endpoints: np.ndarray
    junctions: np.ndarray
    edges: List[np.ndarray]
    main_path: Optional[np.ndarray]


BINARY_THRESHOLD = 127
BINARY_MAX = 255
DEFAULT_PRUNE_LENGTH = 20
DEFAULT_CLOSE_LOOP_MAX_GAP = 0
MAX_PRUNE_ITERATIONS = 100
DEFAULT_PRE_CLOSE_KERNEL_SIZE = 0
DEFAULT_PRE_DILATE_KERNEL_SIZE = 0
DEFAULT_PRE_DILATE_ITERATIONS = 1


def _skeletonize_zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Skeletonize mask using Zhang-Suen algorithm.

    Uses cv2.ximgproc.thinning() if available, otherwise falls back to
    scikit-image skeletonize().

    Args:
        mask: Binary mask (uint8, 0/255)

    Returns:
        Binary skeleton mask (uint8, 0/255)
    """
    # Ensure mask is binary 0/255 for OpenCV
    mask_binary = ((mask > BINARY_THRESHOLD) * BINARY_MAX).astype(np.uint8)

    # Try OpenCV contrib first
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        try:
            skeleton = cv2.ximgproc.thinning(
                mask_binary,
                thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
            )
            return skeleton
        except Exception as e:
            logger.warning(f"OpenCV ximgproc.thinning failed: {e}, trying fallback")
    else:
        logger.debug("OpenCV ximgproc.thinning not available, using scikit-image fallback")

    # Fallback to scikit-image
    if HAS_SKIMAGE:
        try:
            # Convert to boolean (True/False) for scikit-image
            mask_bool_sk = (mask_binary > BINARY_THRESHOLD).astype(bool)
            skeleton_bool = sk_skeletonize(mask_bool_sk)
            # Convert back to 0/255
            return (skeleton_bool.astype(np.uint8) * BINARY_MAX).astype(np.uint8)
        except Exception as e:
            logger.error(f"scikit-image skeletonize failed: {e}")
            return np.zeros_like(mask)
    else:
        logger.error("Neither OpenCV ximgproc.thinning nor scikit-image available. Install opencv-contrib-python or scikit-image.")
        return np.zeros_like(mask)


def _count_neighbors(skeleton: np.ndarray) -> np.ndarray:
    """Count 8-connected neighbors for each pixel in skeleton.

    Args:
        skeleton: Binary skeleton mask (uint8, 0/255)

    Returns:
        Array of neighbor counts (same shape as skeleton)
    """
    # Convert to binary (0/1)
    skeleton_binary = (skeleton > BINARY_THRESHOLD).astype(np.uint8)

    # 3x3 kernel for 8-connectivity (excluding center)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    # Convolve to count neighbors
    neighbor_count = cv2.filter2D(
        skeleton_binary,
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )

    return neighbor_count


def _postprocess_skeleton(
    skeleton: np.ndarray,
    prune_length: int = DEFAULT_PRUNE_LENGTH,
    close_loop_max_gap: int = DEFAULT_CLOSE_LOOP_MAX_GAP,
) -> np.ndarray:
    """Post-process skeleton: remove isolated pixels and prune short branches.

    Args:
        skeleton: Binary skeleton mask (uint8, 0/255)
        prune_length: Minimum branch length to keep (pixels)
        close_loop_max_gap: If > 0, close small gaps between exactly two
            endpoints within this pixel distance.

    Returns:
        Cleaned skeleton mask (uint8, 0/255)
    """
    if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        return skeleton

    # Count neighbors
    neighbor_count = _count_neighbors(skeleton)
    skeleton_binary = (skeleton > BINARY_THRESHOLD).astype(np.uint8)

    # 1. Remove isolated pixels (0 neighbors)
    isolated_mask = (neighbor_count == 0) & (skeleton_binary > 0)
    skeleton_binary[isolated_mask] = 0

    # 2. Prune short branches
    # Identify endpoints (1 neighbor) and junctions (>= 3 neighbors)
    endpoints_mask = (neighbor_count == 1) & (skeleton_binary > 0)
    junctions_mask = (neighbor_count >= 3) & (skeleton_binary > 0)

    # Get endpoint coordinates
    endpoint_coords = np.column_stack(np.where(endpoints_mask))
    junction_coords_set = set(tuple(coord) for coord in np.column_stack(np.where(junctions_mask)))

    if len(endpoint_coords) == 0:
        return (skeleton_binary * BINARY_MAX).astype(np.uint8)

    # Iteratively prune short branches
    changed = True
    max_iterations = MAX_PRUNE_ITERATIONS  # Safety limit
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        # Recompute neighbor count after changes
        neighbor_count = _count_neighbors(
            (skeleton_binary * BINARY_MAX).astype(np.uint8)
        )
        endpoints_mask = (neighbor_count == 1) & (skeleton_binary > 0)
        endpoint_coords = np.column_stack(np.where(endpoints_mask))

        for endpoint in endpoint_coords:
            row, col = endpoint
            # Check if this endpoint is adjacent to a junction
            is_adjacent_to_junction = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbor_row, neighbor_col = row + dr, col + dc
                    if (neighbor_row, neighbor_col) in junction_coords_set:
                        is_adjacent_to_junction = True
                        break
                if is_adjacent_to_junction:
                    break

            # Do NOT prune if adjacent to junction
            if is_adjacent_to_junction:
                continue

            # Trace path from this endpoint
            path_length = 0
            current = (int(row), int(col))
            visited = set()
            path = []

            while True:
                if current in visited:
                    break
                visited.add(current)
                path.append(current)
                path_length += 1

                # Find next neighbor
                next_pixel = None
                neighbor_count_current = 0

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = current[0] + dr, current[1] + dc

                        # Check bounds
                        if nr < 0 or nr >= skeleton_binary.shape[0]:
                            continue
                        if nc < 0 or nc >= skeleton_binary.shape[1]:
                            continue

                        if skeleton_binary[nr, nc] > 0:
                            neighbor_count_current += 1
                            if (nr, nc) not in visited:
                                next_pixel = (nr, nc)

                # Stop if we hit a junction (>= 3 neighbors) or another endpoint
                if neighbor_count_current >= 3:
                    # Hit a junction, don't prune
                    break
                if neighbor_count_current == 1 and path_length > 1:
                    # Hit another endpoint
                    break

                if next_pixel is None:
                    break

                current = next_pixel

            # Prune if path is short and doesn't touch junction
            if path_length < prune_length and path_length > 1:
                # Remove this branch
                for pr, pc in path:
                    skeleton_binary[pr, pc] = 0
                changed = True

    if close_loop_max_gap > 0:
        # Recompute endpoints after pruning and close tiny gaps for loops.
        neighbor_count = _count_neighbors(
            (skeleton_binary * BINARY_MAX).astype(np.uint8)
        )
        endpoints_mask = (neighbor_count == 1) & (skeleton_binary > 0)
        endpoint_coords = np.column_stack(np.where(endpoints_mask))

        if len(endpoint_coords) == 2:
            (r1, c1), (r2, c2) = endpoint_coords
            dr = int(r1) - int(r2)
            dc = int(c1) - int(c2)
            if dr * dr + dc * dc <= close_loop_max_gap * close_loop_max_gap:
                cv2.line(skeleton_binary, (int(c1), int(r1)), (int(c2), int(r2)), 1, 1)

    return (skeleton_binary * BINARY_MAX).astype(np.uint8)


def _cluster_junctions(skeleton: np.ndarray) -> np.ndarray:
    """Cluster nearby junction pixels into single junction points.

    Args:
        skeleton: Binary skeleton mask (uint8, 0/255)

    Returns:
        Array of junction centers as (x, y) coordinates, shape (J, 2), dtype=float32
    """
    if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2)

    neighbor_count = _count_neighbors(skeleton)
    skeleton_binary = (skeleton > BINARY_THRESHOLD).astype(np.uint8)

    # Find junction pixels (>= 3 neighbors)
    junctions_mask = (neighbor_count >= 3) & (skeleton_binary > 0)

    if np.count_nonzero(junctions_mask) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2)

    # Convert to uint8 for connectedComponents
    junctions_uint8 = (junctions_mask.astype(np.uint8) * BINARY_MAX)

    # Find connected components of junction pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        junctions_uint8,
        connectivity=8,
    )

    # Extract centroids (centroids[0] is background, skip it)
    junction_centers = []
    for i in range(1, num_labels):
        # centroids are in (x, y) = (col, row) format from OpenCV
        x, y = centroids[i]
        junction_centers.append([x, y])

    if len(junction_centers) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2)

    return np.array(junction_centers, dtype=np.float32)


def _extract_graph_structure(
    skeleton: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Tuple[int, int]]]:
    """Extract graph structure from skeleton (endpoints, junctions, edges).

    Args:
        skeleton: Binary skeleton mask (uint8, 0/255)

    Returns:
        Tuple of (endpoints, junctions, edges, edge_nodes) where:
        - endpoints: (E, 2) array of (x, y) coordinates
        - junctions: (J, 2) array of (x, y) coordinates
        - edges: List of (Ni, 2) arrays, each representing an edge path
        - edge_nodes: List of (node_i, node_j) indices for each edge
    """
    if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        empty_endpoints = np.array([], dtype=np.float32).reshape(0, 2)
        empty_junctions = np.array([], dtype=np.float32).reshape(0, 2)
        return empty_endpoints, empty_junctions, [], []

    neighbor_count = _count_neighbors(skeleton)
    skeleton_binary = (skeleton > BINARY_THRESHOLD).astype(np.uint8)

    # Find endpoints (1 neighbor)
    endpoints_mask = (neighbor_count == 1) & (skeleton_binary > 0)
    endpoint_coords = np.column_stack(np.where(endpoints_mask))
    # Convert to (x, y) = (col, row)
    if len(endpoint_coords) == 0:
        endpoints = np.array([], dtype=np.float32).reshape(0, 2)
    else:
        endpoints = np.array(
            [[col, row] for row, col in endpoint_coords],
            dtype=np.float32,
        )

    # Get clustered junctions
    junctions = _cluster_junctions(skeleton)

    # Combine all nodes (endpoints + junctions)
    all_nodes = []
    node_types = []  # 'endpoint' or 'junction'

    for ep in endpoints:
        all_nodes.append(tuple(ep))
        node_types.append('endpoint')

    for jn in junctions:
        all_nodes.append(tuple(jn))
        node_types.append('junction')

    # Convert node coordinates to (row, col) for skeleton indexing
    node_coords_rc = []
    for node in all_nodes:
        x, y = node
        # x is column, y is row in our convention
        node_coords_rc.append((int(round(y)), int(round(x))))

    # Extract edges by tracing paths between nodes
    edges = []
    edge_nodes: List[Tuple[int, int]] = []
    visited_edges = set()

    # For each pair of nodes, check if there's a path
    for i, node1_rc in enumerate(node_coords_rc):
        for j, node2_rc in enumerate(node_coords_rc):
            if i >= j:
                continue

            edge_key = (min(i, j), max(i, j))
            if edge_key in visited_edges:
                continue

            # Try to find path between nodes
            path = _trace_path_between_nodes(
                skeleton_binary,
                node1_rc,
                node2_rc,
                node_coords_rc,
            )

            if path is not None and len(path) > 1:
                # Convert path to (x, y) coordinates
                path_xy = np.array([[col, row] for row, col in path], dtype=np.float32)
                edges.append(path_xy)
                edge_nodes.append((i, j))
                visited_edges.add(edge_key)

    # Handle loops (0 endpoints) - extract as closed path
    if len(endpoints) == 0 and len(junctions) == 0 and np.count_nonzero(skeleton_binary) > 0:
        # Find any skeleton pixel and trace the loop
        skeleton_coords = np.column_stack(np.where(skeleton_binary > 0))
        if len(skeleton_coords) > 0:
            start_rc = tuple(skeleton_coords[0])
            loop_path = _trace_loop(skeleton_binary, start_rc)
            if loop_path is not None and len(loop_path) > 1:
                path_xy = np.array([[col, row] for row, col in loop_path], dtype=np.float32)
                edges.append(path_xy)

    return endpoints, junctions, edges, edge_nodes


def _build_main_path(
    endpoints: np.ndarray,
    junctions: np.ndarray,
    edges: List[np.ndarray],
    edge_nodes: List[Tuple[int, int]],
) -> Optional[np.ndarray]:
    """Build a main path by finding the longest path through the graph."""
    if not edges or not edge_nodes:
        return None

    all_nodes = np.vstack([endpoints, junctions]) if len(junctions) > 0 else endpoints
    if all_nodes.size == 0:
        return None

    # Build adjacency with edge lengths as weights.
    adjacency: Dict[int, List[Tuple[int, int, float]]] = {}
    for edge_index, (i, j) in enumerate(edge_nodes):
        length = float(len(edges[edge_index]))
        adjacency.setdefault(i, []).append((j, edge_index, length))
        adjacency.setdefault(j, []).append((i, edge_index, length))

    candidate_nodes = list(range(len(all_nodes)))
    if len(endpoints) >= 2:
        candidate_nodes = list(range(len(endpoints)))

    # Dijkstra from each candidate node to find the farthest candidate.
    best_distance = -1.0
    best_pair = None
    best_prev: Dict[int, Tuple[int, int]] = {}

    for start in candidate_nodes:
        # Standard Dijkstra on sparse adjacency.
        distances = {start: 0.0}
        prev: Dict[int, Tuple[int, int]] = {}
        visited = set()
        queue = [(0.0, start)]

        while queue:
            dist, node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            for neighbor, edge_index, weight in adjacency.get(node, []):
                new_dist = dist + weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    prev[neighbor] = (node, edge_index)
                    queue.append((new_dist, neighbor))
            queue.sort(key=lambda x: x[0])

        for end in candidate_nodes:
            if end == start or end not in distances:
                continue
            if distances[end] > best_distance:
                best_distance = distances[end]
                best_pair = (start, end)
                best_prev = prev

    if best_pair is None:
        return None

    # Reconstruct node path from best_prev.
    start, end = best_pair
    node_path = [end]
    edge_path = []
    current = end
    while current != start:
        if current not in best_prev:
            break
        prev_node, edge_index = best_prev[current]
        node_path.append(prev_node)
        edge_path.append(edge_index)
        current = prev_node
    node_path.reverse()
    edge_path.reverse()

    if not edge_path:
        return None

    # Stitch edge pixel paths in order, aligning orientation to node order.
    stitched: List[np.ndarray] = []
    for idx, edge_index in enumerate(edge_path):
        edge = edges[edge_index]
        node_idx = node_path[idx]
        next_node_idx = node_path[idx + 1]
        node_xy = all_nodes[node_idx]
        next_node_xy = all_nodes[next_node_idx]

        start_dist = np.linalg.norm(edge[0] - node_xy)
        end_dist = np.linalg.norm(edge[-1] - node_xy)
        edge_oriented = edge if start_dist <= end_dist else edge[::-1]

        # Ensure the end aligns with the next node; flip if needed.
        if np.linalg.norm(edge_oriented[-1] - next_node_xy) > np.linalg.norm(
            edge_oriented[0] - next_node_xy
        ):
            edge_oriented = edge_oriented[::-1]

        if stitched:
            stitched.append(edge_oriented[1:])  # drop duplicate junction point
        else:
            stitched.append(edge_oriented)

    return np.concatenate(stitched, axis=0)


def _trace_path_between_nodes(
    skeleton_binary: np.ndarray,
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    all_nodes_rc: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """Trace path between two nodes using BFS.

    Args:
        skeleton_binary: Binary skeleton (0/1)
        start_rc: Start node (row, col)
        end_rc: End node (row, col)
        all_nodes_rc: List of all node coordinates to avoid

    Returns:
        List of (row, col) coordinates forming the path, or None if no path
    """
    from collections import deque

    queue = deque([(start_rc, [start_rc])])
    visited = {start_rc}

    while queue:
        current, path = queue.popleft()

        # Check if we reached the end node
        if current == end_rc:
            return path

        # Explore neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = current[0] + dr, current[1] + dc

                # Check bounds
                if nr < 0 or nr >= skeleton_binary.shape[0]:
                    continue
                if nc < 0 or nc >= skeleton_binary.shape[1]:
                    continue

                # Check if it's a skeleton pixel
                if skeleton_binary[nr, nc] == 0:
                    continue

                next_rc = (nr, nc)

                # Skip if it's another node (except the target)
                if next_rc in all_nodes_rc and next_rc != end_rc:
                    continue

                if next_rc not in visited:
                    visited.add(next_rc)
                    queue.append((next_rc, path + [next_rc]))

    return None


def _trace_loop(
    skeleton_binary: np.ndarray,
    start_rc: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """Trace a closed loop starting from a point.

    Args:
        skeleton_binary: Binary skeleton (0/1)
        start_rc: Start point (row, col)

    Returns:
        List of (row, col) coordinates forming the loop, or None
    """
    path = [start_rc]
    current = start_rc
    visited = {start_rc}
    max_length = np.count_nonzero(skeleton_binary) * 2  # Safety limit

    for _ in range(max_length):
        # Find next neighbor
        next_pixel = None
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = current[0] + dr, current[1] + dc

                # Check bounds
                if nr < 0 or nr >= skeleton_binary.shape[0]:
                    continue
                if nc < 0 or nc >= skeleton_binary.shape[1]:
                    continue

                if skeleton_binary[nr, nc] > 0:
                    next_rc = (nr, nc)
                    # Prefer unvisited, but allow revisiting start to close loop
                    if next_rc not in visited or (len(path) > 2 and next_rc == start_rc):
                        next_pixel = next_rc
                        break

            if next_pixel is not None:
                break

        if next_pixel is None:
            break

        # Check if we closed the loop
        if next_pixel == start_rc and len(path) > 2:
            return path + [start_rc]

        current = next_pixel
        path.append(current)
        visited.add(current)

    return None


def _trace_loop_from_skeleton(
    skeleton: np.ndarray,
    max_start_trials: int = 50,
) -> Optional[List[Tuple[int, int]]]:
    """Trace a loop path from a skeleton by trying degree-2 start pixels.

    Args:
        skeleton: Binary skeleton mask (uint8, 0/255)
        max_start_trials: Maximum number of start points to try

    Returns:
        Loop path as list of (row, col) coordinates, or None
    """
    if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        return None

    skeleton_binary = (skeleton > BINARY_THRESHOLD).astype(np.uint8)
    neighbor_count = _count_neighbors((skeleton_binary * BINARY_MAX).astype(np.uint8))

    # Prefer degree-2 pixels for loop tracing; fallback to any skeleton pixel.
    degree_two = np.column_stack(
        np.where((neighbor_count == 2) & (skeleton_binary > 0))
    )
    candidates = degree_two
    if len(candidates) == 0:
        candidates = np.column_stack(np.where(skeleton_binary > 0))

    for idx in range(min(max_start_trials, len(candidates))):
        row, col = candidates[idx]
        loop = _trace_loop(skeleton_binary, (int(row), int(col)))
        if loop is not None and len(loop) > 2:
            return loop

    return None


def skeletonize_rope(
    mask: np.ndarray,
    config: Optional[dict] = None,
) -> np.ndarray:
    """Extract skeleton from rope mask.

    Args:
        mask: Binary mask of the rope (uint8, 0/255)
        config: Optional configuration dictionary with:
            - method: "zhang_suen" (default)
            - prune_length: int, pixels to prune (default: 10)
            - close_loop_max_gap: int, max pixel gap to close between
              exactly two endpoints (default: 0 = disabled)
            - pre_close_kernel_size: int, closing kernel size before thinning
            - pre_dilate_kernel_size: int, dilation kernel size before thinning
            - pre_dilate_iterations: int, dilation iterations

    Returns:
        Binary mask of the skeleton (uint8, 0/255)
    """
    # Input validation
    if mask is None or not isinstance(mask, np.ndarray):
        logger.warning("Invalid mask input: not a numpy array")
        return np.zeros((0, 0), dtype=np.uint8) if mask is None else np.zeros_like(mask)

    if mask.size == 0:
        logger.warning("Empty mask input")
        return mask.copy()

    if mask.dtype != np.uint8:
        logger.warning(f"Mask dtype is {mask.dtype}, expected uint8. Converting.")
        mask = mask.astype(np.uint8)

    # Check if binary (0 or 255)
    unique_vals = np.unique(mask)
    if not all(v in [0, BINARY_MAX] for v in unique_vals):
        logger.warning("Mask contains non-binary values, thresholding at 127")
        mask = ((mask > BINARY_THRESHOLD) * BINARY_MAX).astype(np.uint8)

    # Config handling
    if config is None:
        config = {}

    method = config.get("method", "zhang_suen")
    prune_length = config.get("prune_length", DEFAULT_PRUNE_LENGTH)
    close_loop_max_gap = int(
        config.get("close_loop_max_gap", DEFAULT_CLOSE_LOOP_MAX_GAP)
    )
    pre_close_kernel_size = int(
        config.get("pre_close_kernel_size", DEFAULT_PRE_CLOSE_KERNEL_SIZE)
    )
    pre_dilate_kernel_size = int(
        config.get("pre_dilate_kernel_size", DEFAULT_PRE_DILATE_KERNEL_SIZE)
    )
    pre_dilate_iterations = int(
        config.get("pre_dilate_iterations", DEFAULT_PRE_DILATE_ITERATIONS)
    )

    if pre_close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (pre_close_kernel_size, pre_close_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    if pre_dilate_kernel_size > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (pre_dilate_kernel_size, pre_dilate_kernel_size),
        )
        mask = cv2.dilate(mask, dilate_kernel, iterations=pre_dilate_iterations)

    # Skeletonization
    try:
        if method == "zhang_suen":
            skeleton = _skeletonize_zhang_suen(mask)
        else:
            logger.warning(f"Unknown method {method}, using zhang_suen")
            skeleton = _skeletonize_zhang_suen(mask)
    except Exception as e:
        logger.error(f"Skeletonization failed: {e}")
        return np.zeros_like(mask)

    # Post-processing
    try:
        skeleton = _postprocess_skeleton(
            skeleton,
            prune_length=prune_length,
            close_loop_max_gap=close_loop_max_gap,
        )
    except Exception as e:
        logger.warning(f"Post-processing failed: {e}, returning unprocessed skeleton")

    return skeleton


def extract_path(skeleton: np.ndarray) -> PathDict:
    """Extract ordered path from skeleton as graph structure.

    Args:
        skeleton: Binary skeleton mask (uint8, 0/255)

    Returns:
        Dictionary with keys:
        - "endpoints": (E, 2) array of (x, y) coordinates
        - "junctions": (J, 2) array of (x, y) coordinates
        - "edges": List of (Ni, 2) arrays, each an edge path
        - "main_path": (N, 2) array or None if complex structure
    """
    # Input validation
    if skeleton is None or not isinstance(skeleton, np.ndarray):
        logger.warning("Invalid skeleton input")
        return {
            "endpoints": np.array([], dtype=np.float32).reshape(0, 2),
            "junctions": np.array([], dtype=np.float32).reshape(0, 2),
            "edges": [],
            "main_path": None,
        }

    if skeleton.size == 0 or np.count_nonzero(skeleton) == 0:
        return {
            "endpoints": np.array([], dtype=np.float32).reshape(0, 2),
            "junctions": np.array([], dtype=np.float32).reshape(0, 2),
            "edges": [],
            "main_path": None,
        }

    try:
        # Extract graph structure
        endpoints, junctions, edges, edge_nodes = _extract_graph_structure(skeleton)

        # Determine a main path even with junctions by finding the longest graph path.
        main_path = _build_main_path(endpoints, junctions, edges, edge_nodes)

        # If the main path is missing or too short, try to recover a full loop
        # directly from the skeleton.
        skeleton_count = int(np.count_nonzero(skeleton))
        if skeleton_count > 0:
            main_path_len = 0 if main_path is None else len(main_path)
            if len(endpoints) == 0 or main_path_len < int(0.7 * skeleton_count):
                loop_path = _trace_loop_from_skeleton(skeleton)
                if loop_path is not None:
                    loop_xy = np.array(
                        [[col, row] for row, col in loop_path], dtype=np.float32
                    )
                    if main_path is None or len(loop_xy) > len(main_path):
                        main_path = loop_xy
                        if not edges:
                            edges = [loop_xy]

        # Handle single pixel case
        if len(endpoints) == 1 and len(junctions) == 0 and len(edges) == 0:
            # Single pixel skeleton
            skeleton_coords = np.column_stack(
                np.where((skeleton > BINARY_THRESHOLD).astype(np.uint8) > 0)
            )
            if len(skeleton_coords) > 0:
                row, col = skeleton_coords[0]
                endpoints = np.array([[col, row]], dtype=np.float32)

        return {
            "endpoints": endpoints,
            "junctions": junctions,
            "edges": edges,
            "main_path": main_path,
        }

    except Exception as e:
        logger.error(f"Path extraction failed: {e}")
        return {
            "endpoints": np.array([], dtype=np.float32).reshape(0, 2),
            "junctions": np.array([], dtype=np.float32).reshape(0, 2),
            "edges": [],
            "main_path": None,
        }
