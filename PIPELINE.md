# Rope Perception Pipeline (Detailed Walkthrough)

This document explains how the rope perception pipeline works, step by step,
using simple language. It also maps each stage to the parameters in
`src/configs/perception_config.yaml` so you can see exactly how configuration
changes affect behavior.

---

# 1) Big Picture

The pipeline turns a camera image into structured rope information:

- A binary rope mask (which pixels are rope vs background)
- Endpoints and crossings
- A thin centerline (skeleton)
- A graph of the rope shape (edges and junctions)
- A final rope state object for downstream logic

The full flow runs per frame in `src/perception/video_processor.py`.

---

# 2) Data Flow at a Glance

```
Camera Frame (BGR image)
  -> Segmentation (mask)
  -> Keypoint Detection (endpoints, crossings)
  -> Skeletonization (1-pixel centerline)
  -> Graph Extraction (edges, junctions, main path)
  -> Rope State (summary)
  -> Visualization (overlays)
```

---

# 3) Module-by-Module Walkthrough

## 3.1 Live Processing: `src/perception/video_processor.py`

The class `LiveVideoProcessor`:

- Captures frames in a background thread.
- Runs the pipeline on the most recent frame.
- Returns a `ProcessingResult` containing every output.

Key config switches:

- `perception.pipeline.disable_keypoint_extraction`
  - If true, skips keypoint detection.
- `perception.pipeline.disable_skeletonization`
  - If true, skips skeletonization and graph extraction.

---

## 3.2 Segmentation: `src/perception/rope_segmentation.py`

Goal: create a **binary mask** where rope pixels are `255` and background is `0`.

### Step A: Preprocessing

Function: `_preprocess_image`

The image is blurred to reduce noise.

Config:
- `perception.segmentation.blur_kernel_size`

### Step B: Color Thresholding (default method)

Function: `_segment_by_color`

The image is converted from BGR to HSV. HSV separates brightness from color,
making it easier to isolate white or black rope.

Config:
- `perception.segmentation.color_range.hsv_lower`
- `perception.segmentation.color_range.hsv_upper`

### Step C: Edge Detection (optional method)

Function: `_segment_by_edges`

Finds edges with the Canny algorithm, then fills enclosed contours.

Config:
- `perception.segmentation.edge_detection.canny_low_threshold`
- `perception.segmentation.edge_detection.canny_high_threshold`

### Step D: Combined Method (optional method)

Function: `_segment_combined`

Runs both color and edge segmentation and merges them (logical OR).

Config:
- `perception.segmentation.method` set to `"combined"`

### Step E: Morphological Cleanup

Function: `_apply_morphology`

Morphology cleans small defects:

- Opening removes small dots (erode then dilate).
- Closing fills small gaps (dilate then erode).

Config:
- `perception.segmentation.morph_operations.opening_kernel_size`
- `perception.segmentation.morph_operations.closing_kernel_size`

### Step F: Contour Filtering

Function: `_filter_contours`

Contours are the outlines of blobs in the mask. We filter:

- Too small contours
- Contours with the wrong aspect ratio
- Small holes (optional)

Config:
- `perception.segmentation.contour_filter.min_area`
- `perception.segmentation.contour_filter.max_area`
- `perception.segmentation.contour_filter.min_aspect_ratio`
- `perception.segmentation.contour_filter.hole_fill_max_area`
- `perception.segmentation.contour_filter.hollow_fill_ratio_threshold`

### Step G: Connected Component Cleanup

Function: `_cleanup_connected_components`

Removes leftover blobs after contour filtering.

Config:
- `perception.segmentation.cleanup.min_area`
- `perception.segmentation.cleanup.keep_largest`

This is important because tiny leftover blobs produce extra skeleton branches
and many false crossings.

### Step H: Confidence Score

Function: `_calculate_confidence`

The mask is scored by:

- Coverage ratio (how much of the image is rope)
- Dominance of the largest contour

Config:
- No direct config parameters, but changes in segmentation affect confidence.

---

## 3.3 Keypoint Detection: `src/perception/keypoint_detection.py`

Goal: find rope endpoints and crossings.

### Step A: Endpoints

Endpoints can be detected using:

- Contour analysis (farthest pair of contour points)
- Skeleton endpoints (pixels with one neighbor)
- Combined (both, then merged)

Config:
- `perception.keypoint_detection.endpoint_detection.method`
  - `"contour_analysis"`, `"skeleton_endpoints"`, or `"combined"`
- `perception.keypoint_detection.endpoint_detection.min_confidence`
- `perception.keypoint_detection.endpoint_detection.merge_distance`

The combined mode is most robust when the rope is curved or partially occluded.

### Step B: Crossings

Crossings are detected on the skeleton by finding junction clusters.
A junction is only accepted if it has multiple **real branches**, not just
noise spurs.

Config:
- `perception.keypoint_detection.crossing_detection.min_confidence`
- `perception.keypoint_detection.crossing_detection.min_area`
- `perception.keypoint_detection.crossing_detection.min_neighbor_count`
- `perception.keypoint_detection.crossing_detection.min_branch_length`
- `perception.keypoint_detection.crossing_detection.min_branch_count`

Branch parameters are the key to avoiding false crossings on straight rope.

### Step C: Keypoint Skeleton Settings

Keypoint detection runs its own skeletonization settings so it can be tuned
independently of the visual path.

Config:
- `perception.keypoint_detection.skeletonization.pre_close_kernel_size`
- `perception.keypoint_detection.skeletonization.pre_dilate_kernel_size`
- `perception.keypoint_detection.skeletonization.pre_dilate_iterations`
- `perception.keypoint_detection.skeletonization.prune_length`
- `perception.keypoint_detection.skeletonization.close_loop_max_gap`

These settings control how connected the skeleton is at crossings.

---

## 3.4 Skeletonization: `src/perception/skeletonization.py`

Goal: convert the mask into a 1-pixel-wide centerline.

### Step A: Pre-Close / Pre-Dilate

Before thinning, the mask can be closed and dilated to ensure continuity.

Config:
- `perception.skeletonization.pre_close_kernel_size`
- `perception.skeletonization.pre_dilate_kernel_size`
- `perception.skeletonization.pre_dilate_iterations`

### Step B: Thinning

Uses Zhang-Suen thinning (OpenCV or scikit-image fallback).

Config:
- `perception.skeletonization.method`

### Step C: Post-Processing

Removes isolated pixels and short branches.

Config:
- `perception.skeletonization.prune_length`
- `perception.skeletonization.close_loop_max_gap`

If prune length is too high, real endpoints disappear.
If close_loop_max_gap is too high, loops may be forced closed.

---

## 3.5 Graph and Path Extraction: `src/perception/skeletonization.py`

Goal: turn the skeleton into a graph.

Outputs:

- `endpoints`: skeleton pixels with one neighbor
- `junctions`: pixels with three or more neighbors
- `edges`: ordered pixel paths between nodes
- `main_path`: the longest path through the graph

Why both edges and main path?

- Loops and crossings cannot be represented by one line.
- `edges` preserves the full shape.
- `main_path` gives a simplified path for control or visualization.

---

## 3.6 Rope State: `src/perception/state_estimation.py`

Goal: produce a clean summary for downstream logic.

`RopeState` includes:

- `endpoints`
- `crossings`
- `knots` (placeholder)
- `path` (main path only)
- `path_graph` (full graph when available)

---

## 3.7 Visualization: `src/perception/visualization.py`

Visualization overlays:

- Rope mask overlay
- Keypoints (endpoints and crossings)
- Rope path or full edge graph

If a loop exists, the visualizer draws **all edges** so the loop is visible.

---

# 4) Configuration Reference (What Each Setting Does)

Segmentation:

- `perception.segmentation.method`
  - `"color_threshold"`, `"edge_detection"`, or `"combined"`
- `perception.segmentation.color_range.hsv_lower`
- `perception.segmentation.color_range.hsv_upper`
- `perception.segmentation.blur_kernel_size`
- `perception.segmentation.pre_box_filter.enabled`
- `perception.segmentation.pre_box_filter.kernel_size`
- `perception.segmentation.morph_operations.opening_kernel_size`
- `perception.segmentation.morph_operations.closing_kernel_size`
- `perception.segmentation.edge_detection.canny_low_threshold`
- `perception.segmentation.edge_detection.canny_high_threshold`
- `perception.segmentation.contour_filter.min_area`
- `perception.segmentation.contour_filter.max_area`
- `perception.segmentation.contour_filter.min_aspect_ratio`
- `perception.segmentation.contour_filter.hole_fill_max_area`
- `perception.segmentation.contour_filter.hollow_fill_ratio_threshold`
- `perception.segmentation.cleanup.min_area`
- `perception.segmentation.cleanup.keep_largest`

Keypoints:

- `perception.keypoint_detection.endpoint_detection.method`
- `perception.keypoint_detection.endpoint_detection.min_confidence`
- `perception.keypoint_detection.endpoint_detection.merge_distance`
- `perception.keypoint_detection.crossing_detection.min_confidence`
- `perception.keypoint_detection.crossing_detection.min_area`
- `perception.keypoint_detection.crossing_detection.min_neighbor_count`
- `perception.keypoint_detection.crossing_detection.min_branch_length`
- `perception.keypoint_detection.crossing_detection.min_branch_count`

Keypoint skeletonization:

- `perception.keypoint_detection.skeletonization.pre_close_kernel_size`
- `perception.keypoint_detection.skeletonization.pre_dilate_kernel_size`
- `perception.keypoint_detection.skeletonization.pre_dilate_iterations`
- `perception.keypoint_detection.skeletonization.prune_length`
- `perception.keypoint_detection.skeletonization.close_loop_max_gap`

Global skeletonization:

- `perception.skeletonization.method`
- `perception.skeletonization.pre_close_kernel_size`
- `perception.skeletonization.pre_dilate_kernel_size`
- `perception.skeletonization.pre_dilate_iterations`
- `perception.skeletonization.prune_length`
- `perception.skeletonization.close_loop_max_gap`

Pipeline toggles:

- `perception.pipeline.disable_keypoint_extraction`
- `perception.pipeline.disable_skeletonization`

---

# 5) Common Issues and Which Parameters Matter

Problem: mask flickers on/off.
- Adjust `perception.segmentation.color_range.hsv_lower` and `hsv_upper`.
- Increase `perception.segmentation.pre_box_filter.kernel_size`.

Problem: skeleton breaks at crossings.
- Increase `perception.keypoint_detection.skeletonization.pre_close_kernel_size`.
- Increase `perception.keypoint_detection.skeletonization.pre_dilate_kernel_size`.
- Reduce `perception.keypoint_detection.skeletonization.prune_length`.
- Set `perception.keypoint_detection.skeletonization.close_loop_max_gap` to 0.

Problem: false crossings on straight rope.
- Increase `perception.keypoint_detection.crossing_detection.min_branch_length`.
- Increase `perception.keypoint_detection.crossing_detection.min_area`.
- Increase `perception.keypoint_detection.crossing_detection.min_neighbor_count`.

Problem: missing true crossings.
- Decrease `perception.keypoint_detection.crossing_detection.min_branch_length`.
- Decrease `perception.keypoint_detection.crossing_detection.min_area`.
- Decrease `perception.keypoint_detection.crossing_detection.min_confidence`.

Problem: missing endpoints.
- Decrease `perception.keypoint_detection.endpoint_detection.min_confidence`.
- Reduce `perception.keypoint_detection.skeletonization.prune_length`.

---

# 6) Mental Model (Short Version)

- Segment the rope into a mask.
- Clean and thin the mask into a skeleton.
- Extract a graph from the skeleton.
- Find endpoints and crossings.
- Summarize everything in a rope state.

Each module is independent, so you can tune or replace steps without rewriting
the rest of the pipeline.
