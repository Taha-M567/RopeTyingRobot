# RopeUntyingRobot — Full Project Workflow

This document explains how every part of the project works, from the
OpenCV perception pipeline to the Isaac Lab simulation and RL training
environment. It is written for someone who is new to the codebase (or to
robotics in general) and wants to understand what each piece does, where
the code lives, and how the pieces connect.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Map](#2-directory-map)
3. [Perception Pipeline](#3-perception-pipeline)
4. [Hardware Interface](#4-hardware-interface)
5. [Utilities](#5-utilities)
6. [Isaac Lab Simulation](#6-isaac-lab-simulation)
7. [RL Environment — RopeReach-SO100-v0](#7-rl-environment--ropereach-so100-v0)
8. [Training and Evaluation Scripts](#8-training-and-evaluation-scripts)
9. [Knot Configuration System](#9-knot-configuration-system)
10. [Tests](#10-tests)
11. [Running the Project](#11-running-the-project)
12. [Phased Roadmap](#12-phased-roadmap)

---

## 1. Project Overview

The goal of this project is to build a system that can **untie knots in
a rope using a robot arm**. That is a hard problem, so the project is
split into independent subsystems that are developed and validated
separately before being combined:

| Subsystem | What it does | Status |
|-----------|-------------|--------|
| **Perception** | Looks at a camera image of a rope and figures out where the rope is, where it crosses over itself, and where its ends are. | Implemented |
| **Simulation** | A physics-based 3D world (Isaac Lab / Isaac Sim) containing a virtual SO-100 robot arm, a table, and a deformable rope. | Implemented |
| **RL Training** | A reinforcement-learning environment where a neural network learns to control the arm. Phase 1 teaches the arm to reach toward the rope. | Implemented (Phase 1) |
| **Hardware** | Camera interface for a real USB camera. | Scaffolded (calibration TODO) |
| **Control / Learning** | Future modules for grasping, manipulation policy, and LeRobot integration. | Planned |

The two main subsystems that are fully functional today are the
**perception pipeline** (pure OpenCV, runs on CPU) and the **Isaac Lab
RL environment** (GPU-accelerated physics + PPO training).

---

## 2. Directory Map

```
RopeUntyingRobot/
|
|-- src/                              # Perception + hardware + utilities
|   |-- perception/                   # OpenCV rope analysis pipeline
|   |   |-- rope_segmentation.py      # Step 1: find rope pixels
|   |   |-- keypoint_detection.py     # Step 2: find endpoints & crossings
|   |   |-- keypoint_mask.py          # Step 2b: label pixels by class
|   |   |-- crossing_analysis.py      # Crossing-specific analysis
|   |   |-- skeletonization.py        # Step 3: thin to centerline + graph
|   |   |-- state_estimation.py       # Step 4: summarise rope state
|   |   |-- visualization.py          # Draw overlays on frames
|   |   +-- video_processor.py        # Orchestrate the pipeline on live video
|   |-- hardware/
|   |   +-- camera.py                 # USB camera capture + calibration stub
|   |-- utils/
|   |   |-- config_loader.py          # Load / save YAML configs
|   |   |-- geometry.py               # 3D math helpers
|   |   +-- logging_config.py         # Project-wide logging setup
|   +-- configs/
|       |-- perception_config.yaml
|       +-- camera_config.yaml
|
|-- isaaclab/RopeUntyingRobot/        # Isaac Lab extension (simulation + RL)
|   |-- assets/                       # Robot + rope model files
|   |   |-- so100.urdf                # SO-100 robot description
|   |   |-- so100_config.py           # URDF-to-USD converter + actuator configs
|   |   |-- so100_from_urdf.usd       # Converted USD (auto-generated)
|   |   |-- *.stl                     # 3D mesh files for each link
|   |   |-- generate_rope_urdf.py     # Generates articulated rope chain URDF
|   |   |-- rope_config.py            # Rope ArticulationCfg + knot config I/O
|   |   |-- rope_chain.urdf           # Generated rope URDF (auto-generated)
|   |   |-- rope_chain_from_urdf.usd  # Converted rope USD (auto-generated)
|   |   +-- knot_configs/             # Saved knot configurations (YAML)
|   |       +-- straight.yaml         # Default straight rope (all joints zero)
|   |-- scripts/
|   |   |-- so100_sandbox.py          # Interactive sandbox (dual camera + perception)
|   |   |-- perception_viewer.py      # Live OpenCV dual-view viewer
|   |   |-- random_agent.py           # Sanity-check with random actions
|   |   |-- zero_agent.py             # Sanity-check with zero actions
|   |   |-- list_envs.py              # Print registered environments
|   |   +-- rsl_rl/
|   |       |-- train.py              # Launch RL training
|   |       |-- play.py               # Evaluate a trained policy
|   |       +-- cli_args.py           # Shared CLI argument helpers
|   +-- source/RopeUntyingRobot/      # Installable Python extension
|       +-- RopeUntyingRobot/
|           |-- __init__.py           # Registers gym envs + UI extension
|           |-- tasks/
|           |   +-- manager_based/ropeuntyingrobot/
|           |       |-- __init__.py              # gym.register("RopeReach-SO100-v0")
|           |       |-- ropeuntyingrobot_env_cfg.py  # Scene + MDP config
|           |       |-- mdp/
|           |       |   |-- __init__.py          # Re-exports everything
|           |       |   |-- observations.py      # rope_com_pos, ee_pos_w
|           |       |   |-- rewards.py           # reaching_rope, close_to_rope
|           |       |   |-- terminations.py      # rope_below_height
|           |       |   +-- events.py            # reset_articulated_rope (knot-aware)
|           |       +-- agents/
|           |           +-- rsl_rl_ppo_cfg.py    # PPO hyperparameters
|           +-- ui_extension_example.py
|
|-- tests/                            # Pytest test suite
|-- examples/
|   +-- live_perception_demo.py
|-- Workflow.md                       # This file
+-- pyproject.toml                    # Python project metadata
```

---

## 3. Perception Pipeline

The perception pipeline takes a single camera frame (a BGR image) and
produces a structured description of the rope: where it is, where its
ends are, where it crosses over itself, and what shape its centerline
follows.

### 3.1 Data Flow

```
Camera Frame (BGR image)
    |
    v
[1] Segmentation -----> RopeMask  (binary mask + confidence)
    |
    v
[2] Keypoint Detection -> List[Keypoint]  (endpoints + crossings)
    |
    v
[2b] Keypoint Mask ----> class mask  (pixels labeled 0-3)
    |
    v
[3] Skeletonization ---> binary skeleton (1-pixel-wide centerline)
    |
    v
[3b] Crossing Analysis -> List[CrossingInfo]  (over/under + patched skeleton)
    |
    v
[4] Graph Extraction --> PathDict  (endpoints, junctions, edges, main_path)
    |
    v
[5] State Estimation --> RopeState  (final summary)
    |
    v
[6] Visualization -----> overlay image (optional)
```

Each step is a separate module. You can swap, tune, or skip individual
steps without rewriting the rest.

### 3.2 Step 1: Segmentation

**File:** `src/perception/rope_segmentation.py`

**What it does:** Decides which pixels in the image belong to the rope
and which are background. The output is a binary mask — a grayscale
image the same size as the input where rope pixels are white (255) and
everything else is black (0).

**Entry point:**
```python
def segment_rope(image: np.ndarray, config: dict | None = None) -> RopeMask
```

**How it works internally:**

1. **Preprocessing** (`_preprocess_image`) — Gaussian blur reduces
   camera noise. Kernel size is set by
   `perception.segmentation.blur_kernel_size` (default 5).

2. **Thresholding** — The actual segmentation. Three methods are
   available, selected by `perception.segmentation.method`:

   - `"color_threshold"` (default) — Converts BGR to HSV colour space,
     then keeps pixels whose H, S, V values fall within the range
     defined by `hsv_lower` and `hsv_upper`. HSV is used because it
     separates brightness from colour, making it easier to detect a
     white or light-coloured rope under varying lighting.

   - `"edge_detection"` — Runs Canny edge detection, then fills
     enclosed contours. Useful when the rope has low colour contrast
     against the background but sharp edges.

   - `"combined"` — Runs both methods and merges results with a
     logical OR.

3. **Morphological cleanup** (`_apply_morphology`) — Two operations
   clean up noise:
   - *Opening* (erode then dilate) removes small dots that are not rope.
   - *Closing* (dilate then erode) fills small gaps in the rope mask.
   Kernel sizes: `opening_kernel_size`, `closing_kernel_size`.

4. **Contour filtering** (`_filter_contours`) — Finds the outlines of
   every white region, then throws away regions that are too small, too
   large, or have the wrong aspect ratio. This removes stray blobs that
   survived morphology. Key config:
   - `min_area` / `max_area` — size bounds
   - `min_aspect_ratio` — elongated shapes pass, round blobs don't
   - `hole_fill_max_area` — fills small holes inside the rope region

5. **Connected component cleanup** (`_cleanup_connected_components`) —
   A final pass that can either remove tiny remaining blobs or keep only
   the single largest connected region (`keep_largest: true`).

6. **Confidence score** (`_calculate_confidence`) — A 0-to-1 number
   estimating how trustworthy the mask is. It considers coverage ratio
   (is too much or too little of the image white?) and whether the mask
   is dominated by a single large contour (good) versus many small
   fragments (bad).

**Output dataclass:**
```python
@dataclass
class RopeMask:
    mask: np.ndarray        # Binary mask, uint8, values 0 or 255
    confidence: float       # 0.0 to 1.0
    image_shape: tuple[int, int]  # (height, width)
```

### 3.3 Step 2: Keypoint Detection

**File:** `src/perception/keypoint_detection.py`

**What it does:** Finds two types of important points on the rope:
- **Endpoints** — the two free ends of the rope.
- **Crossings** — places where the rope passes over or under itself.

**Entry point:**
```python
def detect_keypoints(mask: np.ndarray, config: dict) -> list[Keypoint]
```

**How endpoints are found:**

Three methods, selected by `perception.keypoint_detection.endpoint_detection.method`:

- `"contour_analysis"` — Finds the contour of the rope mask, then
  searches for the pair of contour points that are farthest apart. Those
  two points are the endpoints. Works well for simple, non-looping
  ropes.

- `"skeleton_endpoints"` — First skeletonizes the mask (see Step 3),
  then finds skeleton pixels that have exactly one neighbour. These are
  the tips of the skeleton.

- `"combined"` (default) — Runs both methods, then merges nearby
  duplicates.

**How crossings are found:**

The mask is first skeletonized internally (using its own config section
so it can be tuned independently). Then:

1. Junction pixels are found — these are skeleton pixels with 3 or more
   neighbours in the 8-connected grid.
2. Adjacent junctions are clustered into crossing candidates.
3. Each candidate is validated by checking that it has enough *real
   branches* extending outward (not just short noise spurs). A branch
   must be at least `min_branch_length` pixels long, and there must be
   at least `min_branch_count` valid branches (default 3 — a true
   crossing has at least 3 directions of rope).

**Output dataclass:**
```python
@dataclass
class Keypoint:
    position: tuple[float, float]  # (x, y) in image pixels
    keypoint_type: str             # "endpoint", "crossing", or "knot"
    confidence: float              # 0.0 to 1.0
```

### 3.4 Step 2b: Keypoint Class Mask

**File:** `src/perception/keypoint_mask.py`

**What it does:** Creates a pixel-level label map where every pixel is
assigned a class:

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | `BACKGROUND_CLASS` | Not rope |
| 1 | `ROPE_CLASS` | Rope body |
| 2 | `ENDPOINT_CLASS` | Near a rope endpoint |
| 3 | `CROSSING_CLASS` | Near a rope crossing |

**Entry point:**
```python
def create_keypoint_class_mask(
    rope_mask: np.ndarray,
    keypoints: list[Keypoint],
    config: dict | None = None,
) -> np.ndarray
```

The function draws circles around each keypoint (radii from config:
`endpoint_radius=6`, `crossing_radius=8`) but restricts the circles to
only cover rope pixels — so the class mask never bleeds onto the
background.

This mask is useful for training a semantic segmentation network to
distinguish different parts of the rope (future work).

### 3.5 Step 3: Skeletonization

**File:** `src/perception/skeletonization.py`

**What it does:** Converts the thick rope mask into a 1-pixel-wide
centerline (skeleton). Think of it as shrinking the rope from both sides
until only the middle line remains.

**Entry point:**
```python
def skeletonize_rope(mask: np.ndarray, config: dict | None = None) -> np.ndarray
```

**How it works:**

1. **Pre-processing** — Optional morphological closing and dilation
   before thinning. This ensures small gaps in the mask are connected
   before we thin it down. Config: `pre_close_kernel_size`,
   `pre_dilate_kernel_size`, `pre_dilate_iterations`.

2. **Thinning** — Zhang-Suen thinning algorithm. The code tries
   OpenCV's `cv2.ximgproc.thinning()` first (from opencv-contrib). If
   that module is not installed, it falls back to scikit-image's
   `skeletonize()`.

3. **Post-processing** — Two cleanup operations:
   - *Pruning* removes short branches from the skeleton. A branch is a
     chain of pixels starting at an endpoint (1 neighbour). If the chain
     is shorter than `prune_length` pixels, it is erased. This removes
     noise spurs.
   - *Loop closing* looks at cases where the skeleton has exactly 2
     endpoints that are close together (within `close_loop_max_gap`
     pixels) and connects them, forming a closed loop.

**Output:** A binary image (0/255) the same size as the input mask.

### 3.6 Step 3b: Crossing Analysis

**File:** `src/perception/crossing_analysis.py`

**What it does:** Determines which strand is on top (over) and which is
on the bottom (under) at each crossing point by sampling the original
colour image. The visible colour at the crossing centre belongs to the
top strand. It can also patch the skeleton at crossings, replacing
messy junction clusters with clean traced paths.

**Entry point:**
```python
def analyze_crossing_over_under(
    image: np.ndarray,
    skeleton: np.ndarray,
    rope_mask: np.ndarray,
    config: dict | None = None,
) -> list[CrossingInfo]
```

**How it works (5 internal steps):**

1. **Detect crossing regions** (`detect_crossing_regions`) — Uses the
   distance transform on the rope mask. At a crossing, two strands
   overlap so the mask is roughly twice as wide as a single strand.
   Pixels where the distance-to-background exceeds
   `width_ratio × normal_half_width` are flagged as crossing candidates.
   Connected components are filtered by area and compactness (aspect
   ratio < 4).

2. **Find entry/exit points** (`_find_entry_points`) — For each
   crossing region, finds skeleton pixels just outside the region
   boundary. A real crossing has 4 entry points (two per strand).
   Nearby entries are merged to handle messy skeletons.

3. **Sample branch colours** — Traces the skeleton outward from each
   entry point (up to `color_sample_length` pixels) and samples the
   mean HSV colour of the rope mask in that neighbourhood. This gives a
   colour per branch, sampled outside the crossing where the strands
   are cleanly separated.

4. **Pair branches and classify** — Entries are sorted by angle from
   the crossing centre and paired by angular opposition (roughly 180°
   apart). The pair whose mean colour is closer to the colour sampled
   at the crossing centre is the **over** (top) strand; the other pair
   is the **under** (bottom) strand. Confidence is based on how
   distinguishable the two pairs' colour distances are.

5. **Patch skeleton** (`patch_skeleton_at_crossings`) — Optionally
   replaces the messy skeleton pixels inside each crossing region with
   clean interpolated paths (one for each strand), giving downstream
   graph extraction a clean junction to work with.

**Output dataclasses:**
```python
@dataclass
class CrossingRegion:
    center: tuple[float, float]       # (x, y) centroid
    bbox: tuple[int, int, int, int]   # (r0, c0, r1, c1)
    region_mask: np.ndarray           # Boolean mask (full image size)
    normal_half_width: float          # Estimated single-strand half-width

@dataclass
class CrossingInfo:
    position: tuple[float, float]              # Crossing center (x, y)
    over_strand_path: np.ndarray               # (N, 2) path for top strand
    under_strand_path: np.ndarray              # (N, 2) path for bottom strand
    over_color_hsv: tuple[float, float, float] # Mean HSV of top strand
    under_color_hsv: tuple[float, float, float]# Mean HSV of bottom strand
    center_color_hsv: tuple[float, float, float]# HSV at crossing center
    confidence: float                          # 0.0 to 1.0
    region: CrossingRegion
```

### 3.7 Step 4: Graph Extraction

**File:** `src/perception/skeletonization.py` (same file as Step 3)

**What it does:** Turns the 1-pixel skeleton into a graph structure:
nodes (endpoints and junctions) connected by edges (pixel paths). Then
finds the main path through the graph.

**Entry point:**
```python
def extract_path(skeleton: np.ndarray) -> PathDict
```

**How it works:**

1. **Identify nodes** — Endpoints have 1 neighbour, junctions have 3+.
   Adjacent junction pixels are clustered into single nodes.

2. **Trace edges** — BFS walks along the skeleton from each node,
   following the 1-pixel chain until it reaches another node. Each
   traced chain becomes an edge.

3. **Build main path** — If there are endpoints, Dijkstra's algorithm
   finds the pair of endpoints with the longest path between them
   (measured in pixel count through the graph). The path is
   reconstructed by stitching together the edges along that route. If
   the skeleton forms a closed loop (no endpoints), a loop-tracing
   algorithm follows the skeleton in one direction until it returns to
   the start.

**Output:**
```python
class PathDict(TypedDict):
    endpoints: np.ndarray              # (E, 2) array of (x, y)
    junctions: np.ndarray              # (J, 2) array of (x, y)
    edges: list[np.ndarray]            # List of (N, 2) arrays
    main_path: np.ndarray | None       # Single (N, 2) array or None
```

### 3.8 Step 5: State Estimation

**File:** `src/perception/state_estimation.py`

**What it does:** Combines the keypoints and path information into a
single summary object that downstream code can use.

**Entry point:**
```python
def estimate_rope_state(
    keypoints: list[Keypoint],
    path: np.ndarray | PathDict,
) -> RopeState
```

The function sorts keypoints by type (endpoint, crossing, knot) and
extracts the main path from the `PathDict`. If no main path exists, it
falls back to the longest edge.

**Output:**
```python
@dataclass
class RopeState:
    endpoints: list[tuple[float, float]]
    crossings: list[tuple[float, float]]
    knots: list[tuple[float, float]]
    path: np.ndarray                    # (N, 2) main centerline
    path_graph: PathDict | None         # Full graph if available
```

### 3.9 Step 6: Visualization

**File:** `src/perception/visualization.py`

Drawing functions that overlay perception results onto the original
frame:

- `draw_rope_mask()` — JET colourmap overlay on the rope region.
- `draw_keypoint_mask_overlay()` — Colour-code pixels by class (white =
  rope, green = endpoint, red = crossing).
- `draw_rope_path()` / `draw_rope_edges()` — Draw the centerline or all
  graph edges.
- `visualize_result()` — Combines all of the above plus info text
  (frame number, processing time, keypoint counts).

### 3.10 Orchestration: Live Video Processor

**File:** `src/perception/video_processor.py`

The `LiveVideoProcessor` class ties everything together for real-time
use:

1. A **background thread** captures frames from the camera and puts them
   in a queue (max size 2, so old frames are dropped to stay
   real-time).
2. The main thread calls `process_next_frame()`, which pops the latest
   frame and runs the full pipeline: segmentation, keypoint detection,
   keypoint mask, skeletonization, path extraction, state estimation.
3. The result is returned as a `ProcessingResult` dataclass.

```python
@dataclass
class ProcessingResult:
    frame: np.ndarray
    rope_mask: RopeMask
    keypoints: list[Keypoint]
    keypoint_mask: np.ndarray
    rope_state: RopeState
    processing_time: float
    frame_number: int
```

The pipeline steps can be individually disabled via config:
- `perception.pipeline.disable_keypoint_extraction` — skips keypoints
- `perception.pipeline.disable_skeletonization` — skips skeleton + graph

### 3.11 Perception Config Reference

All perception tuning parameters live in
`src/configs/perception_config.yaml`. Here is what each section
controls:

**Segmentation** (`perception.segmentation.*`):

| Key | Default | What it does |
|-----|---------|-------------|
| `method` | `"color_threshold"` | Segmentation algorithm |
| `color_range.hsv_lower` | `[0, 0, 80]` | Lower HSV bound for rope colour |
| `color_range.hsv_upper` | `[180, 160, 255]` | Upper HSV bound |
| `blur_kernel_size` | `5` | Gaussian blur before segmentation |
| `morph_operations.opening_kernel_size` | `1` | Noise removal (0 = skip) |
| `morph_operations.closing_kernel_size` | `7` | Gap filling |
| `contour_filter.min_area` | `80` | Discard contours smaller than this |
| `contour_filter.min_aspect_ratio` | `1.2` | Discard round blobs |
| `cleanup.min_area` | `300` | Final blob removal threshold |
| `cleanup.keep_largest` | `true` | Keep only the biggest region |

**Keypoint detection** (`perception.keypoint_detection.*`):

| Key | Default | What it does |
|-----|---------|-------------|
| `endpoint_detection.method` | `"combined"` | How to find endpoints |
| `endpoint_detection.min_confidence` | `0.4` | Confidence cutoff |
| `endpoint_detection.merge_distance` | `8` | Merge duplicates within N px |
| `crossing_detection.min_branch_length` | `6` | Branch length to count as real |
| `crossing_detection.min_branch_count` | `3` | Minimum real branches for a crossing |
| `crossing_detection.min_area` | `6` | Junction cluster size cutoff |

**Skeletonization** (`perception.skeletonization.*`):

| Key | Default | What it does |
|-----|---------|-------------|
| `method` | `"zhang_suen"` | Thinning algorithm |
| `prune_length` | `4` | Remove branches shorter than this |
| `close_loop_max_gap` | `0` | Close loops if endpoints are this close (0 = off) |
| `pre_close_kernel_size` | `5` | Morphological close before thinning |
| `pre_dilate_kernel_size` | `3` | Dilation before thinning |

### 3.12 Common Tuning Issues

**Mask flickers between frames:**
- Widen `hsv_lower` / `hsv_upper` to capture more of the rope colour.
- Increase `blur_kernel_size` to smooth out frame-to-frame variation.

**False crossings on straight rope:**
- Increase `crossing_detection.min_branch_length` (e.g. 6 -> 10).
- Increase `crossing_detection.min_branch_count` (e.g. 3 -> 4).

**Missing real crossings:**
- Decrease `crossing_detection.min_branch_length`.
- Decrease `crossing_detection.min_area`.

**Skeleton breaks at crossings:**
- Increase `keypoint_detection.skeletonization.pre_close_kernel_size`.
- Decrease `prune_length`.

**Missing endpoints:**
- Decrease `endpoint_detection.min_confidence`.
- Decrease `prune_length` (aggressive pruning removes real tips).

---

## 4. Hardware Interface

**File:** `src/hardware/camera.py`

The `Camera` class wraps OpenCV's `VideoCapture` for USB cameras.

```python
cam = Camera(camera_id=1)
cam.connect()                 # Opens the device
frame = cam.capture()         # Returns a BGR numpy array
cam.disconnect()              # Releases the device
```

If a `CameraCalibration` dataclass is provided (intrinsic matrix +
distortion coefficients), `capture()` automatically undistorts every
frame.

**Config:** `src/configs/camera_config.yaml`
```yaml
camera:
  device_id: 1
  image_size:
    width: 480
    height: 640
  calibration:
    camera_matrix: null       # TODO: run calibration
    dist_coeffs: null
```

Calibration loading (`load_calibration()`) is a stub — it is not
implemented yet.

---

## 5. Utilities

### 5.1 Config Loader

**File:** `src/utils/config_loader.py`

Two functions:

- `load_config(path) -> dict` — Reads a YAML file and returns a
  dictionary.
- `save_config(config, path)` — Writes a dictionary to YAML, creating
  parent directories if needed.

Every tunable parameter in the project flows through these YAML configs.
No parameters are hardcoded in the processing code.

### 5.2 Geometry

**File:** `src/utils/geometry.py`

Small math functions for 3D transforms:

- `transform_point(point, rotation, translation)` — Applies a rigid
  transform: `R @ p + t`.
- `compute_distance(p1, p2)` — Euclidean distance.
- `angle_between_vectors(v1, v2)` — Angle in radians via dot product.

These are not used by the perception pipeline today (which works in 2D
pixel space) but are ready for when the project moves to 3D world
coordinates with the real robot arm.

### 5.3 Logging

**File:** `src/utils/logging_config.py`

`setup_logging(log_dir, log_level)` configures Python's built-in
`logging` module with a console handler and an optional file handler.
The project uses `logging.getLogger(__name__)` throughout — never
`print()`.

---

## 6. Isaac Lab Simulation

### 6.1 What is Isaac Lab?

Isaac Lab (built on NVIDIA Isaac Sim) is a GPU-accelerated robotics
simulator. It uses PhysX for rigid and deformable body physics, and can
run thousands of parallel environments on a single GPU. The project uses
it to:

1. Simulate the SO-100 robot arm in a virtual scene with a deformable
   rope.
2. Train reinforcement learning policies that control the arm.

### 6.2 Extension Structure

Isaac Lab projects are organised as *extensions* — installable Python
packages that register themselves with the simulator. The extension
lives at `isaaclab/RopeUntyingRobot/source/RopeUntyingRobot/`.

```
source/RopeUntyingRobot/
|-- config/extension.toml    # Extension metadata (name, version, deps)
|-- pyproject.toml            # Build system config
|-- setup.py                  # pip install -e .
+-- RopeUntyingRobot/         # Python package
    |-- __init__.py           # Imports tasks + UI extension
    |-- tasks/
    |   |-- __init__.py       # Auto-discovers task configs
    |   +-- manager_based/
    |       +-- ropeuntyingrobot/   # The RL task
    +-- ui_extension_example.py     # Sample Isaac Sim UI panel
```

When the extension is installed (`pip install -e .`), importing
`RopeUntyingRobot.tasks` triggers the Gymnasium environment
registration, which makes `RopeReach-SO100-v0` available to any script
that calls `gym.make()`.

### 6.3 The SO-100 Robot Arm

**URDF file:** `isaaclab/RopeUntyingRobot/assets/so100.urdf`

The SO-100 is a 6-DOF desktop robot arm (5 arm joints + 1 gripper
joint):

| Joint | Type | What it moves |
|-------|------|--------------|
| `shoulder_pan` | Revolute | Rotates the whole arm left/right |
| `shoulder_lift` | Revolute | Raises/lowers the upper arm |
| `elbow_flex` | Revolute | Bends the elbow |
| `wrist_flex` | Revolute | Tilts the wrist up/down |
| `wrist_roll` | Revolute | Rotates the wrist |
| `gripper` | Revolute | Opens/closes the jaw |

The robot model is described in URDF (Universal Robot Description
Format), which lists each link (rigid body segment) and the joints that
connect them. The 13 STL files in `assets/` provide the 3D mesh
geometry for each link.

**URDF to USD conversion:** Isaac Lab uses Pixar's USD format
internally. The file `assets/so100_config.py` handles the conversion
chain:

```
so100.urdf
   |
   v  (_repair_mesh_paths_if_needed)
so100_mesh_fixed.urdf           # Patched mesh paths
   |
   v  (optional: _strip_collision_blocks)
so100_mesh_fixed_fast.urdf      # No collision meshes (faster to load)
   |
   v  (UrdfConverter)
so100_from_urdf.usd             # Final USD used by Isaac Lab
```

The conversion only runs once. On subsequent launches, the existing USD
file is reused.

**Two ArticulationCfg factories:**

The file provides two functions that create ready-to-use robot
configurations:

1. **`create_so100_articulation_cfg()`** — For the sandbox and
   visualisation. Uses a single actuator group for all joints. Can
   optionally strip collision meshes for faster loading
   (`use_fast_sandbox_urdf=True`).

2. **`create_so100_rl_articulation_cfg()`** — For RL training.
   Separates the arm and gripper into distinct actuator groups with
   different PD gains:

   ```python
   "so100_arm": ImplicitActuatorCfg(
       joint_names_expr=["shoulder_pan", "shoulder_lift",
                         "elbow_flex", "wrist_flex", "wrist_roll"],
       stiffness=20.0, damping=2.0, effort_limit_sim=40.0,
   )
   "so100_gripper": ImplicitActuatorCfg(
       joint_names_expr=["gripper"],
       stiffness=40.0, damping=10.0, effort_limit_sim=20.0,
   )
   ```

   The gripper has higher stiffness so it can grip firmly in future
   phases. It always keeps collision meshes (needed for contact-based
   RL).

**Default joint positions** (`so100_config.py:SO100_DEFAULT_JOINT_POS`):
```python
{
    "shoulder_pan": 0.0,
    "shoulder_lift": 1.0,
    "elbow_flex": -1.2,
    "wrist_flex": 0.3,
    "wrist_roll": 0.0,
    "gripper": 0.15,
}
```

This puts the arm in a neutral pose above the table.

### 6.4 The Articulated Rope Chain

**Files:** `assets/generate_rope_urdf.py`, `assets/rope_config.py`

The rope is modelled as an **articulated chain** — 16 rigid cylindrical
segments connected by 15 two-DOF revolute joints (pitch + yaw per hinge).
This is a rigid-body approximation that is faster than FEM deformable
simulation and simpler to reset and control via joint angles.

**Rope geometry:**
- 16 segments, each a cylinder with radius 0.005 m
- Total length: 0.45 m (~0.028 m per segment)
- 30 joints total: `rope_pitch_0..14` and `rope_yaw_0..14`
- Joint limits: ±120° (±2.094 rad) per axis
- Visual: Green → Yellow → Red → Blue gradient across segments

**URDF generation** (`generate_rope_urdf.py`):
The URDF is generated programmatically, not hand-authored. Each segment
has mass computed from `density=5000 kg/m³`, and each hinge is a
massless link connecting two revolute joints.

**Configuration factory** (`rope_config.py`):
`create_rope_articulation_cfg()` converts the URDF to USD (once, cached)
and returns an `ArticulationCfg` with passive actuators (low stiffness,
low damping). Self-collisions are disabled.

**Key constants** (exported from `rope_config.py`):
- `ROPE_NUM_SEGMENTS = 16`
- `ROPE_ALL_JOINT_NAMES` — interleaved list of 30 joint names
- `ROPE_DEFAULT_JOINT_POS` — all zeros (straight rope)

### 6.5 The Sandbox Scene

**File:** `isaaclab/RopeUntyingRobot/scripts/so100_sandbox.py`

The sandbox script creates a complete scene for testing the robot and
the perception pipeline on simulated images:

| Entity | Type | Details |
|--------|------|---------|
| Ground plane | Kinematic cuboid | 4 x 4 m, dark grey |
| SO-100 arm | Articulation | From URDF, at world origin |
| Table | Kinematic cuboid | 0.6 x 0.4 x 0.06 m at (0.25, 0, -0.02) |
| Rope | Articulated chain | 16 segments, 30 joints, at (0.25, 0, 0.05) |
| Top-down camera | Pinhole sensor | 640x480, mounted on robot base, looks straight down |
| Side camera | Pinhole sensor | 640x480, mounted on robot base offset, looks along +Y |
| Dome light | Light | 3000 intensity |

**Dual-camera system:**

The sandbox has two cameras, each attached to the robot base link:

1. **Top-down camera** — offset `(0.25, 0.0, 0.50)`, rotation
   `(1,0,0,0)` ROS convention. Looks straight down at the table.
   Used for running the perception pipeline (rope segmentation,
   keypoints, skeletonization).

2. **Side camera** — offset `(0.25, -0.60, 0.15)`, rotation
   `(0.7071, 0.7071, 0, 0)` ROS convention (90° about X). Looks
   along +Y toward the scene. Provides a side profile of the arm
   relative to the table and rope — useful for pose estimation.

Both cameras use the same pinhole intrinsics (focal length 24 mm,
horizontal aperture 20.955 mm). Resolution is configurable via
`--camera_width` and `--camera_height` CLI arguments.

The `_compose_dual_view()` helper horizontally concatenates the top-down
perception overlay (left) and raw side-view RGB (right) with text labels.
This composite is saved as `latest.png` and timestamped frame files.

**FOV visualization:** `_spawn_camera_fov_prims()` creates guide-purpose
USD sphere prims outlining both camera frustums: cyan for the top-down
camera, orange for the side camera. These are visible in the viewport
but invisible to camera render products.

**Simulation loop:**

On each step, the script:
1. Holds the arm at its default pose
2. Captures RGB from both cameras
3. Runs the full perception pipeline on the top-down image
4. Extracts the raw side-view image (no perception processing)
5. Composes the dual-view composite
6. Logs rope COM position + perception metrics
7. Optionally saves the composite to disk

### 6.6 Sandbox Keyboard Controller

**Class:** `RopeKeyboardController` in `so100_sandbox.py`

An interactive controller that lets you move and bend the rope during
simulation using keyboard keys:

| Key | Action |
|-----|--------|
| I / K | Move rope forward / backward (X) |
| J / L | Move rope left / right (Y) |
| U / O | Move rope up / down (Z) |
| , / . | Select previous / next hinge joint |
| W / S | Bend pitch of selected joint (+/-) |
| A / D | Bend yaw of selected joint (+/-) |
| R | Reset rope to spawn position (straighten all joints) |
| P | Save current joint angles as a knot config YAML |

Each keypress bends the selected joint by ~10° (0.174 rad). A bright
magenta sphere marker shows which hinge is currently selected.

**Saving knot configs (P key):** Pressing P reads the current joint
positions from the simulation, maps them to joint names, and writes a
timestamped YAML file to `assets/knot_configs/` (e.g.,
`knot_20260331_142530.yaml`). This is the primary workflow for
creating knot configurations — see
[Section 9: Knot Configuration System](#9-knot-configuration-system).

A floating HUD window ("Rope Controls") displays all available keyboard
controls during the simulation.

---

## 7. RL Environment — RopeReach-SO100-v0

### 7.1 What is This Environment?

This is a **Phase 1** reinforcement learning task: teach the robot arm
to move its gripper toward the rope. It does not attempt grasping,
pulling, or knot untying — those are future phases.

The environment is registered with Gymnasium as `RopeReach-SO100-v0`.
It uses Isaac Lab's *manager-based* architecture, where the scene,
observations, actions, rewards, resets, and terminations are each
defined as config classes.

**File:** `isaaclab/.../ropeuntyingrobot_env_cfg.py`

### 7.2 Scene

The scene uses the same robot, table, and articulated rope as the
sandbox, plus one addition:

- A **FrameTransformer** sensor that tracks the position of the
  `gripper` link relative to the robot `base`. This gives the RL agent
  a clean end-effector position signal without the agent needing to
  solve forward kinematics.

The RL scene does **not** include cameras — observations are
proprioceptive (joint states + rope COM + end-effector position). The
scene uses `replicate_physics=True` since the articulated rope chain
(unlike FEM deformable bodies) supports physics replication across
parallel environments.

```python
ee_frame = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper",
            name="ee",
        ),
    ],
)
```

### 7.3 Action Space (5D)

**File:** `ropeuntyingrobot_env_cfg.py` — `ActionsCfg`

The agent controls the 5 arm joints using **joint position commands**.
The gripper is not controlled (stays open at its default position).

```python
arm_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll"],
    scale=0.5,
    use_default_offset=True,
)
```

- `scale=0.5` means the raw action (from the neural network, in
  [-1, 1]) is multiplied by 0.5 before being added to the default joint
  position. This limits how far the arm can move from its rest pose in a
  single step.
- `use_default_offset=True` means actions are *relative to the default
  pose*, not absolute angles.

### 7.4 Observation Space (21D)

**File:** `ropeuntyingrobot_env_cfg.py` — `ObservationsCfg`

The agent receives 21 numbers every step:

| Observation | Dimensions | Source | What it tells the agent |
|-------------|-----------|--------|------------------------|
| `joint_pos_rel` | 5 | Built-in `mdp.joint_pos_rel` | Current arm joint angles relative to default |
| `joint_vel_rel` | 5 | Built-in `mdp.joint_vel_rel` | How fast each joint is moving |
| `rope_com` | 3 | Custom `mdp.rope_com_pos` | Where the rope's centre-of-mass is in world (x,y,z) |
| `ee_pos` | 3 | Custom `mdp.ee_pos_w` | Where the gripper is in world (x,y,z) |
| `last_action` | 5 | Built-in `mdp.last_action` | What action was taken last step |

**Why rope COM instead of all bodies?** The articulated rope chain has
31 bodies (16 segments + 15 hinges). Feeding all 93 coordinates as
observations would be noisy and redundant for a simple reaching task.
The centre-of-mass (mean of all body positions) is a clean 3D target.
Future phases will add individual body positions for more complex tasks.

**Custom observation functions** (`mdp/observations.py`):

```python
def rope_com_pos(env, asset_cfg=SceneEntityCfg("rope")) -> torch.Tensor:
    rope = env.scene[asset_cfg.name]
    return rope.data.body_pos_w.mean(dim=1)    # (num_envs, 3)

def ee_pos_w(env, sensor_cfg=SceneEntityCfg("ee_frame")) -> torch.Tensor:
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.target_pos_w[:, 0, :]   # (num_envs, 3)
```

### 7.5 Reward Design

**File:** `mdp/rewards.py` + `ropeuntyingrobot_env_cfg.py` — `RewardsCfg`

The reward function guides the agent toward the rope:

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| `reaching_rope` | 1.0 | `1 - tanh(distance / 0.1)` | Dense gradient — gives reward for getting closer from anywhere |
| `close_to_rope` | 5.0 | `1.0 if distance < 0.02 else 0.0` | Bonus when the gripper is within 2 cm of the rope COM |
| `action_rate` | -0.01 | `action_rate_l2` | Penalises jerky motions (smooths the trajectory) |
| `joint_vel` | -0.001 | `joint_vel_l1` | Penalises fast joint movement (energy regularisation) |

**Why `tanh`?** The `1 - tanh(d / sigma)` function produces a reward
that is near 1.0 when the distance is near 0, drops smoothly toward 0
as distance grows, and never goes negative. The `sigma` parameter (0.1
metres) controls the steepness — at 10 cm distance the reward is about
0.24, at 20 cm it is about 0.04. This gives a useful learning signal
even when the gripper starts far from the rope.

### 7.6 Reset and Termination

**File:** `mdp/events.py` — `reset_articulated_rope()`

**Resets** (what happens at the start of each episode):

- **Arm joints** — Reset to the default pose multiplied by a random
  scale factor between 0.8 and 1.2. This gives diverse starting
  configurations so the policy generalises instead of memorising one
  trajectory.

- **Rope** — The articulated rope chain is reset via
  `reset_articulated_rope()`, which applies three layers of
  randomization:

  1. **Global position offset** — Random XY shift of up to ±5 cm.
     The Z offset is zero (rope starts on the table).
  2. **Z-axis rotation** — Random angle between 0 and 2π so the
     rope can face any direction.
  3. **Joint angle noise** — Gaussian noise (σ=0.05 rad by default)
     added to each of the 30 joint angles, creating random bends.

  When a **knot configuration** is provided via the `base_joint_pos`
  parameter, the joint angles start from the knot shape instead of
  the default straight configuration. Noise is then added on top, so
  each parallel environment gets a slightly different variation of
  the same knot. See [Section 9](#9-knot-configuration-system) for
  details on defining and using knot configs.

  The reset writes the root state and joint state to the simulation:
  ```python
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  ```

**Terminations** (what ends an episode early):

- **Time out** — 10 seconds (300 control steps at 30 Hz). After this,
  the episode is marked as truncated (not failed) so the value function
  can still bootstrap.

- **Rope fell** — If the rope COM drops below z = -0.05 m, the episode
  ends. This catches cases where the arm accidentally knocks the rope
  off the table.

**Custom termination function** (`mdp/terminations.py`):
```python
def rope_below_height(env, min_height=-0.05, asset_cfg=SceneEntityCfg("rope")):
    rope = env.scene[asset_cfg.name]
    rope_com_z = rope.data.body_pos_w.mean(dim=1)[:, 2]
    return rope_com_z < min_height
```

### 7.7 PPO Training Config

**File:** `agents/rsl_rl_ppo_cfg.py`

The RL algorithm is PPO (Proximal Policy Optimization), implemented by
the RSL-RL library. Key hyperparameters:

| Parameter | Value | Why |
|-----------|-------|-----|
| `actor_hidden_dims` | `[128, 128, 64]` | Larger than CartPole's [32,32] — manipulation needs more capacity |
| `critic_hidden_dims` | `[128, 128, 64]` | Same architecture for the value network |
| `actor_obs_normalization` | `True` | Joint angles and Cartesian positions have very different scales |
| `learning_rate` | `3e-4` | Standard for PPO |
| `entropy_coef` | `0.01` | Encourages exploration early in training |
| `num_learning_epochs` | `8` | PPO updates per batch of experience |
| `num_mini_batches` | `4` | Splits each batch into 4 chunks for SGD |
| `gamma` | `0.99` | Discount factor |
| `max_iterations` | `1500` | Total training iterations |
| `num_steps_per_env` | `24` | Steps collected per env before each update |

Training logs go to `logs/rsl_rl/rope_reach_so100/`.

### 7.8 Simulation Timing

The physics engine runs at **120 Hz** (`sim.dt = 1/120`). The RL agent
acts every 4 physics steps (`decimation = 4`), giving a **30 Hz**
control rate. Each episode is 10 seconds = 300 control steps.

---

## 8. Training and Evaluation Scripts

All scripts are in `isaaclab/RopeUntyingRobot/scripts/`.

### 8.1 Training (`scripts/rsl_rl/train.py`)

Launches PPO training with RSL-RL.

```bash
python scripts/rsl_rl/train.py --task RopeReach-SO100-v0 --num_envs 64 --headless
```

Key arguments:

| Argument | Default | What it does |
|----------|---------|-------------|
| `--task` | (required) | Gymnasium task ID |
| `--num_envs` | From env config (64) | Number of parallel environments |
| `--headless` | False | Run without rendering (faster on servers) |
| `--max_iterations` | From PPO config (1500) | Override training length |
| `--seed` | None | Random seed for reproducibility |
| `--video` | False | Record videos during training |
| `--distributed` | False | Multi-GPU training |

The script:
1. Launches Isaac Sim via `AppLauncher`
2. Creates the environment with `gym.make()`
3. Wraps it for RSL-RL (`RslRlVecEnvWrapper`)
4. Creates an `OnPolicyRunner` and calls `runner.learn()`
5. Saves checkpoints every 100 iterations to `logs/rsl_rl/rope_reach_so100/`

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/rope_reach_so100
```

### 8.2 Evaluation (`scripts/rsl_rl/play.py`)

Loads a trained checkpoint and runs the policy.

```bash
python scripts/rsl_rl/play.py --task RopeReach-SO100-v0 --num_envs 4
```

Key arguments:

| Argument | What it does |
|----------|-------------|
| `--task` | Gymnasium task ID |
| `--num_envs` | Environments to visualise |
| `--real-time` | Slow down to wall-clock time |
| `--video` | Record a video |

The script loads the most recent checkpoint from the training log
directory, exports the policy as JIT and ONNX files (for deployment),
then runs the policy in a loop.

### 8.3 Sanity Checks

- **`random_agent.py`** — Sends uniformly random actions to the
  environment. Useful for verifying the scene spawns correctly,
  observations have the right shape, and rewards compute without
  crashing.

  ```bash
  python scripts/random_agent.py --task RopeReach-SO100-v0 --num_envs 1 --headless
  ```

- **`zero_agent.py`** — Sends zero actions (arm holds default pose).
  Useful for checking that the scene is stable and the rope doesn't
  explode.

### 8.4 Other Scripts

- **`list_envs.py`** — Prints all registered environments in a table.

- **`perception_viewer.py`** — An OpenCV window that polls
  `output/so100_preprocess/latest.png` for modification time changes
  and displays the dual-view composite (top-down perception on the left,
  side-view on the right). Run alongside `so100_sandbox.py --show` to
  see perception output in real time. Auto-resizes if the composite
  exceeds 1920px wide.

  ```bash
  python scripts/perception_viewer.py           # default path
  python scripts/perception_viewer.py --fps 15  # slower refresh
  ```

- **`so100_sandbox.py`** — The interactive sandbox scene with dual
  cameras and keyboard rope controller (see
  [Section 6.5](#65-the-sandbox-scene) and
  [Section 6.6](#66-sandbox-keyboard-controller)).

  ```bash
  python scripts/so100_sandbox.py --show --max_steps 200
  ```

---

## 9. Knot Configuration System

The knot configuration system lets you define specific rope shapes
(knots) as joint angle presets, save them to YAML files, and use them
as episode starting configurations during RL training.

### 9.1 Overview

A knot configuration is a YAML file containing the 30 joint angles
(15 pitch + 15 yaw) that define a rope shape. The primary workflow:

1. Run the sandbox with the keyboard controller
2. Bend the rope into a desired knot shape using the joint controls
3. Press **P** to save the current joint angles as a knot config YAML
4. Reference that YAML in the RL environment's `EventCfg` to train
   against that knot shape

### 9.2 YAML Format

Knot config files live in `assets/knot_configs/`. Each file contains:

```yaml
name: overhand_knot
description: Simple overhand knot for untying training
joint_angles:
  rope_pitch_0: 0.0
  rope_yaw_0: 0.3
  rope_pitch_1: 0.5
  rope_yaw_1: -0.2
  ...all 30 joints...
```

A `straight.yaml` default config (all zeros) is included as a reference.

### 9.3 Loading and Saving

**File:** `assets/rope_config.py`

**Loading:**
```python
from rope_config import load_knot_config

angles = load_knot_config("overhand.yaml")         # relative to knot_configs/
angles = load_knot_config("/absolute/path/knot.yaml")  # absolute path
```

The loader validates that all 30 joint names are present and that values
are within the ±2.094 rad joint limits.

**Saving:**
```python
from rope_config import save_knot_config

save_knot_config(
    joint_angles={"rope_pitch_0": 0.3, ...},
    path="my_knot.yaml",         # saved to knot_configs/my_knot.yaml
    name="my_knot",
    description="A custom knot shape.",
)
```

### 9.4 Using Knot Configs in RL Training

To train against a specific knot shape, load the config and pass it
to the `EventCfg`:

```python
# In ropeuntyingrobot_env_cfg.py:
from rope_config import load_knot_config

@configclass
class EventCfg:
    reset_rope = EventTerm(
        func=mdp.reset_articulated_rope,
        mode="reset",
        params={
            "base_joint_pos": load_knot_config("overhand.yaml"),
            "joint_noise_std": 0.08,  # More noise for diversity
            ...
        },
    )
```

When `base_joint_pos` is set, every environment starts each episode
with the rope in the specified knot shape, plus per-environment
Gaussian noise on the joint angles. This prevents the policy from
overfitting to exactly one configuration while keeping the general
knot topology consistent.

When `base_joint_pos` is `None` (the default), the rope starts
straight with only noise — this preserves backward compatibility with
the existing `RopeReach-SO100-v0` environment.

---

## 10. Tests

**Directory:** `tests/`

The test suite uses **pytest**. All tests are deterministic (no random
seeds, fixed input arrays) and run on CPU.

```bash
pytest tests/ -v
```

### What is tested

| File | Module | Tests |
|------|--------|-------|
| `tests/perception/test_rope_segmentation.py` | Segmentation | `RopeMask` dataclass fields, `segment_rope()` output shape and confidence range, error on invalid input |
| `tests/perception/test_keypoint_mask.py` | Keypoint mask | `create_keypoint_class_mask()` produces correct class values (0-3) |
| `tests/perception/test_crossing_analysis.py` | Crossing analysis | Circular hue mean, HSV color distance, branch pairing by color, crossing region detection (thick strands, uniform width, empty/None masks), skeleton entry points, skeleton patching, integration test with synthetic two-color crossing image |
| `tests/perception/test_skeletonization.py` | Skeletonization | Zhang-Suen thinning, neighbour counting, pruning, junction clustering, graph extraction on lines/T-junctions/curves, performance (<500ms on 640x480) |
| `tests/utils/test_geometry.py` | Geometry | Distance, rotation+translation transform, angle between vectors |

### Test conventions

- Fixtures in `conftest.py` provide `sample_image` (100x100x3 zeros)
  and `sample_mask` (100x100 zeros).
- Assertions use `np.testing` and `pytest.approx` for floating-point
  comparisons.
- No Isaac Lab / GPU tests (simulation requires Isaac Sim runtime).

---

## 11. Running the Project

### 11.1 Prerequisites

| Component | Required for | Install |
|-----------|-------------|---------|
| Python >= 3.10 | Everything | System / conda |
| OpenCV (`cv2`) | Perception | `pip install opencv-python` |
| NumPy 1.x | Perception + Isaac Sim | `pip install "numpy<2"` |
| PyYAML | Config loading | `pip install pyyaml` |
| scikit-image | Skeletonization fallback | `pip install scikit-image` |
| Isaac Lab + Isaac Sim | Simulation + RL | See [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/) |
| RSL-RL >= 3.0.1 | PPO training | `pip install rsl-rl-lib` |
| PyTorch + CUDA | RL training | Bundled with Isaac Lab |

### 11.2 Quick Start

**Run perception tests (no GPU needed):**
```bash
pytest tests/ -v
```

**Run live perception on a USB camera:**
```bash
python examples/live_perception_demo.py
```

**Run the sandbox scene with dual cameras (requires Isaac Lab):**
```bash
# Terminal 1: sandbox with dual-camera view + keyboard rope control
python isaaclab/RopeUntyingRobot/scripts/so100_sandbox.py --show --max_steps 200

# Terminal 2: live viewer (displays top-down + side-view composite)
python isaaclab/RopeUntyingRobot/scripts/perception_viewer.py
```

**Train the RL agent:**
```bash
cd isaaclab/RopeUntyingRobot
python scripts/rsl_rl/train.py --task RopeReach-SO100-v0 --num_envs 64 --headless
```

**Evaluate a trained policy:**
```bash
python scripts/rsl_rl/play.py --task RopeReach-SO100-v0 --num_envs 4
```

---

## 12. Phased Roadmap

The project follows a bottom-up approach — each phase validates a
harder capability before moving to the next.

| Phase | Task | Status |
|-------|------|--------|
| **1** | **Reach the rope** — move gripper to rope COM | Implemented |
| **2** | **Grasp and displace** — pick up a rope end, move to a target | Planned |
| **3** | **Straighten rope** — pull a curved rope into a line | Planned |
| **4** | **Knot untying** — untie a specific pre-tied knot | Planned (knot config system implemented — see [Section 9](#9-knot-configuration-system)) |

Phase 1 validates that the full stack works: scene spawning, action
space, observations, rewards, resets, and the PPO training pipeline. The
later phases add increasingly complex rewards and task-specific
observation terms. Phase 4 can now use the knot configuration system to
initialize episodes with pre-tied knot shapes.
