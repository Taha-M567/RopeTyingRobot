# Image Processing Pipeline: Code-Level Walkthrough

This document explains the current image-processing pipeline in this repository exactly as implemented, with direct references to code locations.

## 1. Where the Pipeline Starts

The live perception pipeline is orchestrated by `LiveVideoProcessor` in `src/perception/video_processor.py:51`.

There are two common entry paths:

1. Demo script path:
`examples/live_perception_demo.py:47` creates `Camera`, connects it, and creates `LiveVideoProcessor` with config and callback.
2. Helper factory path:
`src/perception/video_processor.py:290` (`create_processor_from_config`) loads YAML config, creates camera, connects, and returns a configured processor.

## 2. High-Level Runtime Flow

For each frame, `LiveVideoProcessor.process_next_frame()` executes this exact sequence (`src/perception/video_processor.py:155`):

1. Pull frame from queue (`src/perception/video_processor.py:165`).
2. Apply box filter (noise smoothing) with kernel `(4,4)` (`src/perception/video_processor.py:176`).
3. Segment rope (`segment_rope`) (`src/perception/video_processor.py:177`).
4. Detect keypoints (`detect_keypoints`) (`src/perception/video_processor.py:183`).
5. Build keypoint class mask (`create_keypoint_class_mask`) (`src/perception/video_processor.py:188`).
6. Skeletonize rope mask (`skeletonize_rope`) (`src/perception/video_processor.py:195`).
7. Extract graph/path from skeleton (`extract_path`) (`src/perception/video_processor.py:201`).
8. Estimate rope state (`estimate_rope_state`) (`src/perception/video_processor.py:204`).
9. Package everything into `ProcessingResult` dataclass (`src/perception/video_processor.py:208`).
10. Invoke callback if provided (`src/perception/video_processor.py:219`).

If any exception occurs during this per-frame path, the frame is dropped and `None` is returned (`src/perception/video_processor.py:224`).

## 3. Frame Acquisition and Real-Time Behavior

### 3.1 Camera interface

`Camera` is in `src/hardware/camera.py:28`.

1. `connect()` creates an OpenCV capture object (`cv2.VideoCapture`) using DirectShow backend (`src/hardware/camera.py:49`).
2. `capture()` reads one BGR frame and raises `RuntimeError` if not connected or read fails (`src/hardware/camera.py:66`, `src/hardware/camera.py:69`).
3. Optional undistortion is applied if calibration is present (`src/hardware/camera.py:74`).

### 3.2 Capture thread + queueing

`LiveVideoProcessor.start()` launches `_capture_frames()` in a daemon thread (`src/perception/video_processor.py:101`).

Behavior details:

1. Queue is bounded to size `2` (`src/perception/video_processor.py:81`).
2. Capture thread continuously calls `camera.capture()` (`src/perception/video_processor.py:138`).
3. Queue insertion is non-blocking (`put_nowait`) (`src/perception/video_processor.py:143`).
4. If queue is full, frame is dropped deliberately to maintain freshness/latency (`src/perception/video_processor.py:145`).

This design prioritizes low-latency recent frames over processing every frame.

## 4. Segment 1: Rope Segmentation (`segment_rope`)

Main function: `src/perception/rope_segmentation.py:360`.

Input/Output contract:

1. Input is a BGR image (`np.ndarray`), validated for non-empty (`src/perception/rope_segmentation.py:376`).
2. Output is `RopeMask(mask, confidence, image_shape)` (`src/perception/rope_segmentation.py:410`).
3. `mask` is binary-ish uint8 (0/255) after processing stages.

### 4.1 Method dispatch

`segment_rope` chooses method via `config["method"]` (`src/perception/rope_segmentation.py:384`):

1. `"color_threshold"` -> `_segment_by_color` (`src/perception/rope_segmentation.py:393`)
2. `"edge_detection"` -> `_segment_by_edges` (`src/perception/rope_segmentation.py:395`)
3. `"combined"` -> `_segment_combined` (`src/perception/rope_segmentation.py:397`)
4. `"adaptive"` currently falls back to color method (`src/perception/rope_segmentation.py:399`)
5. Unknown method also falls back to color method (`src/perception/rope_segmentation.py:404`)

### 4.2 Shared preprocessing and filtering

1. `_preprocess_image()` ensures odd blur kernel and applies Gaussian blur (`src/perception/rope_segmentation.py:41`, `src/perception/rope_segmentation.py:46`).
2. `_apply_morphology()` performs:
   - Opening (remove small noise) (`src/perception/rope_segmentation.py:81`)
   - Closing (fill small gaps) (`src/perception/rope_segmentation.py:84`)
3. `_filter_contours()` keeps contours meeting area and rope-like aspect ratio constraints (`src/perception/rope_segmentation.py:89`):
   - Area threshold checks (`src/perception/rope_segmentation.py:122`)
   - Aspect ratio from ellipse fit (fallback to bounding box if fit fails) (`src/perception/rope_segmentation.py:130`, `src/perception/rope_segmentation.py:136`)

### 4.3 Color-threshold method

`_segment_by_color()` (`src/perception/rope_segmentation.py:213`) does:

1. Convert BGR -> HSV (`src/perception/rope_segmentation.py:236`).
2. Read HSV bounds from config (`src/perception/rope_segmentation.py:240`).
3. Blur -> `cv2.inRange()` threshold -> morphology -> contour filtering (`src/perception/rope_segmentation.py:244`, `src/perception/rope_segmentation.py:247`, `src/perception/rope_segmentation.py:250`, `src/perception/rope_segmentation.py:253`).

### 4.4 Edge-detection method

`_segment_by_edges()` (`src/perception/rope_segmentation.py:258`) does:

1. Convert BGR -> grayscale (`src/perception/rope_segmentation.py:283`).
2. Blur -> Canny (`src/perception/rope_segmentation.py:286`, `src/perception/rope_segmentation.py:289`).
3. Morphological closing to connect edge fragments (`src/perception/rope_segmentation.py:296`).
4. Fill closed contours into solid mask (`src/perception/rope_segmentation.py:299`, `src/perception/rope_segmentation.py:308`).
5. Contour filtering for rope-like components (`src/perception/rope_segmentation.py:311`).

### 4.5 Combined method

`_segment_combined()` (`src/perception/rope_segmentation.py:316`) does:

1. Run both color and edge methods (`src/perception/rope_segmentation.py:330`, `src/perception/rope_segmentation.py:331`).
2. OR the masks (`src/perception/rope_segmentation.py:334`).
3. Compute method agreement = intersection/union over nonzero regions (`src/perception/rope_segmentation.py:337`, `src/perception/rope_segmentation.py:344`).
4. Apply additional morphology to the union mask (`src/perception/rope_segmentation.py:351`).

### 4.6 Segmentation confidence score

`_calculate_confidence()` (`src/perception/rope_segmentation.py:151`) combines:

1. Coverage score from rope pixel ratio in image (`src/perception/rope_segmentation.py:170`, `src/perception/rope_segmentation.py:175`).
2. Contour concentration score = largest contour area / total contour area (`src/perception/rope_segmentation.py:193`, `src/perception/rope_segmentation.py:199`).
3. Weighted blend `0.6 * coverage + 0.4 * contour` (`src/perception/rope_segmentation.py:204`).
4. Optional 30% contribution from method agreement (combined mode only) (`src/perception/rope_segmentation.py:207`).

## 5. Segment 2: Keypoint Detection (`detect_keypoints`)

Main function: `src/perception/keypoint_detection.py:214`.

It detects keypoints of type `"endpoint"` and `"crossing"` (and passes through `"knot"` type only if some other producer created it; no knot detector is implemented here).

### 5.1 Input normalization

`detect_keypoints()` ensures:

1. Valid numpy array input (`src/perception/keypoint_detection.py:230`).
2. Binary uint8 mask (0/255) through `_ensure_binary_mask()` (`src/perception/keypoint_detection.py:241`, `src/perception/keypoint_detection.py:39`).

### 5.2 Endpoint detection methods

Configured via `config["endpoint_detection"]["method"]` (`src/perception/keypoint_detection.py:246`).

#### Default: contour analysis

`_detect_endpoints_from_contour()` (`src/perception/keypoint_detection.py:120`) logic:

1. Find external contours (`src/perception/keypoint_detection.py:125`).
2. Sample contour points if huge (`src/perception/keypoint_detection.py:129`).
3. Find farthest point pair by brute-force pairwise squared distance (`src/perception/keypoint_detection.py:83`, `src/perception/keypoint_detection.py:93`).
4. Convert separation distance into confidence normalized by image diagonal (`src/perception/keypoint_detection.py:108`, `src/perception/keypoint_detection.py:114`).
5. Return 2 endpoint keypoints if confidence >= threshold (`src/perception/keypoint_detection.py:139`, `src/perception/keypoint_detection.py:142`).

Fallback behavior:

If contour method yields no endpoints and skeleton exists, it falls back to skeleton-based endpoint detection (`src/perception/keypoint_detection.py:263`).

#### Skeleton endpoint method

`_detect_endpoints_from_skeleton()` (`src/perception/keypoint_detection.py:148`) logic:

1. Compute 8-neighbor counts on skeleton (`src/perception/keypoint_detection.py:156`).
2. Endpoint pixels are `neighbor_count == 1` (`src/perception/keypoint_detection.py:157`).
3. Every endpoint pixel becomes a keypoint with confidence 1.0 (`src/perception/keypoint_detection.py:162`).

### 5.3 Crossing detection method

Configured via `config["crossing_detection"]["method"]` (`src/perception/keypoint_detection.py:249`).

Default `"skeleton_intersection"` runs `_detect_crossings_from_skeleton()` (`src/perception/keypoint_detection.py:176`):

1. Compute 8-neighbor counts.
2. Junction candidate pixels are `neighbor_count >= 3` (`src/perception/keypoint_detection.py:185`).
3. Connected components cluster nearby junction pixels (`src/perception/keypoint_detection.py:191`).
4. Each component centroid becomes one crossing.
5. Confidence is `area / (area + 10.0)` (`src/perception/keypoint_detection.py:200`).

### 5.4 Skeleton reuse behavior inside keypoint detection

`detect_keypoints` attempts to avoid redundant skeletonization:

1. If crossing method is skeleton-based, skeleton is computed once up front (`src/perception/keypoint_detection.py:256`).
2. Endpoint fallback and crossing detection then reuse that skeleton (`src/perception/keypoint_detection.py:263`, `src/perception/keypoint_detection.py:280`).

## 6. Segment 2.5: Class-Labeled Keypoint Mask

Main function: `create_keypoint_class_mask` in `src/perception/keypoint_mask.py:50`.

Class IDs are hard-coded:

1. `0`: background (`src/perception/keypoint_mask.py:15`)
2. `1`: rope (`src/perception/keypoint_mask.py:16`)
3. `2`: endpoint (`src/perception/keypoint_mask.py:17`)
4. `3`: crossing (`src/perception/keypoint_mask.py:18`)

Algorithm:

1. Start with binary rope mask.
2. Initialize class mask with rope pixels set to class `1` (`src/perception/keypoint_mask.py:79`).
3. For each endpoint keypoint, draw circular region with endpoint radius (default 6) clipped to rope pixels (`src/perception/keypoint_mask.py:81`, `src/perception/keypoint_mask.py:87`, `src/perception/keypoint_mask.py:46`).
4. For each crossing keypoint, do same with crossing radius (default 8), overwriting overlapping endpoint labels because crossings are drawn later (`src/perception/keypoint_mask.py:82`, `src/perception/keypoint_mask.py:95`).

This mask is used for downstream visualization and potentially for training labels.

## 7. Segment 3: Skeletonization (`skeletonize_rope`)

Main function: `src/perception/skeletonization.py:492`.

### 7.1 Validation and normalization

1. Invalid/non-array returns empty-like array safely (`src/perception/skeletonization.py:511`).
2. Non-`uint8` converted with warning (`src/perception/skeletonization.py:519`).
3. Non-binary masks are thresholded at 127 (`src/perception/skeletonization.py:524`, `src/perception/skeletonization.py:527`).

### 7.2 Optional pre-thinning morphology

Based on config values:

1. Morphological close if `pre_close_kernel_size > 0` (`src/perception/skeletonization.py:545`).
2. Dilation if `pre_dilate_kernel_size > 0` (`src/perception/skeletonization.py:552`).

Defaults are all disabled in config (`src/configs/perception_config.yaml:48`, `src/configs/perception_config.yaml:49`).

### 7.3 Thinning backend

`_skeletonize_zhang_suen()` (`src/perception/skeletonization.py:43`) backend selection:

1. Prefer `cv2.ximgproc.thinning(... THINNING_ZHANGSUEN)` if available (`src/perception/skeletonization.py:59`, `src/perception/skeletonization.py:61`).
2. Else fallback to `skimage.morphology.skeletonize` (`src/perception/skeletonization.py:72`, `src/perception/skeletonization.py:76`).
3. If neither backend exists, log error and return zeros (`src/perception/skeletonization.py:83`).

Note: `requirements.txt` includes `opencv-python` (not `opencv-contrib-python`) and includes `scikit-image`, so fallback is expected in many environments (`requirements.txt:8`, `requirements.txt:15`).

### 7.4 Post-processing

`_postprocess_skeleton()` (`src/perception/skeletonization.py:113`) performs:

1. Remove isolated pixels (`neighbor_count == 0`) (`src/perception/skeletonization.py:134`).
2. Iterative pruning of short branches shorter than `prune_length` (default 10) (`src/perception/skeletonization.py:115`, `src/perception/skeletonization.py:232`).
3. Branch tracing walks neighbor-to-neighbor until junction/endpoint/stop condition (`src/perception/skeletonization.py:190`).
4. Loop has safety cap `MAX_PRUNE_ITERATIONS = 100` (`src/perception/skeletonization.py:37`, `src/perception/skeletonization.py:151`).

## 8. Segment 4: Graph and Path Extraction (`extract_path`)

Main function: `src/perception/skeletonization.py:579`.

Returned `PathDict` keys (`src/perception/skeletonization.py:25`):

1. `endpoints`: `(E,2)` float array of `(x,y)`
2. `junctions`: `(J,2)` float array of `(x,y)`
3. `edges`: list of paths, each `(Ni,2)` `(x,y)`
4. `main_path`: one selected path or `None`

### 8.1 Graph node detection

`_extract_graph_structure()` (`src/perception/skeletonization.py:284`):

1. Endpoints = skeleton pixels with one neighbor (`src/perception/skeletonization.py:305`).
2. Junctions = connected-component centroids of pixels with `>=3` neighbors (`src/perception/skeletonization.py:241`, `src/perception/skeletonization.py:257`, `src/perception/skeletonization.py:266`).
3. Coordinate convention is explicitly converted from `(row,col)` to `(x,y)=(col,row)` (`src/perception/skeletonization.py:307`, `src/perception/skeletonization.py:330`).

### 8.2 Edge extraction

Edges are discovered by attempting pairwise node-to-node BFS paths (`src/perception/skeletonization.py:336`, `src/perception/skeletonization.py:347`, `src/perception/skeletonization.py:374`).

Important behavior:

1. BFS explores 8-neighborhood (`src/perception/skeletonization.py:404`).
2. BFS will not pass through intermediate nodes except target node (`src/perception/skeletonization.py:423`).
3. Paths are stored in `(x,y)` order (`src/perception/skeletonization.py:356`).

### 8.3 Loop handling

If skeleton has no endpoints and no junctions but has foreground pixels, `_trace_loop()` attempts to recover a closed loop path (`src/perception/skeletonization.py:360`, `src/perception/skeletonization.py:434`).

### 8.4 Main path selection

`extract_path()` sets `main_path` only for simple chains:

1. Exactly 2 endpoints
2. 0 junctions
3. At least one edge

Then the longest edge is used as `main_path` (`src/perception/skeletonization.py:615`, `src/perception/skeletonization.py:617`).

Otherwise `main_path=None` for branched/complex ropes (`src/perception/skeletonization.py:620`).

## 9. Segment 5: Rope State Estimation (`estimate_rope_state`)

Main function: `src/perception/state_estimation.py:32`.

Behavior:

1. Splits keypoints by type into `endpoints`, `crossings`, `knots` (`src/perception/state_estimation.py:48`, `src/perception/state_estimation.py:53`, `src/perception/state_estimation.py:58`).
2. Accepts either legacy path array or new `PathDict` (`src/perception/state_estimation.py:40`, `src/perception/state_estimation.py:65`).
3. Path selection rules for `PathDict`:
   - Use `main_path` if available (`src/perception/state_estimation.py:67`)
   - Else use longest edge if any (`src/perception/state_estimation.py:69`, `src/perception/state_estimation.py:72`)
   - Else empty `(0,2)` array (`src/perception/state_estimation.py:75`)
4. Returns `RopeState` dataclass (`src/perception/state_estimation.py:80`).

## 10. Output Packaging and Visualization

`ProcessingResult` includes:

1. Original (box-filtered) frame
2. `RopeMask`
3. `keypoints`
4. `keypoint_mask`
5. `RopeState`
6. processing time
7. frame number

Defined in `src/perception/video_processor.py:28`.

Visualization (`src/perception/visualization.py`) is separate from inference:

1. `draw_rope_mask` overlays segmentation with JET colormap (`src/perception/visualization.py:20`, `src/perception/visualization.py:36`).
2. `draw_keypoint_mask_overlay` colors class IDs (rope/endpoint/crossing) (`src/perception/visualization.py:44`, `src/perception/visualization.py:69`).
3. `draw_rope_path` draws connected lines from `rope_state.path` (`src/perception/visualization.py:81`, `src/perception/visualization.py:107`).
4. `display_result` calls `cv2.imshow` only when explicitly invoked (`src/perception/visualization.py:166`, `src/perception/visualization.py:192`).

## 11. Configuration: What Parameters Drive Each Stage

The active perception config tree starts at `perception:` in `src/configs/perception_config.yaml:3`.

### 11.1 Segmentation config

Relevant keys:

1. `segmentation.method` (`src/configs/perception_config.yaml:5`)
2. `segmentation.color_range.hsv_lower/hsv_upper` (`src/configs/perception_config.yaml:10`, `src/configs/perception_config.yaml:11`)
3. `segmentation.blur_kernel_size` (`src/configs/perception_config.yaml:20`)
4. `segmentation.morph_operations.*` (`src/configs/perception_config.yaml:23`)
5. `segmentation.edge_detection.*` (`src/configs/perception_config.yaml:28`)
6. `segmentation.contour_filter.*` (`src/configs/perception_config.yaml:33`)

### 11.2 Keypoint detection config

1. `keypoint_detection.endpoint_detection.*` (`src/configs/perception_config.yaml:39`)
2. `keypoint_detection.crossing_detection.*` (`src/configs/perception_config.yaml:42`)
3. Optional skeletonization overrides for keypoint detection (`src/configs/perception_config.yaml:45`)

### 11.3 Keypoint-mask config

1. `keypoint_mask.endpoint_radius` (`src/configs/perception_config.yaml:53`)
2. `keypoint_mask.crossing_radius` (`src/configs/perception_config.yaml:54`)

### 11.4 Skeletonization config

Pipeline-level skeleton settings used in `process_next_frame()`:

1. `skeletonization.method`
2. `skeletonization.prune_length`
3. pre-close/pre-dilate controls (`src/configs/perception_config.yaml:56`)

## 12. Coordinate and Data Conventions

1. OpenCV arrays are indexed as `(row, col)` internally.
2. Public path/keypoint coordinates are represented as `(x, y) = (col, row)` in most outputs, explicitly converted in path extraction (`src/perception/skeletonization.py:307`).
3. Binary masks are expected as uint8 with values `0` or `255`; multiple modules enforce this via thresholding (`src/perception/keypoint_detection.py:39`, `src/perception/keypoint_mask.py:24`, `src/perception/skeletonization.py:524`).

## 13. Test Coverage That Confirms Behavior

### 13.1 Segmentation tests

`tests/perception/test_rope_segmentation.py` validates:

1. Return type/shape/confidence range for valid image (`tests/perception/test_rope_segmentation.py:9`).
2. Error raising for invalid inputs (`tests/perception/test_rope_segmentation.py:21`).

### 13.2 Keypoint-mask tests

`tests/perception/test_keypoint_mask.py` validates:

1. Class-mask shape/dtype correctness (`tests/perception/test_keypoint_mask.py:29`).
2. Endpoint/crossing class overwrite behavior at target pixels (`tests/perception/test_keypoint_mask.py:35`, `tests/perception/test_keypoint_mask.py:37`).

### 13.3 Skeletonization/path tests

`tests/perception/test_skeletonization.py` validates:

1. Binary output and input robustness (`tests/perception/test_skeletonization.py:17`, `tests/perception/test_skeletonization.py:42`).
2. Post-processing behavior: isolated pixel removal and branch pruning (`tests/perception/test_skeletonization.py:113`, `tests/perception/test_skeletonization.py:127`).
3. Graph extraction and `main_path` behavior (`tests/perception/test_skeletonization.py:160`, `tests/perception/test_skeletonization.py:177`).
4. Coordinate convention `(x,y)` expectation (`tests/perception/test_skeletonization.py:242`).
5. Performance sanity check under synthetic 640x480 mask (`tests/perception/test_skeletonization.py:292`).

## 14. Practical Notes About Current Implementation

1. `LiveVideoProcessor` performs skeletonization twice per frame in default config:
   - once inside `detect_keypoints` when crossings use skeleton intersection (`src/perception/keypoint_detection.py:256`)
   - once again for dedicated pipeline skeleton/path stage (`src/perception/video_processor.py:195`)
2. The demo sets `target_fps=1` while comment says "Process at 10 FPS" (`examples/live_perception_demo.py:107`).
3. `state_estimation` can report `knots`, but no knot detector currently generates them in this pipeline (`src/perception/state_estimation.py:58`).

## 15. End-to-End Data Object Summary

For each processed frame, the final object (`ProcessingResult`) contains:

1. `frame`: BGR frame after initial box filter
2. `rope_mask.mask`: binary segmentation
3. `keypoints`: list of `Keypoint(position, keypoint_type, confidence)`
4. `keypoint_mask`: class-labeled segmentation (`0/1/2/3`)
5. `rope_state`:
   - `endpoints`, `crossings`, `knots`
   - `path` as selected polyline (or empty)
6. `processing_time`
7. `frame_number`

See `src/perception/video_processor.py:208` and `src/perception/video_processor.py:28`.


