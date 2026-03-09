# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Robotic rope tying/untying system (VIP research project W26). Two subsystems:
1. **Perception pipeline** (OpenCV, CPU) — fully implemented in `src/perception/`
2. **Isaac Lab simulation** (GPU, PhysX) — RL environment in `isaaclab/RopeUntyingRobot/`

Planned but not yet implemented: control module, learning module (LeRobot), robot arm hardware interface.

## Commands

### Perception (main package)

```bash
pip install -r requirements.txt          # install dependencies
pip install -e .                          # install package in editable mode
pytest                                   # run all tests (verbose, short traceback via pytest.ini)
pytest tests/perception                  # perception tests only
pytest tests/utils                       # utility tests only
pytest -k test_segment                   # run tests matching pattern
python examples/live_perception_demo.py  # live camera demo
```

### Isaac Lab simulation

```bash
# Install the extension (requires Isaac Sim already installed)
cd isaaclab/RopeUntyingRobot && python -m pip install -e source/RopeUntyingRobot

# Lint/format (120-char line limit, ruff config in isaaclab pyproject.toml)
cd isaaclab/RopeUntyingRobot && ruff format source scripts && ruff check source scripts

# Run environments
python isaaclab/RopeUntyingRobot/scripts/zero_agent.py --task RopeReach-SO100-v0
python isaaclab/RopeUntyingRobot/scripts/random_agent.py --task RopeReach-SO100-v0

# RL training (PPO via RSL-RL)
cd isaaclab/RopeUntyingRobot && python scripts/rsl_rl/train.py --task RopeReach-SO100-v0 --num_envs 64 --headless

# Evaluation
cd isaaclab/RopeUntyingRobot && python scripts/rsl_rl/play.py --task RopeReach-SO100-v0 --num_envs 4
```

Training logs go to `isaaclab/RopeUntyingRobot/logs/rsl_rl/rope_reach_so100/`.

```bash
# Monitor training
tensorboard --logdir isaaclab/RopeUntyingRobot/logs/rsl_rl/rope_reach_so100

# Sandbox with perception overlay (run perception_viewer.py alongside for live display)
python isaaclab/RopeUntyingRobot/scripts/so100_sandbox.py --show --max_steps 200
python isaaclab/RopeUntyingRobot/scripts/perception_viewer.py  # separate terminal
```

## Architecture

### Perception pipeline (`src/perception/`)

Sequential stages, each function takes numpy arrays in and returns dataclass outputs:

1. **Segmentation** (`rope_segmentation.py`) — HSV color thresholding → `RopeMask` (binary mask + confidence)
2. **Keypoint detection** (`keypoint_detection.py`) — finds endpoints and crossings → `list[Keypoint]`
3. **Keypoint mask** (`keypoint_mask.py`) — labels pixels: 0=background, 1=rope, 2=endpoint, 3=crossing
4. **Skeletonization** (`skeletonization.py`) — Zhang-Suen thinning → 1-pixel centerline, then graph extraction with edges for loops/branches
5. **State estimation** (`state_estimation.py`) — combines outputs into `RopeState` dataclass
6. **Video processor** (`video_processor.py`) — orchestrates the full pipeline on live camera frames → `ProcessingResult`

### Isaac Lab simulation (`isaaclab/RopeUntyingRobot/`)

- **Scene**: SO-100 6-DOF arm (5 arm joints + 1 gripper) + table + deformable FEM rope + camera
- **Robot asset pipeline**: `so100.urdf` → mesh-path repair → optional collision strip → `UrdfConverter` → `so100_from_urdf.usd` (auto-generated on first run, cached afterward). Config in `assets/so100_config.py` with two factory functions: `create_so100_articulation_cfg()` (sandbox) and `create_so100_rl_articulation_cfg()` (RL, separate arm/gripper actuator groups)
- **Environment**: `RopeReach-SO100-v0` — 5D joint position actions (scale=0.5, relative to default pose), 21D observations (joint pos/vel rel, rope COM, EE pos via FrameTransformer, last action)
- **Reward**: `reaching_rope` (1-tanh(d/0.1), weight 1.0) + `close_to_rope` (bonus at <2cm, weight 5.0) + action/velocity penalties
- **Resets**: arm joints randomized ×[0.8, 1.2], rope XY offset ±5cm. Episode ends at 10s or if rope falls below z=-0.05m
- **RL**: PPO with [128, 128, 64] MLP, 30 Hz control (120 Hz physics, 4x decimation), 10s episodes (300 steps)
- **MDP definitions**: `source/.../mdp/` — observations.py, rewards.py, terminations.py
- **Environment config**: `ropeuntyingrobot_env_cfg.py`
- Deformable rope cannot use `replicate_physics`; each env has an independent rope instance

Environment phases (only Phase 1 implemented): Reach → Grasp → Straighten → Untie.

### Config and utilities

- `src/configs/perception_config.yaml` — all perception tunable parameters (HSV bounds, morphology, pipeline toggles)
- `src/configs/camera_config.yaml` — device ID, resolution, calibration matrices (calibration is TODO)
- `src/utils/config_loader.py` — YAML load/save
- `src/utils/geometry.py` — 3D transforms, distance, angles
- `src/utils/logging_config.py` — project-wide logging setup

## Code Conventions

From `.cursor/rules/main-rule.mdc`:

- Python >= 3.10, PEP8, **88-char line limit** (main package) / **120-char line limit** (Isaac Lab)
- snake_case functions/variables, PascalCase classes, UPPER_CASE constants
- Type hints mandatory on all public functions; docstrings on all public APIs
- Use dataclasses for structured perception outputs
- All tunable parameters in YAML configs — no hardcoded paths or magic numbers
- Perception functions must accept numpy arrays, never read from disk or display images internally
- Strict separation: perception ≠ control ≠ learning
- Use `logging` module, never `print()`; never silently fail
- Tests must be deterministic, pytest style
- Small composable functions; no speculative features

## Perception Tuning Tips

Common issues and which config keys to adjust in `src/configs/perception_config.yaml`:

- **Mask flickers**: widen `hsv_lower`/`hsv_upper`, increase `blur_kernel_size`
- **False crossings on straight rope**: increase `crossing_detection.min_branch_length` or `min_branch_count`
- **Missing crossings**: decrease `min_branch_length` or `min_area`
- **Skeleton breaks at crossings**: increase `keypoint_detection.skeletonization.pre_close_kernel_size`
- **Missing endpoints**: decrease `endpoint_detection.min_confidence` or `prune_length`

## Key Constraints

- Isaac Sim requires `numpy<2` — the requirements.txt pins numpy 2.x but Isaac Sim needs 1.x
- Perception tests run on CPU without Isaac Sim; simulation scripts require Isaac Sim runtime
- Git LFS is used for asset files (URDF, STL, USD)
- `play.py` auto-exports trained policy as JIT and ONNX for deployment
