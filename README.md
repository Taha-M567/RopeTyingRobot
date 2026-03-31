# Rope Untying Robot

VIP research project W26. A robotic system that learns to untie knots in
ropes using reinforcement learning in simulation and an OpenCV perception
pipeline for real-world rope state estimation.

## Core Technologies

- **Isaac Lab / Isaac Sim** — GPU-accelerated physics simulation and RL training
- **OpenCV** — Rope perception, segmentation, and tracking
- **RSL-RL (PPO)** — Reinforcement learning algorithm
- **Python >= 3.10**

## Project Structure

```
RopeUntyingRobot/
├── src/                              # Perception + hardware + utilities
│   ├── perception/                   # OpenCV rope analysis pipeline
│   │   ├── rope_segmentation.py      # Step 1: find rope pixels (HSV/edge)
│   │   ├── keypoint_detection.py     # Step 2: find endpoints & crossings
│   │   ├── keypoint_mask.py          # Step 2b: label pixels by class
│   │   ├── skeletonization.py        # Step 3: thin to centerline + graph
│   │   ├── state_estimation.py       # Step 4: summarise rope state
│   │   ├── crossing_analysis.py      # Crossing-specific analysis
│   │   ├── visualization.py          # Draw overlays on frames
│   │   └── video_processor.py        # Orchestrate pipeline on live video
│   ├── hardware/
│   │   └── camera.py                 # USB camera capture + calibration stub
│   ├── utils/
│   │   ├── config_loader.py          # Load / save YAML configs
│   │   ├── geometry.py               # 3D math helpers
│   │   └── logging_config.py         # Project-wide logging setup
│   └── configs/
│       ├── perception_config.yaml    # All perception tuning parameters
│       └── camera_config.yaml        # Camera device and calibration
│
├── isaaclab/RopeUntyingRobot/        # Isaac Lab simulation extension
│   ├── assets/                       # Robot + rope model files
│   │   ├── so100_config.py           # SO-100 URDF→USD + ArticulationCfg
│   │   ├── rope_config.py            # Rope URDF→USD + knot config I/O
│   │   ├── generate_rope_urdf.py     # Generates articulated rope chain URDF
│   │   ├── knot_configs/             # Saved knot configurations (YAML)
│   │   │   └── straight.yaml         # Default straight rope
│   │   ├── so100.urdf                # SO-100 robot description
│   │   └── *.stl                     # 3D mesh files for each robot link
│   ├── scripts/
│   │   ├── so100_sandbox.py          # Interactive sandbox (dual-camera + perception)
│   │   ├── perception_viewer.py      # Live dual-view OpenCV viewer
│   │   ├── random_agent.py           # Sanity-check: random actions
│   │   ├── zero_agent.py             # Sanity-check: zero actions
│   │   └── rsl_rl/
│   │       ├── train.py              # PPO training launcher
│   │       └── play.py               # Policy evaluation + JIT/ONNX export
│   └── source/RopeUntyingRobot/      # Installable Python extension
│       └── RopeUntyingRobot/
│           └── tasks/manager_based/ropeuntyingrobot/
│               ├── ropeuntyingrobot_env_cfg.py  # Scene + MDP config
│               ├── mdp/              # Observations, rewards, terminations, events
│               └── agents/           # PPO hyperparameters
│
├── tests/                            # Pytest suite (CPU, no Isaac Sim needed)
│   ├── perception/                   # Segmentation, keypoints, skeleton tests
│   └── utils/                        # Geometry tests
│
├── examples/
│   └── live_perception_demo.py       # Live USB camera perception demo
│
├── Workflow.md                       # Full technical documentation
├── CLAUDE.md                         # AI assistant project context
├── requirements.txt                  # Python dependencies
└── pytest.ini                        # Pytest configuration
```

## Installation

**Perception only (no GPU required):**
```bash
pip install -r requirements.txt
pip install -e .
pytest                              # Run all tests
```

**Simulation + RL (requires Isaac Sim):**
```bash
# Install Isaac Lab extension
cd isaaclab/RopeUntyingRobot && pip install -e source/RopeUntyingRobot
```

## Quick Start

**Run perception tests:**
```bash
pytest tests/ -v
```

**Run live perception on a USB camera:**
```bash
python examples/live_perception_demo.py
```

**Run the interactive sandbox (requires Isaac Lab):**
```bash
# Terminal 1: sandbox with dual-camera view
python isaaclab/RopeUntyingRobot/scripts/so100_sandbox.py --show --max_steps 200

# Terminal 2: live viewer
python isaaclab/RopeUntyingRobot/scripts/perception_viewer.py
```

**Train the RL agent:**
```bash
cd isaaclab/RopeUntyingRobot
python scripts/rsl_rl/train.py --task RopeReach-SO100-v0 --num_envs 64 --headless
```

**Evaluate a trained policy:**
```bash
cd isaaclab/RopeUntyingRobot
python scripts/rsl_rl/play.py --task RopeReach-SO100-v0 --num_envs 4
```

## Documentation

See [Workflow.md](Workflow.md) for full technical documentation covering the
perception pipeline, Isaac Lab simulation, RL environment, articulated rope
model, knot configuration system, and training workflow.
