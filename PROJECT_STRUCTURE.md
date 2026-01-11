# Project Structure Template

This document describes the complete repository structure based on the cursor rules.

## Directory Structure

```
VIP_Project_W26/
│
├── .cursor/
│   └── rules/
│       └── main-rule.mdc          # Project coding rules
│
├── src/                            # Main source code
│   │
│   ├── perception/                 # OpenCV-based perception
│   │   ├── __init__.py
│   │   ├── rope_segmentation.py   # Segment rope from images
│   │   ├── keypoint_detection.py   # Detect endpoints, crossings, knots
│   │   ├── skeletonization.py      # Extract rope centerline
│   │   └── state_estimation.py     # Estimate rope state
│   │
│   ├── control/                    # Robot control
│   │   ├── __init__.py
│   │   ├── trajectory_planner.py   # Motion planning
│   │   ├── policy_interface.py     # LeRobot policy wrapper
│   │   └── robot_controller.py     # Robot control interface
│   │
│   ├── learning/                   # LeRobot training
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset handling with versioning
│   │   └── trainer.py              # Policy training with logging
│   │
│   ├── simulation/                 # Isaac Sim
│   │   ├── __init__.py
│   │   └── isaac_sim_env.py        # Simulation environment
│   │
│   ├── hardware/                   # Real hardware
│   │   ├── __init__.py
│   │   ├── camera.py               # Camera interface
│   │   └── robot_arm.py            # Robot arm interface
│   │
│   ├── utils/                      # Shared utilities
│   │   ├── __init__.py
│   │   ├── geometry.py             # Geometric operations
│   │   ├── logging_config.py       # Logging setup
│   │   └── config_loader.py        # YAML/JSON config loading
│   │
│   └── configs/                    # Configuration files
│       ├── camera_config.yaml       # Camera settings
│       ├── robot_config.yaml        # Robot parameters
│       ├── perception_config.yaml   # Perception algorithm params
│       ├── training_config.yaml     # Training hyperparameters
│       └── simulation_config.yaml   # Isaac Sim settings
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── perception/                 # Perception tests
│   │   └── test_rope_segmentation.py
│   ├── control/                    # Control tests
│   ├── learning/                   # Learning tests
│   └── utils/                      # Utility tests
│       └── test_geometry.py
│
├── data/                           # Data directories
│   ├── raw/                        # Raw data
│   │   └── .gitkeep
│   └── processed/                  # Processed datasets
│       └── .gitkeep
│
├── logs/                           # Log files
│   └── .gitkeep
│
├── .gitignore                      # Git ignore rules
├── pytest.ini                      # Pytest configuration
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # Project documentation
└── PROJECT_STRUCTURE.md            # This file
```

## Module Responsibilities

### Perception (`src/perception/`)
- **Purpose**: Computer vision using OpenCV
- **Responsibilities**:
  - Rope segmentation from images
  - Keypoint detection (endpoints, crossings, knots)
  - Skeletonization for path extraction
  - State estimation from perception data
- **Outputs**: Well-defined dataclasses with documented coordinate frames
- **Rules**: Never read from disk internally, never display images unless requested

### Control (`src/control/`)
- **Purpose**: Robot motion control and policy execution
- **Responsibilities**:
  - Trajectory planning with bounds checking
  - Policy interface for LeRobot models
  - Robot control abstraction
- **Rules**: All motion commands must have bounds checking, emergency stop support

### Learning (`src/learning/`)
- **Purpose**: Dataset management and policy training
- **Responsibilities**:
  - Dataset loading with versioning and metadata
  - Policy training with LeRobot
  - Evaluation and metrics logging
- **Rules**: Reproducible (fixed seeds), clear logging, separate train/inference paths

### Simulation (`src/simulation/`)
- **Purpose**: Isaac Sim integration
- **Responsibilities**:
  - Simulation environment setup
  - Coordinate frame definitions
  - Hardware-mirroring interfaces
- **Rules**: Must mirror real hardware interfaces, no simulation-only hacks

### Hardware (`src/hardware/`)
- **Purpose**: Real hardware interfaces
- **Responsibilities**:
  - Camera capture with calibration
  - Robot arm control
- **Rules**: Hardware-agnostic interface, safety checks

### Utils (`src/utils/`)
- **Purpose**: Shared utilities
- **Responsibilities**:
  - Geometric operations
  - Logging configuration
  - Configuration file loading

## Code Standards

### Naming Conventions
- Variables/Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Files: `snake_case.py`

### Type Hints
- All function arguments must have type hints
- All return values must have type hints
- Use `typing` module for complex types

### Documentation
- Every module: top-level docstring
- Every public function/class: docstring with:
  - Purpose
  - Inputs
  - Outputs
  - Assumptions/constraints

### Style
- PEP8 compliant
- Max line length: 88 characters
- Explicit over implicit
- No magic numbers (use named constants)

## Configuration Files

All configuration files are in YAML format and include:
- Clear comments
- Units for all numeric values
- No hardcoded paths

## Testing

- Unit tests required for:
  - Perception utilities
  - Geometry/rope state logic
  - Dataset loading
- Tests must be deterministic
- Use pytest-style tests

## Data Organization

- `data/raw/`: Raw sensor data, unprocessed
- `data/processed/`: Processed datasets for training
- Datasets must be versioned
- Datasets must include metadata

## Logging

- Use `logging` module (not `print`)
- DEBUG: Internal state
- INFO: Milestones
- WARNING/ERROR: Failures
- Logs saved to `logs/` directory
