# Rope Tying Robot

VIP research project W26.

A robotic system capable of tying and untying ropes using learning-based manipulation.

## Core Technologies

- **LeRobot**: Imitation learning and reinforcement learning for manipulation
- **OpenCV**: Rope perception, segmentation, and tracking
- **Python**: Primary programming language (Python >= 3.10)

## Project Structure

```
VIP_Project_W26/
├── src/
│   ├── perception/          # OpenCV-based rope perception
│   │   ├── rope_segmentation.py    # Rope segmentation from images
│   │   ├── keypoint_detection.py   # Endpoint and crossing detection
│   │   ├── skeletonization.py      # Rope centerline extraction
│   │   └── state_estimation.py     # Rope state estimation
│   │
│   ├── hardware/            # Real hardware interfaces
│   │   ├── camera.py              # Camera interface
│   │   └── robot_arm.py           # Robot arm interface
│   │
│   ├── utils/               # Shared utilities
│   │   ├── geometry.py            # Geometric operations
│   │   ├── logging_config.py      # Logging setup
│   │   └── config_loader.py       # Configuration loading
│   │
│   └── configs/             # Configuration files (YAML)
│       ├── camera_config.yaml
│       ├── robot_config.yaml
│       ├── perception_config.yaml
│       ├── training_config.yaml
│       └── simulation_config.yaml
│
├── tests/                   # Test suite
│   ├── perception/         # Perception module tests
│   ├── control/            # Control module tests
│   ├── learning/           # Learning module tests
│   └── utils/              # Utility tests
│
├── requirements.txt        # Python dependencies
├── pytest.ini            # Pytest configuration
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Taha-M567/RopeTyingRobot.git
cd VIP_Project_W26
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install additional dependencies for LeRobot and Isaac Sim as needed.

## Usage

### Running Tests

```bash
pytest
```

### Configuration

Edit configuration files in `src/configs/` to customize:
- Camera settings
- Robot parameters
- Perception algorithms
- Training hyperparameters
- Simulation setup
