"""Setup script for Rope Tying Robot project."""

from setuptools import find_packages, setup

setup(
    name="rope-tying-robot",
    version="0.1.0",
    description="Robotic system for tying and untying ropes using learning-based manipulation",
    author="VIP Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0.0",
        "opencv-python>=4.12.0",
        "torch>=2.9.0",
        "torchvision>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
    },
)
