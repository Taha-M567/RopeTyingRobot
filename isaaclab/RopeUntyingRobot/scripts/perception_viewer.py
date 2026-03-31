"""Live viewer for the SO-100 perception pipeline output.

Run this in a separate terminal while so100_sandbox.py is running
with --show.  Uses OpenCV (already a project dependency).

Displays a dual-view composite: top-down (perception overlay) on the
left and side-view (arm pose) on the right.

Usage:
    python scripts/perception_viewer.py
    python scripts/perception_viewer.py --image path/to/latest.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "so100_preprocess"
    / "latest.png"
)

WINDOW_NAME = "SO100 Perception (Dual View)"
MAX_DISPLAY_WIDTH = 1920


def main() -> None:
    parser = argparse.ArgumentParser(description="Live perception viewer.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_IMAGE),
        help="Path to the latest.png written by so100_sandbox.py --show.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target refresh rate.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    delay_ms = max(1, 1000 // args.fps)
    last_mtime = 0.0

    print(f"Watching: {image_path}")
    print("Press 'q' in the viewer window to quit.")

    if not image_path.exists():
        print("Waiting for sandbox to write first frame...")

    while True:
        if image_path.exists():
            mtime = image_path.stat().st_mtime
            if mtime != last_mtime:
                img = cv2.imread(str(image_path))
                if img is not None:
                    h_img, w_img = img.shape[:2]
                    if w_img > MAX_DISPLAY_WIDTH:
                        scale = MAX_DISPLAY_WIDTH / w_img
                        img = cv2.resize(
                            img,
                            (MAX_DISPLAY_WIDTH, int(h_img * scale)),
                        )
                    cv2.imshow(WINDOW_NAME, img)
                    last_mtime = mtime

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
