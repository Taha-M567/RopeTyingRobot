"""Live viewer for the SO-100 perception pipeline output.

Run this in a separate terminal while so100_sandbox.py is running
with --show.  Uses tkinter (stdlib) so no extra packages are needed.

Usage:
    python scripts/perception_viewer.py
    python scripts/perception_viewer.py --image path/to/latest.png
"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "so100_preprocess"
    / "latest.png"
)


class PerceptionViewer:
    """Tkinter window that polls a PNG file and displays it live."""

    def __init__(self, image_path: Path, refresh_ms: int = 33) -> None:
        self.image_path = image_path
        self.refresh_ms = refresh_ms
        self.last_mtime = 0.0

        self.root = tk.Tk()
        self.root.title("SO100 Perception")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<q>", lambda _: self.root.destroy())

        self._poll()
        self.root.mainloop()

    def _poll(self) -> None:
        try:
            if self.image_path.exists():
                mtime = self.image_path.stat().st_mtime
                if mtime != self.last_mtime:
                    img = Image.open(self.image_path)
                    self._tk_img = ImageTk.PhotoImage(img)
                    self.label.configure(image=self._tk_img)
                    self.root.title(
                        f"SO100 Perception  ({img.width}x{img.height})"
                    )
                    self.last_mtime = mtime
        except Exception:
            pass  # file may be mid-write
        self.root.after(self.refresh_ms, self._poll)


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
    refresh_ms = max(1, 1000 // args.fps)

    print(f"Watching: {image_path}")
    print("Press 'q' in the window to quit.")

    PerceptionViewer(image_path, refresh_ms)


if __name__ == "__main__":
    main()
