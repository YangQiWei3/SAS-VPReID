"""
Simple image viewer for DetXreID-style datasets (two-level folder layout).

Features:
- Auto-play through all images with adjustable interval.
- Pause/Resume playback.
- Manual Previous/Next controls.

Run: python datalook.py --root <dataset_root> [--width 960 --height 720]
"""

import argparse
import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def start_image_scan(root_dir: str, on_image, on_done) -> None:
    """Scan images in a background thread and stream paths via callbacks."""

    def worker() -> None:
        try:
            if not os.path.isdir(root_dir):
                raise FileNotFoundError(f"Dataset root not found: {root_dir}")
            for cur, _, files in os.walk(root_dir):
                for fname in sorted(files):
                    _, ext = os.path.splitext(fname)
                    if ext.lower() not in IMAGE_EXTS:
                        continue
                    on_image(os.path.join(cur, fname))
        except Exception as exc:  # noqa: BLE001
            on_done(exc)
            return
        on_done(None)

    threading.Thread(target=worker, daemon=True).start()


class ImagePlayer:
    def __init__(
        self,
        root_dir: str,
        width: int,
        height: int,
        interval_ms: int = 1,
        scale: float = 3.0,
    ):
        self.image_paths: list[str] = []
        self.total = 0
        self.idx = 0
        self.interval_ms = interval_ms
        self.paused = False
        self.after_job = None
        self.display_size = (width, height)
        self.scale = scale
        self.loading_done = False
        self.loading_error: Exception | None = None
        self.root_dir = root_dir

        self.root = tk.Tk()
        self.root.title("DetXreID Viewer")
        # Bind space key to pause/resume.
        self.root.bind("<space>", lambda _event: self.toggle_pause())
        self._build_ui()
        self.info_label.configure(text="Scanning images ...")

        start_image_scan(self.root_dir, self._on_image_found, self._on_scan_done)
        self._wait_first_image()

    def _build_ui(self) -> None:
        self.img_label = tk.Label(self.root, bd=2, relief="groove")
        self.img_label.pack(padx=10, pady=10)

        self.info_label = tk.Label(self.root, text="", anchor="w", justify="left")
        self.info_label.pack(fill="x", padx=10)

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=5)

        self.pause_btn = tk.Button(controls, text="Pause", width=10, command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=5)

        tk.Button(controls, text="Previous", width=10, command=self.prev_image).pack(side="left", padx=5)
        tk.Button(controls, text="Next", width=10, command=self.next_image_manual).pack(side="left", padx=5)

        tk.Button(controls, text="Quit", width=8, command=self.root.destroy).pack(side="right", padx=5)

        interval_frame = tk.Frame(self.root)
        interval_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(interval_frame, text="Interval (ms):").pack(side="left")
        self.interval_var = tk.IntVar(value=self.interval_ms)
        self.interval_slider = tk.Scale(
            interval_frame,
            from_=0.1,
            to=100,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.interval_var,
            command=self.update_interval,
            length=250,
        )
        self.interval_slider.pack(side="left", padx=5)

    def _on_image_found(self, path: str) -> None:
        self.image_paths.append(path)
        self.total = len(self.image_paths)

    def _on_scan_done(self, error: Exception | None) -> None:
        self.loading_done = True
        self.loading_error = error
        if error:
            # Show a dialog once the first UI cycle is available.
            self.root.after(0, lambda: messagebox.showerror("Error", f"Scan failed:\n{error}"))

    def _wait_first_image(self) -> None:
        """Poll until at least one image is available, then start playback."""
        if self.image_paths:
            self.show_image(self.idx)
            self.schedule_next()
            return
        if self.loading_done and self.loading_error:
            # Already shown dialog; stop polling.
            return
        self.root.after(100, self._wait_first_image)

    def show_image(self, idx: int) -> None:
        if not self.image_paths:
            self.info_label.configure(text="No images yet...")
            return

        path = self.image_paths[idx % len(self.image_paths)]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to open image:\n{path}\n{exc}")
            return

        # Uniformly scale by factor without changing aspect ratio.
        w, h = img.size
        scaled_size = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
        img_resized = img.resize(scaled_size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(img_resized)
        self.img_label.configure(image=photo)
        self.img_label.image = photo  # keep reference
        self.info_label.configure(
            text=f"{idx + 1}/{self.total}  {os.path.basename(path)}  "
            f"scaled x{self.scale:.2f} -> {scaled_size[0]}x{scaled_size[1]}"
        )

    def update_interval(self, _event=None) -> None:
        self.interval_ms = int(self.interval_var.get())

    def schedule_next(self) -> None:
        if self.paused or not self.image_paths:
            return
        self.after_job = self.root.after(self.interval_ms, self.next_image)

    def cancel_scheduled(self) -> None:
        if self.after_job is not None:
            self.root.after_cancel(self.after_job)
            self.after_job = None

    def next_image(self) -> None:
        if not self.image_paths:
            return
        self.idx = (self.idx + 1) % len(self.image_paths)
        self.show_image(self.idx)
        self.schedule_next()

    def next_image_manual(self) -> None:
        self.cancel_scheduled()
        self.next_image()

    def prev_image(self) -> None:
        self.cancel_scheduled()
        if not self.image_paths:
            return
        self.idx = (self.idx - 1) % len(self.image_paths)
        self.show_image(self.idx)
        self.schedule_next()

    def toggle_pause(self) -> None:
        if self.paused:
            self.paused = False
            self.pause_btn.configure(text="Pause")
            self.schedule_next()
        else:
            self.paused = True
            self.pause_btn.configure(text="Resume")
            self.cancel_scheduled()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple image player for DetXreID datasets.")
    parser.add_argument("--root", required=True, help="Path to dataset root (recursively scanned).")
    parser.add_argument("--width", type=int, default=960, help="Max display width.")
    parser.add_argument("--height", type=int, default=720, help="Max display height.")
    parser.add_argument("--interval", type=int, default=5, help="Interval between images in ms.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale factor (e.g., 3 for 3x).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player = ImagePlayer(
        root_dir=args.root,
        width=args.width,
        height=args.height,
        interval_ms=args.interval,
        scale=args.scale,
    )
    player.run()


if __name__ == "__main__":
    main()
