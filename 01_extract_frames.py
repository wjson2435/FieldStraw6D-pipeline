"""
Step 1: extract frames from each sequence's raw video.

Expects, for each sequence folder under --root:
  <root>/<date>/<seq_id>/rgb_raw.mp4

Writes:
  <root>/<date>/<seq_id>/image/000000.png, 000001.png, ...

The first/last CUT_SECONDS of each clip are dropped (camera settling / hand
motion at the start and stop of recording), and only every FRAME_SKIP-th
remaining frame is kept.
"""
import argparse
import math
from pathlib import Path

import cv2


def extract(root: Path, frame_skip: int, cut_seconds: float):
    for video_path in root.rglob("rgb_raw.mp4"):
        image_dir = video_path.parent / "image"
        image_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(math.ceil(fps * cut_seconds))
        end_frame = int(total_frames - fps * cut_seconds)

        if start_frame >= end_frame:
            print(f"[SKIP] Video too short: {video_path}")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        saved_idx = 0
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % frame_skip == 0:
                cv2.imwrite(str(image_dir / f"{saved_idx:06d}.png"), frame)
                saved_idx += 1
            frame_idx += 1

        cap.release()
        print(f"[DONE] {video_path} | saved {saved_idx} frames "
              f"(cut {cut_seconds}s front/back, skip={frame_skip})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                    help="Root folder containing <date>/<seq_id>/rgb_raw.mp4")
    p.add_argument("--frame_skip", type=int, default=5)
    p.add_argument("--cut_seconds", type=float, default=1.0)
    args = p.parse_args()
    extract(Path(args.root), args.frame_skip, args.cut_seconds)
