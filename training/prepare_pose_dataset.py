"""Create reviewable YOLO-pose pseudo-labels from tennis broadcast video.

Only the homography-filtered near/far detections are labeled. Officials and
spectators remain background examples, which is useful for active-player
fine-tuning. Review labels before treating them as ground truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from court_detector import CourtDetector, PlayerPoseDetector


def yolo_pose_row(detection, width: int, height: int, min_keypoint_conf: float) -> str:
    x1, y1, x2, y2 = detection.bbox
    values = [
        0,
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    for x, y, confidence in detection.keypoints:
        if confidence >= min_keypoint_conf and x > 0 and y > 0:
            values.extend((np.clip(x / width, 0, 1), np.clip(y / height, 0, 1), 2))
        else:
            values.extend((0, 0, 0))
    return " ".join(str(int(value)) if index == 0 else f"{float(value):.6f}" for index, value in enumerate(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "tennis_players")
    parser.add_argument("--stride", type=int, default=25, help="Sample every N source frames")
    parser.add_argument("--val-every", type=int, default=5, help="Every Nth temporal block is validation")
    parser.add_argument("--block-size", type=int, default=250, help="Keep neighboring frames in one split")
    parser.add_argument("--min-players", type=int, default=2, choices=(1, 2))
    parser.add_argument("--min-keypoint-conf", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    detector = CourtDetector(PROJECT_ROOT / "models" / "model_tennis_court_det.pt")
    players = PlayerPoseDetector(PROJECT_ROOT / "models" / "yolov8s-pose.pt", conf=0.10, imgsz=1280)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise ValueError(f"Could not open {args.input}")

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    frame_index = saved = 0
    while args.max_samples is None or saved < args.max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.stride:
            frame_index += 1
            continue
        frame, court = detector.track_frame(frame, frame_index, detect_interval=1)
        # Samples are many source frames apart, so video-rate motion gating is
        # intentionally disabled for dataset generation.
        players.reset_tracking()
        detections = players.detect_players(frame, court.matrix)
        if court.matrix is not None and len(detections) >= args.min_players:
            block = frame_index // max(1, args.block_size)
            val_every = max(2, args.val_every)
            split = "val" if block % val_every == val_every - 1 else "train"
            stem = f"{args.input.stem}_{frame_index:07d}"
            image_path = args.output / "images" / split / f"{stem}.jpg"
            label_path = args.output / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
            rows = [yolo_pose_row(item, frame.shape[1], frame.shape[0], args.min_keypoint_conf) for item in detections]
            label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            saved += 1
            if saved % 50 == 0:
                print(f"Saved {saved} reviewed-candidate samples")
        frame_index += 1
    cap.release()
    print(f"Saved {saved} samples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
