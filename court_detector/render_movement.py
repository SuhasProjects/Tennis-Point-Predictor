"""Render tracked coordinates and trained future predictions onto a video."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from training.train_movement import _baseline_at, _feature_at

from .court_reference import CourtReference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = CourtReference()


def render(database: Path, model_path: Path, video_id: int, output: Path) -> None:
    connection = sqlite3.connect(database)
    video = connection.execute(
        "SELECT source_path,fps FROM videos WHERE id=?", (video_id,)
    ).fetchone()
    if video is None:
        raise ValueError(f"Unknown video {video_id}")
    rows = connection.execute(
        "SELECT frame_index,segment_id,near_clean_x,near_clean_y,far_clean_x,far_clean_y,"
        "ball_clean_x,ball_clean_y,near_speed_mps,far_speed_mps,ball_speed_pxps "
        "FROM coordinates WHERE video_id=? ORDER BY frame_index",
        (video_id,),
    ).fetchall()
    connection.close()
    if not rows:
        raise ValueError(f"Video {video_id} has no coordinate rows")
    source, fps = Path(video[0]), float(video[1])
    values = {int(row[0]): row[:8] for row in rows}
    full_rows = {int(row[0]): row for row in rows}
    model = np.load(model_path)
    metadata = json.loads(str(model["metadata"]))
    predictions = {}
    for frame in values:
        features = _feature_at(values, frame, fps, metadata["history_seconds"])
        if features is None:
            continue
        baseline = _baseline_at(
            values, frame, fps, metadata["history_seconds"], metadata["horizon_seconds"]
        )
        design = np.append((features - model["mean"]) / model["scale"], 1.0)
        predictions[frame] = baseline + design @ model["weights"]

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open {source}")
    first_frame, last_frame = min(full_rows), max(full_rows)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not write {output}")
    near_trail: deque[tuple[int, int]] = deque(maxlen=max(30, round(fps * 2)))
    far_trail: deque[tuple[int, int]] = deque(maxlen=max(30, round(fps * 2)))

    x0, y0, panel_width, panel_height = 1010, 45, 240, 500
    scale_x = panel_width / REFERENCE.court_total_width
    scale_y = panel_height / REFERENCE.court_total_height

    def map_court(x: float, y: float) -> tuple[int, int]:
        return round(x0 + x * scale_x), round(y0 + y * scale_y)

    frame_index = first_frame
    try:
        while frame_index <= last_frame:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (1280, 720))
            row = full_rows.get(frame_index)
            if row is not None:
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, (x0, y0), (x0 + panel_width, y0 + panel_height), (15, 45, 24), -1
                )
                cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
                for start, end in np.asarray(
                    REFERENCE.get_important_lines(), dtype=np.float32
                ).reshape(-1, 2, 2):
                    cv2.line(
                        frame, map_court(*start), map_court(*end), (220, 235, 225), 1, cv2.LINE_AA
                    )
                for offset, trail, color in (
                    (2, near_trail, (50, 235, 90)),
                    (4, far_trail, (20, 175, 255)),
                ):
                    if row[offset] is not None:
                        trail.append(map_court(row[offset], row[offset + 1]))
                    for first, second in zip(trail, list(trail)[1:]):
                        cv2.line(frame, first, second, color, 2, cv2.LINE_AA)
                    if trail:
                        cv2.circle(frame, trail[-1], 6, color, -1, cv2.LINE_AA)
                prediction = predictions.get(frame_index)
                if prediction is not None:
                    for offset, color in ((0, (50, 235, 90)), (2, (20, 175, 255))):
                        cv2.circle(
                            frame,
                            map_court(prediction[offset], prediction[offset + 1]),
                            10,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                if row[6] is not None:
                    cv2.circle(
                        frame, (round(row[6]), round(row[7])), 6, (40, 40, 255), -1, cv2.LINE_AA
                    )
                near_speed = "--" if row[8] is None else f"{row[8]:.1f}"
                far_speed = "--" if row[9] is None else f"{row[9]:.1f}"
                cv2.putText(
                    frame,
                    f"near {near_speed} m/s | far {far_speed} m/s",
                    (22, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (245, 255, 245),
                    2,
                    cv2.LINE_AA,
                )
                ball_speed = "--" if row[10] is None else f"{row[10]:.0f}"
                cv2.putText(
                    frame,
                    f"ball image speed {ball_speed} px/s",
                    (22, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (245, 255, 245),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "rings: 0.5 s prediction",
                    (x0, y0 + panel_height + 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 240, 225),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=PROJECT_ROOT / "outputs" / "movement.sqlite"
    )
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "outputs" / "movement_model.npz"
    )
    parser.add_argument("--video-id", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "movement_predictions.mp4"
    )
    args = parser.parse_args()
    render(args.database, args.model, args.video_id, args.output)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
