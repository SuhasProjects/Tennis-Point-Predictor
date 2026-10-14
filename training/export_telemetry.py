"""Export PostgreSQL/SQLite telemetry into dense NumPy training tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SIDES = {"far": 0, "near": 1}
FEATURE_NAMES = [
    "court_x", "court_y", "confidence", "selection_score",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
] + [f"keypoint_{index}_{axis}" for index in range(17) for axis in ("x", "y", "conf")]


def decode_json(value):
    return json.loads(value) if isinstance(value, str) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"), required=False)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("Set DATABASE_URL or pass --database-url")

    from sqlalchemy import create_engine, text

    engine = create_engine(args.database_url)
    query = text(
        """
        SELECT f.frame_index, f.timestamp_ms, f.homography,
               p.player_id, p.confidence, p.selection_score, p.bbox,
               p.court_x, p.court_y, p.keypoints
        FROM frames AS f
        LEFT JOIN player_detections AS p ON p.frame_id = f.id
        WHERE f.video_id = :video_id
        ORDER BY f.frame_index, p.player_id
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(query, {"video_id": args.video_id}).mappings().all()
    engine.dispose()
    if not rows:
        raise SystemExit(f"No frames found for video_id={args.video_id}")

    frame_indices = sorted({int(row["frame_index"]) for row in rows})
    lookup = {frame_index: index for index, frame_index in enumerate(frame_indices)}
    frame_count = len(frame_indices)
    features = np.full((frame_count, 2, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    player_mask = np.zeros((frame_count, 2), dtype=bool)
    homographies = np.full((frame_count, 3, 3), np.nan, dtype=np.float64)
    timestamps_ms = np.full(frame_count, np.nan, dtype=np.float64)

    for row in rows:
        frame_offset = lookup[int(row["frame_index"])]
        timestamps_ms[frame_offset] = float(row["timestamp_ms"])
        if row["homography"] is not None:
            homographies[frame_offset] = np.asarray(
                decode_json(row["homography"]), dtype=np.float64
            )
        if row["player_id"] is None:
            continue
        side_offset = SIDES[row["player_id"]]
        bbox = np.asarray(decode_json(row["bbox"]), dtype=np.float32).reshape(4)
        keypoints = np.asarray(decode_json(row["keypoints"]), dtype=np.float32).reshape(17, 3)
        features[frame_offset, side_offset] = np.concatenate(
            [
                np.asarray(
                    [row["court_x"], row["court_y"], row["confidence"], row["selection_score"]],
                    dtype=np.float32,
                ),
                bbox,
                keypoints.reshape(-1),
            ]
        )
        player_mask[frame_offset, side_offset] = True

    output = args.output or Path("datasets") / "telemetry" / f"video_{args.video_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        timestamps_ms=timestamps_ms,
        homographies=homographies,
        player_features=features,
        player_mask=player_mask,
        player_order=np.asarray(["far", "near"]),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    print(f"Exported {frame_count} frames to {output.resolve()}")


if __name__ == "__main__":
    main()
