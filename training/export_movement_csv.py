"""Export cleaned coordinate rows from SQLite to a portable CSV file."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMNS = (
    "video_id",
    "frame_index",
    "timestamp_ms",
    "segment_id",
    "court_valid",
    "near_clean_x",
    "near_clean_y",
    "near_speed_mps",
    "near_imputed",
    "far_clean_x",
    "far_clean_y",
    "far_speed_mps",
    "far_imputed",
    "ball_clean_x",
    "ball_clean_y",
    "ball_speed_pxps",
    "ball_imputed",
)


def export_csv(database: Path, output: Path, video_id: int | None = None) -> int:
    connection = sqlite3.connect(database)
    query = f"SELECT {','.join(COLUMNS)} FROM coordinates"
    parameters: tuple = ()
    if video_id is not None:
        query += " WHERE video_id=?"
        parameters = (video_id,)
    query += " ORDER BY video_id,frame_index"
    rows = connection.execute(query, parameters)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(row)
            count += 1
    connection.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=PROJECT_ROOT / "outputs" / "movement.sqlite"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "movement.csv")
    parser.add_argument("--video-id", type=int)
    args = parser.parse_args()
    count = export_csv(args.database, args.output, args.video_id)
    print(f"Exported {count} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
