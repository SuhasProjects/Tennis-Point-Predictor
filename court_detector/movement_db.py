"""Compact SQLite storage and causal cleaning for movement coordinates."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY, source_path TEXT NOT NULL, fps REAL NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS coordinates (
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL, timestamp_ms REAL NOT NULL, segment_id INTEGER NOT NULL,
    court_valid INTEGER NOT NULL, scene_cut INTEGER NOT NULL,
    near_x REAL, near_y REAL, near_conf REAL,
    far_x REAL, far_y REAL, far_conf REAL,
    ball_x REAL, ball_y REAL, ball_conf REAL,
    near_clean_x REAL, near_clean_y REAL, near_speed_mps REAL, near_imputed INTEGER NOT NULL DEFAULT 0,
    far_clean_x REAL, far_clean_y REAL, far_speed_mps REAL, far_imputed INTEGER NOT NULL DEFAULT 0,
    ball_clean_x REAL, ball_clean_y REAL, ball_speed_pxps REAL, ball_imputed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (video_id, frame_index)
);
CREATE INDEX IF NOT EXISTS ix_coordinates_video_time ON coordinates(video_id, frame_index);
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_path TEXT NOT NULL, horizon_seconds REAL NOT NULL, history_seconds REAL NOT NULL,
    train_samples INTEGER NOT NULL, validation_samples INTEGER NOT NULL,
    model_mae_m REAL, baseline_mae_m REAL, details_json TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(SCHEMA)
    return connection


def add_video(
    connection: sqlite3.Connection, source: Path, fps: float, width: int, height: int
) -> int:
    cursor = connection.execute(
        "INSERT INTO videos(source_path,fps,width,height) VALUES(?,?,?,?)",
        (str(source.resolve()), float(fps), int(width), int(height)),
    )
    connection.commit()
    return int(cursor.lastrowid)


def clean_video(connection: sqlite3.Connection, video_id: int, *, max_gap: int = 4) -> None:
    """Reject jumps, causally smooth valid points, and forward-fill only short gaps."""
    video = connection.execute("SELECT fps FROM videos WHERE id=?", (video_id,)).fetchone()
    if video is None:
        raise ValueError(f"Unknown video id {video_id}")
    fps = float(video[0])
    rows = connection.execute(
        "SELECT frame_index,segment_id,near_x,near_y,far_x,far_y,ball_x,ball_y "
        "FROM coordinates WHERE video_id=? ORDER BY frame_index",
        (video_id,),
    ).fetchall()
    state: dict[str, tuple[np.ndarray, int, int] | None] = {"near": None, "far": None, "ball": None}
    updates = []
    for row in rows:
        frame, segment = int(row[0]), int(row[1])
        result: list[float | int | None] = []
        for offset, name in ((2, "near"), (4, "far"), (6, "ball")):
            point = (
                None if row[offset] is None else np.asarray(row[offset : offset + 2], dtype=float)
            )
            previous = state[name]
            imputed = 0
            speed = None
            if previous is not None and previous[2] != segment:
                previous = None
                state[name] = None
            # Tennis sprint speeds above roughly 14 m/s are tracking jumps.
            # Court-reference units are centimeters. Ball coordinates remain
            # image pixels and use a separate permissive per-frame limit.
            limit = 55.0 if name == "ball" else 1400.0 / fps
            if point is not None and previous is not None:
                elapsed = max(frame - previous[1], 1)
                if np.linalg.norm(point - previous[0]) / elapsed > limit:
                    point = None
            if point is None and previous is not None and frame - previous[1] <= max_gap:
                point = previous[0].copy()
                imputed = 1
            elif point is not None:
                if previous is not None:
                    elapsed = max((frame - previous[1]) / fps, 1e-6)
                    point = 0.65 * previous[0] + 0.35 * point
                    distance = float(np.linalg.norm(point - previous[0]))
                    speed = distance / elapsed * (1.0 if name == "ball" else 0.01)
                state[name] = (point.copy(), frame, segment)
            result.extend(
                (
                    None if point is None else float(point[0]),
                    None if point is None else float(point[1]),
                    speed,
                    imputed,
                )
            )
        updates.append((*result, video_id, frame))
    connection.executemany(
        "UPDATE coordinates SET near_clean_x=?,near_clean_y=?,near_speed_mps=?,near_imputed=?,"
        "far_clean_x=?,far_clean_y=?,far_speed_mps=?,far_imputed=?,"
        "ball_clean_x=?,ball_clean_y=?,ball_speed_pxps=?,ball_imputed=? "
        "WHERE video_id=? AND frame_index=?",
        updates,
    )
    connection.commit()


def record_model_run(connection: sqlite3.Connection, values: dict) -> None:
    connection.execute(
        "INSERT INTO model_runs(model_path,horizon_seconds,history_seconds,train_samples,"
        "validation_samples,model_mae_m,baseline_mae_m,details_json) VALUES(?,?,?,?,?,?,?,?)",
        (
            values["model_path"],
            values["horizon_seconds"],
            values["history_seconds"],
            values["train_samples"],
            values["validation_samples"],
            values["model_mae_m"],
            values["baseline_mae_m"],
            json.dumps(values, sort_keys=True),
        ),
    )
    connection.commit()


def summarize(connection: sqlite3.Connection) -> dict:
    videos = []
    for video_id, source, fps in connection.execute(
        "SELECT id,source_path,fps FROM videos ORDER BY id"
    ):
        row = connection.execute(
            "SELECT COUNT(*),SUM(near_clean_x IS NOT NULL),SUM(far_clean_x IS NOT NULL),"
            "SUM(ball_clean_x IS NOT NULL),COUNT(DISTINCT segment_id),"
            "MAX(near_speed_mps),MAX(far_speed_mps),MAX(ball_speed_pxps) "
            "FROM coordinates WHERE video_id=?",
            (video_id,),
        ).fetchone()
        frames = int(row[0])
        videos.append(
            {
                "video_id": int(video_id),
                "source": source,
                "fps": float(fps),
                "frames": frames,
                "segments": int(row[4]),
                "near_coverage": float(row[1] or 0) / max(frames, 1),
                "far_coverage": float(row[2] or 0) / max(frames, 1),
                "ball_coverage": float(row[3] or 0) / max(frames, 1),
                "peak_near_mps": row[5],
                "peak_far_mps": row[6],
                "peak_ball_pxps": row[7],
            }
        )
    model = connection.execute(
        "SELECT details_json FROM model_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {"videos": videos, "latest_model": None if model is None else json.loads(model[0])}
