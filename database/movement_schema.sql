-- Compact SQLite schema used by court_detector.movement_analyzer.
-- Video pixels and pose keypoint arrays are intentionally not stored.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    fps REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS coordinates (
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp_ms REAL NOT NULL,
    segment_id INTEGER NOT NULL,
    court_valid INTEGER NOT NULL,
    scene_cut INTEGER NOT NULL,
    near_x REAL, near_y REAL, near_conf REAL,
    far_x REAL, far_y REAL, far_conf REAL,
    ball_x REAL, ball_y REAL, ball_conf REAL,
    near_clean_x REAL, near_clean_y REAL, near_speed_mps REAL,
    near_imputed INTEGER NOT NULL DEFAULT 0,
    far_clean_x REAL, far_clean_y REAL, far_speed_mps REAL,
    far_imputed INTEGER NOT NULL DEFAULT 0,
    ball_clean_x REAL, ball_clean_y REAL, ball_speed_pxps REAL,
    ball_imputed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (video_id, frame_index)
);

CREATE INDEX IF NOT EXISTS ix_coordinates_video_time
    ON coordinates(video_id, frame_index);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_path TEXT NOT NULL,
    horizon_seconds REAL NOT NULL,
    history_seconds REAL NOT NULL,
    train_samples INTEGER NOT NULL,
    validation_samples INTEGER NOT NULL,
    model_mae_m REAL,
    baseline_mae_m REAL,
    details_json TEXT NOT NULL
);
