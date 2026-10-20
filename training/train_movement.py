"""Train a compact ridge model to predict both players' future positions."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _feature_at(values: dict[int, tuple], frame: int, fps: float, history_seconds: float):
    history = max(2, round(history_seconds * fps))
    lags = sorted({0, history // 3, 2 * history // 3, history})
    past = [values.get(frame - lag) for lag in lags]
    if any(item is None for item in past):
        return None
    current = past[0]
    if any(item[1] != current[1] for item in past):
        return None
    if any(any(value is None for value in item[2:6]) for item in past):
        return None
    features: list[float] = []
    for item in past:
        features.extend(float(value) for value in item[2:6])
        features.extend(0.0 if value is None else float(value) for value in item[6:8])
        features.append(float(item[6] is not None and item[7] is not None))
    return np.asarray(features, dtype=float)


def _baseline_at(
    values: dict[int, tuple],
    frame: int,
    fps: float,
    history_seconds: float,
    horizon_seconds: float,
) -> np.ndarray:
    history = max(2, round(history_seconds * fps))
    horizon = max(1, round(horizon_seconds * fps))
    current = np.asarray(values[frame][2:6], dtype=float)
    old = np.asarray(values[frame - history][2:6], dtype=float)
    return current + (current - old) * (horizon / history)


def _video_samples(rows: list[tuple], fps: float, history_seconds: float, horizon_seconds: float):
    values = {int(row[0]): row for row in rows}
    history = max(2, round(history_seconds * fps))
    horizon = max(1, round(horizon_seconds * fps))
    lags = sorted({0, history // 3, 2 * history // 3, history})
    samples = []
    for frame, row in values.items():
        past = [values.get(frame - lag) for lag in lags]
        future = values.get(frame + horizon)
        if future is None or any(item is None for item in past):
            continue
        if any(item[1] != row[1] for item in [*past, future]):
            continue
        # Rows: frame, segment, near x/y, far x/y, ball x/y.
        if any(any(value is None for value in item[2:6]) for item in [*past, future]):
            continue
        features: list[float] = []
        for item in past:
            features.extend(float(value) for value in item[2:6])
            features.extend(0.0 if value is None else float(value) for value in item[6:8])
            features.append(float(item[6] is not None and item[7] is not None))
        target = np.asarray(future[2:6], dtype=float)
        current = np.asarray(row[2:6], dtype=float)
        old = np.asarray(past[-1][2:6], dtype=float)
        baseline = current + (current - old) * (horizon / history)
        samples.append((frame, np.asarray(features), target, baseline))
    return samples


def predict_latest(database: Path, model_path: Path, video_id: int | None = None) -> dict:
    """Predict future near/far coordinates from the latest complete history window."""
    model = np.load(model_path)
    metadata = json.loads(str(model["metadata"]))
    connection = sqlite3.connect(database)
    if video_id is None:
        video = connection.execute("SELECT id,fps FROM videos ORDER BY id DESC LIMIT 1").fetchone()
    else:
        video = connection.execute("SELECT id,fps FROM videos WHERE id=?", (video_id,)).fetchone()
    if video is None:
        connection.close()
        raise ValueError("No matching video")
    selected_id, fps = int(video[0]), float(video[1])
    rows = connection.execute(
        "SELECT frame_index,segment_id,near_clean_x,near_clean_y,far_clean_x,far_clean_y,"
        "ball_clean_x,ball_clean_y FROM coordinates WHERE video_id=? ORDER BY frame_index",
        (selected_id,),
    ).fetchall()
    connection.close()
    if not metadata.get("uses_ball_features", False):
        rows = [(*row[:6], None, None) for row in rows]
    values = {int(item[0]): item for item in rows}
    for frame in sorted(values, reverse=True):
        features = _feature_at(values, frame, fps, metadata["history_seconds"])
        if features is None:
            continue
        design = np.append((features - model["mean"]) / model["scale"], 1.0)
        baseline = _baseline_at(
            values,
            frame,
            fps,
            metadata["history_seconds"],
            metadata["horizon_seconds"],
        )
        prediction = baseline + design @ model["weights"]
        return {
            "video_id": selected_id,
            "source_frame": frame,
            "target_frame": frame + round(metadata["horizon_seconds"] * fps),
            "horizon_seconds": metadata["horizon_seconds"],
            "near": {"court_x": float(prediction[0]), "court_y": float(prediction[1])},
            "far": {"court_x": float(prediction[2]), "court_y": float(prediction[3])},
        }
    raise ValueError("No frame has a complete history window for prediction")


def train(
    database: Path,
    output: Path,
    *,
    history_seconds: float = 0.5,
    horizon_seconds: float = 1.0,
    ridge: float = 10.0,
    use_ball_features: bool = False,
) -> dict:
    connection = sqlite3.connect(database)
    videos = connection.execute("SELECT id,fps FROM videos ORDER BY id").fetchall()
    by_video = []
    for video_id, fps in videos:
        rows = connection.execute(
            "SELECT frame_index,segment_id,near_clean_x,near_clean_y,far_clean_x,far_clean_y,"
            "ball_clean_x,ball_clean_y FROM coordinates WHERE video_id=? ORDER BY frame_index",
            (video_id,),
        ).fetchall()
        if not use_ball_features:
            rows = [(*row[:6], None, None) for row in rows]
        samples = _video_samples(rows, float(fps), history_seconds, horizon_seconds)
        if samples:
            by_video.append((int(video_id), samples))
    connection.close()
    if not by_video:
        raise SystemExit("No complete movement windows. Extract a longer wide-court clip first.")

    if len(by_video) >= 2:
        train_samples = [sample for _, group in by_video[:-1] for sample in group]
        validation = by_video[-1][1]
        split_description = f"video-disjoint: video {by_video[-1][0]} held out"
    else:
        samples = by_video[0][1]
        split = max(1, int(len(samples) * 0.8))
        train_samples, validation = samples[:split], samples[split:]
        split_description = "chronological 80/20 split within one video"
    if not validation or len(train_samples) < 2:
        raise SystemExit("Not enough samples for a train/validation split; use a longer clip.")

    x_train = np.stack([sample[1] for sample in train_samples])
    y_train = np.stack([sample[2] for sample in train_samples])
    train_baselines = np.stack([sample[3] for sample in train_samples])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (x_train - mean) / scale
    design = np.column_stack((normalized, np.ones(len(normalized))))
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[-1, -1] = 0.0
    # Learn only the correction to a strong constant-velocity baseline. This
    # generalizes better across ends/cameras than regressing absolute position.
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ (y_train - train_baselines))

    x_val = np.stack([sample[1] for sample in validation])
    y_val = np.stack([sample[2] for sample in validation])
    baselines = np.stack([sample[3] for sample in validation])
    corrections = np.column_stack(((x_val - mean) / scale, np.ones(len(x_val)))) @ weights
    predictions = baselines + corrections
    model_error_by_player = np.linalg.norm((predictions - y_val).reshape(-1, 2, 2), axis=2) * 0.01
    baseline_error_by_player = np.linalg.norm((baselines - y_val).reshape(-1, 2, 2), axis=2) * 0.01
    model_error = model_error_by_player.reshape(-1)
    baseline_error = baseline_error_by_player.reshape(-1)
    improvement = 1.0 - float(model_error.mean()) / max(float(baseline_error.mean()), 1e-9)
    metrics = {
        "model_path": str(output.resolve()),
        "history_seconds": history_seconds,
        "horizon_seconds": horizon_seconds,
        "ridge": ridge,
        "train_samples": len(train_samples),
        "validation_samples": len(validation),
        "model_mae_m": float(model_error.mean()),
        "model_median_error_m": float(np.median(model_error)),
        "model_p90_error_m": float(np.percentile(model_error, 90)),
        "baseline_mae_m": float(baseline_error.mean()),
        "near_mae_m": float(model_error_by_player[:, 0].mean()),
        "far_mae_m": float(model_error_by_player[:, 1].mean()),
        "improvement_over_velocity_pct": improvement * 100.0,
        "split": split_description,
        "feature_count": int(x_train.shape[1]),
        "model_kind": "velocity_residual_ridge",
        "uses_ball_features": use_ball_features,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path = output.with_suffix(".evaluation.csv")
    with evaluation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frame",
                "near_error_m",
                "far_error_m",
                "near_baseline_error_m",
                "far_baseline_error_m",
            )
        )
        for sample, model_errors, baseline_errors in zip(
            validation, model_error_by_player, baseline_error_by_player, strict=True
        ):
            writer.writerow((sample[0], *model_errors, *baseline_errors))
    metrics["evaluation_path"] = str(evaluation_path.resolve())
    np.savez_compressed(
        output, weights=weights, mean=mean, scale=scale, metadata=np.asarray(json.dumps(metrics))
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=PROJECT_ROOT / "outputs" / "movement.sqlite"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "movement_model.npz"
    )
    parser.add_argument("--history-seconds", type=float, default=0.5)
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--use-ball-features", action="store_true")
    parser.add_argument("--predict-latest", action="store_true")
    args = parser.parse_args()
    metrics = train(
        args.database,
        args.output,
        history_seconds=args.history_seconds,
        horizon_seconds=args.horizon_seconds,
        ridge=args.ridge,
        use_ball_features=args.use_ball_features,
    )
    print(json.dumps(metrics, indent=2))
    if args.predict_latest:
        print(json.dumps(predict_latest(args.database, args.output), indent=2))


if __name__ == "__main__":
    main()
