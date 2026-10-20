import sqlite3
from pathlib import Path

import numpy as np

from court_detector.movement_db import add_video, clean_video, connect, summarize
from court_detector.movement_report import generate_report
from training.export_movement_csv import export_csv
from training.train_movement import predict_latest, train


def _synthetic_database(path: Path) -> None:
    connection = connect(path)
    video_id = add_video(connection, Path("synthetic.mp4"), 30.0, 1280, 720)
    rows = []
    for frame in range(240):
        near_x = 700 + frame * 0.7
        if frame == 50:
            near_x += 1000  # impossible one-frame jump should be removed
        rows.append(
            (
                video_id,
                frame,
                frame / 30 * 1000,
                0,
                1,
                0,
                near_x,
                2500 + frame * 0.1,
                0.9,
                900 - frame * 0.4,
                900 + frame * 0.2,
                0.9,
                300 + frame * 2.0,
                200 + frame,
                0.8,
            )
        )
    connection.executemany(
        "INSERT INTO coordinates(video_id,frame_index,timestamp_ms,segment_id,court_valid,"
        "scene_cut,near_x,near_y,near_conf,far_x,far_y,far_conf,ball_x,ball_y,ball_conf) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    connection.commit()
    clean_video(connection, video_id)
    connection.close()


def test_compact_database_and_training(tmp_path: Path) -> None:
    database = tmp_path / "movement.sqlite"
    model = tmp_path / "movement_model.npz"
    _synthetic_database(database)
    metrics = train(database, model)
    assert metrics["train_samples"] > 100
    assert metrics["validation_samples"] > 20
    assert np.isfinite(metrics["model_mae_m"])
    assert model.is_file()
    assert Path(metrics["evaluation_path"]).is_file()
    prediction = predict_latest(database, model)
    assert prediction["target_frame"] > prediction["source_frame"]
    assert set(prediction) >= {"near", "far"}
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE coordinates SET ball_clean_x=ball_clean_x+9999,ball_clean_y=ball_clean_y-9999"
    )
    connection.commit()
    connection.close()
    # The production model is player-only even when tracked ball data exists.
    assert predict_latest(database, model) == prediction
    report = tmp_path / "report.html"
    generate_report(database, report)
    assert "Tennis Movement Analyzer" in report.read_text(encoding="utf-8")
    csv_path = tmp_path / "movement.csv"
    assert export_csv(database, csv_path) == 240
    assert csv_path.read_text(encoding="utf-8").count("\n") == 241
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM coordinates").fetchone()[0] == 240
    assert (
        connection.execute("SELECT near_imputed FROM coordinates WHERE frame_index=50").fetchone()[
            0
        ]
        == 1
    )
    summary = summarize(connection)
    assert summary["videos"][0]["near_coverage"] == 1.0
