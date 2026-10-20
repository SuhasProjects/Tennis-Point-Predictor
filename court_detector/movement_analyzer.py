"""End-to-end tennis movement extraction, cleaning, training, and reporting."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2

from .ball_detector import BallDetector
from .court_detector import CourtDetector
from .movement_db import add_video, clean_video, connect, record_model_run, summarize
from .movement_report import generate_report
from .player_detector import PlayerPoseDetector
from .visualization import draw_court_overlay, draw_minimap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "outputs" / "movement.sqlite"
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "movement_model.npz"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "movement_report.html"


def extract(args: argparse.Namespace) -> int:
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    court_detector = CourtDetector(
        args.court_model,
        smoothing_factor=args.smoothing,
        max_stale_frames=max(10, args.detect_interval * 2),
    )
    players_detector = PlayerPoseDetector(args.pose_model, conf=args.confidence, imgsz=args.imgsz)
    ball_detector = None
    if not args.no_ball:
        if not args.ball_model.is_file():
            raise FileNotFoundError(f"Ball weights not found: {args.ball_model}; use --no-ball")
        ball_detector = BallDetector(args.ball_model, min_confidence=args.ball_confidence)

    connection = connect(args.database)
    video_id = add_video(connection, source, fps, width, height)
    writer = None
    if args.output_video:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not write {args.output_video}")

    insert = """INSERT INTO coordinates(
        video_id,frame_index,timestamp_ms,segment_id,court_valid,scene_cut,
        near_x,near_y,near_conf,far_x,far_y,far_conf,ball_x,ball_y,ball_conf
    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    batch = []
    segment = 0
    processed = 0
    frame_index = int(args.start_frame)
    started = time.perf_counter()
    ball_trail: deque[tuple[int, int]] = deque(maxlen=10)
    try:
        while args.max_frames is None or processed < args.max_frames:
            ok, raw_frame = capture.read()
            if not ok:
                break
            frame, court = court_detector.track_frame(
                raw_frame, frame_index, detect_interval=args.detect_interval
            )
            if court.scene_cut:
                segment += 1
                players_detector.reset_tracking()
                ball_trail.clear()
                if ball_detector:
                    ball_detector.reset()
            players = players_detector.detect_players(frame, court.matrix)
            ball = ball_detector.detect(frame) if ball_detector else None
            by_side = {player.side: player for player in players}
            near, far = by_side.get("near"), by_side.get("far")
            batch.append(
                (
                    video_id,
                    frame_index,
                    frame_index / fps * 1000.0,
                    segment,
                    int(court.matrix is not None),
                    int(court.scene_cut),
                    None
                    if near is None or near.court_position is None
                    else float(near.court_position[0]),
                    None
                    if near is None or near.court_position is None
                    else float(near.court_position[1]),
                    None if near is None else near.confidence,
                    None
                    if far is None or far.court_position is None
                    else float(far.court_position[0]),
                    None
                    if far is None or far.court_position is None
                    else float(far.court_position[1]),
                    None if far is None else far.confidence,
                    None if ball is None else ball.x,
                    None if ball is None else ball.y,
                    None if ball is None else ball.confidence,
                )
            )
            if len(batch) >= 100:
                connection.executemany(insert, batch)
                connection.commit()
                batch.clear()
            if writer:
                draw_court_overlay(frame, court.matrix)
                players_detector.annotate(frame, players)
                draw_minimap(frame, players)
                if ball:
                    ball_trail.append((round(ball.x), round(ball.y)))
                for index, point in enumerate(ball_trail):
                    cv2.circle(frame, point, max(2, index // 2), (40, 40, 255), -1, cv2.LINE_AA)
                status = f"frame {frame_index} | court {'yes' if court.matrix is not None else 'no'} | ball {'yes' if ball else 'no'}"
                cv2.putText(
                    frame,
                    status,
                    (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
            processed += 1
            frame_index += 1
            if processed % 100 == 0:
                rate = processed / max(time.perf_counter() - started, 1e-6)
                print(f"Processed {processed} frames ({rate:.1f} FPS)")
    finally:
        if batch:
            connection.executemany(insert, batch)
            connection.commit()
        capture.release()
        if writer:
            writer.release()
    clean_video(connection, video_id, max_gap=args.max_gap)
    connection.close()
    print(
        json.dumps(
            {"video_id": video_id, "frames": processed, "database": str(args.database.resolve())},
            indent=2,
        )
    )
    return video_id


def backfill_ball(args: argparse.Namespace) -> None:
    """Run the slower ball model separately and update an existing video's rows."""
    connection = connect(args.database)
    video = connection.execute(
        "SELECT source_path FROM videos WHERE id=?", (args.video_id,)
    ).fetchone()
    frame_range = connection.execute(
        "SELECT MIN(frame_index),MAX(frame_index) FROM coordinates WHERE video_id=?",
        (args.video_id,),
    ).fetchone()
    if video is None or frame_range is None or frame_range[0] is None:
        connection.close()
        raise ValueError(f"No extracted coordinates for video {args.video_id}")
    source = Path(video[0])
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        connection.close()
        raise ValueError(f"Could not open original video: {source}")
    first, last = int(frame_range[0]), int(frame_range[1])
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    detector = BallDetector(args.ball_model, min_confidence=args.ball_confidence)
    scene_cuts = {
        int(row[0])
        for row in connection.execute(
            "SELECT frame_index FROM coordinates WHERE video_id=? AND scene_cut=1",
            (args.video_id,),
        )
    }
    updates = []
    found = 0
    started = time.perf_counter()
    for frame_index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in scene_cuts:
            detector.reset()
        frame = cv2.resize(frame, (1280, 720))
        ball = detector.detect(frame)
        if ball:
            found += 1
        updates.append(
            (
                None if ball is None else ball.x,
                None if ball is None else ball.y,
                None if ball is None else ball.confidence,
                args.video_id,
                frame_index,
            )
        )
        if len(updates) >= 25:
            connection.executemany(
                "UPDATE coordinates SET ball_x=?,ball_y=?,ball_conf=? "
                "WHERE video_id=? AND frame_index=?",
                updates,
            )
            connection.commit()
            updates.clear()
            processed = frame_index - first + 1
            print(f"Ball backfill {processed}/{last - first + 1} frames", flush=True)
    if updates:
        connection.executemany(
            "UPDATE coordinates SET ball_x=?,ball_y=?,ball_conf=? "
            "WHERE video_id=? AND frame_index=?",
            updates,
        )
        connection.commit()
    capture.release()
    clean_video(connection, args.video_id, max_gap=args.max_gap)
    connection.close()
    elapsed = max(time.perf_counter() - started, 1e-6)
    print(
        json.dumps(
            {"video_id": args.video_id, "ball_detections": found, "elapsed_seconds": elapsed},
            indent=2,
        )
    )


def run_train(
    database: Path,
    model: Path,
    history: float,
    horizon: float,
    *,
    use_ball_features: bool = False,
) -> dict:
    from training.train_movement import predict_latest, train

    metrics = train(
        database,
        model,
        history_seconds=history,
        horizon_seconds=horizon,
        use_ball_features=use_ball_features,
    )
    try:
        metrics["latest_prediction"] = predict_latest(database, model)
    except ValueError:
        metrics["latest_prediction"] = None
    connection = connect(database)
    record_model_run(connection, metrics)
    connection.close()
    print(json.dumps(metrics, indent=2))
    return metrics


def add_common_extract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument(
        "--court-model", type=Path, default=PROJECT_ROOT / "models" / "model_tennis_court_det.pt"
    )
    parser.add_argument(
        "--pose-model", type=Path, default=PROJECT_ROOT / "models" / "yolov8s-pose.pt"
    )
    parser.add_argument(
        "--ball-model", type=Path, default=PROJECT_ROOT / "models" / "tracknet_ball.pt"
    )
    parser.add_argument("--no-ball", action="store_true")
    parser.add_argument("--ball-confidence", type=float, default=0.15)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--detect-interval", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=0.20)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_common_extract_arguments(
        commands.add_parser("extract", help="video to compact SQLite coordinates")
    )
    clean = commands.add_parser("clean", help="rerun causal coordinate cleaning")
    clean.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    clean.add_argument("--video-id", type=int, help="omit to clean every video")
    clean.add_argument("--max-gap", type=int, default=4)
    ball = commands.add_parser("backfill-ball", help="add ball coordinates to an extracted video")
    ball.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    ball.add_argument("--video-id", type=int, required=True)
    ball.add_argument(
        "--ball-model", type=Path, default=PROJECT_ROOT / "models" / "tracknet_ball.pt"
    )
    ball.add_argument("--ball-confidence", type=float, default=0.15)
    ball.add_argument("--max-gap", type=int, default=4)
    train = commands.add_parser("train", help="train future player-position regression")
    train.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--history-seconds", type=float, default=0.5)
    train.add_argument("--horizon-seconds", type=float, default=1.0)
    train.add_argument("--use-ball-features", action="store_true")
    predict = commands.add_parser(
        "predict", help="predict from the latest complete coordinate window"
    )
    predict.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--video-id", type=int)
    report = commands.add_parser("report", help="create a standalone HTML dashboard")
    report.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    report.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    export = commands.add_parser("export", help="export clean coordinate rows as CSV")
    export.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    export.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "movement.csv")
    export.add_argument("--video-id", type=int)
    summary = commands.add_parser("summary", help="print extraction and model metrics")
    summary.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    render = commands.add_parser("render", help="render coordinates and predictions to video")
    render.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    render.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    render.add_argument("--video-id", type=int, default=1)
    render.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "movement_predictions.mp4"
    )
    all_command = commands.add_parser(
        "all", help="extract, train, export, report, and render in one command"
    )
    add_common_extract_arguments(all_command)
    all_command.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    all_command.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    all_command.add_argument("--csv", type=Path, default=PROJECT_ROOT / "outputs" / "movement.csv")
    all_command.add_argument(
        "--prediction-video",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "movement_predictions.mp4",
    )
    all_command.add_argument("--history-seconds", type=float, default=0.5)
    all_command.add_argument("--horizon-seconds", type=float, default=1.0)
    all_command.add_argument("--use-ball-features", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        extract(args)
    elif args.command == "clean":
        connection = connect(args.database)
        video_ids = (
            [args.video_id]
            if args.video_id is not None
            else [int(row[0]) for row in connection.execute("SELECT id FROM videos ORDER BY id")]
        )
        for video_id in video_ids:
            clean_video(connection, video_id, max_gap=args.max_gap)
        connection.close()
    elif args.command == "backfill-ball":
        backfill_ball(args)
    elif args.command == "train":
        run_train(
            args.database,
            args.model,
            args.history_seconds,
            args.horizon_seconds,
            use_ball_features=args.use_ball_features,
        )
    elif args.command == "report":
        generate_report(args.database, args.output)
        print(f"Wrote {args.output.resolve()}")
    elif args.command == "predict":
        from training.train_movement import predict_latest

        print(json.dumps(predict_latest(args.database, args.model, args.video_id), indent=2))
    elif args.command == "export":
        from training.export_movement_csv import export_csv

        count = export_csv(args.database, args.output, args.video_id)
        print(f"Exported {count} rows to {args.output.resolve()}")
    elif args.command == "summary":
        connection = connect(args.database)
        print(json.dumps(summarize(connection), indent=2))
        connection.close()
    elif args.command == "render":
        from .render_movement import render

        render(args.database, args.model, args.video_id, args.output)
        print(f"Wrote {args.output.resolve()}")
    else:
        from training.export_movement_csv import export_csv

        from .render_movement import render

        video_id = extract(args)
        run_train(
            args.database,
            args.model,
            args.history_seconds,
            args.horizon_seconds,
            use_ball_features=args.use_ball_features,
        )
        generate_report(args.database, args.report)
        count = export_csv(args.database, args.csv)
        render(args.database, args.model, video_id, args.prediction_video)
        print(f"Wrote {args.report.resolve()}")
        print(f"Wrote {args.prediction_video.resolve()}")
        print(f"Exported {count} rows to {args.csv.resolve()}")


if __name__ == "__main__":
    main()
