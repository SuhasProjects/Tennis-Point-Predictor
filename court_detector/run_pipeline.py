"""Command-line video pipeline for court, player, and database telemetry."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2

try:
    from .court_detector import CourtDetector
    from .db import DatabaseWriter, make_frame_telemetry
    from .player_detector import PlayerPoseDetector
    from .visualization import draw_court_overlay, draw_minimap
except ImportError:  # pragma: no cover - direct script compatibility
    from db import DatabaseWriter, make_frame_telemetry
    from player_detector import PlayerPoseDetector
    from visualization import draw_court_overlay, draw_minimap

    from court_detector import CourtDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "Clipto AI Video Downloader CRAZY 70-Shot Rally! _ Australian Open 2023.mp4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "output_combined_v2.mp4")
    parser.add_argument("--court-model", type=Path, default=PROJECT_ROOT / "models" / "model_tennis_court_det.pt")
    parser.add_argument("--pose-model", type=Path, default=PROJECT_ROOT / "models" / "yolov8s-pose.pt")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--detect-interval", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=0.20)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-video", action="store_true", help="Run inference/DB logging without writing a video")
    parser.add_argument("--no-minimap", action="store_true")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--match-name")
    parser.add_argument("--telemetry-jsonl", type=Path, help="Optional portable per-frame telemetry log")
    return parser


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    court_detector = CourtDetector(
        args.court_model, smoothing_factor=args.smoothing,
        max_stale_frames=max(10, args.detect_interval * 2),
    )
    pose_detector = PlayerPoseDetector(args.pose_model, conf=args.confidence, imgsz=args.imgsz)

    video_writer = None
    if not args.no_video:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not create output video: {args.output}")

    database = None
    if args.database_url:
        database = DatabaseWriter(
            args.database_url, source_path=input_path, fps=fps,
            width=source_width, height=source_height, total_frames=total_frames,
            match_name=args.match_name,
        )

    jsonl_file = None
    if args.telemetry_jsonl:
        args.telemetry_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = args.telemetry_jsonl.open("w", encoding="utf-8")

    frame_index = int(args.start_frame)
    processed = far_detected = near_detected = valid_court = both_detected = 0
    started = time.perf_counter()
    try:
        while args.max_frames is None or processed < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame, court = court_detector.track_frame(
                frame, frame_index, detect_interval=args.detect_interval
            )
            if court.scene_cut:
                pose_detector.reset_tracking()
            players = pose_detector.detect_players(frame, court.matrix)
            sides = {player.side for player in players}
            far_detected += int("far" in sides)
            near_detected += int("near" in sides)
            valid_court += int(court.matrix is not None)
            both_detected += int(court.matrix is not None and sides == {"near", "far"})

            if database is not None:
                database.submit(make_frame_telemetry(frame_index, fps, court, players))
            if jsonl_file is not None:
                jsonl_file.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "timestamp_ms": frame_index / fps * 1000.0,
                            "court": {
                                "matrix": None if court.matrix is None else court.matrix.tolist(),
                                "detected": court.detected,
                                "stale_frames": court.stale_frames,
                                "inlier_count": court.inlier_count,
                                "reprojection_error": court.reprojection_error,
                                "scene_cut": court.scene_cut,
                            },
                            "players": [player.as_record() for player in players],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            if video_writer is not None:
                draw_court_overlay(frame, court.matrix)
                pose_detector.annotate(frame, players)
                if not args.no_minimap and court.matrix is not None:
                    draw_minimap(frame, players)
                status = (
                    f"frame {frame_index} | court {court.inlier_count}/14 err {court.reprojection_error:.1f}px"
                    if court.reprojection_error is not None
                    else f"frame {frame_index} | court unavailable"
                )
                cv2.putText(frame, status, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                video_writer.write(frame)

            processed += 1
            frame_index += 1
            if processed % 100 == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(f"Processed {processed} frames ({processed / elapsed:.1f} FPS)")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if database is not None:
            database.close()
        if jsonl_file is not None:
            jsonl_file.close()

    elapsed = max(time.perf_counter() - started, 1e-6)
    summary = {
        "processed_frames": processed,
        "processing_fps": processed / elapsed,
        "court_coverage": valid_court / max(processed, 1),
        "near_player_coverage": near_detected / max(processed, 1),
        "far_player_coverage": far_detected / max(processed, 1),
        "near_given_court": near_detected / max(valid_court, 1),
        "far_given_court": far_detected / max(valid_court, 1),
        "both_given_court": both_detected / max(valid_court, 1),
        "output": "disabled" if args.no_video else str(args.output.resolve()),
    }
    print(summary)
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
