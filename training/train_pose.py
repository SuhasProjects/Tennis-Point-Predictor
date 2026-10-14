"""Fine-tune a pose model to detect only the two active tennis players."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "training" / "tennis_players.yaml")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "yolov8s-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics fit batch size to GPU memory")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="tennis-player-pose")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_root = Path(dataset_config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (args.data.parent / dataset_root).resolve()
    dataset_config["path"] = dataset_root.as_posix()
    resolved_data = PROJECT_ROOT / "outputs" / "training" / "tennis_players.resolved.yaml"
    resolved_data.parent.mkdir(parents=True, exist_ok=True)
    resolved_data.write_text(
        yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
    )

    model = YOLO(str(args.model))
    model.train(
        data=str(resolved_data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        project=str(PROJECT_ROOT / "outputs" / "training"), name=args.name,
        resume=args.resume, patience=25, cache=False, plots=True,
        degrees=2.0, translate=0.08, scale=0.25, perspective=0.0002,
        fliplr=0.5, mosaic=0.25, close_mosaic=10,
    )


if __name__ == "__main__":
    main()
