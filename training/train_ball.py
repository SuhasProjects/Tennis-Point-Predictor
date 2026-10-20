"""Fine-tune the bundled TrackNet ball model on the public TrackNet dataset."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from court_detector.ball_detector import BallDetector


class TrackNetDataset(Dataset):
    """Create three-frame inputs and classification heatmaps without cached frames."""

    def __init__(self, root: Path, games: set[int], max_samples: int | None = None) -> None:
        self.samples: list[tuple[Path, Path, Path, float, float, int]] = []
        for label_path in sorted(root.rglob("Label.csv")):
            game_name = next(
                (part for part in label_path.parts if part.lower().startswith("game")), ""
            )
            try:
                game = int(game_name.lower().replace("game", ""))
            except ValueError:
                continue
            if game not in games:
                continue
            with label_path.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            for index in range(2, len(records)):
                current, previous, older = records[index], records[index - 1], records[index - 2]
                names = [
                    current.get("file name"),
                    previous.get("file name"),
                    older.get("file name"),
                ]
                if any(not name for name in names):
                    continue
                visibility = int(float(current.get("visibility", 0) or 0))
                x = float(current.get("x-coordinate", "nan") or "nan")
                y = float(current.get("y-coordinate", "nan") or "nan")
                self.samples.append(
                    (
                        label_path.parent / names[0],
                        label_path.parent / names[1],
                        label_path.parent / names[2],
                        x,
                        y,
                        visibility,
                    )
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        current, previous, older, x, y, visibility = self.samples[index]
        images = []
        for path in (current, previous, older):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not load {path}")
            images.append(cv2.resize(image, (640, 360)))
        inputs = np.moveaxis(np.concatenate(images, axis=2).astype(np.float32) / 255.0, 2, 0)
        target = np.zeros((360, 640), dtype=np.int64)
        if visibility and math.isfinite(x) and math.isfinite(y):
            center_x, center_y = round(x / 2), round(y / 2)
            yy, xx = np.ogrid[:360, :640]
            distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
            gaussian = np.exp(-distance / 20.0) * 255.0
            target = gaussian.astype(np.int64)
        return torch.from_numpy(inputs), torch.from_numpy(target.reshape(-1))


def evaluate(model, loader, device, criterion, use_amp: bool) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for inputs, targets in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(inputs.to(device))
                loss = criterion(logits, targets.to(device))
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "datasets" / "tracknet")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "tracknet_ball.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "training" / "tracknet_ball_finetuned.pt",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_data = TrackNetDataset(args.data, set(range(1, 9)), args.max_train_samples)
    validation_data = TrackNetDataset(args.data, {9, 10}, args.max_val_samples)
    if not train_data or not validation_data:
        raise SystemExit(
            f"No samples found under {args.data}. Expected game1..game10 folders containing Label.csv."
        )
    print(f"train={len(train_data)} validation={len(validation_data)} device={args.device}")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = BallDetector(args.model).model.to(args.device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    use_amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(inputs.to(args.device)), targets.to(args.device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        validation_loss = evaluate(model, validation_loader, args.device, criterion, use_amp)
        print(f"epoch={epoch} train_loss={np.mean(losses):.5f} val_loss={validation_loss:.5f}")
        if validation_loss < best:
            best = validation_loss
            torch.save(model.state_dict(), args.output)
    print(f"Best validation loss {best:.5f}; wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
