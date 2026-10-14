"""Render a contact sheet for quick visual review of YOLO pose labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def annotate(image: np.ndarray, label_path: Path) -> np.ndarray:
    height, width = image.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = np.asarray([float(value) for value in line.split()], dtype=np.float32)
        cx, cy, box_width, box_height = values[1:5]
        x1 = int((cx - box_width / 2) * width)
        y1 = int((cy - box_height / 2) * height)
        x2 = int((cx + box_width / 2) * width)
        y2 = int((cy + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 40), 3)
        keypoints = values[5:].reshape(17, 3)
        for x, y, visibility in keypoints:
            if visibility > 0:
                cv2.circle(image, (int(x * width), int(y * height)), 5, (20, 170, 255), -1)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("datasets/tennis_players"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview.jpg"))
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    images = sorted((args.dataset / "images").glob("*/*.*"))
    if not images:
        raise SystemExit(f"No images found under {args.dataset}")
    indices = np.linspace(0, len(images) - 1, min(args.count, len(images)), dtype=int)
    tiles = []
    for index in indices:
        image_path = images[index]
        split = image_path.parent.name
        label_path = args.dataset / "labels" / split / f"{image_path.stem}.txt"
        image = annotate(cv2.imread(str(image_path)), label_path)
        cv2.putText(image, f"{split}: {image_path.stem}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        tiles.append(cv2.resize(image, (320, 180)))

    columns = 4
    blank = np.zeros_like(tiles[0])
    rows = []
    for offset in range(0, len(tiles), columns):
        row = tiles[offset : offset + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.hstack(row))
    contact_sheet = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
