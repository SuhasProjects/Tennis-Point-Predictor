"""Validate YOLO pose labels and report train/validation sample counts."""

from __future__ import annotations

import argparse
from pathlib import Path


def validate_label(path: Path) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 56:  # class + bbox(4) + 17 * (x, y, visibility)
            errors.append(f"{path}:{line_number}: expected 56 values, got {len(parts)}")
            continue
        try:
            values = [float(value) for value in parts]
        except ValueError:
            errors.append(f"{path}:{line_number}: contains a non-numeric value")
            continue
        if values[0] != 0:
            errors.append(f"{path}:{line_number}: class must be 0")
        if any(value < 0 or value > 1 for value in values[1:5]):
            errors.append(f"{path}:{line_number}: bbox is outside normalized range")
        for index in range(5, 56, 3):
            x, y, visibility = values[index : index + 3]
            if x < 0 or x > 1 or y < 0 or y > 1 or visibility not in (0, 1, 2):
                errors.append(f"{path}:{line_number}: invalid keypoint {(index - 5) // 3}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("datasets/tennis_players"))
    args = parser.parse_args()
    all_errors: list[str] = []
    for split in ("train", "val"):
        images = {path.stem for path in (args.dataset / "images" / split).glob("*.*")}
        labels = {path.stem for path in (args.dataset / "labels" / split).glob("*.txt")}
        missing_labels = images - labels
        missing_images = labels - images
        all_errors.extend(f"{split}: missing label for {stem}" for stem in sorted(missing_labels))
        all_errors.extend(f"{split}: missing image for {stem}" for stem in sorted(missing_images))
        for label in (args.dataset / "labels" / split).glob("*.txt"):
            all_errors.extend(validate_label(label))
        print(f"{split}: {len(images)} images, {len(labels)} labels")
    if all_errors:
        raise SystemExit("\n".join(all_errors))
    print("Dataset is valid")


if __name__ == "__main__":
    main()
