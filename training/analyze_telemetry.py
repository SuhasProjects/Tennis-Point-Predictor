"""Summarize coverage gaps and coordinate jumps in pipeline JSONL telemetry."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np


def contiguous_runs(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    runs = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            runs.append((start, previous))
            start = frame
        previous = frame
    runs.append((start, previous))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.telemetry.read_text(encoding="utf-8").splitlines()]
    court_records = [record for record in records if record["court"]["matrix"] is not None]
    print(f"frames={len(records)} court_frames={len(court_records)}")

    for side in ("near", "far"):
        missing = []
        positions = []
        for record in court_records:
            players = {player["player_id"]: player for player in record["players"]}
            if side not in players:
                missing.append(record["frame_index"])
                continue
            position = players[side]["court_position"]
            if position is not None:
                positions.append((record["frame_index"], np.asarray(position, dtype=float)))
        runs = sorted(contiguous_runs(missing), key=lambda run: run[1] - run[0], reverse=True)
        jumps = []
        for (last_frame, last), (frame, current) in pairwise(positions):
            if frame == last_frame + 1:
                jumps.append((float(np.linalg.norm(current - last)), last_frame, frame))
        jumps.sort(reverse=True)
        coverage = None if not court_records else 1.0 - len(missing) / len(court_records)
        coverage_text = "N/A" if coverage is None else f"{coverage:.4%}"
        print(f"{side}: coverage={coverage_text}, missing={len(missing)}")
        print("  longest gaps:", [(start, end, end - start + 1) for start, end in runs[: args.top]])
        print("  largest jumps:", [(round(jump, 1), start, end) for jump, start, end in jumps[: args.top]])


if __name__ == "__main__":
    main()
