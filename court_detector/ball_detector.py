"""Pretrained TrackNet ball inference over three consecutive video frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, value):
        return self.block(value)


class _TrackNet(nn.Module):
    """Architecture used by the public yastrebksv/TrackNet checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        channels = [
            9,
            64,
            64,
            128,
            128,
            256,
            256,
            256,
            512,
            512,
            512,
            256,
            256,
            256,
            128,
            128,
            64,
            64,
            256,
        ]
        self.blocks = nn.ModuleList(
            _ConvBlock(channels[index], channels[index + 1]) for index in range(len(channels) - 1)
        )
        self.pool_after = {1, 3, 6}
        self.up_before = {10, 13, 15}

    def forward(self, value):
        for index, block in enumerate(self.blocks):
            if index in self.up_before:
                value = nn.functional.interpolate(value, scale_factor=2, mode="nearest")
            value = block(value)
            if index in self.pool_after:
                value = nn.functional.max_pool2d(value, 2)
        return value.reshape(value.shape[0], 256, -1)


@dataclass(frozen=True)
class BallDetection:
    x: float
    y: float
    confidence: float


class BallDetector:
    """Find one tennis ball point; returns ``None`` until three frames exist."""

    def __init__(self, weights_path: str | Path, *, min_confidence: float = 0.15) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _TrackNet().to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        # The original module names layers conv1..conv18. This implementation
        # uses a loop while retaining exactly the same tensors.
        remapped = {}
        for key, value in state.items():
            head, separator, tail = key.partition(".")
            if head.startswith("conv") and head[4:].isdigit():
                key = f"blocks.{int(head[4:]) - 1}{separator}{tail}"
            remapped[key] = value
        self.model.load_state_dict(remapped)
        self.model.eval()
        self.min_confidence = float(min_confidence)
        self.frames: deque[np.ndarray] = deque(maxlen=3)
        self.last_point: np.ndarray | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.last_point = None

    def detect(self, frame: np.ndarray) -> BallDetection | None:
        resized = cv2.resize(frame, (640, 360))
        self.frames.append(resized)
        if len(self.frames) < 3:
            return None
        # Checkpoint order is current, previous, pre-previous.
        stacked = np.concatenate(tuple(reversed(self.frames)), axis=2)
        tensor = (
            torch.from_numpy(np.moveaxis(stacked.astype(np.float32) / 255.0, 2, 0))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            classes = probabilities.argmax(dim=1).reshape(360, 640).cpu().numpy()
            peak = probabilities.max(dim=1).values.reshape(360, 640).cpu().numpy()

        mask = np.where(classes >= 128, 255, 0).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates: list[tuple[float, np.ndarray]] = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if 2 <= area <= 120:
                center = centroids[index].astype(np.float32)
                score = float(peak[int(center[1]), int(center[0])])
                if self.last_point is not None:
                    score -= min(float(np.linalg.norm(center - self.last_point)) / 1000.0, 0.2)
                candidates.append((score, center))
        if not candidates:
            self.last_point = None
            return None
        score, center = max(candidates, key=lambda item: item[0])
        raw_confidence = float(peak[int(center[1]), int(center[0])])
        if raw_confidence < self.min_confidence:
            return None
        self.last_point = center
        # The main pipeline normalizes all frames to 1280 x 720.
        return BallDetection(float(center[0] * 2), float(center[1] * 2), raw_confidence)
