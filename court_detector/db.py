"""Non-blocking SQLAlchemy persistence for pipeline telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameTelemetry:
    frame_index: int
    timestamp_ms: float
    homography: list[list[float]] | None
    court_reprojection_error: float | None
    court_inlier_count: int
    court_stale_frames: int
    court_detected: bool
    scene_cut: bool
    players: list[dict[str, Any]]


class DatabaseWriter:
    """Write frame batches on a background thread without blocking inference."""

    _STOP = object()

    def __init__(
        self,
        database_url: str,
        *,
        source_path: str | Path,
        fps: float,
        width: int,
        height: int,
        total_frames: int,
        match_name: str | None = None,
        batch_size: int = 100,
        queue_size: int = 1000,
    ) -> None:
        try:
            from sqlalchemy import (
                JSON,
                BigInteger,
                Boolean,
                Column,
                DateTime,
                Float,
                ForeignKey,
                Integer,
                MetaData,
                String,
                Table,
                UniqueConstraint,
                create_engine,
            )
        except ImportError as error:
            raise RuntimeError(
                "Database recording requires SQLAlchemy and psycopg. Install the project requirements first."
            ) from error

        metadata = MetaData()
        identifier_type = BigInteger().with_variant(Integer, "sqlite")
        self.matches = Table(
            "matches", metadata,
            Column("id", identifier_type, primary_key=True, autoincrement=True),
            Column("name", String(255), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
        self.videos = Table(
            "videos", metadata,
            Column("id", identifier_type, primary_key=True, autoincrement=True),
            Column("match_id", identifier_type, ForeignKey("matches.id", ondelete="SET NULL")),
            Column("source_path", String(2048), nullable=False),
            Column("fps", Float, nullable=False),
            Column("width", Integer, nullable=False),
            Column("height", Integer, nullable=False),
            Column("total_frames", Integer, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
        self.frames = Table(
            "frames", metadata,
            Column("id", identifier_type, primary_key=True, autoincrement=True),
            Column("video_id", identifier_type, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
            Column("frame_index", Integer, nullable=False),
            Column("timestamp_ms", Float, nullable=False),
            Column("homography", JSON),
            Column("court_reprojection_error", Float),
            Column("court_inlier_count", Integer, nullable=False, default=0),
            Column("court_stale_frames", Integer, nullable=False, default=0),
            Column("court_detected", Boolean, nullable=False, default=False),
            Column("scene_cut", Boolean, nullable=False, default=False),
            UniqueConstraint("video_id", "frame_index", name="uq_frames_video_frame"),
        )
        self.player_detections = Table(
            "player_detections", metadata,
            Column("id", identifier_type, primary_key=True, autoincrement=True),
            Column("frame_id", identifier_type, ForeignKey("frames.id", ondelete="CASCADE"), nullable=False),
            Column("player_id", String(8), nullable=False),
            Column("confidence", Float, nullable=False),
            Column("selection_score", Float, nullable=False),
            Column("bbox", JSON, nullable=False),
            Column("foot_image", JSON, nullable=False),
            Column("court_x", Float),
            Column("court_y", Float),
            Column("keypoints", JSON, nullable=False),
            UniqueConstraint("frame_id", "player_id", name="uq_player_detections_frame_player"),
        )

        self.engine = create_engine(database_url, pool_pre_ping=True)
        metadata.create_all(self.engine)
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            match_id = None
            if match_name:
                match_id = connection.execute(
                    self.matches.insert().values(name=match_name, created_at=now).returning(self.matches.c.id)
                ).scalar_one()
            self.video_id = connection.execute(
                self.videos.insert().values(
                    match_id=match_id, source_path=str(Path(source_path).resolve()),
                    fps=float(fps), width=int(width), height=int(height),
                    total_frames=int(total_frames), created_at=now,
                ).returning(self.videos.c.id)
            ).scalar_one()

        self.batch_size = max(1, int(batch_size))
        self.queue: Queue[FrameTelemetry | object] = Queue(maxsize=max(1, queue_size))
        self._error: BaseException | None = None
        self._thread = Thread(target=self._worker, name="telemetry-db-writer", daemon=True)
        self._thread.start()

    def submit(self, telemetry: FrameTelemetry) -> None:
        if self._error is not None:
            raise RuntimeError("Database writer failed") from self._error
        self.queue.put(telemetry)

    def _write_batch(self, batch: list[FrameTelemetry]) -> None:
        with self.engine.begin() as connection:
            frame_rows = [
                {
                    "video_id": self.video_id,
                    "frame_index": item.frame_index,
                    "timestamp_ms": item.timestamp_ms,
                    "homography": item.homography,
                    "court_reprojection_error": item.court_reprojection_error,
                    "court_inlier_count": item.court_inlier_count,
                    "court_stale_frames": item.court_stale_frames,
                    "court_detected": item.court_detected,
                    "scene_cut": item.scene_cut,
                }
                for item in batch
            ]
            statement = self.frames.insert().returning(
                self.frames.c.id, sort_by_parameter_order=True
            )
            frame_ids = list(connection.execute(statement, frame_rows).scalars())
            rows = []
            for frame_id, item in zip(frame_ids, batch, strict=True):
                for player in item.players:
                    court_position = player.get("court_position")
                    rows.append({
                        "frame_id": frame_id, "player_id": player["player_id"],
                        "confidence": player["confidence"], "selection_score": player["selection_score"],
                        "bbox": player["bbox"], "foot_image": player["foot_image"],
                        "court_x": None if court_position is None else court_position[0],
                        "court_y": None if court_position is None else court_position[1],
                        "keypoints": player["keypoints"],
                    })
            if rows:
                connection.execute(self.player_detections.insert(), rows)

    def _worker(self) -> None:
        batch: list[FrameTelemetry] = []
        stopping = False
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self._STOP:
                        stopping = True
                        if batch:
                            self._write_batch(batch)
                        return
                    batch.append(item)  # type: ignore[arg-type]
                    if len(batch) >= self.batch_size:
                        self._write_batch(batch)
                        batch.clear()
                finally:
                    self.queue.task_done()
        except Exception as error:  # noqa: BLE001 - propagate worker failures on close
            self._error = error
            if stopping:
                return
            # Keep draining so producers/close cannot deadlock after a database
            # failure. The main thread receives the original exception.
            while True:
                item = self.queue.get()
                self.queue.task_done()
                if item is self._STOP:
                    return

    def close(self) -> None:
        self.queue.put(self._STOP)
        self.queue.join()
        self._thread.join()
        self.engine.dispose()
        if self._error is not None:
            raise RuntimeError("Database writer failed") from self._error


def make_frame_telemetry(frame_index, fps, court, players) -> FrameTelemetry:
    matrix = None if court.matrix is None else np.asarray(court.matrix, dtype=float).tolist()
    return FrameTelemetry(
        frame_index=int(frame_index), timestamp_ms=float(frame_index / fps * 1000.0),
        homography=matrix, court_reprojection_error=court.reprojection_error,
        court_inlier_count=int(court.inlier_count), court_stale_frames=int(court.stale_frames),
        court_detected=bool(court.detected),
        scene_cut=bool(court.scene_cut),
        players=[player.as_record() for player in players],
    )
