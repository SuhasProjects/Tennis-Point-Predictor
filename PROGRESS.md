# Implementation progress — October 13, 2026

## Completed

- Replaced fragile four-point homography acceptance with multi-candidate scoring,
  RANSAC, geometric validation, and an eight-inlier minimum.
- Added EMA geometry tracking, homography reconstruction from the smoothed
  landmarks, safe stale-state limits, and camera-cut resets.
- Implemented confidence-aware foot projection, one player per court half,
  temporal identity gating, smoothed track state, peripheral-official rejection,
  and a targeted high-resolution far-court crop for tiny players.
- Added pixel-change camera-cut detection for same-color broadcast cuts and
  shortened stale homography reuse so closeups do not inherit old geometry.
- Added clean court overlays, selected-player skeletons, and a top-down minimap.
- Added PostgreSQL/SQLAlchemy tables and a non-blocking bulk writer for matches,
  videos, frames, homographies, poses, boxes, and projected positions.
- Added JSONL telemetry, telemetry analysis, and database-to-NumPy sequence export.
- Added pose dataset preparation, temporal train/validation splitting, label
  validation, contact-sheet preview, and an Ultralytics fine-tuning entry point.
- Created a local `.venv` using the existing CUDA stack and installed SQLAlchemy,
  psycopg, pytest, and Ruff inside the project.

## Verified results

- Full Australian Open source: 3,099 frames at 21.8 FPS without encoding.
- Accepted wide-court frames: 2,467.
- Player coverage given an accepted court: far 100%, near 99.96%, both 99.96%.
- Independent 640×360 window: far 99.7%, near 99.7%, both 99.3% over 300 frames.
- Harder independent 640×360 full match: 3,890 frames at 24.9 FPS, with 1,502
  accepted court frames. Conditional coverage is near 96.54%, far 92.08%, and
  both 88.62%, improved from 93.77%, 80.64%, and 77.62% on the initial run.
- The far-player recovery demo has 100% near/far/both coverage on all 125 valid
  court frames in its 130-frame window.
- Partial-court slow motion/closeups: 0/300 falsely accepted after the inlier fix.
- Automated checks: 6 passing tests and zero Ruff violations.
- Database smoke: 10 frames and 20 players bulk-written and exported to a
  `[10, 2, 59]` NumPy feature tensor.
- Generated pose fine-tuning candidates: 193 images (159 train, 34 validation),
  all structurally valid. See `outputs/dataset_preview.jpg` for a contact sheet.
- Generated a separate 65-image independent-video review set (52 train, 13
  validation), structurally validated without merging it into the primary set.

## Useful artifacts

- `outputs/rally_final.mp4` — latest annotated primary-video result.
- `outputs/second_video_far_fixed.mp4` — low-resolution far-player fallback.
- `outputs/videoplayback2_recovery_demo.mp4` — annotated independent-video
  identity-recovery example.
- `outputs/videoplayback2_telemetry_release.jsonl` — complete independent-video
  telemetry for the exact current code.
- `outputs/dataset_preview.jpg` — generated training-label contact sheet.
- `outputs/videoplayback2_dataset_preview.jpg` — independent-video label review.
- `outputs/full_primary_telemetry.jsonl` — earlier full-run diagnostic telemetry.
- `database/schema.sql` — reviewed PostgreSQL schema.

## Remaining user-input/manual work

- PostgreSQL is running, but creating the application database requires the
  administrator password. Follow `database/README.md`, then set `DATABASE_URL`.
- Pseudo-labels should be visually reviewed/corrected before a long fine-tuning
  run. They are deliberately not presented as hand-labelled ground truth.
- A future point-outcome model needs verified rally/point outcome labels. The
  tracking sequence tensors are ready, but outcome labels were not fabricated.

No Git commit has been created.
