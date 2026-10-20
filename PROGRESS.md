# Implementation progress — October 19, 2026

## Reduced movement product

- Added a one-command movement analyzer with compact SQLite coordinate storage,
  causal jump rejection, short-gap filling, CSV export, and an HTML dashboard.
- Added the public pretrained TrackNet ball checkpoint, three-frame inference, and a
  separate backfill command so slow ball inference never requires rerunning player poses.
- Added 0.5-second player-position prediction as a velocity-residual ridge model and a
  rendered video showing movement trails plus predicted locations.
- Downloaded and extracted the public TrackNet dataset locally: 19,645 usable
  three-frame samples after clip boundaries, split by games into 15,970 training and
  3,675 validation samples. Added an on-demand ball fine-tuning entry point.
- Extracted six public sample clips into 1,545 compact coordinate rows. The player model
  trained on five videos and achieved 0.929 m mean future-position error on the held-out
  sixth video versus 0.962 m for constant velocity (3.4% improvement; 138 windows).
- Generated `outputs/demo_annotated.mp4`, `outputs/demo_predictions.mp4`,
  `outputs/demo_report.html`, `outputs/demo_movement.sqlite`, and CSV/model artifacts.
- Backfilled the held-out clip with 128 raw ball detections; causal short-gap filling
  provides 91.1% cleaned ball-coordinate coverage and a 572 px/s peak image speed.
- Backfilled all five short clips with ball tracks (87.0%–98.1% cleaned coverage). A
  controlled feature ablation measured 0.931 m error with ball input versus 0.929 m
  without it, so the production predictor remains player-only while the product still
  displays ball position, trail, and image-plane speed.
- Verified one complete CPU fine-tuning step through the downloaded TrackNet dataset and
  wrote a test checkpoint, confirming the ball-training path works before a long GPU run.
- Current automated verification: seven passing tests and zero Ruff violations.

# Earlier implementation progress — October 13, 2026

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
