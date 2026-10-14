# Tennis Point Predictor

GPU-accelerated broadcast-tennis pipeline for stable court homography, active
near/far player pose tracking, top-down coordinates, and frame telemetry.

## Current behavior

- Detects 14 court landmarks and validates candidate homographies against all
  visible points before accepting them.
- Smooths projected court geometry, rebuilds a consistent homography from the
  smoothed points, detects camera cuts, and briefly reuses only recent valid
  geometry when inference fails.
- Runs YOLO pose at 1280 px, projects confidence-aware ankle/bbox foot points to
  court space, and selects exactly one tracked player per court half.
- Uses a higher-resolution court-guided far-half crop when a tiny far player is
  missed or the only candidate is peripheral, and smooths track state so one
  torso-only pose cannot cause a multi-frame identity loss.
- Writes match, video, frame, homography, bbox, 17-keypoint pose, and court
  position data through a background SQLAlchemy writer.

Measured smoke checks on the included media:

- Three supplied still images: 14/14 homography inliers, about 2.8–4.7 px median
  landmark error, correct near/far selection despite 6–12 people in view.
- Australian Open wide-court check (170 court frames): 100% far-player and
  99.4% near-player coverage; one blurred near-player frame is deliberately
  dropped rather than switching identity to a bystander.
- Full 3,099-frame Australian Open source: 21.8 processing FPS without video
  encoding; among 2,467 accepted wide-court frames, 100% far-player and 99.96%
  near-player coverage. Replays, closeups, and transitions are excluded by the
  court validator.
- Independent 640×360 match window (300 frames): 99.7% near-player, 99.7%
  far-player, and 99.3% simultaneous coverage after the far-court crop fallback.
- Independent 640×360 full match (`videoplayback2.m4v`, 3,890 frames): among
  1,502 conservatively accepted court frames, 96.5% near-player, 92.1%
  far-player, and 88.6% simultaneous coverage. The initial run on this harder
  video was 93.8%, 80.6%, and 77.6%, respectively.
- A 300-frame slow-motion/closeup segment is rejected as non-wide-court instead
  of producing false homographies from only four or five landmarks.

Coverage measures whether a filtered detection exists, not manually annotated
pose accuracy.

## Setup

The checked local environment uses Python 3.13, CUDA-enabled PyTorch, and an
RTX 4050. A project-local environment is already supported:

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`--system-site-packages` reuses the existing CUDA PyTorch installation rather
than downloading another multi-gigabyte build.

## Run the pipeline

```powershell
.\.venv\Scripts\python.exe -m court_detector.run_pipeline
```

Useful options:

```powershell
.\.venv\Scripts\python.exe -m court_detector.run_pipeline `
  --input data\match.mp4 `
  --output outputs\match_annotated.mp4 `
  --detect-interval 5 `
  --confidence 0.10
```

Use `--max-frames 300` for a quick test, `--start-frame N` to seek, and
`--no-video` for telemetry-only processing.

## PostgreSQL telemetry

PostgreSQL 18 is running on the development machine. Create a database/user in
pgAdmin or `psql`, then set the connection URL without putting a password in the
repository:

```powershell
$env:DATABASE_URL = 'postgresql+psycopg://tennis_app:YOUR_PASSWORD@localhost:5432/tennis_point_predictor'
.\.venv\Scripts\python.exe -m court_detector.run_pipeline `
  --match-name 'Australian Open rally' `
  --no-video
```

The application creates the schema automatically. The equivalent reviewed SQL
is in `database/schema.sql`. Writes are batched on a background thread so the
GPU loop is not held up by individual inserts. A SQLite smoke test is also
possible with `--database-url sqlite:///outputs/telemetry.sqlite`.

## Training

The active-player pose fine-tuning workflow is documented in
`training/README.md`. It includes temporal-block dataset splitting, generation
of homography-filtered review candidates, strict label validation, and an
Ultralytics training entry point. Dataset and training outputs are ignored by
Git.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check court_detector training tests
```

The default tests include robust homography/inverse checks, tracker outlier
recovery, and a GPU integration test over all supplied still images.

## Layout

- `court_detector/` — geometry, pose selection/tracking, visualization, DB writer
- `database/` — PostgreSQL schema
- `training/` — dataset preparation, validation, and pose fine-tuning
- `tests/` — geometry and supplied-image integration tests
- `models/` — court and pose weights
- `data/`, `datasets/`, `outputs/` — local ignored media and generated artifacts
