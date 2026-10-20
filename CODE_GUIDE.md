# Code guide

## What the product does

The reduced product answers: **where will the near and far tennis players be 0.5
seconds from now?** It also displays estimated player speed, ball location, ball trail,
and image-plane ball speed. It does not predict the winner of a point because the current
data has no verified point-outcome labels.

The default model is deliberately small. Court and pose neural networks extract useful
coordinates from video, then ridge regression learns a correction to constant-velocity
player motion. Video pixels are not stored in the training database.

## One frame through the system

1. `court_detector/movement_analyzer.py` reads and resizes a video frame.
2. `court_detector/court_detector.py` detects court landmarks and produces a homography.
3. `court_detector/player_detector.py` selects the active player on each court half. The
   feet are projected through the homography into a canonical top-down court measured in
   roughly centimeters.
4. `court_detector/ball_detector.py` passes the current and two previous frames through
   TrackNet and returns the most likely ball pixel.
5. The analyzer inserts raw coordinates and confidence values into SQLite.
6. `court_detector/movement_db.py` causally rejects large jumps, fills short gaps using
   past values, smooths coordinates, and computes speeds. It never uses future frames to
   clean an earlier frame.
7. `training/train_movement.py` creates four snapshots over the previous 0.5 seconds. It
   extrapolates constant velocity, then trains ridge regression to correct that estimate
   using a position 0.5 seconds in the future as the automatic target.
8. `court_detector/render_movement.py` reads the original video and stored coordinates,
   then draws trails, speeds, and prediction rings. `movement_report.py` creates the HTML
   dashboard.

## Important files

| File | Edit this when you want to… |
| --- | --- |
| `court_detector/movement_analyzer.py` | change CLI behavior or the end-to-end workflow |
| `court_detector/movement_db.py` | change schema, smoothing, jump limits, or speed calculation |
| `training/train_movement.py` | change features, prediction horizon, split, model, or metrics |
| `court_detector/render_movement.py` | change the video overlay and minimap |
| `court_detector/movement_report.py` | change the standalone report |
| `court_detector/ball_detector.py` | change ball inference or TrackNet post-processing |
| `training/train_ball.py` | fine-tune TrackNet on the downloaded public dataset |
| `tests/test_movement_pipeline.py` | verify database cleaning, training, prediction, export, and report |
| `database/movement_schema.sql` | inspect the compact SQLite schema without reading Python |

The movement product uses a lightweight, serverless SQLite database (`outputs/movement.sqlite`).
No background database servers or services are required.

## Stored data

`videos` stores one row per source: path, FPS, dimensions, and creation time.
`coordinates` stores one row per frame: raw and cleaned near/far/ball coordinates,
confidence, court/scene flags, imputation flags, and derived speeds. `model_runs` stores
the JSON metrics from each training run.

`near` and `far` mean court halves, not permanent athlete identities. Player coordinates
use the canonical court plane; ball coordinates remain image pixels. This is why player
speed is approximately meters/second while ball speed is pixels/second.

## Start training

Activate the existing environment and verify it:

```bash
source .venv/bin/activate
python -m pytest
```

Add videos to one database, keeping a representative clip last because the last usable
video becomes the held-out validation set:

```bash
python -m court_detector extract --input data/train_1.mp4 --no-ball
python -m court_detector extract --input data/train_2.mp4 --no-ball
python -m court_detector extract --input data/validation.mp4 --no-ball
python -m court_detector train
python -m court_detector report
```

Player extraction is enough to train the selected model. Add ball tracks separately when
you want the ball overlay, without rerunning court and pose inference:

```bash
python -m court_detector backfill-ball --video-id 1
python -m court_detector render --video-id 1 --output outputs/video_1_predictions.mp4
```

To fine-tune the ball detector on the already downloaded TrackNet data, first run a small
job, then increase epochs on a CUDA machine:

```bash
python training/train_ball.py --epochs 1 --max-train-samples 20 --max-val-samples 10
python training/train_ball.py --epochs 10 --batch 1 --device cuda
```

The full local dataset contains 15,970 training and 3,675 validation windows split by
game. Fine-tuning on CPU is functional but slow.

## Safe first edits

- Change `--history-seconds` and `--horizon-seconds` through the CLI before changing code.
- Change the default ridge penalty in `training/train_movement.py`, then compare held-out
  `model_mae_m`, not training error.
- Add features inside `_video_samples()` and `_feature_at()` together; they must produce
  identical feature layouts during training and prediction.
- Add database columns in both `movement_db.py` and `database/movement_schema.sql`.
- Run `python -m ruff check court_detector training tests` and `python -m pytest` after an
  edit.

The current reference result is 0.929 m mean error over 138 windows from the sixth held-
out video, compared with 0.962 m for constant velocity. The sample is small, so use new
videos and a video-disjoint test set before making an accuracy claim.
