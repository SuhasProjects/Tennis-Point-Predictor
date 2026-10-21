# Implementation Progress

## Milestone: Multi-Surface Scale & 1.0-Second Prediction Horizon (Current)

- **Prediction Horizon Upgrade**:
  - Shifted prediction horizon from 0.5s to **1.0s future forecasting** (`horizon_seconds=1.0`), matching real-world tennis stroke-to-stroke intervals.
  - Dynamically tuned Ridge regularization to $\lambda = 1.0$, allowing the model to capture non-linear recovery arcs without overfitting.
  - Updated video visualizer (`render_movement.py`) with dynamic legend rings and live speed telemetry.

- **Dataset Expansion & Multi-Surface Scaling**:
  - Scaled dataset from 6 initial clips (1,545 frames) to **17 real match sources (6,255 frames)**.
  - Sourced and transcoded clean broadcast rallies across **hard, clay, and grass surfaces**:
    - Hard courts: Australian Open, Shanghai Masters (Tsitsipas, Medvedev), Vienna Open, US Open.
    - Clay courts: Roland Garros / Monte Carlo (Rafael Nadal, Carlos Alcaraz).
    - Grass courts: Wimbledon (Andy Murray, Tomáš Berdych).
  - All coordinate extraction processed via causal SQLite coordinate storage (`outputs/demo_movement.sqlite`).

- **Multi-Surface Average Evaluation & Benchmarks**:
  - Upgraded evaluation in `training/train_movement.py` from holding out a single grass match to an **80/20 chronological split across all 17 matches** (3,188 training windows, 804 held-out validation windows).
  - **Overall Multi-Surface Average**: Model MAE **1.620 m** vs Baseline **2.033 m** (**+20.3% improvement** over physics).
  - **Surface Breakdown**:
    - Hard courts (664 val windows): Model 1.650 m vs Baseline 2.004 m (**+17.7%**).
    - Grass courts (80 val windows): Model 1.736 m vs Baseline 2.155 m (**+19.4%**).
    - Clay courts (60 val windows): Model 1.129 m vs Baseline 2.187 m (**+48.4%**).
  - **Leave-One-Video-Out Cross-Validation (3,992 windows)**: Model MAE **1.523 m** vs Baseline **2.057 m** (**+26.0% overall improvement**).

- **Architecture Streamlining & Low-Risk Cleanup**:
  - Removed obsolete PostgreSQL/telemetry dependencies (`psycopg`, `SQLAlchemy`, `database/schema.sql`, `training/analyze_telemetry.py`).
  - Removed redundant YOLO pose fine-tuning scratch files, focusing the project on its strength: monocular vision-to-kinematic trajectory forecasting.
  - Optimized CPU pose extraction with `--imgsz 640` yielding 4–5x speedup with negligible (< 3 cm) metric drift.

- **Presentation & Documentation Assets**:
  - Generated hero 1.0s prediction GIF: `docs/demo.gif`.
  - Built interactive, zero-dependency HTML analytics report: `docs/index.html` (ready for GitHub Pages).
  - Revamped `README.md` with portfolio badges, ASCII architecture pipeline diagram, benchmark tables, and modular CLI guide.
  - Updated `MODEL_CARD.md`, `CODE_GUIDE.md`, and `PROGRESS.md`.
  - Automated tests: 7/7 unit tests passing cleanly in `pytest`.

---

## Milestone: Reduced Movement Product (October 19, 2026)

- Added a one-command movement analyzer with compact SQLite coordinate storage, causal jump rejection, short-gap filling, CSV export, and an HTML dashboard.
- Integrated public pretrained TrackNet ball checkpoint, three-frame inference, and separate backfill command so ball inference never blocks player pose processing.
- Extracted initial six public sample clips into 1,545 compact coordinate rows with proof-of-concept velocity-residual ridge modeling.
- Backfilled initial clips with ball tracks (87.0%–98.1% cleaned coverage); verified ball feature ablation.
- Automated verification: 7 passing tests and zero Ruff violations.

---

## Milestone: Core Computer Vision Foundation (October 13, 2026)

- Replaced four-point homography acceptance with multi-candidate scoring, RANSAC, geometric validation, and an eight-inlier minimum across 14 court landmarks.
- Added EMA geometry tracking, homography reconstruction from smoothed landmarks, safe stale-state limits, and camera-cut resets.
- Implemented confidence-aware foot projection, one player per court half, temporal identity gating, smoothed track state, and targeted high-resolution far-court crops.
- Added clean court overlays, selected-player skeletons, and top-down minimap rendering.
