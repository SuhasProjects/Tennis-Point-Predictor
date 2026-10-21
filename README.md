# Tennis Movement Analyzer

<div align="center">

<h3>Monocular Broadcast Vision to Metric Kinematics & Trajectory Forecasting</h3>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Database](https://img.shields.io/badge/Storage-SQLite3-003B57.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/Tests-7%20Passed-brightgreen.svg)]()
[![Model Accuracy](https://img.shields.io/badge/Accuracy-%2B20.3%25%20Multi--Surface%20Avg-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br/>

![Tennis Movement Tracking & Trajectory Prediction](docs/demo.gif)

*Autonomous 14-point court homography calibration, YOLOv8 pose-driven foot contact tracking, metric coordinate projection, and **1.0-second future trajectory forecasting** displayed in real-time minimap space.*

<br/>

[**Live Interactive Report**](https://suhasprojects.github.io/Tennis-Point-Predictor/) • [**Architecture**](#system-architecture) • [**Benchmarks**](#empirical-benchmarks) • [**Quick Start**](#quick-start) • [**Code Guide**](CODE_GUIDE.md)

</div>

---

## Overview

Monocular broadcast tennis video presents extreme challenges for kinematic analysis: steep oblique perspective distortion, dynamic camera panning, scale disparity between near and far athletes, and fast athletic motion blur.

**Tennis Movement Analyzer** solves this through a modular, vision-to-kinematics pipeline:
1. **Dynamic Court Calibration**: Detects 14 key court landmarks to compute a planar homography matrix $H$, transforming 2D broadcast pixels into a canonical metric top-down court (in meters).
2. **Confidence-Aware Pose Tracking**: Leverages YOLOv8 keypoints to isolate athletic ankle/foot ground contacts, filtering background ball kids, referees, and crowd members.
3. **Causal Coordinate Filtering**: Implements a zero-leakage SQLite coordinate store that filters kinematic outliers and smooths tracking state strictly from past observations.
4. **Kinematic Trajectory Forecasting**: Outperforms constant-velocity physics baselines by **+20.3% across all surfaces** at a 1.0-second horizon (reaching **+48.4% on clay courts** and **+26.0% in full cross-validation**) by learning court-recovery vectors and momentum damping.

---

## Empirical Benchmarks

The trajectory forecasting model was benchmarked across **6,255 annotated frames (17 match sources)** covering hard, clay, and grass surfaces. Evaluated with an **80/20 chronological split across all 17 matches** (**3,188 training windows**, **804 validation windows**):

### Multi-Surface Performance (1.0-Second Horizon)

| Surface | Validation Windows | Constant-Velocity Baseline | Ridge Residual Model | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Clay Courts** (e.g. Roland Garros) | 60 | `2.187 m` | **`1.129 m`** | **+48.4%** |
| **Grass Courts** (e.g. Wimbledon) | 80 | `2.155 m` | **`1.736 m`** | **+19.4%** |
| **Hard Courts** (e.g. Australian Open / Shanghai) | 664 | `2.004 m` | **`1.650 m`** | **+17.7%** |
| **Multi-Surface Overall Average** | **804** | `2.033 m` | **`1.620 m`** | **+20.3%** |

*In Leave-One-Video-Out Cross Validation across all 3,992 windows, the model achieves **1.523 m MAE** vs **2.057 m baseline** (**+26.0% overall improvement**).*

### Error Distribution (Average Across All Surfaces)

| Metric | Constant-Velocity Baseline | Ridge Residual Model | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | `2.033 m` | **`1.620 m`** | **+20.3%** |
| **Median Prediction Error** | `1.815 m` | **`1.421 m`** | **+21.7%** |
| **90th Percentile Error** | `3.650 m` | **`3.091 m`** | **+15.3%** |
| **Near-Court Player MAE** | `2.029 m` | **`1.617 m`** | **+20.3%** |
| **Far-Court Player MAE** | `2.036 m` | **`1.623 m`** | **+20.3%** |

### Horizon Scaling

Physics baselines diverge rapidly because human athletes do not travel in infinite straight lines—they brake, slide, and recover toward the center of the baseline. Our learned kinematic residual captures this deceleration:

| Forecast Horizon | Constant-Velocity MAE | Model MAE | Average Improvement |
| :---: | :---: | :---: | :---: |
| **0.5 seconds** | `0.962 m` | **`0.817 m`** | `+15.1%` |
| **1.0 second (Default)** | `2.033 m` | **`1.620 m`** | **`+20.3%`** |
| **1.5 seconds** | `4.213 m` | **`3.148 m`** | `+25.3%` |
| **2.0 seconds** | `5.549 m` | **`4.111 m`** | `+25.9%` |

---

## Visual Demonstration

| Roland Garros (Clay Court) | Vienna Open (Hard Court) |
| :---: | :---: |
| ![Monte Carlo Clay](docs/demo_ball_preview.jpg) | ![Vienna Hard Court](docs/demo_preview.jpg) |
| *High contrast clay surface tracking* | *Indoor hard court broadcast homography* |

* Robust across variable court colors (clay, grass, blue acrylic, green indoor).
* Automatic recovery during temporary camera pans and fast cross-court rallies.
* Generates standalone, zero-dependency HTML dashboard reports (`docs/index.html`) with embedded court heatmaps, speed timelines, and trajectory graphs.

---

## System Architecture

```
                    ┌───────────────────────────────────┐
                    │    Monocular Broadcast Video      │
                    └─────────────────┬─────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│ Court Landmark CNN    │                           │ YOLOv8 Pose Estimator │
│ (14 Landmark Points)  │                           │ (Player Keypoints)    │
└───────────┬───────────┘                           └───────────┬───────────┘
            │                                                   │
            ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│ Homography Matrix H   │                           │ Ground Contact Filter │
│ (Metric Calibration)  │                           │ (Near / Far Selection)│
└───────────┬───────────┘                           └───────────┬───────────┘
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Project to Canonical      │
                        │ Court Plane (X, Y meters) │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Causal SQLite Cleaner     │
                        │ (Outlier & Jump Rejection)│
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Kinematic Trajectory      │
                        │ Residual Forecaster (1.0s)│
                        └─────────────┬─────────────┘
                                      ▼
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│ Visual Annotated Video│                           │ Interactive HTML      │
│ (Minimap & Overlays)  │                           │ Analytics Dashboard   │
└───────────────────────┘                           └───────────────────────┘
```

---

## Quick Start

### 1. Environment Setup

Python 3.10 or newer is recommended:

```bash
git clone https://github.com/SuhasProjects/Tennis-Point-Predictor.git
cd Tennis-Point-Predictor

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Run the Full End-to-End Pipeline

Process a match video, clean coordinates, train the trajectory predictor, and render the annotated video with an interactive report in a single command:

```bash
python -m court_detector all \
  --input data/match.mp4 \
  --output-video outputs/match_annotated.mp4 \
  --max-frames 1800
```

### 3. Step-by-Step Workflow (Modular CLI)

You can also run individual stages independently:

```bash
# 1. Extract court homography and player foot coordinates into SQLite
python -m court_detector extract --input data/match.mp4 --no-ball

# 2. Train the 1.0-second kinematic trajectory predictor
python -m court_detector train --horizon-seconds 1.0

# 3. Export clean CSV coordinates
python -m court_detector export --output outputs/movement.csv

# 4. Generate the standalone interactive HTML dashboard report
python -m court_detector report --output outputs/movement_report.html

# 5. Render output video with real-time minimap & prediction rings
python -m court_detector render --video-id 1 --output outputs/prediction_demo.mp4
```

---

## Repository Structure

```
Tennis-Point-Predictor/
├── court_detector/              # Core vision & kinematics engine
│   ├── court_detector.py        # 14-point landmark detector & homography
│   ├── player_detector.py       # YOLOv8 pose ground contact & player assignment
│   ├── ball_detector.py         # TrackNet temporal ball localization
│   ├── movement_analyzer.py     # Unified CLI entry point & pipeline orchestrator
│   ├── movement_db.py           # Zero-leakage causal SQLite coordinate manager
│   ├── movement_report.py       # Self-contained HTML interactive dashboard generator
│   └── render_movement.py       # OpenCV video compositor with minimap & overlays
├── training/                    # Model training & optimization routines
│   ├── train_movement.py        # Ridge kinematic residual trajectory forecaster
│   └── train_ball.py            # TrackNet fine-tuning utilities
├── models/                      # Pretrained neural network weights
├── docs/                        # GitHub Pages assets (demo.gif, index.html)
├── outputs/                     # Generated databases, predictions, and reports
├── tests/                       # Unit and integration test suite
├── requirements.txt             # Core runtime dependencies
└── requirements-dev.txt         # Development and testing dependencies
```

---

## Verification & Testing

Run the automated test suite to verify homography math, causal coordinate cleaning, and trajectory forecasting:

```bash
pytest
```

All 7 core test modules pass in < 2 seconds:
```
.......                                                  [100%]
7 passed in 1.74s
```

---

## Citation & References

* **YOLOv8 Pose**: Ultralytics Real-Time Object Detection and Human Pose Estimation.
* **TrackNet**: Deep learning network for ball tracking in broadcast sports.
* **OpenCV**: Open Source Computer Vision Library (Perspective Transform & Homography).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
