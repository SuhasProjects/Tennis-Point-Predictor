# Tennis Movement Prediction Model Card

## Model Overview

- **Model Architecture**: Physics-informed velocity-residual ridge regression on projected court coordinates.
- **Task**: 1.0-second future trajectory forecasting for near-court and far-court tennis players from monocular broadcast video.
- **Input Features**: Causal sliding history window (0.5s lookback) of planar metric $(x, y)$ coordinates, computed player velocities, and relative court boundary proximities.
- **Target**: Near and far player coordinates on the canonical court plane (measured in meters) at frame $t + 1.0\text{s}$.
- **Regularization**: $L_2$ Ridge Penalty ($\lambda = 1.0$).

---

## Intended Use & Domain

- **Primary Use**: Quantitative player movement analysis, kinematic forecasting, athletic court-coverage visualization, and broadcast augmented-reality minimap overlays.
- **Out of Scope**: Point-winner forecasting or match wagering. The model predicts physical movement dynamics, not subjective match outcomes or unobserved psychological factors.

---

## Benchmark & Multi-Surface Evaluation

Evaluated across **6,255 annotated frames (17 match sources)** covering hard, clay, and grass surfaces.

### Multi-Surface Performance (1.0-Second Horizon)

Evaluated with an **80/20 chronological split across all 17 matches** (3,188 training windows, 804 held-out validation windows):

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

---

## Key Findings

1. **Deceleration & Recovery**: Straight-line physics baselines diverge because tennis athletes rapidly brake, slide, and recover toward the center baseline. The learned model captures this court-geometry attraction vector.
2. **Horizon Scaling**: The model's advantage expands as the prediction horizon grows (from +15.1% at 0.5s to +20.3% at 1.0s and +25.9% at 2.0s).
3. **Ball Features**: Ablation studies show incorporating 2D image-plane ball coordinates did not improve court-plane player trajectory forecasting, preserving a fast, robust player-only kinematic formulation.

---

## Limitations

- **Player Roles**: Tracks `near` and `far` court halves rather than biometric player identities across sets.
- **Homography Sensitivity**: Monocular camera zoom or severe camera shaking can induce minor coordinate drift; mitigated by the 14-point landmark RANSAC tracker and causal jump rejector.
- **Single-Camera 3D Ambiguity**: Player speeds are estimated along the planar court surface. True 3D ball speed cannot be recovered without multi-view camera calibration.
