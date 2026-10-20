# Tennis movement prediction model

## Intended use

The model predicts the near and far players' top-down court positions 0.5 seconds after
the latest observation. It is intended for a local visualization/demo and as a baseline
for later experiments. It does not predict point winners.

## Inputs and target

Each sample contains four causal snapshots spanning the previous 0.5 seconds. A snapshot
contains both players' cleaned court coordinates. The feature layout also reserves ball
image coordinates and an availability flag, but the default model zeros them because the
current ball-feature ablation was slightly worse. The target is both players' cleaned
court coordinates 0.5 seconds later.

The estimator starts with constant-velocity extrapolation and learns a ridge-regression
correction. Coordinates use the canonical court reference, where approximately 100 units
equal one meter. Video frames, appearance embeddings, player identities, and future input
features are not used.

## Current evaluation

The checked demo contains six public sample videos and 1,545 coordinate rows. Five videos
provide 786 training windows. The sixth provides 138 video-disjoint validation windows.

| Estimator | Mean position error |
| --- | ---: |
| Constant velocity | 0.962 m |
| Velocity plus learned correction | 0.929 m |
| Learned correction with ball features | 0.931 m |

The selected model improves on constant velocity by 3.4% on one small held-out source.
Ball coordinates remain useful for the rendered speed/trail display, but did not improve
this player-position target. This establishes that the complete training/evaluation path
works; it is not enough evidence for a general accuracy claim.

## Limitations

- `near` and `far` are court sides, not persistent athlete identities.
- Camera calibration and player-selection errors propagate into training targets.
- Short missing runs are causally forward-filled and marked in SQLite.
- Player speed is an estimate on the court plane. Ball speed remains pixels/second because
  a single broadcast camera does not recover the ball's true 3D path.
- The TrackNet checkpoint and dataset do not state an upstream license. Confirm rights
  before redistribution or commercial use.
