# Active-player pose training

## Recommended movement-product workflow

The quickest finished product does not require fine-tuning the pose network. Extract
compact coordinates and train the included future-position regressor:

```bash
python -m court_detector.movement_analyzer all --input data/match.mp4 \
  --output-video outputs/match_annotated.mp4
```

To accumulate more training sources, run `extract` once per video against the same
database, then retrain and rebuild the report:

```bash
python -m court_detector.movement_analyzer extract --input data/match_2.mp4
python -m court_detector.movement_analyzer train
python -m court_detector.movement_analyzer report
```

Training defaults to player-only features. Add `--use-ball-features` to `train` to include
the stored ball coordinates and availability flags. On the current six-video demo, the
player-only model measured 0.929 m mean error versus 0.931 m with ball features, so the
player-only result is the production artifact.

The database contains frame/time identifiers, court validity, scene segments, feet
coordinates, ball coordinates, confidence, speeds, and imputation flags. It deliberately
does not contain video frames or pose arrays. The learned target is each player's court
position 0.5 seconds in the future. This provides automatic labels from later coordinates
without pretending to predict a point winner.

## Optional ball fine-tuning

Place the public TrackNet dataset at `datasets/tracknet` with `game1` through
`game10` beneath it. Games 1–8 are used for training and games 9–10 for validation:

```bash
python training/train_ball.py --epochs 10 --batch 1
```

For a quick loader/training check before committing GPU time:

```bash
python training/train_ball.py --epochs 1 \
  --max-train-samples 20 --max-val-samples 10
```

Images are decoded and target heatmaps are generated on demand, so this does not create
another directory of derived frames. The 256-class per-pixel output is memory-heavy;
keep batch size at 1 on a laptop GPU.

The training target for ball detection is the 2D ball heatmap. Player detection uses
the pretrained YOLOv8-pose weights (`models/yolov8s-pose.pt`) directly, bypassing the need for
pose re-training.
