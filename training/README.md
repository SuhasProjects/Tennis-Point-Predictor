# Active-player pose training

The training target is a single `player` class containing only the near and far
competitors. Officials, ball kids, and spectators are intentionally left
unlabelled so the fine-tuned model learns that they are background.

## 1. Generate review candidates

From the project root:

```powershell
.\.venv\Scripts\python.exe training\prepare_pose_dataset.py `
  --input "data\Clipto AI Video Downloader CRAZY 70-Shot Rally! _ Australian Open 2023.mp4" `
  --stride 25
```

Frames are split by contiguous temporal blocks, not randomly, to prevent nearly
identical neighboring frames leaking between train and validation. The labels
are pseudo-labels: visually review them before using them as ground truth,
especially low-confidence far-player poses and occluded ankles.

## 2. Validate labels

```powershell
.\.venv\Scripts\python.exe training\validate_pose_dataset.py
.\.venv\Scripts\python.exe training\preview_pose_dataset.py
```

## 3. Fine-tune

```powershell
.\.venv\Scripts\python.exe training\train_pose.py --epochs 100 --imgsz 1280
```

Weights and plots are written under `outputs/training/`. Use a video-disjoint
test set for the final measurement; pseudo-label agreement is not an independent
accuracy metric.

## Export tracked sequences

Database telemetry can be converted to a compressed dense tensor for a later
sequence/point model. The result contains `[frames, 2, 59]` player features,
missing-data masks, timestamps, and homographies:

```powershell
.\.venv\Scripts\python.exe training\export_telemetry.py --video-id 1
```

The player axis is always `[far, near]`. Point-outcome labels are intentionally
not fabricated; add verified rally/point labels before supervised outcome
training.

The primary generated candidate set contains 193 structurally valid samples:
159 training images and 34 validation images from two source videos. The harder
independent video has a separate 65-sample review set at
`datasets/videoplayback2_review` (52 train, 13 validation); inspect
`outputs/videoplayback2_dataset_preview.jpg` before merging it into training.
