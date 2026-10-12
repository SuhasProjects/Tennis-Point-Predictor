# Tennis Point Predictor

Machine-learning tools for detecting tennis courts and players from video.

## Project layout

- `court_detector/` — court, player, and video-processing code
- `models/` — trained court and pose-model weights
- `data/` — local training and test media (not committed)
- `outputs/` — generated videos (not committed)

Run the combined video pipeline from the project root:

```powershell
python court_detector/run_pipeline.py
```
