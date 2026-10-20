# Model provenance

- `model_tennis_court_det.pt`, `yolov8s-pose.pt`, and `yolo26n-pose.pt` were already
  present in this repository.
- `tracknet_ball.pt` is the pretrained checkpoint linked by the public
  [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) README. It was downloaded
  from that README's Google Drive link on October 19, 2026. The upstream repository does
  not include an explicit license, so confirm reuse terms before redistributing the file.

The TrackNet checkpoint expects three consecutive 640×360 frames and emits a per-pixel
ball-location classification map. `court_detector.ball_detector` adapts its output to the
pipeline's 1280×720 coordinate system.
