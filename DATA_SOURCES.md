# Downloaded assets and sample data

## Local sample video

`data/sample_tennis.mp4` and `data/video_input2.mp4` through
`data/video_input8.mp4` (where provided upstream) are from `VideoInput/` in
[ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking), commit
`3a633be55f13667649c385ff76055176bf074c6b`. The repository uses the Unlicense.
The local `data/` directory is Git-ignored, so this sample is available for local testing
without increasing the repository history.

## Ball checkpoint

`models/tracknet_ball.pt` is the pretrained checkpoint linked from
[yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet), commit
`730ea17a324321c553c480179caa8ff13e4cf553`. The repository provides training
and inference code plus a dataset link, but does not state a license. Keep the checkpoint
for local research unless you separately confirm redistribution rights.

The full TrackNet dataset was downloaded from the same upstream Google Drive folder and
extracted under `datasets/tracknet/Dataset` (about 2.5 GB and 19,835 labeled frames). The
downloaded ZIP was removed after extraction to avoid keeping a redundant 2.5 GB copy.
The dataset is Git-ignored. Upstream does not state a license, so use it locally for
research unless you confirm separate distribution terms.

To reproduce the download in a fresh checkout:

```bash
python -m pip install -r requirements-download.txt
gdown --folder \
  https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut \
  -O datasets/tracknet_raw
unzip datasets/tracknet_raw/Dataset.zip -d datasets/tracknet
```

The movement target is generated from future player coordinates extracted from videos in
`data/`, so that workflow remains independent of the TrackNet ball-label dataset.
