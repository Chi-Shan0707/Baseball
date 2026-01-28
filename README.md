## Baseball: simple video classifier

Small, beginner-friendly PyTorch training code for classifying 7-second baseball clips as **strike** vs **ball**.

It implements:
- Per-frame **ResNet-18** feature extraction (2D CNN applied to each frame)
- Two interchangeable temporal heads:
	- **Temporal Average Pooling** (`--arch pool`)
	- **LSTM** (`--arch lstm`)

### Data layout

This repo expects your dataset to look like:

```
dataset/
	videos/
		0.mp4
		1.mp4
		...
	pitchcalls/
		labels.csv
```

Your `labels.csv` should contain at least columns: `id,label` where label is `strike` or `ball`.<br>
`Strike` maps to 1. `Ball` maps to 0.

### Install

Minimal dependencies:
- `torch`
- `torchvision`

Optional (only needed if `torchvision.io.read_video` cannot decode videos on your system):
- `opencv-python`

### Train

**Recommended (skip corrupted videos using health report):**

```bash
python train.py --bad-videos-json video_health_report.json --epochs 10 --batch-size 2
```

Baseline (temporal average pooling):

```
python train.py --arch pool --data-dir ./dataset --epochs 10
```

LSTM head:

```
python train.py --arch lstm --data-dir ./dataset --epochs 10
```

Common flags:
- `--frames 16` (number of frames sampled uniformly per 7s clip, default 16)
- `--batch-size 2` (default 4, reduce if OOM)
- `--lr 1e-4`
- `--freeze-backbone` (faster + less memory)
- `--device cpu` or `--device cuda`
- `--backend auto|torchcodec|torchvision|cv2` (video decoder, default auto)
- `--num-workers 0` (DataLoader workers, 0 is safest)
- `--skip-bad-videos` (replace corrupt videos with synthetic placeholders at runtime)
- `--bad-videos-json path/to/video_health_report.json` (pre-filter bad videos before loading - **most efficient**)

Checkpoints and metrics are saved under `checkpoints/`.

### Handling corrupted videos

If you have a `video_health_report.json` (from running video health checks), you can exclude bad videos **before decoding**:

```bash
python train.py --bad-videos-json video_health_report.json --epochs 10
```

This is more efficient than `--skip-bad-videos` (which tries to decode then falls back). With 77 corrupted videos identified, this will train on the 906 healthy videos only.

Expected JSON format:
```json
{
  "damaged_videos": [
    {"filename": "884.mp4", ...},
    {"filename": "470.mp4", ...}
  ]
}
```

### Smoke test (no videos needed)

If videos are missing or decoding fails, `train.py` automatically falls back to a tiny synthetic dataset so you can verify the pipeline works:

```
python train.py --arch pool --epochs 2 --device cpu
```

### Troubleshooting

- GPU OOM: reduce `--frames` and/or `--batch-size`, or add `--freeze-backbone`.
- Slow training: freezing the backbone helps a lot.
- Video decode errors: install `opencv-python` so the dataset loader can fall back to OpenCV.
- Want gradient accumulation? Start by lowering batch size; if you want, I can add accumulation to `train.py`.

### Dataset source

```
@inproceedings{mlbyoutube2018,
	title={Fine-grained Activity Recognition in Baseball Videos},
	booktitle={CVPR Workshop on Computer Vision in Sports},
	author={AJ Piergiovanni and Michael S. Ryoo},
	year={2018}
}
```

Repo: https://github.com/piergiaj/mlb-youtube/