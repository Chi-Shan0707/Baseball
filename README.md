# Baseball Pitch Classification

A modular PyTorch project for classifying baseball pitch videos (Strike vs Ball) using a ResNet-18 feature extractor with interchangeable temporal heads (Average Pooling or LSTM).

## Project Structure

- `dataset.py`: handles video loading (supports OpenCV and Torchvision), preprocessing, and robustly skips damaged videos.
- `models.py`: contains `ResNetBackbone`, `TemporalPoolHead`, `LSTMHead`, and `FullModel`.
- `train.py`: CLI training script with metrics and checkpoints.
- `utils.py`: helper functions.

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Pandas
- OpenCV (`pip install opencv-python`) - Recommended for robust video loading without FFmpeg setup issues.

## Data Setup

The project expects the following structure:
- Labels CSV: `./dataset/pitchcalls/labels.csv` (Columns: `id`, `clip_id`, `label`)
- Videos: `./dataset/videos/` (Files named `{id}.mp4`)

## usage

### 1. Training with Temporal Average Pooling (Baseline)

The default backend is `opencv`. This effectively implements the "Plan without FFmpeg" (direct FFmpeg command usage) by using OpenCV's internal decoder.

```bash
python train.py --arch pool --data-dir ./dataset/videos --epochs 10
```

### 2. Training with LSTM Head

```bash
python train.py --arch lstm --data-dir ./dataset/videos --epochs 10
```

### 3. Using Torchvision (FFmpeg) Backend

If you have FFmpeg installed and configured with Torchvision, you can use:

```bash
python train.py --arch lstm --backend torchvision --data-dir ./dataset/videos
```

### 4. Smoke Test (CPU)

If you don't have the dataset around, simply point to a non-existent directory. The script will generate a synthetic dataset to verify the pipeline.

```bash
python train.py --data-dir ./dummy_path --epochs 2 --device cpu --batch-size 4
```

## Troubleshooting

- **GPU OOM (Out Of Memory)**:
  - Reduce batch size: `--batch-size 4`
  - Reduce frames per clip: `--frames 8`
  - Freeze backbone: `--freeze-backbone`
- **Video Loading Errors**:
  - The dataset automatically skips damaged videos.
  - If you see many errors, try switching backends: `--backend opencv` vs `--backend torchvision`.

- ```bash
  [h264 @ 0x386feac0] mmco: unref short failure
  ```
  - Cause: Some of the .mp4 video files in the dataset have minor encoding glitches or missing reference frames in their H.264 stream. This is common in web-scraped datasets (like MLB-YouTube).
  - Impact: It is non-fatal. The underlying video decoder (FFmpeg/OpenCV) is simply logging that the stream isn't perfect, but it is successfully recovering and continuing to decode the rest of the video. You can ignore these.

- ```bash
  [av1 @ 0x39955f80] Your platform doesn't support hardware accelerated AV1 decoding.
  [av1 @ 0x39955f80] Failed to get pixel format.
  [av1 @ 0x39955f80] Get current frame error
  ```
  - Cause: Some videos in the dataset use the newer AV1 codec. Your system's FFmpeg/OpenCV installation seems to lack a proper software decoder or hardware support for AV1, causing it to fail completely for these specific files.
  - Impact: These specific videos are failing to load.
  - Status: Handled.<br>
    Your training loop is not crashing. Note the progress bar is moving (2% -> 5% -> 9%).
    This is because dataset.py catches these failures (returns None) and collate_fn filters them out. You are effectively training on the subset of videos that can be decoded (H.264), and skipping the AV1 ones.