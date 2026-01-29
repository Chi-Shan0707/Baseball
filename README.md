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
