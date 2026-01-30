# Baseball Pitch Classification

A modular PyTorch project for classifying baseball pitch videos (Strike vs Ball) using a ResNet-18 feature extractor with interchangeable temporal heads (Average Pooling or LSTM).

## Project Structure

- `dataset.py`: handles video loading (supports OpenCV and Torchvision), preprocessing, and robustly skips damaged videos.
- `models.py`: contains `ResNetBackbone`, `TemporalPoolHead`, `LSTMHead`, and `FullModel`.
- `train.py`: CLI training script with metrics and checkpoints.
- `utils.py`: helper functions.
- `judge_pitch.py`: pitch call

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

### 5. Inference: `judge_pitch.py`

Use `judge_pitch.py` to run inference on one or more local video files. Important: make sure inference arguments match how you trained the model (architecture, number of frames per clip, and device). Mismatched parameters may cause incorrect behavior or poor predictions.

Example usage:

```bash
python judge_pitch.py --device cuda --frames 8 --arch lstm --input-dir ./input_videos --model-path checkpoints/model_best.pth
```

Notes:
- `--arch` must match the architecture used at training time (e.g., `lstm` or `pool`).
- `--frames` should match the `--frames` value used during training. The model expects the same number of frames per clip it was trained on.
- `--device` can be `cuda` or `cpu`; call with `cuda` only if you have a compatible GPU and CUDA configured.
- `--backend` selects the video loader (`opencv` or `torchvision`). If you encounter codec/decoding errors (e.g. AV1), try switching backend or re-encoding the video to H.264.
- Output: predictions are saved to `pitch_judgments.csv`. Videos that fail to decode are skipped and recorded with `Error` in the CSV.

Keep these parameters consistent between `train.py` and `judge_pitch.py` to ensure correct inference behavior.

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
  [av1 @ 0x39955f80] Your platform doesnt support hardware accelerated AV1 decoding.
  [av1 @ 0x39955f80] Failed to get pixel format.
  [av1 @ 0x39955f80] Get current frame error
  ```
  - Cause: Some videos in the dataset use the newer AV1 codec. Your system's FFmpeg/OpenCV installation seems to lack a proper software decoder or hardware support for AV1, causing it to fail completely for these specific files.
  - Impact: These specific videos are failing to load.
  - Status: Handled.<br>
    Your training loop is not crashing. Note the progress bar is moving (2% -> 5% -> 9%).
    This is because dataset.py catches these failures (returns None) and collate_fn filters them out. You are effectively training on the subset of videos that can be decoded (H.264), and skipping the AV1 ones.

---

# Baseball Pitch Classification 

一个模块化的 PyTorch 项目，用于使用 ResNet-18 特征提取器和可互换的时间头部（平均池化或 LSTM）对棒球投球视频进行分类（好球 vs 坏球）。

## 项目结构

- `dataset.py`：处理视频加载（支持 OpenCV 和 Torchvision）、预处理，并稳健地跳过损坏的视频。
- `models.py`：包含 `ResNetBackbone`、`TemporalPoolHead`、`LSTMHead` 和 `FullModel`。
- `train.py`：带有指标和检查点的 CLI 训练脚本。
- `utils.py`：辅助函数。
- `judge_pitch.py`: 投球好坏判罚

## 要求

- Python 3.8+
- PyTorch
- Torchvision
- Pandas
- OpenCV (`pip install opencv-python`) - 推荐用于稳健的视频加载，无需 FFmpeg 设置问题。

## 数据设置

项目期望以下结构：
- 标签 CSV：`./dataset/pitchcalls/labels.csv`（列：`id`、`clip_id`、`label`）
- 视频：`./dataset/videos/`（文件名为 `{id}.mp4`）

## 使用

### 1. 使用时间平均池化训练（基准）

默认后端是 `opencv`。这有效地实现了“无 FFmpeg 计划”（直接 FFmpeg 命令使用），通过使用 OpenCV 的内部解码器。

```bash
python train.py --arch pool --data-dir ./dataset/videos --epochs 10
```

### 2. 使用 LSTM 头部训练

```bash
python train.py --arch lstm --data-dir ./dataset/videos --epochs 10
```

### 3. 使用 Torchvision (FFmpeg) 后端

如果您已安装并配置了 FFmpeg 与 Torchvision，您可以使用：

```bash
python train.py --arch lstm --backend torchvision --data-dir ./dataset/videos
```

### 4. 烟雾测试 (CPU)

如果您没有数据集，只需指向不存在的目录。脚本将生成合成数据集以验证管道。

```bash
python train.py --data-dir ./dummy_path --epochs 2 --device cpu --batch-size 4
```

### 5. 推理：`judge_pitch.py`

使用 `judge_pitch.py` 对一个或多个本地视频文件运行推理。重要：确保推理参数与训练模型的方式匹配（架构、每剪辑帧数和设备）。不匹配的参数可能导致不正确的行为或预测。

示例使用：

```bash
python judge_pitch.py --device cuda --frames 8 --arch lstm --input-dir ./input_videos --model-path checkpoints/model_best.pth
```

注意：
- `--arch` 必须与训练时使用的架构匹配（例如 `lstm` 或 `pool`）。
- `--frames` 应与训练时使用的 `--frames` 值匹配。模型期望与训练时相同的每剪辑帧数。
- `--device` 可以是 `cuda` 或 `cpu`；仅在您有兼容 GPU 和 CUDA 配置时使用 `cuda` 调用。
- `--backend` 选择视频加载器（`opencv` 或 `torchvision`）。如果遇到编解码错误（例如 AV1），请尝试切换后端或将视频重新编码为 H.264。
- 输出：预测保存到 `pitch_judgments.csv`。无法解码的视频将被跳过并在 CSV 中记录为 `Error`。

在 `train.py` 和 `judge_pitch.py` 之间保持这些参数一致，以确保正确的推理行为。

## 故障排除

- **GPU OOM (内存不足)**：
  - 减少批大小：`--batch-size 4`
  - 减少每剪辑帧数：`--frames 8`
  - 冻结骨干：`--freeze-backbone`
- **视频加载错误**：
  - 数据集自动跳过损坏的视频。
  - 如果看到许多错误，请尝试切换后端：`--backend opencv` vs `--backend torchvision`。

- ```bash
  [h264 @ 0x386feac0] mmco: unref short failure
  ```
  - 原因：数据集中的一些 .mp4 视频文件在 H.264 流中有轻微编码故障或缺失参考帧。这在网络抓取数据集（如 MLB-YouTube）中很常见。
  - 影响：非致命。底层视频解码器（FFmpeg/OpenCV）只是记录流不完美，但它成功恢复并继续解码其余视频。您可以忽略这些。

- ```bash
  [av1 @ 0x39955f80] Your platform doesnt support hardware accelerated AV1 decoding.
  [av1 @ 0x39955f80] Failed to get pixel format.
  [av1 @ 0x39955f80] Get current frame error
  ```
  - 原因：数据集中的一些视频使用较新的 AV1 编解码器。您的系统 FFmpeg/OpenCV 安装似乎缺乏适当的软件解码器或硬件支持 AV1，导致这些特定文件完全失败。
  - 影响：这些特定视频无法加载。
  - 状态：已处理。<br>
    您的训练循环不会崩溃。请注意进度条在移动（2% -> 5% -> 9%）。
    这是因为 dataset.py 捕获这些失败（返回 None），collate_fn 过滤它们。您实际上在可以解码的视频子集（H.264）上训练，并跳过 AV1 视频。