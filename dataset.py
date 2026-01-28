"""Dataset utilities for short baseball clips.

This module provides a PyTorch `Dataset` that:
- Loads a short video clip from disk.
- Samples N frames uniformly across the clip.
- Applies basic preprocessing and ImageNet normalization (matching ResNet weights).

Backend choice
-------------
`torchvision.io.read_video` is deprecated in torchvision >= 0.22.

This dataset supports multiple decoding backends:
- **TorchCodec** (recommended when installed): `torchcodec.decoders.VideoDecoder`
- **TorchVision** (legacy / deprecated): `torchvision.io.read_video`
- **OpenCV** fallback: `opencv-python`

By default (`backend="auto"`) we try TorchCodec first (if available), then
TorchVision, then OpenCV.

Expected project data layout (recommended)
-----------------------------------------
Your dataset appears as:
- dataset/videos/0.mp4, 1.mp4, ...
- dataset/pitchcalls/labels.csv  (with columns: id, clip_id, label)

The `id` field maps to the video filename `{id}.mp4`.

All tensors returned are float32 with shape (T, C, H, W).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Set

import torch
from torch import Tensor
from torch.utils.data import Dataset

# TorchCodec (recommended)
try:
    from torchcodec.decoders import VideoDecoder as TorchCodecVideoDecoder  # type: ignore

    _HAS_TORCHCODEC = True
except Exception:
    TorchCodecVideoDecoder = None  # type: ignore
    _HAS_TORCHCODEC = False



try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

from torchvision import transforms


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VideoSample:
    """A single (video_path, label) record."""

    video_path: Path
    label: int


def uniform_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """Return `num_frames` indices uniformly spanning `[0, total_frames-1]`.

    If `total_frames < num_frames`, indices will repeat the last available frame.
    """

    if total_frames <= 0:
        # Caller should handle empty video separately.
        return [0] * num_frames

    if num_frames <= 1:
        return [min(total_frames - 1, 0)]

    if total_frames >= num_frames:
        # linspace-like sampling without numpy dependency
        step = (total_frames - 1) / float(num_frames - 1)
        return [int(round(i * step)) for i in range(num_frames)]

    # Not enough frames: take all frames, then repeat last
    indices = list(range(total_frames))
    indices += [total_frames - 1] * (num_frames - total_frames)
    return indices


def default_frame_transform(image_size: int = 224) -> Callable[[Tensor], Tensor]:
    """Default preprocessing to match ResNet-18 pretrained normalization.

    Input frame: Tensor (C, H, W), dtype float32 in [0, 1].
    Output frame: Tensor (C, image_size, image_size), normalized.
    """

    return transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )



def _read_all_frames_torchcodec(video_path: Path) -> Tensor:
    """Read all frames using TorchCodec.

    Returns:
        video: Tensor of shape (T, H, W, C), dtype uint8.

    Notes:
        TorchCodec's VideoDecoder indexing returns tensors shaped (N, C, H, W).
        We convert to (T, H, W, C) to match the other backends.
    """

    if not _HAS_TORCHCODEC or TorchCodecVideoDecoder is None:
        raise RuntimeError("TorchCodec backend unavailable (pip install torchcodec)")

    decoder = TorchCodecVideoDecoder(str(video_path), device="cpu")
    frames_chw = decoder[:]  # uint8 tensor (T, C, H, W)
    if frames_chw.ndim != 4:
        raise RuntimeError(f"Unexpected TorchCodec frame tensor shape: {tuple(frames_chw.shape)}")

    video = frames_chw.permute(0, 2, 3, 1).contiguous()  # (T, H, W, C)
    return video


def _read_all_frames_cv2(video_path: Path) -> Tensor:
    """Read all frames using OpenCV.

    Returns:
        video: Tensor of shape (T, H, W, C), dtype uint8.

    Notes:
        OpenCV reads in BGR; we convert to RGB.
    """

    if not _HAS_CV2 or cv2 is None:
        raise RuntimeError("OpenCV backend unavailable (install opencv-python)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[Tensor] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    video = torch.stack(frames, dim=0)  # (T, H, W, C), uint8
    return video


def read_video_frames(video_path: Path, backend: str = "auto") -> Tensor:
    """Read all frames from a video file.

    Args:
        video_path: Path to a video file.
        backend: "auto" | "torchcodec" | "cv2".

    Returns:
        Tensor of shape (T, H, W, C), dtype uint8.
    """

    if backend not in {"auto", "torchcodec", "cv2"}:
        raise ValueError(f"Invalid backend: {backend}")

    if backend in {"auto", "torchcodec"}:
        try:
            return _read_all_frames_torchcodec(video_path)
        except Exception:
            if backend == "torchcodec":
                raise


    # Fallback
    return _read_all_frames_cv2(video_path)


def load_bad_video_ids_from_json(json_path: Path) -> Set[str]:
    """Load bad video IDs from a health report JSON.

    Expected JSON format:
        {
          "damaged_videos": [
            {"filename": "884.mp4", ...},
            {"filename": "470.mp4", ...}
          ]
        }

    Returns:
        Set of video IDs (without extension), e.g. {"884", "470"}.
    """

    if not json_path.is_file():
        raise FileNotFoundError(f"Bad videos JSON not found: {json_path}")

    with json_path.open("r") as f:
        data = json.load(f)

    bad_ids: Set[str] = set()
    if "damaged_videos" in data:
        for entry in data["damaged_videos"]:
            if "filename" in entry:
                fname = entry["filename"]
                # Strip extension: "884.mp4" -> "884"
                video_id = Path(fname).stem
                bad_ids.add(video_id)

    return bad_ids


def resolve_dataset_paths(data_dir: Path) -> Tuple[Path, Path]:
    """Resolve expected `videos/` and `pitchcalls/labels.csv` locations.

    Supports passing either:
    - the dataset root (contains `videos/` and `pitchcalls/`), OR
    - the project root (contains `dataset/` folder).
    """

    # Case 1: user passed ./dataset
    videos_dir = data_dir / "videos"
    labels_csv = data_dir / "pitchcalls" / "labels.csv"
    if videos_dir.is_dir() and labels_csv.is_file():
        return videos_dir, labels_csv

    # Case 2: user passed project root
    videos_dir = data_dir / "dataset" / "videos"
    labels_csv = data_dir / "dataset" / "pitchcalls" / "labels.csv"
    if videos_dir.is_dir() and labels_csv.is_file():
        return videos_dir, labels_csv

    raise FileNotFoundError(
        "Could not locate dataset. Expected either:\n"
        "- <data_dir>/videos and <data_dir>/pitchcalls/labels.csv\n"
        "- <data_dir>/dataset/videos and <data_dir>/dataset/pitchcalls/labels.csv"
    )


def load_pitchcalls_samples(
    data_dir: Path, bad_videos_json: Optional[Path] = None
) -> List[VideoSample]:
    """Load (video_path, label) samples for the pitchcalls dataset.

    The CSV format is expected to include: `id`, `label`.
    Video path is resolved as: `<videos_dir>/{id}.mp4`.

    Labels:
        "strike" -> 1
        "ball"   -> 0

    Args:
        data_dir: Path to dataset root.
        bad_videos_json: Optional path to a health report JSON listing bad videos.
                         If provided, videos in that list are skipped upfront.

    Returns:
        List of `VideoSample`.
    """

    videos_dir, labels_csv = resolve_dataset_paths(data_dir)

    # Load bad video IDs if JSON provided
    bad_ids: Set[str] = set()
    if bad_videos_json is not None:
        bad_ids = load_bad_video_ids_from_json(bad_videos_json)
        print(f"[INFO] Loaded {len(bad_ids)} bad video IDs from {bad_videos_json.name}")

    samples: List[VideoSample] = []
    skipped_count = 0
    with labels_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "id" not in row or "label" not in row:
                raise ValueError(
                    "labels.csv must contain columns 'id' and 'label' (got: "
                    f"{list(row.keys())})"
                )

            video_id = row["id"].strip()
            label_str = row["label"].strip().lower()

            if label_str not in {"strike", "ball"}:
                raise ValueError(f"Unexpected label '{label_str}' in labels.csv")

            # Skip if in bad video list
            if video_id in bad_ids:
                skipped_count += 1
                continue

            label = 1 if label_str == "strike" else 0
            video_path = videos_dir / f"{video_id}.mp4"
            if not video_path.is_file():
                # Keep going but warn. Train script can decide whether to drop.
                continue

            samples.append(VideoSample(video_path=video_path, label=label))

    if skipped_count > 0:
        print(f"[INFO] Pre-filtered {skipped_count} bad videos from bad_videos_json")

    if not samples:
        raise RuntimeError(
            f"No valid samples found. Checked {labels_csv} and {videos_dir}"
        )

    return samples


class PitchCallVideoDataset(Dataset[Tuple[Tensor, int]]):
    """Video dataset returning a fixed number of frames per clip.

    Each item returns:
        frames: Tensor (T, C, H, W), float32
        label: int (0 or 1)
    """

    def __init__(
        self,
        samples: Sequence[VideoSample],
        num_frames: int = 32,
        image_size: int = 224,
        backend: str = "auto",
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        skip_bad_videos: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.num_frames = int(num_frames)
        self.backend = backend
        self.transform = transform or default_frame_transform(image_size=image_size)
        self.skip_bad_videos = bool(skip_bad_videos)
        self.skipped_count = 0
        self.image_size = int(image_size)

        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        sample = self.samples[index]

        try:
            # Read all frames (short 7s clips, so this is typically fine)
            video = read_video_frames(sample.video_path, backend=self.backend)
            # video: (T, H, W, C), uint8

            total_frames = int(video.shape[0])
            indices = uniform_frame_indices(total_frames, self.num_frames)

            selected = video[indices]  # (T, H, W, C)
            # Convert to (T, C, H, W) and float32 in [0, 1]
            frames = selected.permute(0, 3, 1, 2).contiguous().float() / 255.0

            # Apply transforms per-frame for clarity (beginner-friendly).
            processed: List[Tensor] = []
            for t in range(frames.shape[0]):
                processed.append(self.transform(frames[t]))

            clip = torch.stack(processed, dim=0)  # (T, C, H, W)
            return clip, int(sample.label)
        except Exception as e:
            if self.skip_bad_videos:
                # Silently skip this video and return a synthetic placeholder
                self.skipped_count += 1
                print(
                    f"[WARN] Skipping bad video {index}/{len(self)} "
                    f"(path={sample.video_path.name}): {str(e)[:80]}"
                )
                # Return synthetic random clip so training can continue
                clip = torch.rand(
                    (self.num_frames, 3, self.image_size, self.image_size),
                    dtype=torch.float32,
                )
                return clip, int(sample.label)
            else:
                print(
                    f"[ERROR] Failed to decode video {index}/{len(self)}: "
                    f"{sample.video_path}\nReason: {str(e)[:150]}"
                )
                raise




class SyntheticVideoDataset(Dataset[Tuple[Tensor, int]]):
    """Tiny synthetic dataset for smoke tests when no videos are available."""

    def __init__(
        self,
        num_samples: int = 64,
        num_frames: int = 8,
        image_size: int = 224,
        num_classes: int = 2,
        seed: int = 0,
    ) -> None:
        self.num_samples = int(num_samples)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.generator = torch.Generator().manual_seed(int(seed))

        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        _ = index
        frames = torch.rand(
            (self.num_frames, 3, self.image_size, self.image_size),
            generator=self.generator,
            dtype=torch.float32,
        )
        label = int(torch.randint(0, self.num_classes, (1,), generator=self.generator))
        return frames, label
