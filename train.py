"""Training script for per-frame ResNet-18 + temporal classifier.

Examples:
    python train.py --arch pool --data-dir ./dataset --epochs 10
    python train.py --arch lstm --data-dir ./dataset --epochs 10

If real videos are not available, the script automatically falls back to a tiny
synthetic dataset so you can do a CPU smoke test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from dataset import PitchCallVideoDataset, SyntheticVideoDataset, load_pitchcalls_samples
from models import FullModel
from utils import binary_classification_metrics, save_checkpoint, save_json, set_seed


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    arch: str
    frames: int
    batch_size: int
    lr: float
    epochs: int
    freeze_backbone: bool
    device: str
    num_workers: int
    backend: str
    skip_bad_videos: bool
    bad_videos_json: Path | None
    seed: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a simple video classifier")

    parser.add_argument("--data-dir", type=str, default="", help="Path to dataset root")
    parser.add_argument(
        "--arch",
        type=str,
        default="pool",
        choices=["pool", "lstm"],
        help="Temporal classifier head",
    )
    parser.add_argument("--frames", type=int, default=16, help="Number of frames sampled")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ResNet backbone (faster, less memory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. If you see 'worker killed by signal: Killed', "
            "set this to 0 (safest) or 1."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torchcodec", "torchvision", "cv2"],
        help="Video decoding backend: auto (try torchcodec, then torchvision, then cv2), or force one.",
    )
    parser.add_argument(
        "--skip-bad-videos",
        action="store_true",
        help="Skip corrupted videos with a warning instead of crashing (uses synthetic placeholders).",
    )
    parser.add_argument(
        "--bad-videos-json",
        type=str,
        default=None,
        help=(
            "Path to JSON file listing bad videos (e.g., video_health_report.json). "
            "If provided, those videos are excluded upfront for max efficiency."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return TrainConfig(
        data_dir=Path(args.data_dir) if args.data_dir else Path("."),
        arch=args.arch,
        bad_videos_json=Path(args.bad_videos_json) if args.bad_videos_json else None,
        frames=args.frames,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        freeze_backbone=bool(args.freeze_backbone),
        device=args.device,
        num_workers=int(args.num_workers),
        backend=str(args.backend),
        skip_bad_videos=bool(args.skip_bad_videos),
        seed=args.seed,
    )


def split_indices(n: int, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Deterministic train/val split."""

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    val_n = int(round(n * val_ratio))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    return train_idx, val_idx


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    """Run evaluation and return loss + metrics."""

    model.eval()

    total_loss = 0.0
    total_items = 0
    all_logits: List[Tensor] = []
    all_targets: List[Tensor] = []

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        logits = model(frames)
        loss = loss_fn(logits, labels)

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = binary_classification_metrics(logits_cat, targets_cat)

    metrics["loss"] = (total_loss / total_items) if total_items > 0 else 0.0
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One training epoch. Returns average loss."""

    model.train()

    total_loss = 0.0
    total_items = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        # 1) Forward
        logits = model(frames)
        loss = loss_fn(logits, labels)

        # 2) Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 3) Update parameters
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return (total_loss / total_items) if total_items > 0 else 0.0


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Falls back to a synthetic dataset if real videos cannot be found.
    """

    try:
        samples = load_pitchcalls_samples(cfg.data_dir, bad_videos_json=cfg.bad_videos_json)
        dataset = PitchCallVideoDataset(
            samples=samples,
            num_frames=cfg.frames,
            backend=cfg.backend,
            skip_bad_videos=cfg.skip_bad_videos,
        )

        train_idx, val_idx = split_indices(len(dataset), seed=cfg.seed)
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        print(f"Loaded {len(dataset)} real samples")
        if cfg.skip_bad_videos:
            print("[INFO] Bad video skipping is ENABLED. Corrupted clips will be replaced with synthetic placeholders.")

    except Exception as e:
        print("WARNING: Falling back to synthetic dataset because real data was not usable.")
        print(f"Reason: {e}")

        train_ds = SyntheticVideoDataset(num_samples=64, num_frames=min(cfg.frames, 8), seed=cfg.seed)
        val_ds = SyntheticVideoDataset(num_samples=32, num_frames=min(cfg.frames, 8), seed=cfg.seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        persistent_workers=(cfg.num_workers > 0),
    )

    return train_loader, val_loader


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    train_loader, val_loader = build_dataloaders(cfg)

    model = FullModel(arch=cfg.arch, freeze_backbone=cfg.freeze_backbone)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Simple epoch-based scheduler (kept intentionally beginner-friendly)
    step_size = max(1, cfg.epochs // 3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    checkpoints_dir = Path("checkpoints")
    best_path = checkpoints_dir / f"best_{cfg.arch}.pt"
    results_path = checkpoints_dir / f"final_eval_{cfg.arch}.json"

    best_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        val_metrics = evaluate(model, val_loader, device, loss_fn)

        # 4) Scheduler step (once per epoch)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.3f} | prec={val_metrics['precision']:.3f} | "
            f"rec={val_metrics['recall']:.3f} | f1={val_metrics['f1']:.3f}"
        )

        # Save best checkpoint based on F1
        if val_metrics["f1"] > best_f1:
            best_f1 = float(val_metrics["f1"])
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                args=cfg,
            )

    # Final evaluation: load best checkpoint and report metrics
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
        print(f"Loaded best checkpoint from {best_path} (epoch={ckpt.get('epoch')})")

    final_metrics = evaluate(model, val_loader, device, loss_fn)

    # Make config JSON-serializable (e.g., Path -> str)
    config_json = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in cfg.__dict__.items()
    }
    save_json(results_path, {"config": config_json, "metrics": final_metrics})
    print(f"Saved final evaluation to {results_path}")


if __name__ == "__main__":
    main()
