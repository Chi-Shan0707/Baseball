"""Small utilities for training: reproducibility, metrics, checkpointing."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # numpy is optional; training does not require it.
        pass

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Good defaults for deterministic-ish behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_classification_metrics(logits: Tensor, targets: Tensor) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1 for binary classification.

    Args:
        logits: (N, 2) model outputs
        targets: (N,) integer labels in {0,1}

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """

    if logits.ndim != 2 or logits.size(1) != 2:
        raise ValueError("Expected logits shape (N, 2)")
    if targets.ndim != 1:
        targets = targets.view(-1)

    preds = logits.argmax(dim=1)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    args: Optional[Any] = None,
) -> None:
    """Save a training checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }

    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    if args is not None:
        if is_dataclass(args):
            payload["args"] = asdict(args)
        else:
            try:
                payload["args"] = vars(args)
            except Exception:
                payload["args"] = str(args)

    torch.save(payload, str(path))


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint into a model (and optionally optimizer)."""

    ckpt = torch.load(str(path), map_location=map_location)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    return ckpt


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save a JSON file with pretty formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
