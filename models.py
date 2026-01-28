"""Model components for per-frame ResNet features + temporal classification heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor


class ResNetBackbone(nn.Module):
    """Per-frame ResNet-18 feature extractor.

    This module applies a standard 2D ResNet-18 to each frame independently.

    Input:
        x: Tensor of shape (B, T, C, H, W)

    Output:
        feats: Tensor of shape (B, T, D) where D=512 for ResNet-18.

    Notes:
        - We remove the final FC layer and use global average pooled features.
        - To process per-frame efficiently, we reshape (B,T) -> (B*T) before
          forwarding through the CNN, then reshape back.
    """

    def __init__(self, freeze: bool = False, weights: Optional[str] = "default") -> None:
        super().__init__()
        self.freeze = bool(freeze)

        # Lazy import so this file still imports even if torchvision isn't installed.
        try:
            import torchvision
            from torchvision.models import ResNet18_Weights

            if weights == "default":
                w = ResNet18_Weights.DEFAULT
            elif weights is None:
                w = None
            else:
                # Allow passing a string to be explicit.
                w = ResNet18_Weights.DEFAULT

            try:
                resnet = torchvision.models.resnet18(weights=w)
            except Exception:
                # Common case: offline environment, weights download fails.
                resnet = torchvision.models.resnet18(weights=None)

        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "torchvision is required for ResNetBackbone. Install torchvision."
            ) from e

        # Remove the final classification layer.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (N, 512, 1, 1)
        self.feature_dim = 512

        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "ResNetBackbone":
        super().train(mode)
        # If frozen, keep backbone in eval mode to avoid BN stat updates.
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got shape={tuple(x.shape)}")

        b, t, c, h, w = x.shape
        x_2d = x.reshape(b * t, c, h, w)

        feats = self.backbone(x_2d)  # (B*T, 512, 1, 1)
        feats = feats.flatten(1)  # (B*T, 512)

        feats = feats.reshape(b, t, self.feature_dim)
        return feats


class TemporalPoolHead(nn.Module):
    """Temporal average pooling head.

    Input:
        feats: Tensor (B, T, D)
    Output:
        logits: Tensor (B, num_classes)
    """

    def __init__(self, feature_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, feats: Tensor) -> Tensor:
        if feats.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got shape={tuple(feats.shape)}")
        pooled = feats.mean(dim=1)
        return self.classifier(pooled)


class LSTMHead(nn.Module):
    """Single-layer LSTM temporal head.

    Input:
        feats: Tensor (B, T, D)
    Output:
        logits: Tensor (B, num_classes)

    We take the last hidden state as the sequence representation.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 256,
        num_classes: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bool(bidirectional)

        self.lstm = nn.LSTM(
            input_size=int(feature_dim),
            hidden_size=int(hidden_size),
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        out_dim = hidden_size * (2 if self.bidirectional else 1)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, feats: Tensor) -> Tensor:
        if feats.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got shape={tuple(feats.shape)}")

        _out, (h_n, _c_n) = self.lstm(feats)
        # h_n: (num_layers * num_directions, B, hidden)

        if self.bidirectional:
            # Last layer forward and backward states
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            h = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            h = h_n[-1]

        return self.classifier(h)


class FullModel(nn.Module):
    """Convenience wrapper: ResNetBackbone + a temporal head.

    Args:
        arch: "pool" or "lstm"
        num_classes: number of output classes (default 2)
        freeze_backbone: freeze ResNet parameters
    """

    def __init__(
        self,
        arch: Literal["pool", "lstm"] = "pool",
        num_classes: int = 2,
        freeze_backbone: bool = False,
        lstm_hidden: int = 256,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = ResNetBackbone(freeze=freeze_backbone)
        if arch == "pool":
            self.head: nn.Module = TemporalPoolHead(
                feature_dim=self.backbone.feature_dim,
                num_classes=num_classes,
            )
        elif arch == "lstm":
            self.head = LSTMHead(
                feature_dim=self.backbone.feature_dim,
                hidden_size=lstm_hidden,
                num_classes=num_classes,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("arch must be 'pool' or 'lstm'")

        self.arch = arch
        self.num_classes = int(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits
