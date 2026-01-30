import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time

from dataset import VideoDataset, collate_fn
from models import FullModel
from utils import set_seed, save_checkpoint, calculate_metrics

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch.

    Notes:
    - Use `total_examples` to count the actually processed samples because `collate_fn`
      may filter out invalid samples (so `len(loader.dataset)` can be larger than
      the number of examples actually used in this epoch).
    - Returns (epoch_loss, metrics). If no examples were processed, `epoch_loss` will be None.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_examples = 0
    
    pbar = tqdm(loader, desc="Training")
    for frames, labels in pbar:
        if frames is None:  # Handled by collate_fn but check just in case
            continue

        frames = frames.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Stats (accumulate by number of examples in this batch)
        batch_size = frames.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

        # Preds for metrics
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = (running_loss / total_examples) if total_examples > 0 else None
    metrics = calculate_metrics(all_labels, all_preds)
    return epoch_loss, metrics
"""
total_examples：在 evaluate 中用来累计已经处理的样本数（用于计算平均 loss）。因为有时会跳过无效样本，用 total_examples 更准确。🔢

frames.size(0)：返回当前 batch 的第 0 维大小，也就是 batch 大小 B。在这里 frames 形状是 (B, T, C, H, W)，因此：

frames.size(0) → B（该 batch 的样本数）
frames.size(1) → T（每个样本的帧数）
"""
def evaluate(model, loader, criterion=None, device='cpu'):
    """Evaluate model on a dataloader.

    Args:
        model: nn.Module
        loader: DataLoader
        criterion: loss function or None (if None, loss is not computed)
        device: device string or torch.device

    Returns:
        epoch_loss (float or None), metrics (dict)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_examples = 0
    
    with torch.no_grad():
        for frames, labels in loader:
            if frames is None:
                continue
            
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * frames.size(0)
            total_examples += frames.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = (running_loss / total_examples) if (criterion is not None and total_examples > 0) else None
    metrics = calculate_metrics(all_labels, all_preds)
    return epoch_loss, metrics

def main():
    parser = argparse.ArgumentParser(description="Train video classification model.")
    parser.add_argument('--data-dir', type=str, default='./dataset/videos', help='Path to videos')
    parser.add_argument('--labels-file', type=str, default='./dataset/pitchcalls/labels.csv', help='Path to labels CSV')
    parser.add_argument('--arch', type=str, default='pool', choices=['pool', 'lstm'], help='Model architecture')
    parser.add_argument('--frames', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze ResNet backbone')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--backend', type=str, default='opencv', choices=['opencv', 'torchvision'], help='Video loading backend')
    args = parser.parse_args()
    
    # Set seed
    set_seed(20260130)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Dataset
    print("Loading dataset...")
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"Warning: {args.data_dir} not found. Creating synthetic dataset for smoke test.")
        # Create dummy data for smoke test to work without files?
        # The user asked for "default synthetic splitter if none given" or "Provide clear defaults so a user can run a smoke test... if no videos available"
        # I'll handle this by generating random tensors if dataset fails to load logic?
        # Actually, let's implement a 'SyntheticDataset' class or just mock inside dataset if path missing?
        # User said: "provide default synthetic splitter if none given" - likely refers to Train/Val split.
        # But also: "Provide clear defaults so a user can run a smoke test... using a tiny synthetic dataset"
        pass

    try:
        full_dataset = VideoDataset(
            data_dir=args.data_dir, 
            labels_file=args.labels_file, 
            frames_per_clip=args.frames,
            backend=args.backend
        )
        
        # Split Train/Val (80/20)
        train_size = int(0.6 * len(full_dataset))
        val_size =   int(0.2 * len(full_dataset))
        test_size =  len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0) # workers=0 for simplicity/compatibility
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        """
    num_workers 是 PyTorch DataLoader 中的一个参数，用于指定用于数据加载的子进程数量。它控制并行加载数据的进程数，以提高数据加载效率。

设置为 0 表示使用主进程（单进程）进行数据加载，没有额外的子进程。这在以下情况下是有效的：

调试时，避免多进程带来的复杂性。
兼容性问题（如某些环境不支持多进程）。
小数据集或简单场景下，单进程足够且更稳定。
        """
        print(f"Dataset loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to Synthetic Dataset for smoke test.")
        
        # Synthetic dataset implementation
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, length=100):
                self.length = length
            def __len__(self):
                return self.length
            def __getitem__(self, idx):
                # (T, C, H, W)
                return torch.randn(args.frames, 3, 224, 224), torch.randint(0, 2, (1,)).item()
        
        train_dataset = SyntheticDataset(100)
        val_dataset = SyntheticDataset(20)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=None) # No collate needed for clean synthetic
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    print(f"Initializing model: {args.arch}")
    model = FullModel(arch=args.arch, num_classes=2, freeze_backbone=args.freeze_backbone)
    model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_acc = 0.0
    
    # Training Loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f} | F1: {train_metrics['f1']:.4f}")
        
        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        # Scheduler
        scheduler.step()
        
        # Save best
        is_best = val_metrics['accuracy'] > best_acc
        if is_best:
            best_acc = val_metrics['accuracy']
            print("New best model!")
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        
    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")


    print("#######################################################\nStarting Test Evaluation")
    print(f"Model after {args.epochs} epochs:\n")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Acc: {test_metrics['accuracy']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")


    print("-------------------------------------------------")
    print("\nReloading best model for final test evaluation...")
    # 构建模型实例：
    # - `FullModel(arch=args.arch, num_classes=2, freeze_backbone=False)`
    #   * `arch=args.arch`：从命令行参数选择模型变体（例如 'pool' 或 'lstm'），决定时间维度上的聚合方式。
    #   * `num_classes=2`：二分类任务（例如球/好球）。
    #   * `freeze_backbone=False`：是否冻结 ResNet 背骨（训练时可选）。
    # 注意：新创建的 `nn.Module`（包括其参数和 buffer）默认位于 CPU（即参数类型为 `torch.FloatTensor`），
    # 而训练/评估循环中我们会把输入 `frames` 用 `frames.to(device)` 移到 `device`（例如 GPU），
    # 如果模型仍在 CPU 而输入在 CUDA，会出现设备/类型不一致错误，例如：
    # "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same"。
    # 因此在加载权重或运行前明确调用 `model.to(device)`，把模型的参数和 buffers 转到相同设备，
    # 保证权重、模型与输入在同一设备/类型上。注意我们也在加载时用 `torch.load(..., map_location=device)`
    # 以确保 checkpoint 的张量被映射到同一设备。
    model = FullModel(arch=args.arch, num_classes=2, freeze_backbone=False)
    model = model.to(device)  # 将模型参数和 buffers 转移到 `device`（cuda 或 cpu），以保持输入/权重一致性
    ckpt_path = os.path.join('checkpoints', 'model_best.pth')
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        # If file contains a full checkpoint dict, extract state_dict
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        print(f"Loaded weights from {ckpt_path}")

        # Evaluate loaded best model
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Acc: {test_metrics['accuracy']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")

    except Exception as e:
        print(f"Failed to load best model weights from {ckpt_path}: {e}")
        print("Skipping reload of best model.")

if __name__ == '__main__':
    main()
