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
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for frames, labels in pbar:
        if frames is None: # Handled by collate_fn but check just in case
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
        
        # Stats
        running_loss += loss.item() * frames.size(0)
        
        # Preds for metrics
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"loss": loss.item()})
        
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)
    return epoch_loss, metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in loader:
            if frames is None:
                continue
            
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * frames.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
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
    set_seed(42)
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
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0) # workers=0 for simplicity/compatibility
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        print(f"Dataset loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
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
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        
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

if __name__ == '__main__':
    main()
