import os
import random
import numpy as np
import torch
import shutil

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints'):
    """
    Save a checkpoint of the model and optimizer state.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth'))

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1 score.
    Args:
        y_true: list or array of true labels
        y_pred: list or array of predicted labels
    Returns:
        dict containing metrics
    """
    # Use sklearn if available, else manual
    # For a small project, manual is fine but sklearn is standard.
    # I'll implement manual to avoid extra dependency if not requested, 
    # but sklearn is usually assumed. 
    # Let's do a simple manual calculation for binary classification for zero dependency
    # or just use sklearn if user installs it. 
    # Given requirements: "minimal dependency list (torch, torchvision, opencv-python optional)"
    # So I will implement manually to avoid sklearn dependency.
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    # Assuming binary 0/1, but let's handle multi-class generally or just strictly binary as per prompt default.
    # Prompt says: "Default to binary classification (2 classes)"
    
    # Convert to numpy arrays for easier handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0.0
    
    # For binary metrics, usually class 1 is positive.
    # Let's assume class 1 is positive.
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
