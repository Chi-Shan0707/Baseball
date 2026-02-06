import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

# --- 1. 引入你现有的模块 ---
# 我们复用 dataset.py 的加载逻辑和 utils.py 的指标计算
from dataset import VideoDataset, collate_fn
from models import FullModel
from utils import calculate_metrics

# --- 2. 定义各种自定义图像处理策略 ---
def get_diagnostic_transform(mode='normal'):
    """
    根据模式生成不同的图像预处理流程 (Transform)。
    
    Args:
        mode (str): 
            'normal': 正常处理 (Resize -> Norm)
            'block_center': 遮挡中心 (模拟看不见捕手手套)
            'block_periphery': 遮挡四周 (模拟只看中心)
            'noise': 添加高斯噪声
    """
    # 基础尺寸调整
    transform_list = [transforms.Resize((224, 224))]

    # --- 核心：这里是我们可以“动以此手脚”的地方 ---
    if mode == 'normal':
        pass # 不做额外操作
        
    elif mode == 'block_center':
        # 遮挡中心 100x100 区域 (假设捕手手套在这里)
        def erase_center(img):
            # img 是 PIL Image。用 PIL 在中心画黑色矩形以遮挡
            left = 62
            top = 62
            right = left + 100
            bottom = top + 100
            draw = ImageDraw.Draw(img)
            draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))
            return img
        transform_list.append(transforms.Lambda(erase_center))
        
    elif mode == 'block_periphery':
        # 遮挡四周，只留中心 
        def keep_center_only(img):
            # 使用 PIL：创建黑色背景并把中心区域粘回去，保留宽=100, 高=160
            w, h = img.size  # 预期 224x224
            center_w = 100
            center_h = 160
            left = (w - center_w) // 2
            top = (h - center_h) // 2
            # 裁剪中心并粘回到黑底
            center = img.crop((left, top, left + center_w, top + center_h))
            new = Image.new(img.mode, (w, h), (0, 0, 0))
            new.paste(center, (left, top))
            return new
        transform_list.append(transforms.Lambda(keep_center_only))

    elif mode == 'grayscale':
        # 转灰度 (测试颜色是否重要)
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    # --- 必须保留的后续步骤 ---
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)

def main():
    parser = argparse.ArgumentParser(description="Model Diagnosis Tool")
    parser.add_argument('--data-dir', type=str, default='./dataset/videos', help='Video directory')
    # 注意：这里需要带标签的CSV，因为我们要计算 Accuracy
    parser.add_argument('--labels-file', type=str, default='./dataset/pitchcalls/labels.csv', help='Labels CSV')
    parser.add_argument('--model-path', type=str, default='checkpoints/model_best.pth', help='Path to checkpoint')
    parser.add_argument('--arch', type=str, default='pool', choices=['pool', 'lstm'], help='Model architecture')
    parser.add_argument('--frames', type=int, default=16, help='Frames per clip')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 诊断模式开关
    parser.add_argument('--mode', type=str, default='normal', 
                        choices=['normal', 'block_center', 'block_periphery', 'grayscale'],
                        help='Diagnostic mode to apply to images')
    parser.add_argument('--temporal-mode', type=str, default='full',
                        choices=['full', 'second_half', 'first_half'],
                        help='Temporal slicing: full (all frames), second_half (后半截), first_half (前半截)')

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"--- Starting Diagnosis ---")
    print(f"Mode: {args.mode}")
    print(f"Temporal Mode: {args.temporal_mode}")
    print(f"Device: {device}")

    # 1. 获取特定的 Transform
    custom_transform = get_diagnostic_transform(args.mode)

    # 2. 加载数据集
    # 我们直接使用 dataset.py 里的 VideoDataset，但是传入我们的 custom_transform
    # 它会自动处理视频读取、采样、以及遇到坏视频返回 None 的逻辑
    # 如果我们要只取前/后半截，并且用户指定了 --frames 为目标帧数
    # 那么先从 dataset 请求两倍帧数，再在推理时切片为目标帧数，保证最终输入长度为 args.frames
    desired_frames = args.frames
    if args.temporal_mode in ['first_half', 'second_half']:
        dataset_frames = desired_frames * 2
    else:
        dataset_frames = desired_frames

    val_dataset = VideoDataset(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        frames_per_clip=dataset_frames,
        transform=custom_transform, # <--- 注入自定义处理
        backend='opencv' # 默认使用 OpenCV，稳健
    )

    # 使用 dataset.py 里的 collate_fn 来处理 None (坏视频)
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                        collate_fn=collate_fn, num_workers=0)

    print(f"Dataset loaded. Total samples: {len(val_dataset)}")

    # 3. 加载模型
    model = FullModel(arch=args.arch, num_classes=2, freeze_backbone=False)
    model = model.to(device)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} (Best Acc: {checkpoint.get('best_acc', '?')})")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Checkpoint not found at {args.model_path}")
        return

    model.eval()

    # 4. 推理循环
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for frames, labels in tqdm(loader):
            if frames is None:
                continue # collate_fn 已经处理了大部分，但双重保险
            
            # 时序切片：根据 temporal_mode 只取部分帧
            # frames 形状: (B, T, C, H, W)
            if args.temporal_mode == 'second_half':
                t = frames.size(1)
                frames = frames[:, t//2:, :, :, :]  # 取后半截
            elif args.temporal_mode == 'first_half':
                t = frames.size(1)
                frames = frames[:, :t//2, :, :, :]  # 取前半截
            # 'full' 则不做任何操作
            
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. 统计指标
    # 使用 utils.py 里的 calculate_metrics
    metrics = calculate_metrics(all_labels, all_preds)
    
    valid_count = len(all_labels)
    skipped_count = len(val_dataset) - valid_count
    
    print("\n" + "="*40)
    print(f" DIAGNOSIS REPORT: {args.mode.upper()} + {args.temporal_mode.upper()}")
    print("="*40)
    print(f"Processed Videos : {valid_count}")
    print(f"Skipped/Error    : {skipped_count}")
    print("-" * 40)
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1 Score         : {metrics['f1']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()