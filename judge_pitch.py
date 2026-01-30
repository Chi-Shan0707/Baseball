import argparse
import os
import torch
import torch.nn.functional as F
from models import FullModel
from dataset import VideoDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

#Define a minimal Dataset for inference that reuses logic where possible
#or just re-implements the loading logic to avoid dependency on labels.csv
class InferenceDataset(VideoDataset):
    """
    Inherits from VideoDataset but adapts for inference (no labels file needed).
    """
    def __init__(self, input_dir, frames_per_clip=16, backend='opencv'):
        # We don't call super().__init__ because it requires a labels file.
        # instead we initialize what we need.
        self.data_dir = input_dir
        self.frames_per_clip = frames_per_clip
        self.backend = backend
        
        # Discover files
        valid_exts = ['.mp4', '.avi', '.mov', '.mkv']
        self.video_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        self.video_files.sort()
        
        # Check backend availability (copied logic)
        try:
            import cv2
            self.has_opencv = True
        except ImportError:
            self.has_opencv = False
            
        try:
            from torchvision.io import read_video
            self.has_torchvision = True
        except ImportError:
            self.has_torchvision = False
            
        if self.backend == 'opencv' and not self.has_opencv:
            print("OpenCV not found, falling back to torchvision.")
            self.backend = 'torchvision'
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.video_files)
        
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.data_dir, video_name)
        
        try:
            # We reuse the _load_video method from VideoDataset.
            # But wait, VideoDataset methods are instance methods relying on 'self' attributes.
            # We have set self.backend, self.frames_per_clip, self.transform.
            # We need to make sure we have everything _load_video needs.
            # _load_video calls _load_video_opencv or _load_video_torchvision.
            # They call _sample_frames, _get_indices.
            # All these are available if we inhereted correct methods.
            
            # Since we didn't call super().__init__, we need to make sure we didn't miss anything.
            # VideoDataset methods used: _load_video, _load_video_opencv, _load_video_torchvision, _sample_frames, _get_indices
            
            # We can just call the method:
            frames = self._load_video(video_path)
            
            if frames is None:
                return video_name, None
                
            return video_name, frames
            
        except Exception as e:
            # print(f"Error loading {video_name}: {e}")
            return video_name, None

def main():
    parser = argparse.ArgumentParser(description="Judge Pitch: Predict Strike/Ball from Videos")
    parser.add_argument('--input-dir', type=str, default='./input_videos', help='Directory containing video files')
    parser.add_argument('--model-path', type=str, default='checkpoints/model_best.pth', help='Path to model checkpoint')
    parser.add_argument('--arch', type=str, default='pool', choices=['pool', 'lstm'], help='Model architecture used for training')
    parser.add_argument('--frames', type=int, default=16, help='Number of frames to sample')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--backend', type=str, default='opencv', choices=['opencv', 'torchvision'])
    args = parser.parse_args()

    # Create input directory if it doesn't exist to avoid crash, though user should provide it
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    print(f"Loading model from {args.model_path}...")
    device = torch.device(args.device)
    
    # Initialize model
    # Note: We need to know if the model was an LSTM or Pool model. 
    # Usually checkpoints save the architecture name, but our save_checkpoint in utils.py 
    # only saved state_dict. So user must specify --arch matching the training.
    model = FullModel(arch=args.arch, num_classes=2, freeze_backbone=False ) # Freeze doesn't matter for inference
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device,weights_only=  True)
        # Check if 'state_dict' key exists (standard) or if it's the state_dict itself
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} with Best Acc: {checkpoint.get('best_acc', '?')}")
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    print(f"Processing videos in {args.input_dir}...")
    
    # Setup Dataset
    dataset = InferenceDataset(args.input_dir, frames_per_clip=args.frames, backend=args.backend)

    # Custom collate to handle videos that failed to decode (where __getitem__ returns (name, None)).
    # Default collate will raise on NoneTypes (TypeError), so we explicitly handle that case:
    # 即处理破烂视频的情况
    def _inference_collate(batch):
        # batch is a list of tuples (video_name, frames_or_None)
        video_names = [item[0] for item in batch]
        frames_list = [item[1] for item in batch]
        # If any sample failed to load/decode, return names and None so the loop can skip gracefully
        if any(f is None for f in frames_list):
            return video_names, None
        import torch
        frames = torch.stack(frames_list, dim=0)
        return video_names, frames

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_inference_collate)
    
    results = []
    
    with torch.no_grad():
        for video_name, frames in tqdm(loader):
            video_name = video_name[0] # Tuple of 1
            
            if frames is None:
                print(f"[Skipped] {video_name}: Could not load/decode.")
                results.append({'video': video_name, 'prediction': 'Error', 'confidence': 0.0})
                continue
                
            frames = frames.to(device) # (1, T, C, H, W)
            
            outputs = model(frames) # (1, 2)
            probs = F.softmax(outputs, dim=1)
            
            # Class 0: Ball, Class 1: Strike (Based on dataset.py label_map)
            # label_map = {'ball': 0, 'strike': 1}
            prob_strike = probs[0, 1].item()
            pred_idx = torch.argmax(probs, dim=1).item()
            
            pred_label = "Strike" if pred_idx == 1 else "Ball"
            confidence = prob_strike if pred_idx == 1 else (1 - prob_strike)
            
            print(f"[{pred_label}] {video_name} ({confidence:.1%})")
            
            results.append({
                'video': video_name, 
                'prediction': pred_label, 
                'confidence': confidence,
                'strike_prob': prob_strike
            })

    # Save results to CSV
    df = pd.DataFrame(results)
    output_file = 'pitch_judgments.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
