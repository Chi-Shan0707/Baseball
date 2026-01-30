import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Plan A: With direct FFmpeg usage via Torchvision (requires ffmpeg installed)
# Plan B: Without direct FFmpeg usage (using OpenCV) - Recommended by user prompt hint.

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from torchvision.io import read_video
    HAS_TORCHVISION_IO = True
except ImportError:
    HAS_TORCHVISION_IO = False

class VideoDataset(Dataset):
    """
    Dataset class for loading video clips and labels.
    
    Supports two backends for video loading:
    1. 'opencv': Uses cv2.VideoCapture (Plan B - without ffmpeg dependency/issues).
    2. 'torchvision': Uses torchvision.io.read_video (Plan A - with ffmpeg).
    """
    def __init__(self, 
                 data_dir, 
                 labels_file, 
                 frames_per_clip=32, 
                 transform=None, 
                 backend='opencv'):
        """
        Args:
            data_dir (str): Path to the directory containing video files.
            labels_file (str): Path to the CSV file with labels.
            frames_per_clip (int): Number of frames to sample uniformly.
            transform (callable, optional): Transform to be applied on a sample.
            backend (str): 'opencv' or 'torchvision'. Defaults to 'opencv' if available.
        """
        self.data_dir = data_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        
        # Load labels
        # Assuming CSV has columns: id, clip_id, label
        self.labels_df = pd.read_csv(labels_file)
        
        # Map labels to integers
        # "ball" -> 0, "strike" -> 1
        self.label_map = {'ball': 0, 'strike': 1}
        self.labels_df['label_idx'] = self.labels_df['label'].map(self.label_map)
        
        # Determine backend
        if backend == 'opencv' and not HAS_OPENCV:
            print("OpenCV not found, falling back to torchvision.")
            backend = 'torchvision'
        
        if backend == 'torchvision' and not HAS_TORCHVISION_IO:
            print("Tochvision IO not found, cannot load videos with this backend.")
            # If both missing, this will fail at runtime load_video
            
        self.backend = backend
        print(f"VideoDataset using backend: {self.backend}")

        # Default transform (ImageNet stats)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Returns:
            frames (torch.Tensor): Processed video frames (T, C, H, W)
            label (int): Class index
        """
        if idx >= len(self):
            return None # Should not happen usually
            
        row = self.labels_df.iloc[idx]
        video_id = row['id']
        label = row['label_idx']
        
        # Construct video path
        # Assuming files are named like '0.mp4', '1.mp4' based on 'id' column
        video_path = os.path.join(self.data_dir, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            # Try clip_id if id fails? No, prompts said dataset/videos : 0.mp4...
            # But let's be safe.
            return None
        
        try:
            frames = self._load_video(video_path)
            if frames is None:
                return None
                
            # Frames are (T, C, H, W)
            return frames, label
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

    def _load_video(self, path):
        if self.backend == 'opencv':
            return self._load_video_opencv(path)
        else:
            return self._load_video_torchvision(path)

    def _load_video_torchvision(self, path):
        # Plan A: With ffmpeg (via torchvision)
        try:
            # read_video returns (T, H, W, C), audio, info
            # pts_unit='sec' means we read in seconds, but simple read works fine
            vframes, _, _ = read_video(path, pts_unit='sec', output_format='TCHW') 
            # Note: output_format='TCHW' is available in newer torchvision, 
            # if older, it might be THWC. Latest 0.18+ supports it.
            # User has 0.16 or something from prompt: 0.24.1. So TCHW might work if specified or default THWC.
            # read_video default is THWC (T, H, W, C) in uint8 [0, 255]
            
            # Let's check dimensions carefully.
            # Convert to float and [0,1]
            if vframes.shape[0] == 0:
                return None
                
            # If THWC
            if vframes.shape[-1] == 3:
                vframes = vframes.permute(0, 3, 1, 2) # (T, C, H, W)
            
            # Sampling
            vframes = self._sample_frames(vframes)
            
            # Apply transforms per frame
            # ToTensor expects (C, H, W) or (H, W, C) numpy or PIL. 
            # But our transform starts with Resize.
            # Input to transforms can be Tensor (C, H, W).
            
            # vframes is Tensor uint8.
            # Scale to float [0,1] before transform if using ToTensor?
            # transforms.ToTensor() handles uint8 -> float scaled.
            # But here we have a batch of frames.
            
            processed_frames = []
            for t in range(vframes.shape[0]):
                frame = vframes[t] # (C, H, W)
                # Ensure it's treated as image. 
                # ToTensor expects HWC if numpy, or CHW/HWC if Tensor? 
                # Actually ToTensor converts PIL/numpy to Tensor. If it's already Tensor, we might just need Normalize.
                # My default transform has ToTensor().
                
                # To be consistent with OpenCV path (which usually yields numpy),
                # let's convert to PIL or leave as Tensor.
                # Using torchvision.transforms on Tensors is supported.
                # However, ToTensor may complain if it's already a Tensor.
                
                # Let's standardize: Convert to PIL image for the transform chain defined in __init__
                img = transforms.ToPILImage()(frame) # frame is (C, H, W) uint8
                processed_frames.append(self.transform(img))
                
            return torch.stack(processed_frames) # (T, C, H, W)
            
        except Exception as e:
            print(f"Torchvision read failed for {path}: {e}")
            return None

    def _load_video_opencv(self, path):
        # Plan B: Without ffmpeg (opencv direct)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
            
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV returns BGR (H, W, C)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
            
        if len(frames) == 0:
            return None

       
       

        # Convert list of numpy arrays to list of PIL Images or Tensors
        # Let's use PIL to be consistent with the Transform pipeline
        
        # Sample first
        indices = self._get_indices(len(frames))
        sampled_frames = [frames[i] for i in indices]
        
        processed_frames = []
        for frame_np in sampled_frames:
            img = Image.fromarray(frame_np)
            processed_frames.append(self.transform(img))
            
        return torch.stack(processed_frames)

    def _sample_frames(self, vframes):
        # vframes is a Tensor (T, ...)
        total_frames = vframes.shape[0]
        indices = self._get_indices(total_frames)
        return vframes[indices]

    def _get_indices(self, total_frames):
        if total_frames <= self.frames_per_clip:
            # Pad or take all. For now, let's just take all and maybe copy last?
            # Or simplified: if not enough, duplicate.
            indices = np.arange(total_frames)
            if total_frames < self.frames_per_clip:
                # Pad with last frame
                diff = self.frames_per_clip - total_frames
                indices = np.concatenate([indices, np.full(diff, total_frames - 1)])
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)
        return indices

def collate_fn(batch):
    """
    Collate function to filter out None samples (damaged videos).
    """
    # Filter Nones
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
        
    frames, labels = zip(*batch)
    
    # Stack
    frames = torch.stack(frames) # (Batch, T, C, H, W)
    labels = torch.tensor(labels) # (Batch,)
    
    return frames, labels
