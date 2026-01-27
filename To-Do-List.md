I'll help you create a comprehensive training plan for your baseball pitch classification model. Let me first explore your workspace to understand what you have. 

Read [](file:///home/chishan/Projects/Baseball/youtube-videos)

Read [](file:///home/chishan/Projects/Baseball/mlb-youtube/README.md#1-1), lines 1 to 50

Read [](file:///home/chishan/Projects/Baseball/mlb-youtube/experiments)

Ran terminal command:  ls -1 /home/chishan/Projects/Baseball/youtube-videos | wc -l

Ran terminal command:  ls -1 /home/chishan/Projects/Baseball/youtube-videos | head -5 | xargs -I {} file "/home/chishan/Projects/Baseball/youtube-videos/{}"

Read [](file:///home/chishan/Projects/Baseball/mlb-youtube/experiments/train_segmented.py#1-1), lines 1 to 50

Read [](file:///home/chishan/Projects/Baseball/mlb-youtube/experiments/models.py#1-1), lines 1 to 80

Now let me create a comprehensive training plan for you: 

Created 9 todos

## Baseball Pitch Classification: Complete Training Plan

You currently have **~54 video clips** of baseball pitches with [strike] and [ball] labels. Here's your comprehensive training roadmap:

---

## 📋 COMPLETE TO-DO LIST

### **Step 1: Data Preparation & Organization**
- **What**: Create proper dataset structure with train/validation/test splits
- **How**: 
  - Create folder structure: `data/strike/` and `data/ball/`
  - OR create annotation file (CSV/JSON) mapping video filenames to labels
  - Split: 70% train, 15% validation, 15% test
- **Why**: Organized data is essential for training

### **Step 2: Extract Frames from Videos**
- **What**: Convert videos to frame sequences
- **How**: Use OpenCV or FFmpeg
  - Extract at 10-30 FPS per video
  - Save as JPG/PNG images
- **Code**: `cv2.VideoCapture()` or `ffmpeg -i video.mp4 -vf fps=15 frames/%04d.jpg`

### **Step 3: YOLO Object Detection (Optional but Recommended)**
- **What**: Detect and localize pitcher, batter, ball in frames
- **How**: 
  - Use YOLOv8 or YOLOv5 pre-trained on COCO
  - Detect "person" class for pitcher/batter
  - Crop detected regions to focus model attention
- **Why**: Reduces background noise, improves accuracy
- **Tools**: `ultralytics` library for YOLOv8

### **Step 4: Feature Extraction with I3D/3D CNN**
- **What**: Extract spatio-temporal features from video clips
- **How**: 
  - Use I3D model (you have `pytorch-i3d/` folder)
  - Load pre-trained I3D weights (Kinetics-400)
  - Extract 1024-dim features per clip
- **Why**: Captures motion patterns distinguishing strikes from balls

### **Step 5: LSTM Temporal Modeling**
- **What**: Model temporal dependencies in frame sequences
- **How**: 
  - Feed I3D features into 2-3 layer LSTM (hidden size: 256-512)
  - Use bidirectional LSTM for better context
  - Add dropout (0.3-0.5) for regularization
- **Why**: Learns pitch trajectory and timing patterns

### **Step 6: Build Final Classification Model**
- **Architecture**: 
  ```
  Video Frames → I3D (3D CNN) → LSTM → FC Layer → [Strike/Ball]
  ```
- **Loss**: Binary Cross-Entropy or Categorical Cross-Entropy
- **Output**: 2 classes (Strike, Ball)

### **Step 7: Training Pipeline Setup**
- **Data Augmentation**: Random crop, horizontal flip, temporal jittering
- **Optimizer**: Adam (lr=1e-4) or SGD with momentum
- **Scheduler**: ReduceLROnPlateau or CosineAnnealing
- **Batch size**: 8-16 (depending on GPU memory)

### **Step 8: Train and Evaluate**
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Epochs**: 50-100 with early stopping (patience=10)
- **Validation**: Monitor on validation set every epoch
- **Testing**: Final evaluation on test set

### **Step 9: Model Optimization**
- Test on unseen videos
- Analyze misclassifications
- Fine-tune hyperparameters
- Consider ensemble or attention mechanisms

---

## 🔢 **MINIMUM DATASET SIZE**

**Your current 54 clips is INSUFFICIENT.** Here's what you need:

| Scenario | Minimum Clips | Recommended | Optimal |
|----------|--------------|-------------|---------|
| **Basic Model** | 200-300 per class | 500 per class | 1000+ per class |
| **Your Case (2 classes)** | **400-600 total** | **1000 total** | **2000+ total** |

**Why you need more:**
- Deep learning models (I3D + LSTM) require substantial data
- 54 clips → ~38 train, ~8 val, ~8 test (too small!)
- Risk of severe overfitting
- Poor generalization to unseen pitches

**What to do:**
1. ✅ **Download more clips** (aim for 500-1000 total)
2. ✅ **Data augmentation** (flips, crops, speed variations) can help
3. ✅ **Transfer learning** from pre-trained I3D (reduces data requirements)
4. ✅ **Start simple**: Try training with current data but expect ~60-70% accuracy max

---

## 🚀 **RECOMMENDED APPROACH FOR YOUR 54 CLIPS**

Since you have limited data, use this **transfer learning approach**:

1. **Use pre-trained I3D** on Kinetics-400 (already includes sports actions)
2. **Freeze I3D layers**, only train LSTM + classifier
3. **Heavy augmentation**: temporal crop, speed perturbation, flips
4. **Simple LSTM**: 1-2 layers, not too deep
5. **Expect**: 60-75% accuracy with 54 clips, 85-95% with 500+ clips

---

Would you like me to help you:
1. Create the dataset organization script?
2. Set up the frame extraction pipeline?
3. Build the I3D + LSTM training script?
4. Download more video clips first?