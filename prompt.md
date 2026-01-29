You will generate a small, modular, and well-documented PyTorch project that implements a per-frame ResNet-18 feature extractor plus two interchangeable temporal classifiers: (A) Temporal Average Pooling head, and (B) LSTM head. The resulting code must be beginner-friendly, highly readable, and include clear comments and docstrings at key places.

Version: 
torch                       2.9.1+cu130      pypi_0              pypi
torchaudio                  2.9.1+cu130      pypi_0              pypi
torchcodec                  0.9.0            pypi_0              pypi
torchvision                 0.24.1+cu130     pypi_0              pypi
(MLenv) chishan@LAPTOP-7N8BKOTJ:~/Projects/Baseball$ ffmpeg -version
ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers
built with gcc 13 (Ubuntu 13.2.0-23ubuntu3)



Project structure (create these files):

dataset.py : dataset class that loads a short video clip (or frames) and returns a tensor of shape (T, C, H, W). Implement frame sampling to sample N frames uniformly from each 7s clip and simple preprocessing using torchvision.transforms. Use torchvision.io.read_video or cv2 with a safe fallback and document dependency choices.
When encounter damaged videos, skip automatically. You can choose ffmpeg, but I recommend not, so please give me two plans: with ffmpeg and without ffmpeg

[my dataset is in ./dataset/videos : almost 1000 7-second clips    in ./dataset/pitchcalls/labels.csv  each video's result "strike" or "ball"]

models.py : contain three modules:
ResNetBackbone — loads torchvision.models.resnet18(pretrained=True), removes the final fc, and returns per-frame features. Allow option to freeze backbone.

TemporalPoolHead — accepts per-frame embeddings (T, D) and returns classification logits using temporal average pooling.

LSTMHead — accepts per-frame embeddings (T, D) and uses a single-layer (optionally bidirectional) LSTM + linear head to return logits.

FullModel — wrapper that composes ResNetBackbone + either TemporalPoolHead or LSTMHead depending on arch argument ('pool' or 'lstm').

train.py : CLI training script with argparse flags:
--data-dir, --arch (pool or lstm), --frames (e.g., 32), --batch-size, --lr, --epochs, --freeze-backbone (bool), --device.

Implement dataset split (train/val) from folder or CSV; provide default synthetic splitter if none given.
Training loop with clear, commented steps: forward, loss, backward, optimizer step, scheduler step, and periodic validation.

You must:
Make sure the package version fits the usage, and the usage fits the package version. You must check the consistency of the parameters. You must make sure you process videos properly --- skip bad ones, rescale to th same size, .....
Make sure you finally build a feasible, easy-to -read pipeline.

Save best model checkpoint to checkpoints/.

Compute metrics: accuracy, precision, recall, F1; print them per epoch and save final evaluation results.
utils.py : small helpers for metrics, checkpoint save/load, and a reproducible seed setter.
README.md : short usage examples and minimal dependency list (torch, torchvision, opencv-python optional).
Coding style and requirements:

Use explicit type hints and docstrings for each class/function.
Add comments explaining the purpose of each major block and non-obvious lines.
Keep functions short and modular; avoid large monolithic scripts.
Make the ResNetBackbone compute features for each frame by applying the 2D ResNet per frame (i.e., reshape batch/time to batch*T, run through backbone, then reshape back to (batch, T, D)).
Normalize inputs using the same mean/std as ResNet pretrained weights.
Default to binary classification (2 classes), use nn.CrossEntropyLoss.
Provide clear defaults so a user can run a smoke test with CPU using a tiny synthetic dataset (e.g., random tensors) if no videos available.
Include example commands in README.md:
Training baseline (pool): python train.py --arch pool --data-dir /path/to/data --epochs 10
Training LSTM: python train.py --arch lstm --data-dir /path/to/data --epochs 10
Mention common troubleshooting tips (GPU OOM -> reduce frames/batch, freeze backbone, use gradient accumulation).