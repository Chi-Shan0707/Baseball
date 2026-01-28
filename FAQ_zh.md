# 常见问题解答 / FAQ

## 问题1：为什么跳过错误视频后仍然显示FFmpeg错误？

**原因**：`[h264 @ 0x...] mmco: unref short failure` 是FFmpeg解码器打印到**stderr**的警告消息，不是Python异常。

- 你的 `try/except` 只能捕获Python异常（如`RuntimeError`、`ValueError`）
- FFmpeg的C库直接写入stderr，Python无法阻止这些消息
- **这些消息不会让程序崩溃**，只是日志噪音

**解决方案**：
1. **最优**：使用 `--bad-videos-json` 提前过滤，避免尝试解码损坏的视频
2. 忽略stderr输出（这些只是警告）
3. 可选：重定向stderr（但不推荐，会隐藏真正的错误）

```bash
# 推荐：提前过滤77个损坏视频
python train.py --bad-videos-json video_health_report.json --epochs 10
```

---

## 问题2：为什么更换到TorchCodec后仍有torchvision弃用警告？

**可能原因**：

### A) TorchCodec未正确安装
检查是否安装：
```bash
python -c "import torchcodec; print(torchcodec.__version__)"
```

如果报错，需要安装：
```bash
conda install -c pytorch torchcodec
# 或
pip install torchcodec
```

### B) Backend回退到torchvision
你的配置是 `backend="auto"`，会按顺序尝试：
1. torchcodec（如果失败或未安装）
2. **torchvision**（触发弃用警告）
3. cv2

如果TorchCodec解码第一个视频失败，会自动回退到torchvision。

**解决方案**：
```bash
# 强制使用torchcodec（确保安装成功）
python train.py --backend torchcodec --bad-videos-json video_health_report.json

# 或使用cv2避免警告
python train.py --backend cv2 --bad-videos-json video_health_report.json
```

### C) Import时触发警告
某些版本的torchvision会在**import时**就发出警告，即使你不使用它的视频功能。这是无害的。

---

## 问题3：如何使用video_health_report.json提前跳过损坏视频？

**答案**：使用新增的 `--bad-videos-json` 参数！

### 优势：
✅ **更高效**：在加载CSV时就过滤，不会尝试解码  
✅ **更干净**：避免77个解码警告消息  
✅ **更准确**：训练只用906个真实视频，不会用随机噪声污染数据

### 用法：

```bash
# 方式1：完全排除77个损坏视频（推荐）
python train.py --bad-videos-json video_health_report.json --epochs 10 --batch-size 2

# 方式2：如果video_health_report.json不在当前目录
python train.py --bad-videos-json /path/to/video_health_report.json --epochs 10

# 方式3：双重保险（预过滤 + 运行时跳过）
python train.py --bad-videos-json video_health_report.json --skip-bad-videos --epochs 10
```

### 工作原理：
1. 从JSON读取77个损坏视频的文件名（如`"884.mp4"`, `"470.mp4"`）
2. 在`load_pitchcalls_samples()`中，跳过这些video_id
3. 数据集只包含906个健康视频
4. 不会尝试解码损坏视频 → 无FFmpeg警告 → 训练更快

### JSON格式要求：
```json
{
  "damaged_videos": [
    {"filename": "884.mp4", ...},
    {"filename": "470.mp4", ...}
  ]
}
```

你的`video_health_report.json`已经符合这个格式！

---

## 推荐运行命令

```bash
# 最优配置：预过滤 + 低内存 + TorchCodec
python train.py \
  --bad-videos-json video_health_report.json \
  --backend torchcodec \
  --batch-size 2 \
  --frames 16 \
  --epochs 10 \
  --freeze-backbone

# 如果TorchCodec有问题，用cv2
python train.py \
  --bad-videos-json video_health_report.json \
  --backend cv2 \
  --batch-size 2 \
  --frames 16 \
  --epochs 10 \
  --freeze-backbone
```

---

## 总结

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| FFmpeg警告仍出现 | C库直接写stderr，Python无法捕获 | 用`--bad-videos-json`预过滤 |
| torchvision警告 | TorchCodec未安装或回退 | 检查安装，强制`--backend torchcodec`或用`cv2` |
| 如何用health report | 需要新功能 | 使用`--bad-videos-json video_health_report.json` |
