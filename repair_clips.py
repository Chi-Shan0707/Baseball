#!/usr/bin/env python3
"""
一键修复损坏的棒球视频
支持所有类型的损坏修复
"""

import subprocess
import json
import os
import shutil
from pathlib import Path
import logging

class VideoRepairer:
    def __init__(self, video_dir, output_dir=None):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir) if output_dir else self.video_dir / "repaired"
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('video_repair.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 损坏视频分类
        self.damaged_videos = {
            'dts_error': [],      # 时间戳错误
            'h264_error': [],     # H.264参考帧错误
            'nal_error': [],      # NAL单元损坏
            'audio_error': [],    # 音频错误
            'unknown': []         # 未知错误
        }
    
    def load_damaged_list(self, json_file):
        """从JSON文件加载损坏视频列表"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for video_info in data['damaged_videos']:
            filename = video_info['filename']
            errors = video_info['errors']
            
            # 根据错误信息分类
            if any('non monotonically increasing dts' in e for e in errors):
                self.damaged_videos['dts_error'].append(filename)
            elif any('mmco: unref' in e for e in errors):
                self.damaged_videos['h264_error'].append(filename)
            elif any('NAL unit' in e for e in errors):
                self.damaged_videos['nal_error'].append(filename)
            elif any('aac' in e.lower() for e in errors):
                self.damaged_videos['audio_error'].append(filename)
            else:
                self.damaged_videos['unknown'].append(filename)
        
        self.logger.info(f"加载损坏视频分类: {json.dumps({k: len(v) for k, v in self.damaged_videos.items()}, indent=2)}")
    
    def repair_dts_error(self, video_file):
        """修复时间戳错误"""
        output_file = self.output_dir / video_file
        cmd = [
            'ffmpeg', '-y', '-i', str(self.video_dir / video_file),
            '-fflags', '+genpts',        # 生成正确的时间戳
            '-c:v', 'copy',              # 复制视频流
            '-c:a', 'copy',              # 复制音频流
            str(output_file)
        ]
        return self._run_repair(cmd, video_file, "DTS时间戳修复")
    
    def repair_h264_error(self, video_file):
        """修复H.264参考帧错误"""
        output_file = self.output_dir / video_file
        cmd = [
            'ffmpeg', '-y', '-i', str(self.video_dir / video_file),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-x264-params', 'ref=3:bframes=2',  # 优化参考帧设置
            '-c:a', 'aac', '-b:a', '128k',
            str(output_file)
        ]
        return self._run_repair(cmd, video_file, "H.264参考帧修复")
    
    def repair_nal_error(self, video_file):
        """修复NAL单元损坏（最复杂的修复）"""
        temp_dir = self.output_dir / 'temp_frames'
        temp_dir.mkdir(exist_ok=True)
        
        video_stem = Path(video_file).stem
        output_file = self.output_dir / video_file
        
        # 第一步：尝试直接修复
        cmd1 = [
            'ffmpeg', '-y', '-err_detect', 'ignore_err',
            '-i', str(self.video_dir / video_file),
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '25',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_file)
        ]
        
        if self._run_repair(cmd1, video_file, "NAL直接修复"):
            # 检查修复结果
            if self._verify_video(output_file):
                return True
        
        # 第二步：如果直接修复失败，尝试逐帧提取
        self.logger.warning(f"直接修复失败，尝试逐帧提取: {video_file}")
        
        # 提取所有能读取的帧
        frame_pattern = str(temp_dir / f"{video_stem}_%04d.png")
        cmd2 = [
            'ffmpeg', '-y', '-err_detect', 'ignore_err',
            '-i', str(self.video_dir / video_file),
            '-vsync', '0',  # 不进行帧率同步
            frame_pattern
        ]
        
        subprocess.run(cmd2, capture_output=True, text=True)
        
        # 统计提取的帧数
        frames = list(temp_dir.glob(f"{video_stem}_*.png"))
        
        if len(frames) >= 10:  # 至少需要10帧
            # 从PNG重新创建视频
            cmd3 = [
                'ffmpeg', '-y', '-framerate', '30',
                '-i', str(temp_dir / f"{video_stem}_%04d.png"),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                str(output_file)
            ]
            
            result = self._run_repair(cmd3, video_file, "NAL逐帧修复")
            
            # 清理临时文件
            for frame in frames:
                frame.unlink()
            
            return result
        else:
            self.logger.error(f"无法提取足够帧: {video_file} (仅{len(frames)}帧)")
            return False
    
    def repair_audio_error(self, video_file):
        """修复音频错误"""
        output_file = self.output_dir / video_file
        cmd = [
            'ffmpeg', '-y', '-i', str(self.video_dir / video_file),
            '-c:v', 'copy',          # 保持视频不变
            '-c:a', 'aac', '-b:a', '128k',  # 重新编码音频
            str(output_file)
        ]
        return self._run_repair(cmd, video_file, "音频修复")
    
    def repair_unknown_error(self, video_file):
        """修复未知错误（尝试通用修复）"""
        output_file = self.output_dir / video_file
        cmd = [
            'ffmpeg', '-y', '-err_detect', 'ignore_err',
            '-i', str(self.video_dir / video_file),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_file)
        ]
        return self._run_repair(cmd, video_file, "通用修复")
    
    def _run_repair(self, cmd, video_file, repair_type):
        """执行修复命令"""
        self.logger.info(f"开始{repair_type}: {video_file}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ {repair_type}成功: {video_file}")
                
                # 验证修复后的视频
                if self._verify_video(self.output_dir / video_file):
                    return True
                else:
                    self.logger.warning(f"⚠️ 修复后验证失败: {video_file}")
                    return False
            else:
                self.logger.error(f"❌ {repair_type}失败: {video_file}")
                self.logger.error(f"错误输出: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ {repair_type}超时: {video_file}")
            return False
        except Exception as e:
            self.logger.error(f"🚨 {repair_type}异常: {video_file} - {str(e)}")
            return False
    
    def _verify_video(self, video_path):
        """验证修复后的视频"""
        cmd = ['ffprobe', '-v', 'error', '-i', str(video_path), '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and not result.stderr
    
    def repair_all(self):
        """修复所有损坏视频"""
        repair_methods = {
            'dts_error': self.repair_dts_error,
            'h264_error': self.repair_h264_error,
            'nal_error': self.repair_nal_error,
            'audio_error': self.repair_audio_error,
            'unknown': self.repair_unknown_error
        }
        
        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }
        
        total = sum(len(videos) for videos in self.damaged_videos.values())
        processed = 0
        
        for error_type, videos in self.damaged_videos.items():
            if error_type in repair_methods:
                repair_func = repair_methods[error_type]
                
                for video in videos:
                    processed += 1
                    self.logger.info(f"进度: {processed}/{total} - {video}")
                    
                    # 检查源文件是否存在
                    if not (self.video_dir / video).exists():
                        self.logger.warning(f"源文件不存在: {video}")
                        results['skipped'].append(video)
                        continue
                    
                    # 执行修复
                    if repair_func(video):
                        results['success'].append(video)
                    else:
                        results['failed'].append(video)
        
        # 生成修复报告
        self._generate_report(results)
        return results
    
    def _generate_report(self, results):
        """生成修复报告"""
        report = {
            'summary': {
                'total_damaged': len(results['success']) + len(results['failed']) + len(results['skipped']),
                'repaired_success': len(results['success']),
                'repaired_failed': len(results['failed']),
                'skipped': len(results['skipped']),
                'success_rate': len(results['success']) / (len(results['success']) + len(results['failed'])) if (len(results['success']) + len(results['failed'])) > 0 else 0
            },
            'success_videos': results['success'],
            'failed_videos': results['failed'],
            'skipped_videos': results['skipped']
        }
        
        # 保存报告
        with open(self.output_dir / 'repair_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        self.logger.info("="*50)
        self.logger.info("修复完成!")
        self.logger.info(f"成功修复: {len(results['success'])} 个视频")
        self.logger.info(f"修复失败: {len(results['failed'])} 个视频")
        self.logger.info(f"跳过: {len(results['skipped'])} 个视频")
        self.logger.info(f"成功率: {report['summary']['success_rate']:.2%}")
        self.logger.info(f"修复后的视频保存在: {self.output_dir}")
        self.logger.info("="*50)

# 使用示例
if __name__ == "__main__":
    # 1. 创建修复器
    repairer = VideoRepairer(
        video_dir="./dataset/videos",
        output_dir="./dataset/videos_repaired"
    )
    
    # 2. 加载损坏列表（从你的JSON文件）
    repairer.load_damaged_list("video_health_report.json")  # 你的JSON文件名
    
    # 3. 开始修复所有视频
    results = repairer.repair_all()
    
    # 4. 可选：用修复后的视频替换原始视频
    # shutil.copytree("./dataset/videos_repaired", "./dataset/videos", dirs_exist_ok=True)