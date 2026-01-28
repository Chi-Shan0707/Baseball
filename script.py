# video_checker.py
import subprocess
import json
import os

class VideoChecker:
    def __init__(self):
        self.results = []
    
    def check_video(self, video_path):
        """检查单个视频文件"""
        result = {
            'filename': os.path.basename(video_path),
            'path': video_path,
            'status': 'unknown',
            'errors': [],
            'info': {}
        }
        
        try:
            # 1. 检查是否可以读取
            cmd = ['ffmpeg', '-v', 'error', '-i', video_path, '-f', 'null', '-']
            process = subprocess.run(
                cmd, 
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                text=True,
                timeout=30
            )
            
            if process.stderr:
                result['errors'] = process.stderr.split('\n')
                result['status'] = 'damaged'
            else:
                result['status'] = 'healthy'
            
            # 2. 获取视频信息
            info_cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            info_process = subprocess.run(
                info_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if info_process.returncode == 0:
                result['info'] = json.loads(info_process.stdout)
            
        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['errors'].append('检查超时（可能文件过大或严重损坏）')
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
        
        self.results.append(result)
        return result
    
    def batch_check(self, directory, extensions=('.mp4', '.avi', '.mov')):
        """批量检查目录下的所有视频"""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extensions):
                    video_path = os.path.join(root, file)
                    print(f"检查: {file}")
                    self.check_video(video_path)
        
        return self.generate_report()
    
    def generate_report(self):
        """生成检查报告"""
        total = len(self.results)
        healthy = sum(1 for r in self.results if r['status'] == 'healthy')
        damaged = sum(1 for r in self.results if r['status'] == 'damaged')
        
        report = {
            'summary': {
                'total_videos': total,
                'healthy': healthy,
                'damaged': damaged,
                'damage_rate': damaged / total if total > 0 else 0
            },
            'damaged_videos': [
                {
                    'filename': r['filename'],
                    'errors': r['errors'][:5]  # 只显示前5个错误
                }
                for r in self.results if r['status'] == 'damaged'
            ]
        }
        
        return report

# 使用示例
if __name__ == '__main__':
    checker = VideoChecker()
    
    # 检查单个视频
    # result = checker.check_video('./dataset/videos/pitch_001.mp4')
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 批量检查
    report = checker.batch_check('./dataset/videos')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 保存报告
    with open('video_health_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)