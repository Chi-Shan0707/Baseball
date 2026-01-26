#!/usr/bin/env python3
"""
下载 MLB YouTube 视频
解决方案：
1. 添加 cookies 和 user-agent 来规避403错误
2. 使用分片重试和更稳定的下载策略
"""

import os
import json
import subprocess
import time
from pathlib import Path


def download_video(ytid, yturl, save_dir, max_attempts=5):
    """下载单个视频，带重试机制"""
    output_path = save_dir / f'{ytid}.mkv'
    
    if output_path.exists():
        print(f'✓ 视频 {ytid} 已存在，跳过')
        return True
    
    # 构建 yt-dlp 命令 - 只下载视频（无音频），速度更快
    cmd = [
        'yt-dlp',
        # 使用代理（VPN）连接YouTube
        '--proxy', 'http://127.0.0.1:7897',  # 显式指定代理
        '--no-check-certificate',  # 跳过SSL证书验证（代理可能导致证书问题）
        '-f', 'bestvideo[ext=mp4]/bestvideo/best',  # 只下载视频，不要音频
        '-o', str(output_path),
        
        # 网络稳定性设置
        '--socket-timeout', '30',
    
        '--retries', '10',
        '--fragment-retries', '10',
        '--concurrent-fragments', '4',
        '--retry-sleep', '5',
        
        # 规避403错误
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        '--add-header', 'Accept-Language:en-US,en;q=0.9',
        
        # 安静模式，只显示进度
        '--progress',
        '--no-warnings',
        
        
        yturl
    ]
    
    for attempt in range(1, max_attempts + 1):
        print(f'[{attempt}/{max_attempts}] 下载 {ytid}...')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f'✓ 视频 {ytid} 下载完成')
                return True
            else:
                print(f'✗ 失败 (code {result.returncode})')
                if '--verbose' in cmd or attempt == max_attempts:
                    print(f'stderr: {result.stderr[:500]}')
                    
        except subprocess.TimeoutExpired:
            print(f'✗ 超时')
        except Exception as e:
            print(f'✗ 异常: {e}')
        
        # 清理临时文件
        for temp_file in save_dir.glob(f'{ytid}*'):
            if temp_file.suffix in ['.part', '.ytdl', '.temp']:
                temp_file.unlink(missing_ok=True)
                print(f'  清理: {temp_file.name}')
        
        if attempt < max_attempts:
            wait = min(30, 5 * attempt)
            print(f'  等待 {wait}s 后重试...\n')
            time.sleep(wait)
    
    print(f'✗ 视频 {ytid} 下载失败，已达最大重试次数\n')
    return False


def main():
    save_dir = Path('../youtube-videos')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open('data/mlb-youtube-segmented.json', 'r') as f:
        data = json.load(f)
    
    total = len(data)
    success = 0
    
    print(f'开始下载 {total} 个视频\n')
    
    for idx, (key, entry) in enumerate(data.items(), 1):
        yturl = entry['url']
        ytid = yturl.split('=')[-1]
        
        print(f'[{idx}/{total}] {ytid}')
        if download_video(ytid, yturl, save_dir):
            success += 1
    
    print(f'\n完成: {success}/{total} 成功')


if __name__ == '__main__':
    main()
