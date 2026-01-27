#!/usr/bin/env python3
"""
下载 MLB YouTube 视频片段
从 mlb-youtube-segmented.json 读取视频信息，下载指定时间段的片段
"""

import os
import json
import subprocess
from pathlib import Path


def download_clip(clip_id, url, start_time, end_time, save_dir, max_attempts=3):
    """下载单个视频片段"""
    # 从 URL 提取视频 ID
    video_id = url.split('=')[-1]
    
    # 输出文件名：clip_id_video_id.mp4
    output_path = save_dir / f'{clip_id}_{video_id}.mp4'
    
    if output_path.exists():
        print(f'✓ 片段 {clip_id} 已存在，跳过')
        return True
    
    # 计算片段时长
    duration = end_time - start_time
    
    # 构建 yt-dlp 命令
    cmd = [
        'yt-dlp',
        # 使用代理
        '--proxy', 'http://127.0.0.1:7897',
        '--no-check-certificate',
        
        # 下载指定时间段的视频片段
        '--download-sections', f'*{start_time}-{end_time}',
        
        # 视频格式选择
        '-f', 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        
        # 输出路径
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
        
        # 进度显示
        '--progress',
        '--no-warnings',
        
        url
    ]
    
    for attempt in range(1, max_attempts + 1):
        print(f'[{attempt}/{max_attempts}] 下载片段 {clip_id} ({duration:.1f}s)...')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and output_path.exists():
                print(f'✓ 片段 {clip_id} 下载完成')
                return True
            else:
                print(f'✗ 失败 (code {result.returncode})')
                if attempt == max_attempts:
                    print(f'stderr: {result.stderr[:500]}')
                    
        except subprocess.TimeoutExpired:
            print(f'✗ 超时')
        except Exception as e:
            print(f'✗ 异常: {e}')
        
        # 清理临时文件
        for temp_file in save_dir.glob(f'{clip_id}*'):
            if temp_file.suffix in ['.part', '.ytdl', '.temp']:
                temp_file.unlink(missing_ok=True)
                print(f'  清理: {temp_file.name}')
        
        if attempt < max_attempts:
            print(f'  等待 5s 后重试...\n')
            import time
            time.sleep(5)
    
    print(f'✗ 片段 {clip_id} 下载失败\n')
    return False


def main():
    # 设置路径
    save_dir = Path('../youtube-videos')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取 JSON 数据
    json_path = Path('data/mlb-youtube-segmented.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 只处理前 1000 个片段
    max_clips = 1000
    clips = list(data.items())[:max_clips]
    
    total = len(clips)
    success = 0
    failed = []
    
    print(f'开始下载 {total} 个视频片段\n')
    print('=' * 60)
    
    for idx, (clip_id, info) in enumerate(clips, 1):
        url = info['url']
        start = info['start']
        end = info['end']
        
        print(f'\n[{idx}/{total}] Clip ID: {clip_id}')
        print(f'  URL: {url}')
        print(f'  时间: {start:.2f}s - {end:.2f}s')
        
        if download_clip(clip_id, url, start, end, save_dir):
            success += 1
        else:
            failed.append(clip_id)
        
        print('-' * 60)
    
    # 输出统计信息
    print('\n' + '=' * 60)
    print(f'下载完成!')
    print(f'成功: {success}/{total}')
    print(f'失败: {len(failed)}/{total}')
    
    if failed:
        print(f'\n失败的片段 ID:')
        for clip_id in failed[:20]:  # 只显示前20个失败的
            print(f'  - {clip_id}')
        if len(failed) > 20:
            print(f'  ... 以及其他 {len(failed) - 20} 个')


if __name__ == '__main__':
    main()
