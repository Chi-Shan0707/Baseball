#!/usr/bin/env python3
"""
下载 MLB YouTube 视频片段
从 mlb-youtube-segmented.json 读取视频信息，下载指定时间段的片段
"""

import os
import csv
import json
import subprocess
from pathlib import Path



def get_pitch_label(info):
#JSON 文件里每个片段有 labels 字段（列表），常见值有 ["strike"] 或 ["ball"]。
    """获取投球标签"""
    labels = info.get('labels', [])
    # labels_lc = [label.lower() for label in labels] 本就全小写
    if 'strike' in labels:
        return 'strike'
    elif 'ball' in labels:
        return 'ball'
    else :
        return None 

def download_clip(save_idx,clip_id, url, start_time, end_time, save_dir, max_attempts=3):
    """下载单个视频片段"""
    ## 从 URL 提取视频 ID
    ##video_id = url.split('=')[-1]
    
    # 输出文件名：clip_id.mp4
    output_path = save_dir / f'{save_idx}.mp4'
    
    if output_path.exists():
        print(f'✓ 片段 {save_idx} 已存在，跳过')
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
        print(f'[{attempt}/{max_attempts}] 下载片段 {save_idx} ({duration:.1f}s)...')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and output_path.exists():
                print(f'✓ 片段 {save_idx} 下载完成')
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
        for temp_file in save_dir.glob(f'{save_idx}*'):
            if temp_file.suffix in ['.part', '.ytdl', '.temp']:
                temp_file.unlink(missing_ok=True)
                print(f'  清理: {temp_file.name}')
        
        if attempt < max_attempts:
            print(f'  等待 5s 后重试...\n')
            import time
            time.sleep(5)
    
    print(f'✗ 片段 {save_idx} 下载失败\n')
    return False

def append_csv(id,clip_id,label,csv_path):
    # file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        # if not file_exists:
        #     w.writerow(['id','clip_id', 'label'])
        w.writerow([id,clip_id, label])

def main():
    # 设置路径
    save_dir = Path('../dataset/videos')
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
    

    

    csv_path= Path('../dataset/pitchcalls/labels.csv')
    
    if csv_path.exists():
        save_idx = 0
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f): #遍历的是数据行，不访问head
                save_idx = int(row['id']) + 1

    else :
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id','clip_id', 'label'])
        save_idx = 0
    
    print(f'从 {csv_path} 继续编号，起始 ID: {save_idx}\n')

    for idx, (clip_id, info) in enumerate(clips, 1):
        
        label = get_pitch_label(info)
        if label is None:
            print(f'跳过片段 {clip_id}，无有效标签')
            continue

        url = info['url']
        start = info['start']
        end = info['end']
        
        if start >= end or start < 0 or end < 0 :
            continue
        
        print(f'\n[{idx}/{total}] Clip ID: {clip_id}')
        print(f'  URL: {url}')
        print(f'  时间: {start:.2f}s - {end:.2f}s')
        
        if csv_path.exists():
            """
    说明：open(..., newline=...) 是什么？ 💡
作用：控制 Python 在读写时如何处理行结束符（换行符），例如 \n、\r\n 等。
和 csv 模块的关系：使用 csv 时推荐传 newline=''，因为 csv 模块自己负责写入正确的行结束符；如果不这么做（例如默认 None），在 Windows 上写 CSV 可能会出现额外空行。
                """
            with open(csv_path, newline='') as f:
                if any(row.get('clip_id') == clip_id for row in csv.DictReader(f)):
                    print(f'✓ 片段 {clip_id} 已存在，跳过')
                    continue

        if download_clip(save_idx, clip_id, url, start, end, save_dir):
            success += 1
            append_csv(save_idx,clip_id,label,csv_path)
            save_idx += 1
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
