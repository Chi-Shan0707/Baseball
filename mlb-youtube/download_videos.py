import os
import json
import string
import random
import subprocess


save_dir = '../youtube-videos/'
with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)
    for entry in data.values():
        yturl = entry['url']
        ytid = yturl.split('=')[-1]

        if os.path.exists(os.path.join(save_dir, ytid+'.mkv')):
            continue

        # Prefer best video+audio, merge to mkv, use node if available for JS runtime
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo+bestaudio/best',  # 最佳音视频合并
            yturl,                              # YouTube链接
            '-o', os.path.join(save_dir, f'{ytid}.%(ext)s'),  # 输出路径（带动态扩展名）
            '--merge-output-format', 'mkv',     # 合并后输出mkv格式
            '--js-runtime', 'node'             # 指定JS运行时为node（关键，匹配之前的安装）
        ]

        result = subprocess.run(
            cmd,
            check=True,  # 命令执行失败时抛出异常，方便定位问题
            capture_output=True,  # 捕获stdout/stderr输出
            text=True  # 输出转为字符串（而非字节流），方便打印
        )
        # 打印下载成功的日志
        print(f'视频 {ytid} 下载完成\n{result.stdout}')
