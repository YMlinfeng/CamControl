"""
文件用途：
    校验 CSV 文件中指定列的视频是否符合特定的规格：
    - 帧数 (Total Frames) == 77
    - 帧率 (FPS) == 15
    
使用方法：
    1. 确保系统安装了 ffmpeg (需调用 ffprobe)。
    2. 运行脚本：python check_video_specs.py
    3. 脚本会列出所有不符合条件的视频路径及其具体参数。
"""

import pandas as pd
import os
import subprocess
from tqdm import tqdm

def get_video_specs(video_path):
    """使用 ffprobe 获取视频的 fps 和总帧数"""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,nb_frames',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')
        if len(output) < 2:
            return None, None
        
        # 处理 FPS (ffprobe 返回的是分数形式如 "15/1")
        fps_str = output[0]
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str)
            
        # 处理帧数
        total_frames = int(output[1]) if output[1].isdigit() else 0
        
        return fps, total_frames
    except Exception:
        return None, None

def verify_csv_videos():
    # --- 配置区域 ---
    csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o2.csv"
    columns_to_check = ['sd2.0_gen', 'camclone_gen', 'ltx_gen', 'ours_gen']
    target_fps = 15.0
    target_frames = 77
    # ----------------

    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV 文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    issues = []

    print(f"开始校验视频规格 (期望: {target_frames}帧, {target_fps}FPS)...")

    for col in columns_to_check:
        if col not in df.columns:
            print(f"⚠️ 列 {col} 不存在，跳过。")
            continue

        # 过滤掉空值
        valid_paths = df[col].dropna().unique()
        
        for path in tqdm(valid_paths, desc=f"校验 {col}"):
            path = str(path).strip()
            if not path or not os.path.exists(path):
                # 如果路径不存在，视为严重问题记录下来
                issues.append({
                    "column": col,
                    "path": path,
                    "reason": "文件不存在"
                })
                continue

            fps, frames = get_video_specs(path)

            if fps is None or frames is None:
                issues.append({
                    "column": col,
                    "path": path,
                    "reason": "无法读取元数据"
                })
            elif int(round(fps)) != target_fps or frames != target_frames:
                issues.append({
                    "column": col,
                    "path": path,
                    "reason": f"规格不符 (实际: {frames}帧, {fps:.2f}fps)"
                })

    # --- 输出结果 ---
    print("\n" + "="*50)
    print("📋 校验统计报告")
    print("="*50)

    if not issues:
        print(f"✅ 完美！所有视频均符合 {target_frames}帧 和 {target_fps}FPS 的要求。")
    else:
        print(f"❌ 发现 {len(issues)} 个异常视频：\n")
        # 按列分组显示
        current_col = ""
        for issue in issues:
            if issue['column'] != current_col:
                current_col = issue['column']
                print(f"--- 列: {current_col} ---")
            print(f"  [!] {issue['reason']}")
            print(f"      路径: {issue['path']}")
            print("-" * 30)

    print("\n校验完成。")

if __name__ == "__main__":
    verify_csv_videos()