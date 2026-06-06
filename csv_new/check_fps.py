#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: check_fps.py

功能说明:
    读取 summary.csv,遍历其中所有视频列 (ours / ltx / sd2.0 / camclone / ref),
    检查每个视频的帧率是否为 25fps,并把帧率不等于 25 的视频列出来。

    使用 ffprobe 读取帧率 (需要系统已安装 ffmpeg)。
    若没有 ffprobe,可改用脚本中提供的 OpenCV 版本 (见注释)。

    带实时进度输出: 有 tqdm 用进度条,无 tqdm 自动降级为逐条打印。

用法:
    python check_fps.py
================================================================================
"""

import os
import csv
import sys
import time
import subprocess
from fractions import Fraction

# 尝试导入 tqdm,没有就降级
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

input_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/csv_new/summary.csv"

# 容差: 由于 ffprobe 有时返回 25/1, 有时 30000/1001 等,做近似比较
TARGET_FPS = 25.0
TOL = 0.01

# 需要检查的列 (index 列不是路径,跳过)
video_columns = ["ours", "ltx", "sd2.0", "camclone", "ref"]


def get_fps(path):
    """用 ffprobe 获取视频帧率,返回 float;失败返回 None"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        if not out:
            return None
        # 形如 "25/1" 或 "30000/1001"
        return float(Fraction(out))
    except Exception:
        return None


# ---- 如果没有 ffprobe,可用下面的 OpenCV 版本替换 get_fps ----
# import cv2
# def get_fps(path):
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         return None
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     return fps if fps > 0 else None
# -----------------------------------------------------------


# ==========================================
# 第一步: 先把所有要检查的任务收集起来 (方便统计总数 + 显示进度)
# ==========================================
tasks = []        # (列名, index, 路径)
missing = []      # 路径不存在的 (列名, index, 路径)

with open(input_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = row["index"]
        for col in video_columns:
            path = row.get(col, "").strip()
            if not path:
                # CSV 中本就置空 (例如 sd2.0 缺失),跳过
                continue
            if not os.path.exists(path):
                missing.append((col, idx, path))
                continue
            tasks.append((col, idx, path))

total = len(tasks)
print(f"准备检查 {total} 个视频文件 (另有 {len(missing)} 个路径缺失已跳过)\n")

# ==========================================
# 第二步: 逐个检查,带实时进度
# ==========================================
bad_videos = []   # (列名, index, 路径, 实际fps)
start = time.time()

if HAS_TQDM:
    iterator = tqdm(tasks, total=total, unit="vid", ncols=80)
    for col, idx, path in iterator:
        fps = get_fps(path)
        if fps is None:
            bad_videos.append((col, idx, path, "读取失败"))
        elif abs(fps - TARGET_FPS) > TOL:
            bad_videos.append((col, idx, path, round(fps, 4)))
        # 把当前发现的问题数显示在进度条后缀
        iterator.set_postfix(bad=len(bad_videos))
else:
    # 无 tqdm: 手动打印进度,用 \r 原地刷新
    for i, (col, idx, path) in enumerate(tasks, 1):
        fps = get_fps(path)
        if fps is None:
            bad_videos.append((col, idx, path, "读取失败"))
        elif abs(fps - TARGET_FPS) > TOL:
            bad_videos.append((col, idx, path, round(fps, 4)))

        elapsed = time.time() - start
        speed = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / speed if speed > 0 else 0
        sys.stdout.write(
            f"\r  进度: {i}/{total} ({i*100//total}%)  "
            f"已用 {elapsed:.0f}s  预计剩余 {eta:.0f}s  "
            f"发现异常 {len(bad_videos)} 个   "
        )
        sys.stdout.flush()
    print()  # 换行

# ==========================================
# 输出结果
# ==========================================
print("\n" + "=" * 60)
print("📊 帧率检查结果")
print("=" * 60)
print(f"  共检查视频数: {total}")
print(f"  非 25fps 视频数: {len(bad_videos)}")
print(f"  路径缺失数 (跳过): {len(missing)}")
print(f"  总耗时: {time.time() - start:.1f}s")
print("=" * 60)

if bad_videos:
    print("\n⚠️  以下视频帧率不是 25fps:")
    print(f"{'列名':<10}{'index':<40}{'帧率'}")
    print("-" * 60)
    for col, idx, path, fps in bad_videos:
        print(f"{col:<10}{idx:<40}{fps}")
else:
    print("\n✅ 所有视频帧率均为 25fps。")

if missing:
    print(f"\n(注: 有 {len(missing)} 个路径在 CSV 中有记录但文件不存在,已跳过)")