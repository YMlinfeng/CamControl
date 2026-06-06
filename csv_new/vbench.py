#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: generate_summary_csv.py

功能说明:
    汇总 output_25 目录下多个方法 (ours / ltx / sd2.0 / camclone / ref) 的视频文件,
    生成一个统一的 CSV 表格,方便后续对比、查看或导入可视化工具。

    CSV 列说明:
        - index    : 视频文件名 (以 ours 文件夹下的 mp4 为准,共 208 个)
        - ours     : ours 方法对应视频的完整路径
        - ltx      : ltx 方法对应视频的完整路径
        - sd2.0    : sd2.0 方法对应视频的完整路径 (该文件夹仅 135 个,缺失则置空)
        - camclone : camclone 方法对应视频的完整路径
        - ref      : 参考视频的完整路径

用法:
    直接运行即可:
        python generate_summary_csv.py

    如需修改输入/输出路径,请编辑下方的路径变量 (base / output_csv)。

依赖:
    仅使用 Python 标准库 (os, csv),无需额外安装。
================================================================================
"""

import os
import csv

base = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25"
ours_dir   = os.path.join(base, "ours")
ltx_dir    = os.path.join(base, "ltx")
sd_dir     = os.path.join(base, "sd2.0")
cam_dir    = os.path.join(base, "camclone")
ref_dir    = os.path.join(base, "ref")

output_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/csv_new/summary.csv"

# 1. 以 ours 文件夹下的 mp4 作为 index (排序保证顺序稳定)
index_list = sorted([f for f in os.listdir(ours_dir) if f.endswith(".mp4")])
print(f"共找到 {len(index_list)} 个 index (来自 ours)")

# 用于统计各列实际存在的文件数量
stats = {"ours": 0, "ltx": 0, "sd2.0": 0, "camclone": 0, "ref": 0}

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # 写表头
    writer.writerow(["index", "ours", "ltx", "sd2.0", "camclone", "ref"])

    for name in index_list:
        ours_path = os.path.join(ours_dir, name)
        ltx_path  = os.path.join(ltx_dir, name)
        sd_path   = os.path.join(sd_dir, name)
        cam_path  = os.path.join(cam_dir, name)
        ref_path  = os.path.join(ref_dir, name)

        # sd2.0 可能缺失，缺失则置空
        sd_val = sd_path if os.path.exists(sd_path) else ""

        writer.writerow([name.replace(".mp4", ""), ours_path, ltx_path, sd_val, cam_path, ref_path])

        # 统计实际存在的文件数
        if os.path.exists(ours_path): stats["ours"] += 1
        if os.path.exists(ltx_path):  stats["ltx"] += 1
        if sd_val:                    stats["sd2.0"] += 1
        if os.path.exists(cam_path):  stats["camclone"] += 1
        if os.path.exists(ref_path):  stats["ref"] += 1

# ==========================================
# 输出统计信息
# ==========================================
total = len(index_list)
print("\n" + "=" * 40)
print("📊 输出统计")
print("=" * 40)
print(f"  总行数 (index): {total}")
print(f"  ours     实际存在: {stats['ours']:>4} / {total}  (缺失 {total - stats['ours']})")
print(f"  ltx      实际存在: {stats['ltx']:>4} / {total}  (缺失 {total - stats['ltx']})")
print(f"  sd2.0    实际存在: {stats['sd2.0']:>4} / {total}  (缺失 {total - stats['sd2.0']})")
print(f"  camclone 实际存在: {stats['camclone']:>4} / {total}  (缺失 {total - stats['camclone']})")
print(f"  ref      实际存在: {stats['ref']:>4} / {total}  (缺失 {total - stats['ref']})")
print("=" * 40)
print(f"✅ CSV 已生成: {output_csv}")