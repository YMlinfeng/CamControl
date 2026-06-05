"""
文件用途：
    本脚本用于读取 o7.csv 中的数据，提取原视频和生成的四个模型的视频。
    1. 解析 ref_videos 列，提取真实的参考视频路径 (GUIDE_VIDEO)。
    2. 按 [ref_videos, sd2.0, camclone, ltx, ours] 的顺序将这 5 个视频横向拼接。
    3. 为保证拼接不报错，强制将所有视频等比例缩放至高度 512 像素，并对齐帧率。
    4. 拼接好的视频存入 /eval_dataset_200/concat/ 目录。
    5. 原 CSV 内容完全保留，并新增一列 'all_concat' 写入新生成的视频路径，最后保存为 o8.csv。

使用方法：
    1. 确保系统已安装 ffmpeg 工具。
    2. 运行环境需要安装 pandas, tqdm。
    3. 运行命令：python <文件名>.py
"""

import os
import json
import subprocess
import pandas as pd
from tqdm import tqdm

def extract_ref_video(json_str):
    """从类似 [{"type": "GUIDE_VIDEO", "value": "/path/video.mp4"}] 的字符串中提取路径"""
    if pd.isna(json_str) or not str(json_str).strip():
        return ""
    try:
        data = json.loads(str(json_str))
        for item in data:
            if item.get("type") == "GUIDE_VIDEO":
                return str(item.get("value", ""))
    except Exception:
        pass
    return ""

def concat_videos():
    # --- 配置区域 ---
    INPUT_CSV = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o7.csv"
    OUTPUT_CSV = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o8.csv"
    CONCAT_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/concat"
    
    # 按照你的要求，依次为 sd2.0, camclone, ltx, ours
    MODELS = ['sd2.0', 'camclone', 'ltx', 'ours']
    
    os.makedirs(CONCAT_DIR, exist_ok=True)
    # ----------------

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入 CSV 文件: {INPUT_CSV}")
        return

    print("🚀 开始读取 CSV 文件并准备视频拼接任务...")
    df = pd.read_csv(INPUT_CSV)
    
    # 初始化新列
    if "all_concat" not in df.columns:
        df["all_concat"] = pd.NA

    stats = {"total": len(df), "success": 0, "missing_files": 0, "error": 0}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🎥 拼接进度"):
        vid_index = str(row.get("index", f"video_{idx}"))
        
        # 1. 提取 5 个视频的路径
        ref_path = extract_ref_video(row.get("ref_videos", ""))
        
        # 组装待拼接路径列表：第 1 个是 ref，后 4 个是生成结果
        paths = [ref_path]
        for x in MODELS:
            paths.append(str(row.get(f"{x}_gen", "")).strip())
            
        # 2. 检查 5 个视频是否全都存在
        all_exist = True
        for p in paths:
            if not p or p == "nan" or not os.path.exists(p):
                all_exist = False
                break
                
        if not all_exist:
            stats["missing_files"] += 1
            df.at[idx, "all_concat"] = ""
            continue
            
        # 3. 配置输出路径
        out_path = os.path.join(CONCAT_DIR, f"{vid_index}_concat5.mp4")
        
        # 4. 构建超级 FFmpeg 命令 (横向拼接 5 个视频)
        cmd = ["ffmpeg", "-y", "-v", "error"]
        
        # 传入 5 个输入源
        for p in paths:
            cmd.extend(["-i", p])
            
        # 构建复杂的 FilterGraph：
        # - scale=-2:512 (高度统一为512，宽度等比缩放且为2的倍数)
        # - fps=15 (统一帧率防止不同步)
        # - hstack=inputs=5 (横向拼接 5 个视频)
        filter_complex = ""
        for i in range(5):
            filter_complex += f"[{i}:v]scale=-2:512,fps=15[v{i}];"
            
        filter_complex += "".join([f"[v{i}]" for i in range(5)]) + "hstack=inputs=5[v]"
        
        cmd.extend([
            "-filter_complex", filter_complex, 
            "-map", "[v]", 
            "-an",  # 丢弃音频流，避免报错
            "-c:v", "libx264", 
            "-crf", "23", 
            "-preset", "fast", 
            out_path
        ])
        
        # 5. 执行拼接
        try:
            subprocess.run(cmd, check=True)
            # 记录成功的路径
            df.at[idx, "all_concat"] = out_path
            stats["success"] += 1
        except Exception as e:
            tqdm.write(f"❌ 拼接失败 [Index: {vid_index}]: {e}")
            df.at[idx, "all_concat"] = ""
            stats["error"] += 1
            
        # 实时保存，防止中断
        if stats["success"] % 5 == 0:
            df.to_csv(OUTPUT_CSV, index=False)

    # 完整保存
    df.to_csv(OUTPUT_CSV, index=False)
    
    # --- 6. 统计报告 ---
    print("\n" + "="*50)
    print("🎉 五屏同框视频拼接任务全部完成！")
    print("="*50)
    print(f"📊 总任务行数     : {stats['total']}")
    print(f"✅ 成功拼接输出   : {stats['success']}")
    print(f"⚠️ 因缺失文件跳过 : {stats['missing_files']}")
    print(f"❌ FFmpeg 执行报错 : {stats['error']}")
    print("="*50)
    print(f"📁 拼接视频目录 : {CONCAT_DIR}")
    print(f"💾 新的 CSV 文件已保存至 : {OUTPUT_CSV}")
    print("==================================================")

if __name__ == "__main__":
    concat_videos()