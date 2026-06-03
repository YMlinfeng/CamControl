"""
文件用途：
    1. 处理指定 CSV 中的视频路径：将 {x}_gen 重命名为 {x}_ori_gen，并修改路径中的文件夹名。
    2. 视频预处理：对 {x}_ori_gen 中的视频进行截取（前77帧）和帧率转换（FPS=15）。
    3. 结果保存：处理后的视频存入新目录，并将新路径写入 CSV 的 {x}_gen 列。
    4. 健壮性：自动跳过不存在的路径或空单元格，并输出最终处理统计报告。

使用方法：
    1. 确保系统已安装 ffmpeg。
    2. 检查脚本开头的路径配置是否正确。
    3. 运行：python <文件名>.py
    4. 结果将保存为 o2.csv，不影响原始 o1.csv。
"""

import pandas as pd
import os
import subprocess
from tqdm import tqdm

def process_eval_videos():
    # --- 配置区域 ---
    input_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o1.csv"
    output_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o2.csv"
    eval_root = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200"
    models = ['ltx', 'ours', 'sd2.0']
    # ----------------

    if not os.path.exists(input_csv):
        print(f"❌ 错误: 找不到输入 CSV 文件 {input_csv}")
        return

    # 读取 CSV
    df = pd.read_csv(input_csv)
    
    # 初始化统计字典
    stats = {x: {"total": 0, "processed": 0, "missing": 0, "failed": 0} for x in models}

    for x in models:
        old_gen_col = f"{x}_gen"
        ori_gen_col = f"{x}_ori_gen"
        
        if old_gen_col not in df.columns:
            print(f"⚠️ 跳过: 列 {old_gen_col} 不在 CSV 中。")
            continue

        print(f"\n🚀 正在处理模型: {x}")

        # 1. 重命名列 {x}_gen -> {x}_ori_gen
        df.rename(columns={old_gen_col: ori_gen_col}, inplace=True)

        # 2. 修正路径逻辑：将路径中的 /{x}/ 替换为 /{x}_ori/
        # 注意：使用 pd.notna 确保不处理空值
        df[ori_gen_col] = df[ori_gen_col].apply(
            lambda p: str(p).replace(f"/{x}/", f"/{x}_ori/") if pd.notna(p) and str(p).strip() != "" else p
        )

        processed_paths = []
        stats[x]["total"] = len(df)

        # 3. 遍历视频进行物理处理
        for index, src_path in enumerate(tqdm(df[ori_gen_col], desc=f"Progress ({x})")):
            # 逻辑：如果路径为空或文件不存在，写入空值并跳过
            if pd.isna(src_path) or str(src_path).strip() == "" or not os.path.exists(str(src_path)):
                processed_paths.append("")
                stats[x]["missing"] += 1
                continue

            # 确定输出目录和路径
            out_dir = os.path.join(eval_root, x)
            os.makedirs(out_dir, exist_ok=True)
            video_name = os.path.basename(str(src_path))
            dst_path = os.path.join(out_dir, video_name)

            # FFmpeg 处理：截取前77帧，FPS转为15
            # -y: 覆盖已存在文件; -v error: 只输出错误日志; -frames:v 77: 截取帧数; -r 15: 设置帧率
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(src_path),
                '-frames:v', '77',
                '-r', '15',
                dst_path
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                processed_paths.append(dst_path)
                stats[x]["processed"] += 1
            except subprocess.CalledProcessError:
                processed_paths.append("")
                stats[x]["failed"] += 1

        # 4. 将新生成的路径存入新增的 {x}_gen 列
        df[old_gen_col] = processed_paths

    # 保存新的 CSV
    df.to_csv(output_csv, index=False)

    # --- 输出统计结果报告 ---
    print("\n" + "="*30 + " 处理报告 " + "="*30)
    print(f"{'模型':<10} | {'总计':<8} | {'成功':<8} | {'缺失(跳过)':<10} | {'失败':<8}")
    print("-" * 70)
    for x, s in stats.items():
        print(f"{x:<10} | {s['total']:<8} | {s['processed']:<8} | {s['missing']:<10} | {s['failed']:<8}")
    print("="*70)
    print(f"✅ 处理完成！结果已保存至: {output_csv}")

if __name__ == "__main__":
    process_eval_videos()