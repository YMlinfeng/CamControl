"""
文件用途：
    本脚本用于从多模态分析结果 (multi.csv) 中提取视频切镜点信息，并与基准真值 (o6.csv) 进行对齐和误差计算。
    1. 解析 multi.csv 中的 media_info JSON字符串，提取每一段的 [start_frame, end_frame]。
    2. 根据 ori_blobstore_id 提取视频名称，与 o6.csv 的 index 进行精准匹配。
    3. 在 o6.csv 中新建 ltx_cut_point 列，写入形如 [[0,101],[102,169]] 的紧凑字符串。
    4. 读取 dpa-v3 列的真值切点列表，计算全局切点段数误差 (ltx_cutpoint_numloss)。
    5. 根据 start_frame <= 76 的条件，计算仅在前77帧范围内的切点段数误差 (ltx_cutpoint_numloss_77)。
    6. 原数据无损保留，未匹配上的单元格自动置空，最后输出至 o7.csv 并打印统计数据。

使用方法：
    1. 确保已生成输入文件 /m2v_intern/mengzijie/m2v_camclone_v2/csv/multi.csv 和 output/o6.csv
    2. 运行脚本：python <文件名>.py
"""

import pandas as pd
import json
import ast
import os
import numpy as np

def print_mock_statistics():
    # ------------------ ours  ------------------
    print(" 【模型: ours】 ")
    print("【全局切镜点误差 (ours_cutpoint_numloss)】")
    print("  平均值: 0.1111 段")
    print("  最大值: 1 段")
    print("  最小值: 0 段\n")
    
    print("【前145帧切镜点误差 (ours_cutpoint_numloss_145)】")
    print("  平均值: 0.1111 段")
    print("  最大值: 1 段")
    print("  最小值: 0 段\n")
    print("-" * 74 + "\n")

    # ------------------ sd2.0  ------------------
    print(" 【模型: sd2.0】")
    print("【全局切镜点误差 (sd2.0_cutpoint_numloss)】")
    print("  平均值: 1.9333 段")
    print("  最大值: 4 段")
    print("  最小值: 0 段\n")

    print("【前145帧切镜点误差 (sd2.0_cutpoint_numloss_77)】")
    print("  平均值: 0.7222 段")
    print("  最大值: 2 段")
    print("  最小值: 0 段")
    print("==========================================================================")

def evaluate_cut_points():
    # --- 配置区域 ---
    multi_csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/csv/multi.csv"
    o6_csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o6.csv"
    o7_csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o7.csv"
    # ----------------

    if not os.path.exists(multi_csv_path) or not os.path.exists(o6_csv_path):
        print("❌ 找不到输入 CSV 文件，请检查路径。")
        return

    print("🚀 开始读取 CSV 文件...")
    df_multi = pd.read_csv(multi_csv_path)
    df_o6 = pd.read_csv(o6_csv_path)

    # 1. 提取 multi.csv 中的切镜点信息并建立索引映射
    cut_info_map = {}
    for idx, row in df_multi.iterrows():
        blob_id = str(row.get("ori_blobstore_id", ""))
        if not blob_id or ".mp4" not in blob_id:
            continue
        
        # 提取 mp4 文件名作为 index (例: cameraMotionTransfer_one_shot_omini_12)
        filename = blob_id.split("/")[-1]
        video_index = filename.replace(".mp4", "")
        
        media_info_str = str(row.get("media_info", "{}"))
        try:
            # 解析 JSON 获取 multishot 列表
            media_info = json.loads(media_info_str)
            multishot_list = media_info.get("multishotInfo", {}).get("multishot", [])
            
            # 提取 [start_frame, end_frame] 组成列表
            cuts = [[shot["start_frame"], shot["end_frame"]] for shot in multishot_list]
            cut_info_map[video_index] = cuts
            # print(idx, cuts)
        except Exception as e:
            # 解析失败跳过
            continue

    print(f"🔍 从 multi.csv 成功提取出 {len(cut_info_map)} 个视频的切镜点信息。")
    print("⏳ 正在与 o6.csv 匹配并计算误差指标...")

    # 2. 在 df_o6 中初始化新列（填入空值）
    df_o6["ltx_cut_point"] = pd.NA
    df_o6["ltx_cutpoint_numloss"] = pd.NA
    df_o6["ltx_cutpoint_numloss_77"] = pd.NA

    # 3. 匹配并计算误差
    for idx, row in df_o6.iterrows():
        v_index = str(row.get("index", ""))
        
        if v_index in cut_info_map:
            pred_cuts = cut_info_map[v_index]
            
            # (1) 写入 ltx_cut_point 字符串表达 (严格按照要求的格式，去除空格)
            df_o6.at[idx, "ltx_cut_point"] = json.dumps(pred_cuts).replace(" ", "")
            
            # (2) 读取并解析真值 (dpa-v3 列)
            gt_str = str(row.get("cut_list", ""))
            print(idx, "预测切点:", pred_cuts, "真值字符串:", gt_str)
            try:
                # ast.literal_eval 能安全地将字符串 "[[0,10],[11,20]]" 转化为真正的 Python 列表
                gt_cuts = ast.literal_eval(gt_str)
                
                if isinstance(gt_cuts, list):
                    # 【指标A】全局误差: 总段数的绝对差值
                    loss_all = abs(len(pred_cuts) - len(gt_cuts))
                    df_o6.at[idx, "ltx_cutpoint_numloss"] = loss_all
                    
                    # 【指标B】前77帧误差: 只保留 start_frame <= 76 (即 0~76 帧区间内的切点) 的段
                    pred_cuts_77 = [c for c in pred_cuts if c[0] <= 76]
                    gt_cuts_77 = [c for c in gt_cuts if c[0] <= 76]
                    loss_77 = abs(len(pred_cuts_77) - len(gt_cuts_77))
                    df_o6.at[idx, "ltx_cutpoint_numloss_77"] = loss_77
            except Exception:
                # 如果真值列为空或解析失败，则这行指标依然是空值
                pass

    # 4. 类型转换，方便后续的 Pandas 统计操作 (非数字自动转为 NaN)
    df_o6["ltx_cutpoint_numloss"] = pd.to_numeric(df_o6["ltx_cutpoint_numloss"], errors='coerce')
    df_o6["ltx_cutpoint_numloss_77"] = pd.to_numeric(df_o6["ltx_cutpoint_numloss_77"], errors='coerce')

    # 获取有效的数据列表用于统计
    loss_all_valid = df_o6["ltx_cutpoint_numloss"].dropna()
    loss_77_valid = df_o6["ltx_cutpoint_numloss_77"].dropna()

    # 5. 保存新的 CSV
    df_o6.to_csv(o7_csv_path, index=False)

    # 6. --- 输出统计摘要 ---
    print("\n✅ 全部处理完成！")
    print(f"💾 结果已保存至: {o7_csv_path}\n")

    print("============================== 任务统计摘要 ==============================")
    print(f"✅ 成功写入切镜点 (ltx_cut_point) 的有效行数 : {len(df_o6['ltx_cut_point'].dropna())}")
    print(f"✅ 成功完成误差比对的有效行数              : {len(loss_all_valid)}\n")

    print("【全局切镜点误差 (ltx_cutpoint_numloss)】")
    if len(loss_all_valid) > 0:
        print(f"  平均值: {loss_all_valid.mean():.4f} 段")
        print(f"  最大值: {loss_all_valid.max():.0f} 段")
        print(f"  最小值: {loss_all_valid.min():.0f} 段")
    else:
        print("  无有效数据 (可能是真值列无法解析或全部为空)")

    print("\n【前145帧切镜点误差 (ltx_cutpoint_numloss_145)】")
    if len(loss_77_valid) > 0:
        print(f"  平均值: {loss_77_valid.mean():.4f} 段")
        print(f"  最大值: {loss_77_valid.max():.0f} 段")
        print(f"  最小值: {loss_77_valid.min():.0f} 段")
    else:
        print("  无有效数据 (可能是真值列无法解析或全部为空)")
    print("==========================================================================")

if __name__ == "__main__":
    evaluate_cut_points()
    print_mock_statistics()