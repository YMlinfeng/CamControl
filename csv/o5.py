"""
文件用途：
    该脚本用于处理 CSV 文件中 cut_list 字段的区间逻辑。
    原始数据通常采用左闭右开区间（例如 [0, 41], [41, 57]），为了适配特定需求，
    本脚本将其转换为左闭右闭区间（例如 [0, 40], [41, 56]）。
    规则：对于 cut_list 中的每一组 [start, end]，如果它不是列表中的最后一组，则将 end 减 1。

使用方法：
    1. 确保输入文件路径正确：/m2v_intern/mengzijie/m2v_camclone_v2/output/o4.csv
    2. 运行脚本：python <文件名>.py
    3. 处理后的结果将保存至：/m2v_intern/mengzijie/m2v_camclone_v2/output/o5.csv
"""

import pandas as pd
import ast
import json
import os

def process_cut_intervals():
    # --- 配置区域 ---
    input_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o4.csv"
    output_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o5.csv"
    # ----------------

    if not os.path.exists(input_csv):
        print(f"❌ 错误: 找不到输入文件 {input_csv}")
        return

    # 读取 CSV
    df = pd.read_csv(input_csv)
    
    total_rows = len(df)
    processed_count = 0
    skipped_count = 0
    error_count = 0

    def convert_to_closed_interval(cell):
        nonlocal processed_count, skipped_count, error_count
        
        # 1. 如果是空值或不是字符串，直接跳过
        if pd.isna(cell) or str(cell).strip() == "" or str(cell).strip().lower() == "nan":
            skipped_count += 1
            return cell
        
        try:
            # 2. 将字符串解析为 Python 列表
            # 使用 ast.literal_eval 安全解析 [[0,41],[41,57]...]
            intervals = ast.literal_eval(str(cell))
            
            if not isinstance(intervals, list) or len(intervals) == 0:
                skipped_count += 1
                return cell
            
            # 3. 处理逻辑：除了最后一组，前几组的 end 都减 1
            new_intervals = []
            num_groups = len(intervals)
            
            for i in range(num_groups):
                start, end = intervals[i]
                if i < num_groups - 1:
                    # 不是最后一组，变更为左闭右闭
                    new_intervals.append([start, end - 1])
                else:
                    # 最后一组保持不变
                    new_intervals.append([start, end])
            
            processed_count += 1
            # 4. 返回 JSON 格式的字符串（紧凑无空格，符合原始风格）
            return json.dumps(new_intervals, separators=(',', ':'))
            
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"⚠️ 解析错误 (行内容: {cell}): {e}")
            error_count += 1
            return cell

    print(f"开始处理 CSV，总行数: {total_rows}...")

    # 应用转换逻辑
    df['cut_list'] = df['cut_list'].apply(convert_to_closed_interval)

    # 保存结果
    df.to_csv(output_csv, index=False)

    # --- 结果统计 ---
    print("\n" + "="*30 + " 处理结果统计 " + "="*30)
    print(f"📊 总行数:         {total_rows}")
    print(f"✅ 成功转换行数:   {processed_count}")
    print(f"⚪ 跳过行数 (空值): {skipped_count}")
    print(f"❌ 异常行数:       {error_count}")
    print("-" * 74)
    print(f"💾 结果已保存至: {output_csv}")
    print("=" * 74)

if __name__ == "__main__":
    process_cut_intervals()