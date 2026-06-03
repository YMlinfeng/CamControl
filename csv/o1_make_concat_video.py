"""
文件用途：
    该脚本用于校验 CSV 文件中指定的视频路径是否存在。
    通过检查 CSV 中的路径列，统计并列出所有文件缺失的路径。

使用方法：
    1. 修改 `csv_path` 为你需要检查的 CSV 文件路径。
    2. 在 `columns_to_check` 中填入需要校验的列名。
    3. 设置 `REMOVE_MISSING` 参数：
        - True: 仅清除不存在的路径字符串（单元格置空），保留整行其他数据，并保存。
        - False: 仅输出测试报告，不修改文件。
    4. 运行：python <文件名>.py
"""

import pandas as pd
import os
from tqdm import tqdm

def verify_video_paths():
    # --- 配置区域 ---
    # 定义 CSV 路径
    csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o1.csv"
    
    # 控制参数：如果开启，将清除不存在的路径（单元格内容置空），但保留该行
    REMOVE_MISSING = True 
    
    # 定义需要检查的列名
    columns_to_check = [
        # 'ltx_gen', 
        # 'ltx_concat', 
        # 'ours_gen', 
        # 'ours_concat', 
        # 'camclone_gen', 
        # 'camclone_concat',
        'sd2.0_gen',
        'sd2.0_concat'
    ]
    # ----------------

    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到 CSV 文件 {csv_path}")
        return

    # 读取 CSV
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    print(f"开始测试... 总计行数: {total_rows}")
    print(f"待检查列: {', '.join(columns_to_check)}")
    print(f"清除模式: {'开启 (将清除无效单元格)' if REMOVE_MISSING else '关闭 (仅报告)'}")
    print("-" * 50)

    # 存储统计结果
    missing_files = {col: [] for col in columns_to_check}
    total_missing_count = 0
    modified_flag = False

    # 遍历每一行进行检查
    # 使用 tqdm 显示进度条
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="校验进度"):
        for col in columns_to_check:
            path = row[col]
            
            # 检查路径是否为空或文件是否存在
            if pd.isna(path) or not os.path.exists(str(path)):
                missing_files[col].append({
                    'row_index': index,
                    'video_index': row.get('index', 'N/A'),
                    'path': path
                })
                total_missing_count += 1
                
                # 如果开启清除模式，将该单元格内容清空
                if REMOVE_MISSING:
                    df.at[index, col] = ""
                    modified_flag = True

    # --- 输出测试报告 ---
    print("\n" + "="*20 + " 测试报告 " + "="*20)
    
    if total_missing_count == 0:
        print("✅ 完美！所有列中的所有视频文件均已找到。")
    else:
        print(f"❌ 警告: 共发现 {total_missing_count} 个位置路径缺失！\n")
        
        for col, missing_list in missing_files.items():
            missing_num = len(missing_list)
            status = "OK" if missing_num == 0 else f"缺失 {missing_num}"
            print(f"列 [{col}]: {status}")
            
            # 如果该列有缺失，列出前 5 个缺失的具体路径供参考
            if missing_num > 0:
                for item in missing_list[:5]:
                    print(f"   - 行 {item['row_index']} (Index: {item['video_index']}): {item['path']}")
                if missing_num > 5:
                    print(f"   - ... 还有 {missing_num - 5} 个缺失项未列出")
                print("-" * 30)

        # --- 执行保存操作 ---
        if REMOVE_MISSING and modified_flag:
            print(f"\n正在更新 CSV 文件...")
            df.to_csv(csv_path, index=False)
            print(f"♻️ 已将 {total_missing_count} 个无效路径从单元格中清除并保存。")

    print("=" * 50)

if __name__ == "__main__":
    verify_video_paths()