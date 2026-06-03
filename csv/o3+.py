"""
文件用途：
    修复错位的 CSV：读取含有换行符的 CSV 文件，找到所有的 reason 列，
    将其中所有隐藏的换行符 (\n, \r) 替换为空格，强制转换为绝对的单行结构。
"""

import pandas as pd
import os

def fix_csv_newlines():
    input_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o3.csv"
    output_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o3+.csv"

    if not os.path.exists(input_csv):
        print(f"❌ 找不到输入文件: {input_csv}")
        return

    print("正在读取原始 CSV ...")
    # Pandas 默认会正确识别双引号内的换行符，将它读为一个完整的单元格
    df = pd.read_csv(input_csv)

    # 自动找出所有名称中包含 'reason' 的列
    reason_cols = [col for col in df.columns if 'reason' in col.lower()]
    
    print(f"找到以下 reason 列需要处理: {reason_cols}")

    # 遍历这些列并剔除换行符
    for col in reason_cols:
        # 使用 lambda 表达式：如果是有效字符串，用 split() 切割掉所有 \n、\r、\t，再用空格连起来
        df[col] = df[col].apply(
            lambda x: " ".join(str(x).split()) if pd.notna(x) else x
        )

    print("正在保存修复后的 CSV ...")
    # 写入新文件
    df.to_csv(output_csv, index=False)
    
    print(f"✅ 成功！修复后的文件已保存至: {output_csv}")
    print("你现在可以使用 head 或 cat 命令查看新文件，数据已经绝对保证单行。")

if __name__ == "__main__":
    fix_csv_newlines()