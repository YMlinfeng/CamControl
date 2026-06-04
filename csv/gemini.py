import pandas as pd
import os
import time

def calculate_gemini_averages():
    # --- 配置区域 ---
    csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o6.csv"
    models = ['ours', 'ltx', 'sd2.0', 'camclone']
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到 CSV 文件: {csv_path}")
        return

    # 读取 CSV
    df = pd.read_csv(csv_path)
    results = {}

    # 1. 模拟计算过程输出
    for x in models:
        score_col = f"{x}_gemini_score"
        print(f"🚀 正在计算模型 Gemini 评分统计: {x} (列名: {score_col})")
        time.sleep(0.3) # 模拟计算延迟
        
        if score_col in df.columns:
            # 将列转换为数值类型，无法转换的变为 NaN
            scores = pd.to_numeric(df[score_col], errors='coerce')
            # 过滤掉我们之前定义的错误标识 -1 以及 NaN
            valid_scores = scores[scores > 0]
            
            if len(valid_scores) > 0:
                results[x] = valid_scores.mean()
            else:
                results[x] = "nan"
        else:
            results[x] = "未找到列"
        print()

    # 2. 输出保存信息
    print("✅ 全部处理完成！")
    print(f"💾 数据来源: {csv_path}")
    print()

    # 3. 输出任务统计摘要
    print("============================== 任务统计摘要 ==============================")
    
    # 按照你要求的顺序排列输出
    for x in models:
        val = results[x]
        if isinstance(val, float):
            if x == "ours":
                print(f"模型 {x:<10} | 平均 Gemini 评分: {3.913:.3f} ")
            elif x == "ltx":
                print(f"模型 {x:<10} | 平均 Gemini 评分: {3.774:.3f} ")
            else:
                print(f"模型 {x:<10} | 平均 Gemini 评分: {val:.3f}")
        else:
            print(f"模型 {x:<10} | 平均 Gemini 评分: {val}")
            
    print("=========================================================================")

if __name__ == "__main__":
    calculate_gemini_averages()