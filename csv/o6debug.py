import os
import numpy as np
import pandas as pd
import json

# ================= 配置区 =================
CSV_PATH = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o5+.csv"
REFERENCE_COL = "dpa-v3"
MODELS = ['sd2.0', 'camclone', 'ltx', 'ours']
# ==========================================

def print_step(msg):
    print(f"\n{'='*20} {msg} {'='*20}")

def debug_comprehensive():
    # --- Step 1: CSV 文件基础检查 ---
    print_step("STEP 1: CSV 结构检查")
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到 CSV 文件: {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"✅ 成功读取 CSV. 总行数: {len(df)}")
    print(f"现有列名: {list(df.columns)}")
    
    for col in [REFERENCE_COL] + [f"{x}_npz" for x in MODELS]:
        if col not in df.columns:
            print(f"⚠️ 警告: 列 '{col}' 不在 CSV 中")

    # --- Step 2: 路径和字符检查 ---
    print_step("STEP 2: 路径和隐藏字符检查 (取前1行)")
    sample_row = df.iloc[0]
    for col in [REFERENCE_COL] + [f"{x}_npz" for x in MODELS]:
        if col in df.columns:
            raw_val = sample_row[col]
            path_str = str(raw_val).strip() if pd.notna(raw_val) else ""
            
            # 检查是否有隐藏的 \r (Windows换行符)
            if "\r" in str(raw_val):
                print(f"❌ 发现隐藏换行符 \\r ! 列: {col}")
            
            if path_str == "":
                print(f"⚪ 列 '{col}' 首行为空")
            elif not os.path.exists(path_str):
                print(f"❌ 文件不存在: [{col}] -> {repr(path_str)}")
            else:
                print(f"✅ 路径有效: [{col}] -> {path_str[:50]}...")

    # --- Step 3: NPZ 数据内部结构检查 ---
    print_step("STEP 3: NPZ 内部数据格式检查 (取首个有效对)")
    
    # 找到第一行所有模型和参考都有路径的数据
    test_row = None
    for i in range(len(df)):
        if pd.notna(df.iloc[i][REFERENCE_COL]):
            test_row = df.iloc[i]
            break
            
    if test_row is not None:
        ref_path = str(test_row[REFERENCE_COL]).strip()
        try:
            ref_data = np.load(ref_path)
            print(f"📄 参考文件 ({REFERENCE_COL}): {ref_path}")
            print(f"   - 包含的 Key: {list(ref_data.files)}")
            if "extrinsics" in ref_data:
                print(f"   - extrinsics 形状: {ref_data['extrinsics'].shape}")
            else:
                print(f"   ❌ 错误: 参考文件中找不到 'extrinsics' 键！")
        except Exception as e:
            print(f"❌ 无法读取参考 NPZ: {e}")

        for x in MODELS:
            gen_col = f"{x}_npz"
            if gen_col in df.columns and pd.notna(test_row[gen_col]):
                gen_path = str(test_row[gen_col]).strip()
                try:
                    gen_data = np.load(gen_path)
                    print(f"📄 模型文件 ({x}): {gen_path}")
                    if "extrinsics" in gen_data:
                        print(f"   - extrinsics 形状: {gen_data['extrinsics'].shape}")
                    else:
                        print(f"   ❌ 错误: 模型 {x} 的 NPZ 中找不到 'extrinsics' 键！")
                except Exception as e:
                    print(f"❌ 无法读取模型 {x} 的 NPZ: {e}")

    # --- Step 4: 模拟数学计算 ---
    print_step("STEP 4: 模拟数学计算逻辑")
    # 复制你源码里的核心函数
    def w2c_to_c2w_local(w2c):
        try:
            F_n = w2c.shape[0]
            w2c_4 = np.tile(np.eye(4)[None], (F_n, 1, 1))
            w2c_4[:, :3, :4] = w2c
            return np.linalg.inv(w2c_4)
        except Exception as e:
            return f"C2W转换失败: {e}"

    if test_row is not None:
        try:
            ref_w2c = np.load(str(test_row[REFERENCE_COL]).strip())["extrinsics"]
            res = w2c_to_c2w_local(ref_w2c)
            if isinstance(res, str):
                print(f"❌ 数学计算报错: {res}")
            else:
                print(f"✅ 数学计算 (W2C->C2W) 成功. 结果形状: {res.shape}")
        except Exception as e:
            print(f"❌ 计算流程中断: {e}")

    # --- Step 5: 索引映射逻辑检查 ---
    print_step("STEP 5: DataFrame 索引一致性检查")
    if df.index.is_monotonic_increasing and df.index[0] == 0:
        print("✅ DataFrame 索引是标准的 (0, 1, 2...)")
    else:
        print("⚠️ 警告: DataFrame 索引不连续。这可能导致 map() 映射失败。")
        print(f"起始索引: {df.index[0]}, 结束索引: {df.index[-1]}")

    print_step("DEBUG 结束")

if __name__ == "__main__":
    debug_comprehensive()