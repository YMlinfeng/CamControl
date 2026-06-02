import pandas as pd
import json
import os

def diagnose_missing_paths():
    csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o1.csv"
    
    # 你反馈的 15 个失败编号
    failed_indices = [
        "cameraMotionTransfer_one_shot208", "cameraMotionTransfer_one_shot465", 
        "cameraMotionTransfer_one_shot493", "cameraMotionTransfer_one_shot235",
        "cameraMotionTransfer_one_shot347", "cameraMotionTransfer_one_shot458", 
        "cameraMotionTransfer_one_shot156", "cameraMotionTransfer_one_shot437",
        "cameraMotionTransfer_one_shot355", "cameraMotionTransfer_one_shot51", 
        "cameraMotionTransfer_one_shot374", "cameraMotionTransfer_one_shot81",
        "cameraMotionTransfer_one_shot163", "cameraMotionTransfer_one_shot324", 
        "cameraMotionTransfer_one_shot352"
    ]

    if not os.path.exists(csv_path):
        print(f"错误: 找不到 CSV 文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    print("="*80)
    print(f"{'Index 编号':<40} | {'检查结果'}")
    print("-"*80)

    for target_idx in failed_indices:
        row = df[df['index'] == target_idx]
        
        if row.empty:
            print(f"{target_idx:<40} | ❌ CSV 中根本没找到这个 index")
            continue
        
        row = row.iloc[0]
        
        # 1. 检查参考视频路径
        try:
            ref_v_data = json.loads(row['ref_videos'])
            ref_path = ref_v_data[0]['value']
            ref_exists = os.path.exists(ref_path)
        except:
            ref_path = "JSON解析失败"
            ref_exists = False

        # 2. 检查 LTX 生成视频路径
        ltx_path = str(row['ltx_gen'])
        ltx_exists = os.path.exists(ltx_path)

        # 3. 检查 OURS 生成视频路径
        ours_path = str(row['ours_gen'])
        ours_exists = os.path.exists(ours_path)

        # 输出诊断信息
        print(f"📍 {target_idx}")
        if not ref_exists:
            print(f"  FAILED -> 参考视频不存在: {ref_path}")
        if not ltx_exists:
            print(f"  FAILED -> LTX生成视频不存在: {ltx_path}")
        if not ours_exists:
            print(f"  FAILED -> OURS生成视频不存在: {ours_path}")
        if ref_exists and ltx_exists and ours_exists:
            print(f"  ✅ 奇怪：诊断显示路径都存在，请检查是否是权限问题。")
        print("-" * 40)

if __name__ == "__main__":
    diagnose_missing_paths()