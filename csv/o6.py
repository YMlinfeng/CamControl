"""
文件用途：
    批量计算真实 (dpa-v3) 与生成 ({x}_npz) 相机位姿的误差指标。
    脚本会遍历 sd2.0, camclone, ltx, ours 四个模型，提取位姿误差，
    并将以下三个结果写入新列：
    1. camera_motion_consistency (综合运动一致性) -> 写入 {x}_npz_value
    2. rot_err_deg_mean (平均旋转误差)            -> 写入 {x}_rot_err_deg_mean
    3. trans_err_mean (平均平移误差)              -> 写入 {x}_trans_err_mean

使用方法：
    1. 输入 CSV：/m2v_intern/mengzijie/m2v_camclone_v2/output/o5.csv
    2. 输出 CSV：/m2v_intern/mengzijie/m2v_camclone_v2/output/o6.csv
    3. 逻辑：对比 dpa-v3 列与 {x}_npz 列，计算出的误差存入对应新列。
    4. 脚本已处理路径不存在的情况，若路径缺失将跳过该行，不报错。
"""

# ★ 必须在 import numpy 之前设置，防止多进程时线程冲突卡死
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DOMAIN_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


# ════════════════ 相机位姿数学计算模块 (完整保留，未做任何修改) ════════════════

def w2c_to_c2w(w2c):
    F_n = w2c.shape[0]
    w2c_4 = np.tile(np.eye(4)[None], (F_n, 1, 1))
    w2c_4[:, :3, :4] = w2c
    return np.linalg.inv(w2c_4)

def orthonormalize_R(R):
    U, _, Vt = np.linalg.svd(R)
    R_o = U @ Vt
    if np.linalg.det(R_o) < 0:
        U[:, -1] *= -1
        R_o = U @ Vt
    return R_o

def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def compute_pose_errors(real_w2c, gen_w2c):
    F_n = min(len(real_w2c), len(gen_w2c))
    if F_n == 0:
        return np.array([]), np.array([]), np.array([])

    real_c2w = w2c_to_c2w(real_w2c[:F_n])
    gen_c2w = w2c_to_c2w(gen_w2c[:F_n])

    real_rel = np.linalg.inv(real_c2w[0])[None] @ real_c2w
    gen_rel = np.linalg.inv(gen_c2w[0])[None] @ gen_c2w

    real_t = real_rel[:, :3, 3]
    gen_t = gen_rel[:, :3, 3]

    real_len = np.linalg.norm(np.diff(real_t, axis=0), axis=1).sum() if F_n > 1 else 0.0
    gen_len = np.linalg.norm(np.diff(gen_t, axis=0), axis=1).sum() if F_n > 1 else 0.0
    scale = real_len / gen_len if gen_len > 1e-6 else 1.0
    gen_t_aligned = gen_t * scale

    trans_err = np.linalg.norm(real_t - gen_t_aligned, axis=1)

    rot_err = np.zeros(F_n)
    real_R = np.empty_like(real_rel[:, :3, :3])
    gen_R = np.empty_like(gen_rel[:, :3, :3])
    for i in range(F_n):
        real_R[i] = orthonormalize_R(real_rel[i, :3, :3])
        gen_R[i] = orthonormalize_R(gen_rel[i, :3, :3])
        rot_err[i] = rot_angle_deg(real_R[i] @ gen_R[i].T)

    real_pose34 = np.concatenate([real_R, real_t[..., None]], axis=2)
    gen_pose34 = np.concatenate([gen_R, gen_t_aligned[..., None]], axis=2)
    motion_err = np.linalg.norm(
        real_pose34.reshape(F_n, -1) - gen_pose34.reshape(F_n, -1), axis=1)

    return rot_err, trans_err, motion_err


# ════════════════ 多进程单任务工作函数 (完整保留) ════════════════

def process_single_pair(task):
    task_id = task["task_id"]
    real_npz = task["real_npz"]
    gen_npz = task["gen_npz"]
    extra_data = task.get("extra_data", {}) # 用于保留 CSV 原有列数据

    result = {"task_id": task_id, "real_npz": real_npz, "gen_npz": gen_npz}
    result.update(extra_data)

    try:
        # 判断路径是否存在，不存在则跳过
        if not real_npz or not os.path.exists(real_npz):
            result["metrics_status"] = "missing_real"
            return result
        if not gen_npz or not os.path.exists(gen_npz):
            result["metrics_status"] = "missing_gen"
            return result

        real_w2c = np.load(real_npz)["extrinsics"]
        gen_w2c = np.load(gen_npz)["extrinsics"]
        rot_err, trans_err, motion_err = compute_pose_errors(real_w2c, gen_w2c)

        if len(rot_err) == 0:
            result["metrics_status"] = "empty"
            return result

        result.update({
            "metrics_status": "ok",
            "pose_frames": int(len(rot_err)),
            "rot_err_deg_mean": float(np.mean(rot_err)),
            "rot_err_deg_median": float(np.median(rot_err)),
            "rot_err_deg_max": float(np.max(rot_err)),
            "trans_err_mean": float(np.mean(trans_err)),
            "trans_err_median": float(np.median(trans_err)),
            "trans_err_max": float(np.max(trans_err)),
            "camera_motion_consistency": float(np.mean(motion_err)),
        })
    except Exception as e:
        result["metrics_status"] = f"fail: {e!r}"
    
    return result


# ════════════════ 核心封装函数 (完整保留) ════════════════

def evaluate_pose_metrics_batch(
    out_csv: str,
    mode: str = "csv",
    csv_path: str = None,
    real_col: str = None,
    gen_col: str = None,
    real_dir: str = None,
    gen_dir: str = None,
    workers: int = 16
):
    tasks = []

    # 1. 准备任务列表
    if mode == "csv":
        if not csv_path or not real_col or not gen_col:
            raise ValueError("CSV 模式下必须提供 csv_path, real_col, gen_col 参数")
        
        df = pd.read_csv(csv_path)
        for c in [real_col, gen_col]:
            if c not in df.columns:
                raise ValueError(f"CSV 缺少列 '{c}'")

        for i, row in df.iterrows():
            tasks.append({
                "task_id": i,
                "real_npz": str(row[real_col]).strip() if pd.notna(row[real_col]) else "",
                "gen_npz": str(row[gen_col]).strip() if pd.notna(row[gen_col]) else "",
                "extra_data": {"original_index": i} # 仅保留索引用于写回
            })

    elif mode == "folder":
        # (保留 Folder 逻辑不变)
        if not real_dir or not gen_dir:
            raise ValueError("Folder 模式下必须提供 real_dir 和 gen_dir 参数")
        real_files = {f for f in os.listdir(real_dir) if f.endswith('.npz')}
        gen_files = {f for f in os.listdir(gen_dir) if f.endswith('.npz')}
        common_files = real_files.intersection(gen_files)
        for i, filename in enumerate(common_files):
            tasks.append({
                "task_id": i,
                "real_npz": os.path.join(real_dir, filename),
                "gen_npz": os.path.join(gen_dir, filename),
                "extra_data": {"filename": filename}
            })
    
    total = len(tasks)
    if total == 0: return pd.DataFrame()

    # 2. 多进程执行
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_single_pair, t): t for t in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    return pd.DataFrame(results)


# ════════════════ 适配你的任务逻辑 ════════════════

def main():
    # --- 你的特定任务配置 ---
    INPUT_CSV = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o5+.csv"
    OUTPUT_CSV = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o6.csv"
    REAL_COL = "dpa-v3"
    MODELS = ['sd2.0', 'camclone', 'ltx', 'ours']
    WORKERS = 16

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入文件: {INPUT_CSV}")
        return

    # 读取原始数据
    final_df = pd.read_csv(INPUT_CSV)
    
    # 循环为每个模型计算误差
    for model_x in MODELS:
        gen_col = f"{model_x}_npz"
        target_val_col = f"{model_x}_npz_value"
        target_rot_col = f"{model_x}_rot_err_deg_mean"
        target_trans_col = f"{model_x}_trans_err_mean"
        
        if gen_col not in final_df.columns:
            print(f"⚠️ 列 {gen_col} 不在 CSV 中，跳过。")
            continue

        print(f"\n🚀 正在计算模型误差: {model_x} (对比 {REAL_COL} vs {gen_col})")
        
        # 调用原始封装函数
        res_df = evaluate_pose_metrics_batch(
            out_csv="tmp.csv", # 实际上我们直接用返回的 DataFrame
            mode="csv",
            csv_path=INPUT_CSV,
            real_col=REAL_COL,
            gen_col=gen_col,
            workers=WORKERS
        )

        # 将计算出的指标映射回主表
        if not res_df.empty:
            # 1. 综合运动一致性
            if "camera_motion_consistency" in res_df.columns:
                mapping_motion = res_df.set_index("original_index")["camera_motion_consistency"].to_dict()
                final_df[target_val_col] = final_df.index.map(mapping_motion)
            else:
                final_df[target_val_col] = np.nan
                
            # 2. 平均旋转误差
            if "rot_err_deg_mean" in res_df.columns:
                mapping_rot = res_df.set_index("original_index")["rot_err_deg_mean"].to_dict()
                final_df[target_rot_col] = final_df.index.map(mapping_rot)
            else:
                final_df[target_rot_col] = np.nan
                
            # 3. 平均平移误差
            if "trans_err_mean" in res_df.columns:
                mapping_trans = res_df.set_index("original_index")["trans_err_mean"].to_dict()
                final_df[target_trans_col] = final_df.index.map(mapping_trans)
            else:
                final_df[target_trans_col] = np.nan
        else:
            final_df[target_val_col] = np.nan
            final_df[target_rot_col] = np.nan
            final_df[target_trans_col] = np.nan

    # 保存最终结果
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 全部处理完成！")
    print(f"💾 结果已保存至: {OUTPUT_CSV}")

    # 打印简要统计
    print("\n" + "="*20 + " 任务统计摘要 (各指标均值) " + "="*20)
    for model_x in MODELS:
        val_col = f"{model_x}_npz_value"
        rot_col = f"{model_x}_rot_err_deg_mean"
        trans_col = f"{model_x}_trans_err_mean"
        
        if val_col in final_df.columns:
            mean_err = final_df[val_col].mean()
            mean_rot = final_df[rot_col].mean() if rot_col in final_df.columns else np.nan
            mean_trans = final_df[trans_col].mean() if trans_col in final_df.columns else np.nan
            
            print(f"模型 {model_x:<10} | 综合误差: {mean_err:<8.4f} | 旋转误差(度): {mean_rot:<8.4f} | 平移误差: {mean_trans:<8.4f}")
    print("="*75)

if __name__ == "__main__":
    main()