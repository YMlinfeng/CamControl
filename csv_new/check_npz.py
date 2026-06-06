#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

# ====== 在这里填两个要对比的目录 ======
DIR_A = "/ytech_milm_disk2/lishujuan/motion-test/OmniDirector/eval_dataset_200/eval_poses/ytech_milm_disk2/lishujuan/motion-test/OmniDirector/eval_dataset_200/results/m2v_intern/mengzijie/m2v_camclone_v2/output_25/camclone"
DIR_B = "./camclone"   # 改成你的另一个目录
# =====================================


def list_npz(d):
    """返回该目录下所有 .npz 文件名（不含路径）的集合"""
    if not os.path.isdir(d):
        print(f"[ERROR] 目录不存在: {d}")
        sys.exit(1)
    return {f for f in os.listdir(d) if f.endswith(".npz")}


def compare_npz(path_a, path_b):
    """
    对比两个 npz 文件是否完全相同。
    返回 (is_equal: bool, message: str)
    """
    try:
        a = np.load(path_a, allow_pickle=True)
        b = np.load(path_b, allow_pickle=True)
    except Exception as e:
        return False, f"读取失败: {e}"

    keys_a = set(a.files)
    keys_b = set(b.files)

    if keys_a != keys_b:
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        return False, f"key 不一致 (仅A有: {sorted(only_a)} | 仅B有: {sorted(only_b)})"

    for k in sorted(keys_a):
        arr_a = a[k]
        arr_b = b[k]

        # 形状不同
        if arr_a.shape != arr_b.shape:
            return False, f"key '{k}' 形状不同: {arr_a.shape} vs {arr_b.shape}"

        # dtype 不同（仅警告，仍继续比较数值）
        if arr_a.dtype != arr_b.dtype:
            # 数值型仍可比较；object/str 型用 array_equal
            pass

        # 数值类型用 array_equal（完全相同，包括 NaN 位置一致才算等可用 equal_nan）
        if np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number):
            if not np.array_equal(arr_a, arr_b, equal_nan=True):
                # 给出最大差异，便于判断是“完全不同”还是“浮点微差”
                try:
                    diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
                    max_diff = np.nanmax(diff)
                    return False, f"key '{k}' 数值不同, 最大绝对差 = {max_diff:.6g}"
                except Exception:
                    return False, f"key '{k}' 数值不同"
        else:
            # 非数值（object、字符串等）
            if not np.array_equal(arr_a, arr_b):
                return False, f"key '{k}' 内容不同 (非数值类型)"

    return True, "完全相同"


def main():
    print("=" * 70)
    print(f"目录 A: {DIR_A}")
    print(f"目录 B: {DIR_B}")
    print("=" * 70)

    files_a = list_npz(DIR_A)
    files_b = list_npz(DIR_B)

    common = sorted(files_a & files_b)
    only_a = sorted(files_a - files_b)
    only_b = sorted(files_b - files_a)

    print(f"A 中 npz 数量: {len(files_a)}")
    print(f"B 中 npz 数量: {len(files_b)}")
    print(f"同名文件数量: {len(common)}")
    if only_a:
        print(f"[WARN] 仅 A 有的文件: {only_a}")
    if only_b:
        print(f"[WARN] 仅 B 有的文件: {only_b}")
    print("-" * 70)

    if not common:
        print("没有同名的 npz 文件可对比。")
        return

    same_cnt = 0
    diff_cnt = 0
    for idx, name in enumerate(common, 1):
        pa = os.path.join(DIR_A, name)
        pb = os.path.join(DIR_B, name)
        equal, msg = compare_npz(pa, pb)
        status = "✅ SAME" if equal else "❌ DIFF"
        print(f"[{idx}/{len(common)}] {status}  {name}  ->  {msg}", flush=True)
        if equal:
            same_cnt += 1
        else:
            diff_cnt += 1

    print("-" * 70)
    print(f"对比完成: 相同 {same_cnt} 个, 不同 {diff_cnt} 个, 共 {len(common)} 个")
    print("=" * 70)


if __name__ == "__main__":
    main()