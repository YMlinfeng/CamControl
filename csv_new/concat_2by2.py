#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: cut_and_concat_parallel.py

功能说明:
    读取 summary.csv,对视频做两步处理 (多进程 + 多 GPU 并行):

    【第一步:裁剪对齐 (cut)】
        - 对每一个 index (CSV 的每一行),取该行 5 个方法列
          (ours / ltx / sd2.0 / camclone / ref) 下的视频;
        - 以这几个视频里"帧数最少"的那个为基准长度,
          把同一行的每个视频都裁剪到该长度 (取前 N 帧);
        - 裁剪后视频存到 /m2v_intern/mengzijie/VBench/testdataset/{x}/ 下;
        - 保存路径写回 CSV 的 {x}_cut 列。  (x ∈ ours/ltx/sd2.0/camclone/ref)

    【第二步:左右拼接 (concat)】
        - 把 ref_cut 依次与 {y}_cut 做左右拼接,ref 在左,y 在右;
        - !! 只缩放 ref: 把 ref 的"高"对齐到右侧 y 视频的高 (保持 ref 宽高比),
             y 视频的尺寸保持不变 !!
        - 结果存到 /m2v_intern/mengzijie/VBench/testdataset/{y}_concat/ 下;
        - 保存路径写回 CSV 的 {y}_concat 列。  (y ∈ ours/ltx/sd2.0/camclone)

    并行策略:
        - 用 ProcessPoolExecutor 开 NUM_WORKERS 个进程;
        - 每个任务 (= 一行) 绑定一张 GPU: gpu_id = 任务序号 % NUM_GPUS;
        - ffmpeg 用 h264_nvenc 在指定 GPU 上编码 (-gpu N)。

    过程: 实时进度 + 末尾统计汇总 + 失败 case 写入 log.txt。

依赖:
    需要 ffmpeg / ffprobe,且 ffmpeg 编译时带 NVENC (h264_nvenc)。
    若环境无 NVENC,把 USE_GPU 设为 False,会退回 CPU 的 libx264。

用法:
    python cut_and_concat_parallel.py
================================================================================
"""

import os
import csv
import time
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 配置
# ==========================================
INPUT_CSV    = "/m2v_intern/mengzijie/m2v_camclone_v2/csv_new/summary.csv"
OUTPUT_CSV   = "/m2v_intern/mengzijie/m2v_camclone_v2/csv_new/summary_cut_concat.csv"
DATASET_ROOT = "/m2v_intern/mengzijie/VBench/testdataset"
LOG_PATH     = os.path.join(DATASET_ROOT, "log.txt")

CUT_COLS    = ["ours", "ltx", "sd2.0", "camclone", "ref"]   # 需要裁剪的 5 列
CONCAT_COLS = ["ours", "ltx", "sd2.0", "camclone"]          # 需与 ref 拼接的 4 列

NUM_GPUS    = 8     # GPU 数量
NUM_WORKERS = 8     # 并行进程数 (一般 = GPU 数)
USE_GPU     = True  # True: h264_nvenc;  False: libx264 (CPU)

# tqdm 可选
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ==========================================
# 工具函数 (会在子进程中调用,需放在模块顶层以便 pickle)
# ==========================================
def get_frame_count(path):
    """获取视频帧数。优先用 nb_frames,失败再用 时长×帧率,最后才逐帧 count。"""
    # 方法1: 直接读 nb_frames (最快)
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=nb_frames",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        if out.isdigit() and int(out) > 0:
            return int(out)
    except Exception:
        pass

    # 方法2: 时长 × 平均帧率
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=avg_frame_rate,duration",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip().splitlines()
        rate, dur = None, None
        for line in out:
            line = line.strip()
            if "/" in line:            # avg_frame_rate 形如 25/1
                num, den = line.split("/")
                if float(den) != 0:
                    rate = float(num) / float(den)
            else:
                try:
                    dur = float(line)
                except ValueError:
                    pass
        if rate and dur:
            return int(round(rate * dur))
    except Exception:
        pass

    # 方法3: 逐帧 count (最慢,兜底)
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_frames", "-show_entries", "stream=nb_read_frames",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        if out.isdigit() and int(out) > 0:
            return int(out)
    except Exception:
        pass

    return None


def _enc_args(gpu):
    """返回编码器参数"""
    if USE_GPU:
        return ["-c:v", "h264_nvenc", "-gpu", str(gpu), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p"]


def run_cmd(cmd):
    """执行命令,成功 (True,'') 失败 (False, 错误片段)"""
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode(errors="ignore")[-500:]
    except Exception as e:
        return False, str(e)


def crop_video(src, dst, n_frames, gpu):
    """裁剪 src 到前 n_frames 帧"""
    cmd = ["ffmpeg", "-y", "-i", src, "-frames:v", str(n_frames), "-an"] \
          + _enc_args(gpu) + [dst]
    return run_cmd(cmd)


def concat_lr(ref, y, dst, gpu):
    """
    左右拼接: ref 在左, y 在右。
    只缩放 ref: 把 ref 的高对齐到 y 的高 (oh=ih=y高),
    宽按 ref 自身宽高比 (mdar) 计算 → y 尺寸完全不变。
      input0 = ref (main, 被缩放)
      input1 = y   (ref, 作为参照, 不变)
    scale2ref 中 ih/oh 指参照(y)的高, mdar 指 main(ref) 的显示宽高比。
    """
    flt = ("[0:v][1:v]scale2ref=w=oh*mdar:h=ih[refs][yo];"
           "[refs][yo]hstack=inputs=2")
    cmd = ["ffmpeg", "-y", "-i", ref, "-i", y,
           "-filter_complex", flt, "-an"] + _enc_args(gpu) + [dst]
    return run_cmd(cmd)


# ==========================================
# 子进程任务: 裁剪一整行
# ==========================================
def cut_one_row(task):
    """
    task = (idx, name, srcs, gpu)
        srcs: dict {col: src_path}  仅含非空列
    返回 dict: index / cut(成功的 {col:dst}) / logs / success / fail
    """
    idx, name, srcs, gpu = task
    res = {"index": idx, "cut": {}, "logs": [], "success": 0, "fail": 0}

    # 1) 统计每个视频帧数
    frame_counts = {}
    for col, src in srcs.items():
        if not os.path.exists(src):
            res["logs"].append(f"[cut][缺文件] index={idx} col={col} path={src}")
            res["fail"] += 1
            continue
        fc = get_frame_count(src)
        if not fc or fc <= 0:
            res["logs"].append(f"[cut][读帧失败] index={idx} col={col} path={src}")
            res["fail"] += 1
            continue
        frame_counts[col] = fc

    if not frame_counts:
        res["logs"].append(f"[cut][整行无有效视频] index={idx}")
        return res

    # 2) 取最小帧数为基准
    min_frames = min(frame_counts.values())

    # 3) 逐列裁剪
    for col in frame_counts:
        src = srcs[col]
        dst = os.path.join(DATASET_ROOT, col, f"{name}.mp4")
        ok, err = crop_video(src, dst, min_frames, gpu)
        if ok:
            res["cut"][col] = dst
            res["success"] += 1
        else:
            res["logs"].append(
                f"[cut][ffmpeg失败] index={idx} col={col} "
                f"min_frames={min_frames}\n        src={src}\n        err={err}")
            res["fail"] += 1
    return res


# ==========================================
# 子进程任务: 拼接一整行
# ==========================================
def concat_one_row(task):
    """
    task = (idx, name, ref_cut, ycuts, gpu)
        ycuts: dict {y: y_cut_path}
    返回 dict: index / concat(成功的 {y:dst}) / logs / success / fail / skip
    """
    idx, name, ref_cut, ycuts, gpu = task
    res = {"index": idx, "concat": {}, "logs": [],
           "success": 0, "fail": 0, "skip": 0}

    if not ref_cut or not os.path.exists(ref_cut):
        res["logs"].append(f"[concat][缺ref_cut] index={idx} ref_cut={ref_cut}")
        res["skip"] += len(CONCAT_COLS)
        return res

    for y in CONCAT_COLS:
        y_cut = ycuts.get(y, "")
        if not y_cut or not os.path.exists(y_cut):
            res["logs"].append(f"[concat][缺{y}_cut] index={idx} {y}_cut={y_cut}")
            res["skip"] += 1
            continue
        dst = os.path.join(DATASET_ROOT, f"{y}_concat", f"{name}.mp4")
        ok, err = concat_lr(ref_cut, y_cut, dst, gpu)
        if ok:
            res["concat"][y] = dst
            res["success"] += 1
        else:
            res["logs"].append(
                f"[concat][ffmpeg失败] index={idx} y={y}\n"
                f"        ref={ref_cut}\n        y={y_cut}\n        err={err}")
            res["fail"] += 1
    return res


# ==========================================
# 进度迭代器 (兼容有无 tqdm)
# ==========================================
def progress(futures_iter, total, desc):
    if HAS_TQDM:
        yield from tqdm(futures_iter, total=total, desc=desc, ncols=90, unit="row")
    else:
        start = time.time()
        for i, item in enumerate(futures_iter, 1):
            el = time.time() - start
            eta = (total - i) / (i / el) if el > 0 else 0
            print(f"\r  [{desc}] {i}/{total} ({i*100//total}%)  "
                  f"已用 {el:.0f}s  剩余 {eta:.0f}s   ", end="", flush=True)
            yield item
        print()


# ==========================================
# 主流程
# ==========================================
def main():
    # 建输出目录
    for x in CUT_COLS:
        os.makedirs(os.path.join(DATASET_ROOT, x), exist_ok=True)
    for y in CONCAT_COLS:
        os.makedirs(os.path.join(DATASET_ROOT, f"{y}_concat"), exist_ok=True)

    # 读 CSV
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames[:]

    for x in CUT_COLS:
        if f"{x}_cut" not in fieldnames:
            fieldnames.append(f"{x}_cut")
    for y in CONCAT_COLS:
        if f"{y}_concat" not in fieldnames:
            fieldnames.append(f"{y}_concat")

    total_rows = len(rows)
    print(f"共读取 {total_rows} 行;并行进程 {NUM_WORKERS},GPU {NUM_GPUS},"
          f"编码 {'NVENC' if USE_GPU else 'libx264'}\n")

    all_logs = []
    stats = {"rows": total_rows,
             "cut_success": 0, "cut_fail": 0, "cut_skip_empty": 0,
             "concat_success": 0, "concat_fail": 0, "concat_skip": 0}

    # 用 index 快速定位行
    row_by_idx = {r["index"]: r for r in rows}

    # ---------- 第一步: 裁剪 (并行) ----------
    print("=" * 60)
    print("第一步:裁剪对齐 (cut)")
    print("=" * 60)

    cut_tasks = []
    for n, row in enumerate(rows):
        idx = row["index"]
        name = os.path.splitext(idx)[0]
        srcs = {}
        for x in CUT_COLS:
            src = (row.get(x) or "").strip()
            if not src:
                stats["cut_skip_empty"] += 1   # 空单元格 (如 sd2.0 缺失)
                continue
            srcs[x] = src
        gpu = n % NUM_GPUS
        cut_tasks.append((idx, name, srcs, gpu))

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = [ex.submit(cut_one_row, t) for t in cut_tasks]
        for fut in progress(as_completed(futs), len(futs), "cut"):
            r = fut.result()
            all_logs.extend(r["logs"])
            stats["cut_success"] += r["success"]
            stats["cut_fail"]    += r["fail"]
            row = row_by_idx[r["index"]]
            for col, dst in r["cut"].items():
                row[f"{col}_cut"] = dst

    # ---------- 第二步: 拼接 (并行) ----------
    print("\n" + "=" * 60)
    print("第二步:左右拼接 (concat)  ref 在左(按 y 高缩放),y 在右(尺寸不变)")
    print("=" * 60)

    concat_tasks = []
    for n, row in enumerate(rows):
        idx = row["index"]
        name = os.path.splitext(idx)[0]
        ref_cut = (row.get("ref_cut") or "").strip()
        ycuts = {y: (row.get(f"{y}_cut") or "").strip() for y in CONCAT_COLS}
        gpu = n % NUM_GPUS
        concat_tasks.append((idx, name, ref_cut, ycuts, gpu))

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = [ex.submit(concat_one_row, t) for t in concat_tasks]
        for fut in progress(as_completed(futs), len(futs), "concat"):
            r = fut.result()
            all_logs.extend(r["logs"])
            stats["concat_success"] += r["success"]
            stats["concat_fail"]    += r["fail"]
            stats["concat_skip"]    += r["skip"]
            row = row_by_idx[r["index"]]
            for y, dst in r["concat"].items():
                row[f"{y}_concat"] = dst

    # ---------- 写 CSV ----------
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for col in fieldnames:
                row.setdefault(col, "")
            writer.writerow(row)

    # ---------- 写日志 ----------
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        if all_logs:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"# 失败日志 生成于 {ts}  共 {len(all_logs)} 条\n\n")
            f.write("\n".join(all_logs) + "\n")
        else:
            f.write("无失败记录,全部处理成功。\n")

    # ---------- 统计汇总 ----------
    print("\n" + "=" * 60)
    print("📊 处理统计汇总")
    print("=" * 60)
    print(f"  总行数 (index):        {stats['rows']}")
    print(f"  ── 裁剪 (cut) ──")
    print(f"     成功:               {stats['cut_success']}")
    print(f"     失败:               {stats['cut_fail']}")
    print(f"     空单元格(跳过):     {stats['cut_skip_empty']}")
    print(f"  ── 拼接 (concat) ──")
    print(f"     成功:               {stats['concat_success']}")
    print(f"     失败:               {stats['concat_fail']}")
    print(f"     跳过(缺cut素材):    {stats['concat_skip']}")
    print("=" * 60)
    print(f"  新 CSV 已保存: {OUTPUT_CSV}")
    print(f"  失败日志:      {LOG_PATH}  (共 {len(all_logs)} 条)")
    print("=" * 60)


if __name__ == "__main__":
    main()