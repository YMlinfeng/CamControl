#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: cut_videos_by_index.py

功能:
    每一行 (index) 对比 ours/ltx/sd2.0/camclone/ref 五列视频,取帧数最少者为基准,
    把该行所有可用视频统一裁到该最短帧数 (取前 N 帧),保存到
    /m2v_intern/mengzijie/VBench/testdataset/{列名}/,并写回 {列名}_cut 列。

本版改动 (针对"无法读取帧数"全部失败的排查):
    - 启动时自检 ffprobe / ffmpeg 是否可用,并打印版本。
    - probe / crop 失败时,把"真实错误原因"同时打印到终端 + 写入 log.txt。
    - 新增 DEBUG 开关:打开后打印每次 ffprobe 的原始返回值,便于定位。

依赖: 系统 ffmpeg/ffprobe;仅用 Python 标准库。
用法: python cut_videos_by_index.py
================================================================================
"""

import os
import csv
import sys
import time
import shutil
import subprocess
from multiprocessing import Process, Value, Manager

# ============== 配置区 ==============
INPUT_CSV   = "/m2v_intern/mengzijie/m2v_camclone_v2/csv_new/summary.csv"
OUTPUT_CSV  = "/m2v_intern/mengzijie/m2v_camclone_v2/csv_new/summary_cut.csv"
OUTPUT_BASE = "/m2v_intern/mengzijie/VBench/testdataset"
LOG_FILE    = "/m2v_intern/mengzijie/m2v_camclone_v2/csv_new/log.txt"

COLUMNS     = ["ours", "ltx", "sd2.0", "camclone", "ref"]
NUM_WORKERS = 8
USE_GPU     = False
DEBUG       = False     # True: 打印每次 ffprobe 原始返回,排查时用;稳定后可设 False
# ===================================


def log_err(msg, errors):
    """同时打印到终端 + 收集到共享 errors 列表"""
    print(msg, flush=True)
    errors.append(msg)


def _ffprobe(args):
    """
    调用 ffprobe。
    返回 (stdout字符串, 错误原因或None)。
    成功: (out, None);失败: ("", 原因)
    """
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error"] + args,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        out = proc.stdout.decode(errors="ignore").strip()
        err = proc.stderr.decode(errors="ignore").strip()
        if proc.returncode != 0:
            return "", f"ffprobe退出码{proc.returncode}: {err or '(无stderr)'}"
        return out, None
    except FileNotFoundError:
        return "", "找不到 ffprobe 命令 (未安装或不在 PATH)"
    except Exception as e:
        return "", f"调用异常: {repr(e)}"


def get_frame_count(path, idx, col, errors):
    """获取帧数;失败返回 None,并把真实原因打印+记录"""
    # 方法1: 读容器 nb_frames
    out, err = _ffprobe([
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    if DEBUG:
        print(f"[debug probe1] {col}/{idx} -> out='{out}' err='{err}'", flush=True)
    if out.isdigit() and int(out) > 0:
        return int(out)

    # 方法2: 实际数包
    out2, err2 = _ffprobe([
        "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ])
    if DEBUG:
        print(f"[debug probe2] {col}/{idx} -> out='{out2}' err='{err2}'", flush=True)
    if out2.isdigit() and int(out2) > 0:
        return int(out2)

    # 两种都失败,带出真实原因
    reason = err or err2 or f"ffprobe返回非数字: nb_frames='{out}' packets='{out2}'"
    log_err(f"[probe] index={idx} col={col} path={path} :: {reason}", errors)
    return None


def crop_video(src, dst, n_frames, use_gpu):
    """裁前 n_frames 帧。返回 (成功?, 错误信息)"""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-frames:v", str(n_frames)]
    if use_gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "17", "-preset", "medium"]
    cmd += ["-pix_fmt", "yuv420p", dst]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            return False, proc.stderr.decode(errors="ignore")[:500]
        return True, ""
    except FileNotFoundError:
        return False, "找不到 ffmpeg 命令 (未安装或不在 PATH)"
    except Exception as e:
        return False, repr(e)


def worker(wid, tasks, gpu_id, counter, results, errors, use_gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for task in tasks:
        idx = task["index"]
        videos = task["videos"]
        try:
            counts = {}
            for col, path in videos.items():
                c = get_frame_count(path, idx, col, errors)
                if c:
                    counts[col] = c

            if not counts:
                log_err(f"[skip]  index={idx} 该行无可用视频,整行跳过", errors)
                continue

            min_frames = min(counts.values())

            for col in counts:
                src = videos[col]
                fname = idx if idx.endswith(".mp4") else idx + ".mp4"
                dst = os.path.join(OUTPUT_BASE, col, fname)
                ok, msg = crop_video(src, dst, min_frames, use_gpu)
                if ok:
                    results[f"{idx}\t{col}"] = dst
                else:
                    log_err(f"[crop]  index={idx} col={col} src={src} :: {msg}", errors)

        except Exception as e:
            log_err(f"[error] index={idx} {repr(e)}", errors)
        finally:
            with counter.get_lock():
                counter.value += 1


def self_check():
    """启动自检:ffprobe/ffmpeg 是否存在;输入CSV是否存在"""
    print("=" * 60)
    print("启动自检")
    print("=" * 60)

    ok = True
    for tool in ("ffprobe", "ffmpeg"):
        path = shutil.which(tool)
        if path:
            try:
                ver = subprocess.run([tool, "-version"],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE).stdout.decode().splitlines()[0]
            except Exception:
                ver = "(版本获取失败)"
            print(f"  ✅ {tool}: {path}  | {ver}")
        else:
            print(f"  ❌ {tool}: 未找到!请确认已安装并在 PATH 中。")
            ok = False

    if not os.path.exists(INPUT_CSV):
        print(f"  ❌ 输入CSV不存在: {INPUT_CSV}")
        ok = False
    else:
        print(f"  ✅ 输入CSV: {INPUT_CSV}")

    print("=" * 60 + "\n")
    if not ok:
        print("自检未通过,已退出。请先解决上面标 ❌ 的问题。")
        sys.exit(1)


def main():
    self_check()

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # 构建任务,同时统计 CSV 里记录但文件实际不存在的情况
    tasks = []
    missing_cnt = 0
    for row in rows:
        idx = row["index"]
        videos = {}
        for col in COLUMNS:
            p = (row.get(col) or "").strip()
            if not p:
                continue
            if os.path.exists(p):
                videos[col] = p
            else:
                missing_cnt += 1
                print(f"[missing] index={idx} col={col} 文件不存在: {p}", flush=True)
        tasks.append({"index": idx, "videos": videos})

    total = len(tasks)
    print(f"\n共 {total} 个 index 待处理,{NUM_WORKERS} 进程并行,"
          f"GPU编码={'开' if USE_GPU else '关'},DEBUG={'开' if DEBUG else '关'}")
    if missing_cnt:
        print(f"(注意: 有 {missing_cnt} 个 CSV 中记录的路径实际不存在)\n")

    for col in COLUMNS:
        os.makedirs(os.path.join(OUTPUT_BASE, col), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    manager = Manager()
    results = manager.dict()
    errors  = manager.list()
    counter = Value("i", 0)

    chunks = [tasks[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    procs = []
    for wid in range(NUM_WORKERS):
        p = Process(target=worker,
                    args=(wid, chunks[wid], wid, counter, results, errors, USE_GPU))
        p.start()
        procs.append(p)

    start = time.time()
    while any(p.is_alive() for p in procs):
        done = counter.value
        elapsed = time.time() - start
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        # 进度行用 stderr,避免和上面的错误 print 抢同一行
        sys.stderr.write(
            f"\r进度 {done}/{total} ({done * 100 // max(total,1)}%)  "
            f"已用 {elapsed:.0f}s 剩余~{eta:.0f}s 失败 {len(errors)}   ")
        sys.stderr.flush()
        time.sleep(0.5)

    for p in procs:
        p.join()
    sys.stderr.write(f"\r进度 {counter.value}/{total} (100%) 全部结束        \n")

    err_list = list(errors)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"# 失败记录 {time.strftime('%Y-%m-%d %H:%M:%S')}  共 {len(err_list)} 条\n\n")
        for e in err_list:
            f.write(e + "\n")

    new_fields = list(fieldnames)
    for col in COLUMNS:
        if f"{col}_cut" not in new_fields:
            new_fields.append(f"{col}_cut")

    res = dict(results)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in rows:
            idx = row["index"]
            for col in COLUMNS:
                row[f"{col}_cut"] = res.get(f"{idx}\t{col}", "")
            writer.writerow(row)

    per_col = {c: 0 for c in COLUMNS}
    for key in res:
        per_col[key.split("\t", 1)[1]] += 1

    print("\n" + "=" * 50)
    print("📊 处理统计")
    print("=" * 50)
    print(f"  总 index 数      : {total}")
    print(f"  成功裁剪视频总数 : {len(res)}")
    print(f"  失败/跳过条目数  : {len(err_list)}")
    for col in COLUMNS:
        print(f"    {col:<10} 成功: {per_col[col]}")
    print(f"  总耗时           : {time.time() - start:.1f}s")
    print("=" * 50)
    print(f"✅ 新 CSV : {OUTPUT_CSV}")
    print(f"📝 日志   : {LOG_FILE}")
    if err_list:
        print(f"⚠️  有 {len(err_list)} 条失败,详见 log.txt / 上方终端输出")


if __name__ == "__main__":
    main()