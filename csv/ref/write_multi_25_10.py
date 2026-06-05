import os
import re
import json
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


CSV_PATH = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o7.csv"
# ref 视频保存目录
DST_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25/ref"
TARGET_FPS = 25.0
MAX_DURATION = 10        # 秒，只保留前十秒
NUM_WORKERS = 32       # 并行进程数，按机器 CPU 核数调整


def parse_mp4_path(cell):
    """
    从 ref_videos 单元格中解析出 mp4 路径。
    兼容：
      [{"id": 1, "type": "GUIDE_VIDEO", "value": "/path/to/xxx.mp4"}]
      "/path/to/xxx.mp4"
      /path/to/xxx.mp4
    """
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "value" in item:
                    if str(item["value"]).endswith(".mp4"):
                        return item["value"]
        elif isinstance(obj, dict) and "value" in obj:
            return obj["value"]
        elif isinstance(obj, str) and obj.endswith(".mp4"):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass

    m = re.search(r"(/[^\"'\[\]\s]+\.mp4)", text)
    if m:
        return m.group(1)

    return None


def convert_to_fps(src_path, dst_path, fps, max_duration):
    """用 ffmpeg 把视频重编码到指定 fps，并只保留前 max_duration 秒"""
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-r", str(fps),
        "-t", str(max_duration),
        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        dst_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        cmd_fallback = [
            "ffmpeg", "-y", "-i", src_path,
            "-r", str(fps),
            "-t", str(max_duration),
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            dst_path,
        ]
        result = subprocess.run(
            cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))


def process_one(task):
    """处理单个任务，返回 (状态, 信息)"""
    i, ref_path, dst_path = task
    try:
        convert_to_fps(ref_path, dst_path, TARGET_FPS, MAX_DURATION)
        return ("ok", f"[{i}] OK  {ref_path} -> {dst_path}")
    except Exception as e:
        return ("fail", f"[{i}] FAIL {ref_path}: {e}")


def main():
    df = pd.read_csv(CSV_PATH)

    assert "ref_videos" in df.columns, "CSV 中找不到 ref_videos 列"
    assert "index" in df.columns, "CSV 中找不到 index 列"

    os.makedirs(DST_DIR, exist_ok=True)

    # 先收集所有有效任务
    tasks = []
    skip = 0
    for i, row in df.iterrows():
        ref_path = parse_mp4_path(row["ref_videos"])
        idx_name = str(row["index"]).strip()

        if not ref_path or not os.path.exists(ref_path):
            print(f"[{i}] [skip] ref_video 无效: {ref_path}")
            skip += 1
            continue
        if not idx_name:
            print(f"[{i}] [skip] index 为空")
            skip += 1
            continue

        dst_path = os.path.join(DST_DIR, f"{idx_name}.mp4")
        tasks.append((i, ref_path, dst_path))

    print(f"共 {len(tasks)} 个任务，使用 {NUM_WORKERS} 个并行进程 ...")

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_one, t) for t in tasks]
        for future in as_completed(futures):
            status, msg = future.result()
            print(msg)
            if status == "ok":
                ok += 1
            else:
                fail += 1

    print(
        f"\n完成：成功 {ok} 个，失败 {fail} 个，跳过 {skip} 个，输出目录: {DST_DIR}"
    )


if __name__ == "__main__":
    main()