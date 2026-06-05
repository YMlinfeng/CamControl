"""
文件用途：
    遍历各模型输出文件夹，按 index 拼成 2x3 带标签网格。
    布局：
       第一行：[Reference] [sd2.0]   [camclone]
       第二行：[Reference] [ltx]     [ours]

    关键策略：
    - 以 ours 视频的高度 H 为统一基准，所有视频(含 ref)缩放到高度 = H，
      宽度按各自宽高比自动决定（不变形、不裁剪、不补黑边到固定盒子）。
      => ref 可以比生成视频宽很多或窄很多，都没关系，反正“高”严格对齐。
    - 缩放用显式宽度 scale=Wi:H（Wi 由 Python 按宽高比算定并取偶），
      不用 -2，避免 ffmpeg 自行取整与 pad 目标错位（之前报错的根因）。
    - hstack 每行三个视频（高相同 -> 可直接横拼）。
    - 由于 ref 宽度差异，上下两行总宽可能不同：
      取两行最大宽度 max_w，对较窄的行 pad 黑边居中补齐，再 vstack。
    - 缺失视频用黑底视频补位（高=H，宽默认取 ours 宽度）。
    - 多线程并行拼接。
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 配置区域 ---
CONCAT_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25/concat"
CAM_DIR    = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25/camclone"
LTX_DIR    = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25/ltx"
OURS_DIR   = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25/ours"
SD2_DIR    = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25/sd2.0"
REF_DIR    = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25/ref"

NUM_WORKERS = 16      # 并行进程数，按机器调整
# ----------------


def get_video_resolution(video_path):
    """获取视频确切的宽高"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", video_path,
    ]
    try:
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        w, h = map(int, output.split("x"))
        return w, h
    except Exception:
        return None, None


def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ]
    try:
        return float(subprocess.check_output(cmd).decode("utf-8").strip())
    except Exception:
        return None


def get_video_fps(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0", video_path,
    ]
    try:
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        num, den = output.split("/")
        den = float(den)
        if den == 0:
            return None
        return float(num) / den
    except Exception:
        return None


def even(x):
    """四舍五入到最近的偶数（libx264 要求宽高为偶数），最小为 2"""
    v = int(round(x / 2.0)) * 2
    return max(2, v)


def build_index_list():
    indices = []
    for fn in os.listdir(REF_DIR):
        if fn.endswith(".mp4"):
            indices.append(os.path.splitext(fn)[0])
    return sorted(indices)


def process_one(vid_index):
    """
    返回 (status, msg)
    所有视频高度统一为 H(ours 高度)，宽度按宽高比自适应。
    上下两行宽度不同则对窄行 pad 黑边补到 max_w，再 vstack。
    """
    # 路径顺序: REF, SD2, CAM, LTX, OURS（与下方 labels 一一对应）
    paths = [
        os.path.join(REF_DIR, f"{vid_index}.mp4"),
        os.path.join(SD2_DIR, f"{vid_index}.mp4"),
        os.path.join(CAM_DIR, f"{vid_index}.mp4"),
        os.path.join(LTX_DIR, f"{vid_index}.mp4"),
        os.path.join(OURS_DIR, f"{vid_index}.mp4"),
    ]
    exist_flags = [os.path.exists(p) for p in paths]

    if not any(exist_flags):
        return ("error", f"[{vid_index}] 所有视频均缺失")

    # 基准高度 H：优先 ours(index 4)，缺失则取任意存在视频
    if exist_flags[4]:
        base_path = paths[4]
    else:
        base_path = next(p for p, e in zip(paths, exist_flags) if e)

    bw, bh = get_video_resolution(base_path)
    if bw is None or bh is None:
        return ("error", f"[{vid_index}] 无法读取基准分辨率: {base_path}")

    H = even(bh)
    if H == 0:
        return ("error", f"[{vid_index}] 基准高度无效: {base_path}")

    # ours 宽度（用于缺失视频的默认宽度），缺失则用基准等比宽
    ours_w_default = even(bw * H / bh)

    # 黑图补位所需 时长/帧率
    duration, fps = None, None
    for p, e in zip(paths, exist_flags):
        if e:
            if duration is None:
                duration = get_video_duration(p)
            if fps is None:
                fps = get_video_fps(p)
            if duration is not None and fps is not None:
                break
    if duration is None:
        duration = 5.0
    if fps is None:
        fps = 25.0

    # 计算每路缩放后的宽度 Wi（高统一为 H，宽按宽高比；缺失则默认 ours 宽）
    widths = []
    for p, e in zip(paths, exist_flags):
        if e:
            ww, hh = get_video_resolution(p)
            if ww is None or hh is None or hh == 0:
                widths.append(ours_w_default)
            else:
                widths.append(even(ww * H / hh))
        else:
            widths.append(ours_w_default)

    w_ref, w_sd2, w_cam, w_ltx, w_ours = widths

    top_width = w_ref + w_sd2 + w_cam        # 第一行总宽
    bottom_width = w_ref + w_ltx + w_ours    # 第二行总宽
    max_w = even(max(top_width, bottom_width))

    missing = [p for p, e in zip(paths, exist_flags) if not e]
    out_path = os.path.join(CONCAT_DIR, f"{vid_index}_grid_v5.mp4")

    cmd = ["ffmpeg", "-y", "-v", "error"]
    for p, e, wi in zip(paths, exist_flags, widths):
        if e:
            cmd.extend(["-i", p])
        else:
            # 缺失 -> 黑底视频补位（宽 wi，高 H）
            cmd.extend([
                "-f", "lavfi",
                "-i", f"color=c=black:s={wi}x{H}:r={fps}:d={duration}",
            ])

    f_size = max(18, H // 14)
    draw_text_base = (
        f"drawtext=fontcolor=white:fontsize={f_size}:x=15:y=15:"
        f"box=1:boxcolor=black@0.5:text="
    )

    filter_complex = ""

    # input 0 (Reference)：缩放到 w_ref x H，加标签，split 给上下两行
    filter_complex += (
        f"[0:v]scale={w_ref}:{H},setpts=PTS-STARTPTS,"
        f"{draw_text_base}'Reference',split=2[ref_t][ref_b];"
    )

    # input 1~4：缩放到各自 Wi x H，加标签
    labels = ['sd2.0', 'camclone', 'ltx', 'ours']
    for i in range(1, 5):
        filter_complex += (
            f"[{i}:v]scale={widths[i]}:{H},setpts=PTS-STARTPTS,"
            f"{draw_text_base}'{labels[i - 1]}'[v{i}];"
        )

    # 每行 hstack（高相同 -> 可直接横拼）
    filter_complex += "[ref_t][v1][v2]hstack=inputs=3[top];"
    filter_complex += "[ref_b][v3][v4]hstack=inputs=3[bottom];"

    # 两行宽度可能不同 -> pad 到 max_w（居中），再 vstack
    filter_complex += (
        f"[top]pad={max_w}:{H}:(ow-iw)/2:0:black[topp];"
    )
    filter_complex += (
        f"[bottom]pad={max_w}:{H}:(ow-iw)/2:0:black[bottomp];"
    )
    filter_complex += "[topp][bottomp]vstack=inputs=2,format=yuv420p[v]"

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        out_path,
    ])

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        return ("error", f"[{vid_index}] 拼接失败: {err}")

    if missing:
        return ("success", f"[{vid_index}] OK(含黑图补位 {missing}) -> {out_path}")
    return ("success", f"[{vid_index}] OK -> {out_path}")


def main():
    os.makedirs(CONCAT_DIR, exist_ok=True)

    indices = build_index_list()
    print(f"共 {len(indices)} 个 index，使用 {NUM_WORKERS} 个并行进程 ...")

    stats = {"success": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_one, idx): idx for idx in indices}
        for future in tqdm(as_completed(futures), total=len(futures), desc="🎥 拼接进度"):
            status, msg = future.result()
            stats[status] += 1
            if status != "success":
                tqdm.write(f"❌ {msg}")

    print("\n" + "=" * 50)
    print("✅ 2x3 网格拼接处理完毕")
    print(f"   成功: {stats['success']} | 错误: {stats['error']}")
    print("=" * 50)


if __name__ == "__main__":
    main()