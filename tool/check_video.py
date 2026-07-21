import subprocess
import json
import os
import sys


def get_video_info(video_path):
    if not os.path.exists(video_path):
        print(f"错误：文件不存在 -> {video_path}")
        return

    # 调用 ffprobe 获取 JSON 格式的完整信息
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("错误：未找到 ffprobe，请先安装 ffmpeg")
        return
    except subprocess.CalledProcessError:
        print(f"错误：ffprobe 解析失败 -> {video_path}")
        return

    info = json.loads(result.stdout)

    # 找到视频流
    video_stream = next(
        (s for s in info["streams"] if s["codec_type"] == "video"), None
    )
    if video_stream is None:
        print("错误：未找到视频流")
        return

    fmt = info["format"]

    width = video_stream.get("width")
    height = video_stream.get("height")

    # FPS（形如 "25/1"，需要计算）
    fps_str = video_stream.get("avg_frame_rate", "0/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0

    duration = float(fmt.get("duration", 0))
    frame_count = video_stream.get("nb_frames", "N/A")
    # 部分视频无 nb_frames，则用 fps * duration 估算
    if frame_count == "N/A" and fps > 0:
        frame_count = round(fps * duration)

    codec = video_stream.get("codec_name", "N/A")
    bit_rate = fmt.get("bit_rate")
    bit_rate_str = f"{int(bit_rate) / 1000:.1f} kbps" if bit_rate else "N/A"

    size_bytes = int(fmt.get("size", os.path.getsize(video_path)))
    size_mb = size_bytes / (1024 * 1024)

    h = int(duration // 3600)
    m = int((duration % 3600) // 60)
    s = duration % 60

    print("=" * 50)
    print(f"文件路径 : {video_path}")
    print(f"文件大小 : {size_mb:.2f} MB ({size_bytes} 字节)")
    print(f"分辨率   : {width} x {height}")
    print(f"帧率 FPS : {fps:.2f}")
    print(f"总帧数   : {frame_count}")
    print(f"时长     : {duration:.2f} 秒  ({h:02d}:{m:02d}:{s:05.2f})")
    print(f"编码格式 : {codec}")
    print(f"码率     : {bit_rate_str}")
    print("=" * 50)


if __name__ == "__main__":
    # path
    path = "/ytech_m2v4_hdd/mengzijie/recam/content/16_reCam_human2_00425_背面远景.mp4"
    get_video_info(path)
