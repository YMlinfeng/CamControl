import os
import re
import csv
import json
import decord
import numpy as np
import pandas as pd
import imageio
from decord import VideoReader, cpu


CSV_PATH = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o7.csv"
# 重采样后视频的保存目录
SAVE_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/output/resampled_videos"
# 保存视频的帧率（重采样目标 fps）
SAVE_FPS = 15.0


class FrameSampler:
    def __init__(self):
        self.sample_fps = 15.0
        self.sample_type = "fps"
        self.max_num_frames = 77
        self.max_fps = 60.0  # read_video 中用到的 self.max_fps，按需修改

    def get_frame_indexes(self, path):
        """精简后的 read_video，只负责算出 frame_indexes"""
        reader = decord.VideoReader(path, ctx=decord.cpu(0))
        length = len(reader)
        # length = min(77, length)
        # 取平均帧率和最大帧率之间的小值
        ori_fps = reader.get_avg_fps()
        fps = min(reader.get_avg_fps(), self.max_fps)
        num_frames = 77
        # nums_frames = n

        # 整个视频长度对应的 stride
        frame_stride_full = max(1, round((length - 1) / (num_frames - 1)))

        # fps 是视频自身的 fps, sample_fps = 15
        frame_stride = max(1, round(fps / self.sample_fps))
        print(ori_fps, fps, path, length, frame_stride_full, frame_stride)
        frame_stride = min(frame_stride, frame_stride_full)

        num_frames = min(num_frames, self.max_num_frames)

        start_frame = 0
        if num_frames == 1:
            frame_indexes = [start_frame]
        else:
            frame_indexes = range(
                start_frame, start_frame + num_frames * frame_stride, frame_stride
            )
            frame_indexes = [min(idx, length - 1) for idx in frame_indexes]

        n = frame_indexes[-1]
        fps_new = ori_fps / n * 77
        n_new = int(round(77 / fps_new * 15))
        index_new = np.linspace(0, 76, n_new, dtype=int)
        print(f"frame_index:{frame_indexes}")
        print(f"index_new:{index_new}, {len(index_new)}")
        index_new[index_new >= 77] = 76

        # print(fps, path, length, frame_stride_full, frame_stride)
        return list(frame_indexes), list(index_new)


def parse_mp4_path(cell):
    """
    从 ref_videos 单元格中解析出 mp4 路径。
    兼容以下几种形式：
      "/path/to/xxx.mp4"
      [{"id": 1, "type": "GUIDE_VIDEO", "value": "/path/to/xxx.mp4"}]
      /path/to/xxx.mp4
    """
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None

    # 1) 优先尝试当作 JSON 解析（形如 [{"value": "...mp4"}]）
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

    # 2) 兜底：用正则直接抓取以 / 开头、以 .mp4 结尾的路径
    m = re.search(r"(/[^\"'\[\]\s]+\.mp4)", text)
    if m:
        return m.group(1)

    return None


def save_video_from_frames(video_array, output_path, fps):
    frame_count = video_array.shape[0]

    # 1. Create the directory first
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 2. Write directly to the output_path instead of using BytesIO.
    # We explicitly set format="FFMPEG" to ensure the correct plugin is used.
    with imageio.get_writer(
        output_path,
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        ffmpeg_params=["-crf", "12"],
        pixelformat="yuv420p",
    ) as writer:
        for i in range(frame_count):
            writer.append_data(video_array[i])

    print(f"Video successfully saved to {output_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    assert "ref_videos" in df.columns, "CSV 中找不到 ref_videos 列"
    assert "camclone_gen" in df.columns, "CSV 中找不到 camclone_gen 列"

    os.makedirs(SAVE_DIR, exist_ok=True)

    sampler = FrameSampler()
    results = []

    for i, row in df.iterrows():
        ref_path = parse_mp4_path(row["ref_videos"])
        gen_path = str(row["camclone_gen"]).strip()

        frame_indexes, index_new = sampler.get_frame_indexes(ref_path)
        vr = VideoReader(gen_path, ctx=cpu(0))
        frames = vr.get_batch(index_new).asnumpy()

        # 保存重采样后的视频
        base_name = os.path.splitext(os.path.basename(gen_path))[0]
        save_path = os.path.join(SAVE_DIR, f"{base_name}_resampled.mp4")
        save_video_from_frames(frames, save_path, SAVE_FPS)

        results.append(
            {
                "ref_video": ref_path,
                "camclone_gen": gen_path,
                "resampled_video": save_path,
                "frame_indexes": json.dumps(frame_indexes),
                "index_new": json.dumps([int(x) for x in index_new]),
            }
        )
        print(f"[{i}] {ref_path} -> {len(frame_indexes)} frames: {frame_indexes}")
        print(f"[{i}] 保存重采样视频 -> {save_path} ({len(frames)} frames)")
        print("done")

    # 保存结果
    out_path = os.path.join(os.path.dirname(CSV_PATH), "frame_indexes.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ref_video",
                "camclone_gen",
                "resampled_video",
                "frame_indexes",
                "index_new",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n完成，共处理 {len(results)} 条，结果已保存到: {out_path}")


if __name__ == "__main__":
    main()