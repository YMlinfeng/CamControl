"""
文件用途：
    1. 使用 decord 高速读取视频，使用 imageio (FFMPEG) 质量写出视频。
    2. 匹配 o7.csv 中的参考视频与 ours 目录下的视频。
    3. 逻辑：
        - 在每一帧的左上角打印该视频的帧数(n)和帧率(fps)。
        - 将处理后的两个视频左右拼接。
    4. 结果保存至 /m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/test 下。
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import imageio
from decord import VideoReader, cpu
from tqdm import tqdm

def extract_ref_video_path(json_str):
    """解析 CSV 单元格中的 JSON 提取路径"""
    if pd.isna(json_str) or not str(json_str).strip():
        return ""
    try:
        data = json.loads(str(json_str))
        for item in data:
            if item.get("type") == "GUIDE_VIDEO":
                return str(item.get("value", ""))
    except Exception:
        pass
    return ""

def draw_metadata(frame, text):
    """在帧的左上角绘制带阴影的文字元数据"""
    # frame 是 RGB 格式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255) # 白色
    pos = (20, 40)
    # 绘制黑色描边/阴影增加可读性
    cv2.putText(frame, text, pos, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, font, font_scale, color, thickness)
    return frame

def center_crop_and_resize(ref_frame, target_w, target_h):
    """
    对 ref_frame 进行中心裁剪以匹配 target 的长宽比，并缩放到 target 的尺寸。
    """
    h_ref, w_ref = ref_frame.shape[:2]
    target_aspect = target_w / target_h
    ref_aspect = w_ref / h_ref

    if ref_aspect > target_aspect:
        # Ref 太宽，裁左右
        new_w = int(h_ref * target_aspect)
        start_w = (w_ref - new_w) // 2
        cropped = ref_frame[:, start_w : start_w + new_w]
    else:
        # Ref 太高，裁上下
        new_h = int(w_ref / target_aspect)
        start_h = (h_ref - new_h) // 2
        cropped = ref_frame[start_h : start_h + new_h, :]

    # 缩放到目标尺寸
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized

def save_video_from_frames(video_array, output_path, fps):
    """按照要求的 imageio 方式保存视频"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with imageio.get_writer(
        output_path, 
        fps=fps, 
        format="FFMPEG", 
        codec="libx264", 
        ffmpeg_params=["-crf", "12"], 
        pixelformat="yuv420p"
    ) as writer:
        for i in range(video_array.shape[0]):
            writer.append_data(video_array[i])

def process_and_concat():
    # --- 配置 ---
    INPUT_CSV = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o7.csv"
    OURS_DIR = "/ytech_milm_disk2/lishujuan/motion-test/OmniDirector/eval_dataset_200/ours"
    SAVE_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/output_25/ours"
    # ------------

    df = pd.read_csv(INPUT_CSV)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🎥 拼接进度"):
        vid_index = str(row.get("index", ""))
        ref_path = extract_ref_video_path(row.get("ref_videos", ""))
        ours_path = os.path.join(OURS_DIR, f"{vid_index}.mp4")

        if not os.path.exists(ref_path) or not os.path.exists(ours_path):
            continue

        if True:
            # 1. 使用 decord 读取视频
            vr_ref = VideoReader(ref_path, ctx=cpu(0))
            vr_ours = VideoReader(ours_path, ctx=cpu(0))
            
            n_ref, n_ours = len(vr_ref), len(vr_ours)
            fps_ref = vr_ref.get_avg_fps() or 30.0
            fps_ours = vr_ours.get_avg_fps() or 30.0

            if n_ours > n_ref:
                index_list = np.linspace(0, n_ours - 1, n_ref, dtype=int).tolist()
            else:
                index_list = list(range(n_ours))
                
            n4 = int(round(len(index_list) / fps_ref * 25.0))
            index_list_new = np.linspace(0, index_list[-1], int(n4), dtype=int).tolist()
            # 统一处理帧数（取交集防止溢出）
            # min_frames = min(n_ref, n_ours)
            
            # 批量获取原始帧
            # frames_ref = vr_ref.get_batch(range(len(index_list))).asnumpy()
            frames_ours = vr_ours.get_batch(index_list_new).asnumpy()
            
            # 3. 保存拼接视频
            # video_array = np.array(frames_ours)
            output_name = os.path.join(SAVE_DIR, f"{vid_index}.mp4")
            save_video_from_frames(frames_ours, output_name, 25.0)

       

if __name__ == "__main__":
    process_and_concat()