"""
文件用途：
    1. 扫描 /m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/test 目录下的所有 mp4 文件。
    2. 根据文件名（index）在 o7.csv 中查找对应的参考视频（ref_videos）。
    3. 逻辑：
        - 缩放 Ref 视频，使其高度与 Test 视频一致（保持比例）。
        - 在每一帧左上角打印各自视频的原始帧数和帧率。
        - 左右横向拼接并保存。
    4. 结果保存至 /m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/test_final。
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255) 
    pos = (20, 40)
    cv2.putText(frame, text, pos, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, font, font_scale, color, thickness)
    return frame

def resize_ref_to_target_height(ref_frame, target_h):
    """调整 Ref 高度与目标一致，保持长宽比"""
    h_ref, w_ref = ref_frame.shape[:2]
    if h_ref == target_h:
        return ref_frame
    scale = target_h / h_ref
    new_w = int(w_ref * scale)
    return cv2.resize(ref_frame, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)

def save_video_from_frames(video_array, output_path, fps):
    """使用 imageio 保存视频"""
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

def process_from_dir():
    # --- 配置 ---
    INPUT_CSV = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o7.csv"
    # 现在从 test 目录读取视频
    SOURCE_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/test"
    # 保存到新的目录，避免递归读取
    SAVE_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/test_final"
    # ------------

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到 CSV 文件: {INPUT_CSV}")
        return

    # 预加载 CSV 建立 索引->参考视频 的映射，提高查询速度
    df = pd.read_csv(INPUT_CSV)
    ref_map = {}
    for _, row in df.iterrows():
        idx_val = str(row.get("index", ""))
        if idx_val:
            ref_map[idx_val] = extract_ref_video_path(row.get("ref_videos", ""))

    # 获取目录下所有视频文件
    video_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')]
    if not video_files:
        print(f"空目录: {SOURCE_DIR}")
        return

    print(f"🚀 开始处理 {len(video_files)} 个视频文件...")

    for filename in tqdm(video_files, desc="🎥 处理中"):
        vid_index = filename.replace(".mp4", "")
        test_path = os.path.join(SOURCE_DIR, filename)
        
        # 反查参考视频
        ref_path = ref_map.get(vid_index, "")

        if not ref_path or not os.path.exists(ref_path):
            # tqdm.write(f"⚠️ 跳过 {filename}: 未在 CSV 中找到参考视频或文件不存在")
            continue

        try:
            # 1. 读取视频
            vr_ref = VideoReader(ref_path, ctx=cpu(0))
            vr_test = VideoReader(test_path, ctx=cpu(0))
            
            n_ref, n_test = len(vr_ref), len(vr_test)
            fps_ref = vr_ref.get_avg_fps() or 30.0
            fps_test = vr_test.get_avg_fps() or 30.0

            # 2. 帧数对齐逻辑 (沿用之前的逻辑)
            if n_test > n_ref:
                # 对测试视频进行重采样对齐原视频长度
                index_list = np.linspace(0, n_test - 1, n_ref, dtype=int).tolist()
                ref_indices = list(range(n_ref))
                final_fps = fps_ref
            else:
                # 截断原视频对齐测试视频长度
                index_list = list(range(n_test))
                ref_indices = list(range(n_test))
                final_fps = fps_test
            
            frames_ref = vr_ref.get_batch(ref_indices).asnumpy()
            frames_test = vr_test.get_batch(index_list).asnumpy()
            
            h_test = frames_test.shape[1]
            final_frames = []
            
            # 3. 逐帧处理
            for i in range(len(index_list)):
                f_ref = frames_ref[i]
                f_test = frames_test[i]
                
                # 缩放 Ref 适配 Test 高度
                f_ref_resized = resize_ref_to_target_height(f_ref, h_test)
                
                # 水印标签 (注意：这里标签改为 Test 以适应目录名)
                f_ref_processed = draw_metadata(f_ref_resized, f"Ref: {n_ref}f / {fps_ref:.1f}fps")
                f_test_processed = draw_metadata(f_test, f"Test: {n_test}f / {fps_test:.1f}fps")
                
                # 横向拼接
                combined = np.hstack((f_ref_processed, f_test_processed))
                final_frames.append(combined)
            
            # 4. 保存
            video_array = np.array(final_frames)
            output_name = os.path.join(SAVE_DIR, f"{vid_index}_final_concat.mp4")
            save_video_from_frames(video_array, output_name, final_fps)

        except Exception as e:
            tqdm.write(f"❌ 处理 {filename} 失败: {e}")

if __name__ == "__main__":
    process_from_dir()