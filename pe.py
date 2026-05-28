import torch
import os
import pandas as pd

os.environ["HF_HOME"] = "/m2v_intern/mengzijie/m2v_camclone_v2"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

CAM_MODEL_ID     = "chancharikm/qwen2.5-vl-7b-cam-motion"
CONTENT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
FPS = 8.0

# ---------- 公共 processor ----------
processor = AutoProcessor.from_pretrained(CONTENT_MODEL_ID)

# ---------- 模型 1：运镜 ----------
cam_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CAM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},          # 运镜模型固定在 GPU 0
).eval()

# ---------- 模型 2：场景内容 ----------
content_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CONTENT_MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 1},          # 场景内容模型固定在 GPU 1；如果只有单卡请改成 {"": 0}
).eval()


# ---------- 通用推理函数 ----------
@torch.inference_mode()
def run_qwen_vl(model, video_path: str, question: str, max_new_tokens: int = 384) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{video_path}", "fps": FPS},
            {"type": "text",  "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ---------- 生成内容描述（场景+主体+背景） ----------
def generate_content_description(video_path):
    # 使用强格式化Prompt逼迫模型模仿目标风格
    q = (
        "Describe the visual content of this video in detail. Do NOT describe camera movement. "
        "Write a single continuous paragraph following this exact structure and phrasing style: "
        "1. Overall summary: Start with 'The video depicts...' or 'The video features...' and summarize the scene and mood. "
        "2. Subject details: Start with 'The main subject is...' or 'The main subjects are...'. Describe their appearance, clothing, expressions, and specific movements. "
        "3. Background/Setting: Start with 'The background...' or 'The setting is...'. Detail the environment, weather, and lighting."
    )
    # 给定稍长一点的 max_new_tokens 以免长描述被截断
    return run_qwen_vl(content_model, video_path, q, max_new_tokens=400)


# ---------- 生成运镜与画幅描述 ----------
def generate_camera_description(video_path):
    q = (
        "Analyze the cinematography of this video. "
        "Write 2 to 3 sentences starting exactly with 'The camera is...'. "
        "Describe whether the camera is stationary or moving (e.g., panning, tilting, tracking). "
        "Include the shot scale (e.g., wide shot, medium close-up, close-up) and focus/depth of field (e.g., blurred background, sharp focus)."
    )
    return run_qwen_vl(cam_model, video_path, q, max_new_tokens=128)


# ---------- 组装成仿样例的单段落 Caption ----------
def generate_full_caption(video_path: str) -> str:
    content_desc = generate_content_description(video_path)
    camera_desc = generate_camera_description(video_path)
    
    # 拼接并清理多余的空格，形成一个完整的长段落
    full_text = f"{content_desc} {camera_desc}"
    full_text = " ".join(full_text.split())
    return full_text


if __name__ == "__main__":
    # 示例运行单条测试
    test_video = "/m2v_intern/public_datasets/Camera_Dataset/Testset/Data/all_videos_0511/00202.mp4"
    if os.path.exists(test_video):
        caption = generate_full_caption(test_video)
        print("\n=== Generated PE Caption ===\n")
        print(caption)
        print("\n============================\n")

    """
    # ====== 附加：批量处理脚本示例 ======
    # 如果你需要批量跑 CSV 并输出相同的格式，可以解除以下注释使用：
    
    input_csv = "input.csv"   # 包含 ref_video_path, video_path 的输入文件
    output_csv = "output.csv"
    
    df = pd.read_csv(input_csv)
    captions = []
    
    for idx, row in df.iterrows():
        vid_path = row['video_path']
        print(f"Processing {idx+1}/{len(df)}: {vid_path}")
        try:
            cap = generate_full_caption(vid_path)
        except Exception as e:
            print(f"Error processing {vid_path}: {e}")
            cap = ""
        captions.append(cap)
        
    df['caption'] = captions
    # 保证包含 ref_video_path, video_path, caption 列
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Batch processing completed. Saved to {output_csv}")
    """