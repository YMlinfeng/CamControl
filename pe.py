import torch
import os
import pandas as pd
import csv
from tqdm import tqdm

# 设置缓存目录
os.environ["HF_HOME"] = "/m2v_intern/mengzijie/m2v_camclone_v2"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

CAM_MODEL_ID     = "chancharikm/qwen2.5-vl-7b-cam-motion"
CONTENT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
FPS = 8.0

print("Loading Processor...")
processor = AutoProcessor.from_pretrained(CONTENT_MODEL_ID)

# A100 80G 显存非常充裕（两个 7B Bf16 占~28G），直接全部加载到显卡 0
print("Loading Camera Motion Model to GPU 0...")
cam_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CAM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},  # 单卡 A100 固定在 GPU 0
).eval()

print("Loading Scene Content Model to GPU 0...")
content_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CONTENT_MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},  # 同样放在 GPU 0
).eval()
print("All Models loaded successfully on a single A100 80G!\n")


@torch.inference_mode()
def run_qwen_vl(model, video_path: str, question: str, max_new_tokens: int = 512) -> str:
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


def generate_content_description(video_path):
    # 【Few-Shot 提示法】提供你 CSV 中的真实标注作为例子，限制其只生成场景、主体、背景
    few_shot_prompt = """Describe the visual content of this video in detail (excluding camera movement). Please strictly follow the writing style, structure, and detail level of the examples below.

Example 1:
The video depicts a tense and dramatic scene inside a vintage car, featuring two men. One man, dressed in a white shirt and dark suit, is seated on the left side of the car, while the other man, wearing a red shirt with a blue collar and a dark suit, is seated on the right. The main subjects are two men. The man on the left is wearing a white shirt and a dark suit, and he is seated calmly, observing the situation. The man on the right is wearing a red shirt and is visibly distressed. The background is blurred, indicating that the car is moving. The outside scenery is not clearly visible, but it appears to be a natural, outdoor setting. The lighting suggests it is daytime.

Example 2:
A man in a dark sweater and black pants stands on a blue platform, facing a large black door. He initially looks down, then slowly raises his arms and spreads them wide, smiling. The scene is minimalistic. The main subject is a man wearing a dark sweater and black pants. His movements are slow and deliberate. The background remains static throughout the video, with no changes or movements.

Now, please write the visual content description for the provided video in the exact same style:"""
    return run_qwen_vl(content_model, video_path, few_shot_prompt, max_new_tokens=400)


def generate_camera_description(video_path):
    # 【Few-Shot 提示法】提供真实的运镜样例，限制其只输出画幅、焦段和运镜描述
    few_shot_prompt = """Describe the camera motion, shot scale, and focus in this video. Please strictly follow the exact writing style of the examples below.

Example 1:
The camera is stationary, positioned inside the car, capturing a medium close-up view of the two men. The focus is on the two main subjects, with the background intentionally blurred to emphasize the tension and drama within the car.

Example 2:
The camera is stationary, capturing the scene from a fixed, frontal view. There is no camera movement, maintaining a steady and focused perspective on the man's actions.

Example 3:
The camera is stationary, providing a wide-angle view of the scene from a slightly elevated angle, capturing both the subjects and the diner's exterior in a single frame.

Now, please write the camera description for the provided video in the exact same style, starting with "The camera is":"""
    return run_qwen_vl(cam_model, video_path, few_shot_prompt, max_new_tokens=150)


def generate_full_caption(video_path: str) -> str:
    content_desc = generate_content_description(video_path)
    camera_desc = generate_camera_description(video_path)
    
    # 强制清理：以防模型把“Example x”或多余的换行带出来
    full_text = f"{content_desc} {camera_desc}"
    full_text = full_text.replace("Example 1:", "").replace("Example 2:", "").replace("Example 3:", "")
    full_text = " ".join(full_text.split())
    return full_text


if __name__ == "__main__":
    # 读写路径配置
    input_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/testset_3_arc_left.csv"
    output_csv = "/m2v_intern/mengzijie/m2v_camclone_v2/testset_3_arc_left_with_captions.csv"
    
    print(f"Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if 'video_path' not in df.columns:
        raise ValueError("The input CSV does not contain a 'video_path' column.")
    
    qwen_captions = []
    
    print(f"Start processing {len(df)} videos...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating PE Captions"):
        vid_path = row['video_path']
        
        # 防止坏路径或空值
        if pd.isna(vid_path) or not os.path.exists(str(vid_path)):
            print(f"\n[Warning] File missing or invalid path: {vid_path}")
            qwen_captions.append("")
            continue
            
        try:
            # 串行跑两个模型
            cap = generate_full_caption(str(vid_path))
            qwen_captions.append(cap)
        except Exception as e:
            print(f"\n[Error] Failed on video {vid_path}. Error msg: {e}")
            qwen_captions.append("")
            
    df['qwen_caption'] = qwen_captions
    
    print(f"\nSaving updated data to: {output_csv}")
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    
    print("Done! 🎉")