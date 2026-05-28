import torch
import os
os.environ["HF_HOME"] = "/m2v_intern/mengzijie/m2v_camclone_v2"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

CAM_MODEL_ID     = "chancharikm/qwen2.5-vl-7b-cam-motion"
CONTENT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
FPS = 8.0

# ---------- 公共 processor（两个模型共用同一个 tokenizer/processor）----------
processor = AutoProcessor.from_pretrained(CONTENT_MODEL_ID)

# ---------- 模型 1：运镜 ----------
cam_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CAM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},          # 固定在 GPU 0
).eval()

# ---------- 模型 2：场景内容 ----------
content_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CONTENT_MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},          # 固定在 GPU 1；只有单卡就改成 {"": 0}
).eval()


# ---------- 通用推理函数 ----------
@torch.inference_mode()
def run_qwen_vl(model, video_path: str, question: str, max_new_tokens: int = 128) -> str:
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


# ---------- 三种描述：运镜 / 场景 / 主体动作 ----------
def describe_camera_motion(video_path):
    q = ("Describe the camera motion in this video using cinematography terms "
         "(pan, tilt, dolly, truck, pedestal, zoom, roll, static, handheld, etc.). "
         "Be concise, one or two sentences.")
    return run_qwen_vl(cam_model, video_path, q, max_new_tokens=96)

def describe_scene(video_path):
    q = ("Describe the visual content of this video in detail: "
         "the setting/environment, lighting, color palette, weather/time of day, "
         "and overall atmosphere. Do NOT describe camera movement. "
         "Output 2-3 sentences.")
    return run_qwen_vl(content_model, video_path, q, max_new_tokens=200)

def describe_subject(video_path):
    q = ("Describe the main subject(s) in this video: "
         "who/what they are, their appearance (clothing, age, expression), "
         "and what action they are performing. "
         "Do NOT describe camera movement. Output 2-3 sentences.")
    return run_qwen_vl(content_model, video_path, q, max_new_tokens=200)


# ---------- 组装成你需要的结构化 caption ----------
def full_caption(video_path: str) -> dict:
    return {
        "video": video_path,
        "camera": describe_camera_motion(video_path),
        "scene":  describe_scene(video_path),
        "subject": describe_subject(video_path),
    }


if __name__ == "__main__":
    result = full_caption("/ytech_m2v6_hdd/liujiwen/video_data/4105067608.mp4")
    print("== Camera  ==", result["camera"])
    print("== Scene   ==", result["scene"])
    print("== Subject ==", result["subject"])