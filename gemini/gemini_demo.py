"""
调用 gemini-3.1-pro-preview 进行图片内容分析
依赖: pip install openai
"""

import base64
import mimetypes
from pathlib import Path
from openai import OpenAI

# ============ 配置区 ============
BASE_URL = "http://kigress-gateway-sgp.internal/kling-shanmingyang03-67256/v1"
API_KEY = "9b0ecb67-b43f-47f5-bc7c-3273538b6261"
USER_KEY = "kling-shanmingyang03-67256"
MODEL_ID = "gemini-3.1-pro-preview"
BIZ_SCENE = "offline"

IMAGE_PATH = "/ytech_m2v2_hdd/liujiwen/ID_Encoder/aio_omini/camera-clone-control/dataset_visualization_en.png"
PROMPT = "请详细分析这张图片的内容，包括它展示的信息、结构、关键要点等。"
# ================================

# 初始化客户端
client = OpenAI(
    api_key="dummy",
    base_url=BASE_URL,
    default_headers={
        "x-api-key": API_KEY,
        "x-ks-user-key": USER_KEY,
        "x-ks-llm-model": MODEL_ID,
        "x-ks-biz-scene": BIZ_SCENE,
    },
)


def encode_image_to_data_url(image_path: str) -> str:
    """把本地图片转成 data URL (base64)"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def analyze_image(image_path: str, prompt: str, max_tokens: int = 1024) -> str:
    data_url = encode_image_to_data_url(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    try:
        answer = analyze_image(IMAGE_PATH, PROMPT, max_tokens=2048)
        print("=" * 60)
        print("模型分析结果：")
        print("=" * 60)
        print(answer)
    except Exception as e:
        print(f"调用失败: {e}")