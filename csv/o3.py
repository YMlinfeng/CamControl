"""
文件用途：
    1. 调用 Gemini-3.1-Pro 模型对视频运镜一致性进行评分。
    2. 遍历 CSV 中的 {x}_concat 列（x 包含 ltx, ours, sd2.0, camclone）。
    3. 核心改进：强制将模型返回的打分依据（reason）转换为单行文本，防止 CSV 数据错位。
    4. 健壮性：处理路径不存在的情况，自动跳过并填入空值，支持断点续传。

使用方法：
    1. 配置 API_KEY 和相关网关信息。
    2. 输入路径：/m2v_intern/mengzijie/m2v_camclone_v2/output/o2.csv
    3. 输出路径：/m2v_intern/mengzijie/m2v_camclone_v2/output/o3+.csv
    4. 运行：python <文件名>.py
"""

import os
import base64
import mimetypes
import re
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# ============ 你的公司 API 配置区 ============
BASE_URL = "http://kigress-gateway-sgp.internal/kling-shanmingyang03-67256/v1"
API_KEY = "9b0ecb67-b43f-47f5-bc7c-3273538b6261"
USER_KEY = "kling-shanmingyang03-67256"
MODEL_ID = "gemini-3.1-pro-preview"
BIZ_SCENE = "offline"

# 初始化 OpenAI 客户端
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

# ============ 精密设计的运镜打分 Prompt ============
PROMPT = """
[角色设定]
你现在不是一个普通的视频分析助手，而是一个极度专业的“3D摄像机轨迹解算器”（Camera Tracker）和资深电影摄影指导。

[任务描述]
我向你提供了一个左右拼接（Side-by-Side）的低分辨率压缩测试视频。
- 视频左半侧（Left）：参考视频（Reference video）
- 视频右半侧（Right）：AI生成的视频（Generated video）

你的唯一任务：极其严苛地评估右侧视频的**摄像机运动轨迹（运镜跟随度，Camera Movement）**是否与左侧完美一致。

[🔴 绝对禁止项（极其重要）]
大模型极易被以下因素干扰，你必须屏蔽以下所有信息，它们绝对不参与扣分：
1. 绝对忽略：画质崩坏、清晰度极低、压缩噪点、画面闪烁、生成伪影（视频经过了极端压缩以适应传输，请无视糊和马赛克）。
2. 绝对忽略：画面中主体（如人物、动物）的自身动作差异。注意：即使人物在走动或转头，只要摄像机没动，这就是“静止运镜”！不要把人物动作误认为运镜！
3. 绝对忽略：背景内容不一致、光影色彩不同、美学质量差。

[🟢 关注重点（只看全局位移）]
你只是一台“机器眼”，只能感知全局像素的几何位移：
1. 摄像机是在 推(Zoom in)、拉(Zoom out)、摇(Pan左右/Tilt上下)、移(Truck/Pedestal)，还是处于绝对静止？
2. 对比左边和右边：运动的【方向】是否相同？运动的【时机】（什么时候起幅，什么时候落幅）是否同步？运动的【速度和幅度】是否一致？

[评分标准 (1分 到 5分)]
- 5分 (很优)：完美的运镜跟随。右侧的摄像机运动方向、时机、速度与左侧严丝合缝，如同同一个摄像机拍摄。
- 4分 (略优)：优秀。方向和时机基本一致，但速度或幅度有非常细微的偏差。
- 3分 (中等)：及格。运镜大方向对上了，但存在可见的不同步、明显的幅度差异，或者一侧流畅一侧卡顿。
- 2分 (略差)：糟糕。运镜意图错误，比如左侧在左摇，右侧却静止；或者左侧是推进，右侧只是人物变大但镜头没推。
- 1分 (极差)：完全不相关的摄像机运动，轨迹彻底偏离。

[输出格式要求]
请先简明扼要地给出你的运镜轨迹对比分析作为“打分依据”，最后，**必须在最后一行以精确的格式输出你的最终打分**，格式必须严格为：
FINAL_SCORE: X
（注意：X 只能是 1, 2, 3, 4, 5 中的一个纯数字整数）
"""

def compress_video_for_api(input_path: str, output_path: str):
    """极限压缩视频以绕过网关限制"""
    cmd = [
        "ffmpeg", "-y", 
        "-i", input_path,
        "-vf", "scale='min(512,iw)':-2",  
        "-crf", "38",                     
        "-preset", "ultrafast",           
        "-an",                            
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError(f"FFmpeg 处理失败: {e}")

def encode_video_to_data_url(video_path: str) -> str:
    mime, _ = mimetypes.guess_type(video_path)
    if mime is None or not mime.startswith('video'):
        mime = "video/mp4"
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def analyze_video_camera_movement(video_path: str, max_tokens: int = 1024) -> tuple:
    """调用模型并返回处理成单行的 reason"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video_path = temp_video.name

    try:
        compress_video_for_api(video_path, temp_video_path)
        data_url = encode_video_to_data_url(temp_video_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=0.1 
        )
        
        response_text = resp.choices[0].message.content
        
        # 提取分数
        match = re.search(r"FINAL_SCORE:\s*([1-5])", response_text)
        score = int(match.group(1)) if match else -1
        
        # 提取理由并强制转换为单行
        reason_raw = re.sub(r"FINAL_SCORE:\s*[1-5]", "", response_text).strip()
        # 核心逻辑：将所有换行符替换为空格，并将多个空格合并为一个
        reason_single_line = " ".join(reason_raw.split())
        
        return score, reason_single_line

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

def process_csv_multi_model(input_csv: str, output_csv: str):
    if not os.path.exists(input_csv):
        print(f"❌ 找不到输入 CSV: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    models = ['ltx', 'ours', 'sd2.0', 'camclone']
    stats = {m: {"total": 0, "success": 0, "missing": 0, "error": 0} for m in models}

    for m in models:
        concat_col = f"{m}_concat"
        score_col = f"{m}_gemini_score"
        reason_col = f"{m}_gemini_reason"

        if concat_col not in df.columns:
            continue

        if score_col not in df.columns:
            df[score_col] = pd.NA
        if reason_col not in df.columns:
            df[reason_col] = pd.NA

        print(f"\n🎬 评测模型: {m}")
        stats[m]["total"] = len(df)

        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Model: {m}"):
            # 断点续传：已有分数则跳过
            if pd.notna(row[score_col]) and row[score_col] != -1:
                stats[m]["success"] += 1
                continue

            video_path = row[concat_col]

            # 检查路径
            if pd.isna(video_path) or str(video_path).strip() == "" or not os.path.exists(str(video_path)):
                df.at[index, score_col] = ""
                df.at[index, reason_col] = ""
                stats[m]["missing"] += 1
                continue

            try:
                score, reason = analyze_video_camera_movement(str(video_path))
                df.at[index, score_col] = score
                df.at[index, reason_col] = reason
                
                if score != -1:
                    stats[m]["success"] += 1
                else:
                    stats[m]["error"] += 1
                
                # 及时保存，防止崩溃丢失
                df.to_csv(output_csv, index=False)
                
            except Exception as e:
                tqdm.write(f"❌ 异常: {video_path} -> {e}")
                df.at[index, score_col] = -1
                df.at[index, reason_col] = f"Error: {str(e)}".replace("\n", " ")
                stats[m]["error"] += 1
                df.to_csv(output_csv, index=False)

    # --- 输出最终统计报告 ---
    print("\n" + "="*30 + " 任务统计报告 " + "="*30)
    print(f"{'模型':<10} | {'总数':<8} | {'成功':<8} | {'路径缺失':<8} | {'模型异常':<8}")
    print("-" * 65)
    for m, s in stats.items():
        if s["total"] > 0:
            print(f"{m:<10} | {s['total']:<8} | {s['success']:<8} | {s['missing']:<8} | {s['error']:<8}")
    print("="*65)
    print(f"✅ 任务完成！单行格式 CSV 已保存至: {output_csv}")

if __name__ == "__main__":
    INPUT_PATH = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o2.csv"
    OUTPUT_PATH = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o3+.csv"
    
    process_csv_multi_model(INPUT_PATH, OUTPUT_PATH)