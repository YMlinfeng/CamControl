import os
import base64
import mimetypes
import re
import pandas as pd
import subprocess
import tempfile
import textwrap
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
    except FileNotFoundError:
        raise RuntimeError("系统未安装 ffmpeg，请安装。")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg 压缩视频失败: {e}")

def encode_video_to_data_url(video_path: str) -> str:
    """把本地视频转成 data URL (base64) 用于 API 传输"""
    mime, _ = mimetypes.guess_type(video_path)
    if mime is None or not mime.startswith('video'):
        mime = "video/mp4"

    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def analyze_video_camera_movement(video_path: str, max_tokens: int = 1024) -> tuple:
    """调用大模型分析运镜，并提取 1-5 的分数 以及 打分依据"""
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video_path = temp_video.name

    try:
        tqdm.write(f"正在进行临时压缩以绕过网关限制...")
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

        tqdm.write(f"正在向 {MODEL_ID} 发送推理请求...")
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=0.1 
        )
        
        response_text = resp.choices[0].message.content
        tqdm.write("--- 模型推理解析 ---")
        tqdm.write(response_text)
        tqdm.write("-" * 20)
        
        # 匹配 1-5 的整数得分
        match = re.search(r"FINAL_SCORE:\s*([1-5])", response_text)
        if match:
            score = int(match.group(1))
            # 清理打分依据文本，把带 FINAL_SCORE 的那一行剔除掉
            reason = re.sub(r"FINAL_SCORE:\s*[1-5]", "", response_text).strip()
            return score, reason
        else:
            tqdm.write("⚠️ 警告：无法从模型回复中提取格式化的分数，默认返回 -1")
            return -1, response_text

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

def calculate_current_avg(df):
    """计算当前有效的平均分（排除 -1 和空值，有效分为 1-5）"""
    valid_scores = pd.to_numeric(df["gemini_cam_score"], errors='coerce').dropna()
    valid_scores = valid_scores[valid_scores > 0]
    if len(valid_scores) > 0:
        return round(valid_scores.mean(), 3)
    return 0.0

def export_to_txt(df, txt_path: str):
    """把视频编号、分数、打分依据美观地输出到单独的 TXT 文件"""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" 📽️ 视频运镜打分校验报告 (1-5分制)\n")
        f.write("=" * 70 + "\n\n")
        
        for index, row in df.iterrows():
            score = row.get("gemini_cam_score")
            # 只有已打分的才输出
            if pd.notna(score) and score != -1:
                video_name = Path(str(row.get("concat", ""))).name
                reason = str(row.get("gemini_cam_reason", "无依据"))
                
                f.write(f"【视频名称】: {video_name}\n")
                f.write(f"【最终打分】: {score} 分\n")
                f.write("【打分依据】:\n")
                
                # 使用 textwrap 对大段文字进行自动换行处理 (限宽 80 字符)，方便阅读
                wrapped_lines = textwrap.wrap(reason, width=80)
                for line in wrapped_lines:
                    f.write(f"    {line}\n")
                
                f.write("\n" + "-" * 70 + "\n\n")

def process_csv(input_csv_path: str, output_csv_path: str):
    """读取包含 'concat' 路径的 CSV，打分并写入新列和 TXT"""
    if not os.path.exists(input_csv_path):
        print(f"找不到 CSV 文件: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)
    
    if "concat" not in df.columns:
        raise ValueError("CSV 表头中没有找到 'concat' 列！")

    # 新增或初始化分数与理由列
    if "gemini_cam_score" not in df.columns:
        df["gemini_cam_score"] = pd.NA
    if "gemini_cam_reason" not in df.columns:
        df["gemini_cam_reason"] = pd.NA

    # 生成对应的 TXT 校验文件路径 (与输出CSV同目录)
    txt_output_path = output_csv_path.replace(".csv", "_check.txt")

    total_videos = len(df)
    scored_mask = df["gemini_cam_score"].notna()
    already_scored = scored_mask.sum()
    pending = total_videos - already_scored
    current_avg = calculate_current_avg(df)

    print("\n" + "="*45)
    print("📊 [打分任务概览]")
    print(f"🔹 视频总数: {total_videos}")
    print(f"🔹 已打分数: {already_scored}  (未打分: {pending})")
    print(f"🔹 当前有效平均分: {current_avg}")
    print("="*45 + "\n")

    with tqdm(total=total_videos, desc="整体进度", unit="个", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        
        for index, row in df.iterrows():
            video_path = row["concat"]
            
            if pd.isna(video_path) or not str(video_path).strip():
                pbar.update(1)
                continue
                
            if pd.notna(row["gemini_cam_score"]):
                pbar.update(1)
                continue

            video_name = Path(video_path).name
            tqdm.write(f"\n[{index+1}/{total_videos}] 正在处理视频: {video_name}")
            
            try:
                score, reason = analyze_video_camera_movement(video_path)
                tqdm.write(f"✅ [{video_name}] 本次得分: {score}")
                
                # 更新 Dataframe 对应的单元格
                df.at[index, "gemini_cam_score"] = score
                df.at[index, "gemini_cam_reason"] = reason
                
                # 保存 CSV 并同步导出 TXT
                df.to_csv(output_csv_path, index=False)
                export_to_txt(df, txt_output_path)
                
                new_avg = calculate_current_avg(df)
                pbar.set_postfix({"最新分": score, "均分": new_avg})
                
            except Exception as e:
                tqdm.write(f"❌ 处理视频 {video_name} 时发生错误: {e}")
                df.at[index, "gemini_cam_score"] = -1
                df.at[index, "gemini_cam_reason"] = f"处理出错: {str(e)}"
                
                df.to_csv(output_csv_path, index=False)
                export_to_txt(df, txt_output_path)
                pbar.set_postfix({"错误行": index})

            pbar.update(1)

    final_avg = calculate_current_avg(df)
    print("\n🎉 全部处理完成！")
    print(f"📈 最终运镜一致性平均分: {final_avg}")
    print(f"💾 CSV 结果已保存至: {output_csv_path}")
    print(f"📄 TXT 校验已保存至: {txt_output_path}")

if __name__ == "__main__":
    INPUT_CSV = "/m2v_intern/mengzijie/m2v_camclone_v2/t.csv" 
    OUTPUT_CSV = "/m2v_intern/mengzijie/m2v_camclone_v2/tt.csv"
    
    process_csv(INPUT_CSV, OUTPUT_CSV)