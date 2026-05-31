import time
import os
from google import genai
from google.genai import types

# ==================== API KEY 配置区域（二选一） ====================

# 方式一：直接硬编码（将您的 API Key 粘贴到下方引号中）
API_KEY = "AIzaSyCyiiApnmw_2PCPUGkl8_smHXRC0U-GOZE" 

# 方式二：从本地文件读取（推荐，更安全）
# 如果您想用这种方式，请取消下方 4 行代码的注释，并注释掉上方的 1 行硬编码。
# 并在该 Python 脚本的同级目录下，新建一个名为 `api_key.txt` 的文本文件，里面只写入您的 API Key。
# if os.path.exists("api_key.txt"):
#     with open("api_key.txt", "r", encoding="utf-8") as f:
#         API_KEY = f.read().strip()

# ===================================================================

# 初始化 Client。传入我们配置好的 API Key
client = genai.Client(api_key=API_KEY)

def wait_for_files_active(files):
    """
    因为视频文件通常较大，上传后需要等待后台服务将其处理为‘ACTIVE’状态才能进行推理。
    """
    print("正在等待视频文件处理完成...")
    for f in files:
        current_file = client.files.get(name=f.name)
        while current_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            current_file = client.files.get(name=f.name)
        
        if current_file.state.name == "FAILED":
            raise ValueError(f"文件 {f.display_name} 处理失败。")
    print("\n所有视频处理已就绪！")

try:
    # 1. 上传第一个视频
    print("正在上传视频 A...")
    video_a = client.files.upload(file="/m2v_intern/mengzijie/m2v_camclone_v2/00202.mp4")
    
    # 2. 上传第二个视频
    print("正在上传视频 B...")
    video_b = client.files.upload(file="/m2v_intern/mengzijie/m2v_camclone_v2/00217.mp4")
    
    # 3. 等待两个视频就绪
    wait_for_files_active([video_a, video_b])
    
    # 4. 构建对比 Prompt
    prompt = """
    你是一位专业的视频后期与画质评估专家。请仔细对比我上传的两个视频，并从以下几个维度输出一份客观详尽的对比报告：
    1. 画质与清晰度：哪一个视频的分辨率更高、细节保留更完整？是否存在明显的压缩伪影或噪点？
    2. 色彩与光影：两者的色彩饱和度、白平衡、对比度以及曝光控制表现如何？
    3. 稳定性与镜头表现：哪一个视频的镜头更稳定（如防抖效果）?
    4. 综合建议：基于以上对比，哪一个视频的整体质量更好，好在哪里？
    
    请在报告中明确指出‘视频 A’和‘视频 B’分别对应哪一个输入。
    """
    
    # 5. 调用 Gemini 1.5 Pro 进行对比推理
    print("正在调用 Gemini 进行对比分析...")
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview", # 您也可以选择 "gemini-3.1-pro" 等最新模型
        contents=[
            video_a, # 输入第一个视频
            video_b, # 输入第二个视频
            prompt   # 传入您的对比指令
        ]
    )
    
    # 输出对比结果
    print("\n====== 对比报告 ======")
    print(response.text)

finally:
    # 6. 清理上传的文件（可选，Files API 上传的文件默认会在 2 天后自动删除）
    print("\n正在清理上传的文件...")
    if 'video_a' in locals():
        client.files.delete(name=video_a.name)
    if 'video_b' in locals():
        client.files.delete(name=video_b.name)
    print("清理完成。")