import pandas as pd
import shutil
import os

# 输入CSV文件路径和目标文件夹路径
csv_file_path = '/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/outputs/lumiere_xxl/test_t2v_lumeire_xxl_53M_8_23000064_1kprompt/results.csv'  # 修改为实际的CSV文件路径
target_folder_path = '/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/outputs/lumiere_xxl/test_t2v_lumeire_xxl_53M_8_23000064_1kprompt/m2vbench'  # 修改为实际的目标文件夹路径

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 确保目标文件夹存在
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

# 遍历DataFrame的每一行
for index, row in df.iterrows():
    # 替换prompts列中的空格为下划线
    new_video_name = row['prompts'].replace(' ', '_')
    
    # 获取原视频路径
    original_video_path = row['videos']
    
    # 构建新的视频路径
    new_video_path = os.path.join(target_folder_path, new_video_name + os.path.splitext(original_video_path)[-1])
    
    # 复制视频
    shutil.copy2(original_video_path, new_video_path)
    print(f"Copied '{original_video_path}' to '{new_video_path}'")

print("Done!")
