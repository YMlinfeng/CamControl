import os
import csv

# 指定目标基础路径
base_dir = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200"

# 获取基础路径下的所有子目录
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 支持的视频格式（可根据你的实际情况增删）
video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    
    # 1. 获取该子目录下所有的视频文件，并按文件夹中的字母/数字顺序排序
    files = sorted(os.listdir(subdir_path))
    video_paths = []
    for f in files:
        if f.lower().endswith(video_exts):
            # 获取绝对路径
            abs_path = os.path.join(subdir_path, f)
            video_paths.append(abs_path)
    
    # 如果该子目录下没有视频，则跳过
    if not video_paths:
        continue

    # 2. 生成 CSV 文件
    csv_filename = f"{subdir}.csv"
    
    # 写入 CSV
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # 写入表头（键）
        header = f"{subdir}_generated"
        writer.writerow([header])
        
        # 写入视频路径（逐行）
        for vp in video_paths:
            writer.writerow([vp])

    print(f"✅ 成功生成文件: {csv_filename} ，包含 {len(video_paths)} 个视频路径。")