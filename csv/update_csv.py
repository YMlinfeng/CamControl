import os
import pandas as pd

# ==================== 路径配置 ====================
csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/t.csv"
concat_dir = "/m2v_intern/mengzijie/m2v_camclone_v2/test_dir/new_camclone_100_v2/concat"
generated_dir = "/m2v_intern/mengzijie/m2v_camclone_v2/test_dir/new_camclone_100_v2/generated"

def get_sorted_video_paths(directory):
    """读取目录下所有视频文件，排序并返回绝对路径列表"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"找不到目录: {directory}")
        
    # 过滤出常见的视频文件（防止混入隐藏文件如 .DS_Store）
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    files = [f for f in os.listdir(directory) if f.lower().endswith(valid_exts)]
    
    # 按名字字母/数字顺序排序，保证写入顺序一致
    files.sort()
    
    # 拼成绝对路径
    return [os.path.join(directory, f) for f in files]

def main():
    # 1. 获取视频路径列表
    concat_paths = get_sorted_video_paths(concat_dir)
    generated_paths = get_sorted_video_paths(generated_dir)

    # 2. 读取现有的 CSV 文件
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 3. 获取数量
    csv_rows = len(df)
    concat_count = len(concat_paths)
    generated_count = len(generated_paths)

    print("=== 数量情况 ===")
    print(f"CSV 现有数据行数: {csv_rows}")
    print(f"concat 目录视频数量: {concat_count}")
    print(f"generated 目录视频数量: {generated_count}")

    # 检查：如果视频数量少于 CSV 行数，就会因为长度不一致报错，所以需要拦截
    if concat_count < csv_rows or generated_count < csv_rows:
        raise ValueError(f"❌ 视频数量不足！CSV需要 {csv_rows} 行，但某个目录视频不够。")

    print(f"⚠️ 将自动截取按文件名排序后的前 {csv_rows} 个视频写入...")

    # 4. 根据 CSV 的行数，精准截取前 N 个视频
    concat_paths = concat_paths[:csv_rows]
    generated_paths = generated_paths[:csv_rows]

    # 5. 写入新列
    df['concat'] = concat_paths
    df['generated'] = generated_paths

    # 6. 保存回 CSV
    # 直接覆盖原文件，index=False 防止把行号作为新的一列写入
    df.to_csv(csv_path, index=False)
    print(f"✅ 成功将前 {csv_rows} 个视频的 'concat' 和 'generated' 两列写入: {csv_path}")

if __name__ == "__main__":
    main()