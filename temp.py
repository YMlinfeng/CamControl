import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- 配置参数 ---

# 1. 输入的 CSV 文件路径
CSV_FILE_PATH = '/m2v_intern/public_datasets/Camera_Dataset/benchmark/csv/realestate10k_testset_sample_1k.csv'

# 2. 保存处理后 .npy 文件的目标文件夹
#    脚本会自动创建此文件夹（如果不存在）
OUTPUT_DIR = '/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/demo/realestate10k_processed_poses_77'

# 3. 需要采样的行数
NUM_SAMPLES = 77

# --- 脚本主逻辑 ---

def process_pose_file(txt_path, num_samples):
    """
    处理单个相机姿态txt文件。

    Args:
        txt_path (str): 相机姿态txt文件的路径。
        num_samples (int): 要等间隔采样的行数。

    Returns:
        np.ndarray: 形状为 [num_samples, 4, 4] 的姿态矩阵数组，如果处理失败则返回 None。
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(txt_path):
            # print(f"错误: 文件 {txt_path} 未找到，已跳过。") # 在tqdm循环中打印会扰乱进度条，建议注释
            return None

        # 打开文件并读取除第一行（标题行）外的所有行
        with open(txt_path, 'r') as f:
            lines = f.readlines()[1:]

        if not lines:
            return None
            
        total_lines = len(lines)
        # 如果总行数少于采样数，也无法处理
        if total_lines < num_samples:
            return None

        # 计算需要采样的行的索引
        indices_to_sample = np.linspace(0, total_lines - 1, num=num_samples, dtype=int)
        
        sampled_lines = [lines[i] for i in indices_to_sample]

        all_matrices = []
        for line in sampled_lines:
            parts = line.strip().split()
            if len(parts) < 12:
                continue # 跳过格式不正确的行

            pose_flat = np.array(parts[-12:], dtype=float)
            pose_16 = np.concatenate([pose_flat, [0.0, 0.0, 0.0, 1.0]])
            matrix_4x4 = pose_16.reshape(4, 4)
            all_matrices.append(matrix_4x4)
        
        # 如果由于行格式问题导致最终矩阵数量不足，则认为处理失败
        if len(all_matrices) != num_samples:
            return None

        return np.array(all_matrices)

    except Exception as e:
        # print(f"处理文件 {txt_path} 时发生未知错误: {e}")
        return None

def main():
    """
    主函数，读取CSV，处理文件，并将结果路径添加回CSV。
    """
    # 检查CSV文件是否存在
    if not os.path.exists(CSV_FILE_PATH):
        print(f"错误: CSV文件未找到 -> {CSV_FILE_PATH}")
        return

    # 创建输出目录，并确保路径是绝对路径，以便在CSV中存储明确的路径
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    absolute_output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"输出的NPY文件将保存到: {absolute_output_dir}")

    # 读取CSV文件
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"错误: 无法读取CSV文件 -> {CSV_FILE_PATH}")
        return

    # 初始化一个列表来存储每个生成的 .npy 文件路径
    generated_cam_paths = []

    # 使用 tqdm 显示进度条，遍历DataFrame的每一行
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理姿态文件并生成NPY"):
        pose_txt_path = row['camera_pose_txt']
        
        # 处理单个文件
        final_array = process_pose_file(pose_txt_path, NUM_SAMPLES)
        
        if final_array is not None:
            # 构建输出文件路径
            base_name = os.path.basename(pose_txt_path)
            file_name_no_ext, _ = os.path.splitext(base_name)
            output_filename = f"{file_name_no_ext}.npy"
            # 使用绝对路径以确保路径的明确性
            output_path = os.path.join(absolute_output_dir, output_filename)
            
            # 保存 .npy 文件
            np.save(output_path, final_array)
            
            # 将成功的路径添加到列表中
            generated_cam_paths.append(output_path)
        else:
            # 如果处理失败，添加一个NaN (Not a Number) 作为占位符
            generated_cam_paths.append(np.nan)

    print("\n所有文件处理完毕！")
    
    # 将生成的路径列表作为新列添加到DataFrame中
    print("正在向DataFrame添加 'cam_path' 列...")
    df['cam_path'] = generated_cam_paths

    # 构建新的CSV文件名并保存
    # 在原文件名基础上添加后缀 _with_paths，保存在原CSV文件相同的目录下
    csv_dir = os.path.dirname(CSV_FILE_PATH)
    csv_basename = os.path.basename(CSV_FILE_PATH)
    csv_name_no_ext, ext = os.path.splitext(csv_basename)
    output_csv_filename = f"{csv_name_no_ext}_with_paths.csv"
    output_csv_path = os.path.join(csv_dir, output_csv_filename)
    
    print(f"正在保存带有 'cam_path' 列的新CSV文件到: {output_csv_path}")
    # 使用 index=False 来避免将DataFrame的索引写入CSV文件
    df.to_csv(output_csv_path, index=False)
    
    # 最终的总结信息
    print("\n--- 处理结果总结 ---")
    successful_count = df['cam_path'].notna().sum()
    total_count = len(df)
    print(f"总共处理了 {total_count} 条记录。")
    print(f"成功生成了 {successful_count} 个 .npy 文件。")
    print(f"失败或跳过了 {total_count - successful_count} 条记录。")
    print(f"所有 .npy 文件保存在: {absolute_output_dir}")
    print(f"包含新路径列的CSV文件已保存至: {output_csv_path}")
    print("--- 任务完成 ---")


if __name__ == "__main__":
    main()