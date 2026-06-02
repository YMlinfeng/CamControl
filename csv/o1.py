import pandas as pd
import os

def process_csv():
    # 定义输入和输出路径
    input_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o1.csv"
    output_dir = "/m2v_intern/mengzijie/m2v_camclone_v2/output"
    output_path = os.path.join(output_dir, "o1.csv")

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 读取 CSV 文件
    print(f"正在读取: {input_path}")
    df = pd.read_csv(input_path)

    # 确保 index 列是字符串类型，防止拼接出错
    df['index'] = df['index'].astype(str)

    # # (1) ltx_gen
    # df['ltx_gen'] = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/ltx/" + df['index'] + ".mp4"
    
    # # (2) ltx_concat
    # df['ltx_concat'] = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/ltx_concat/" + df['index'] + ".mp4"
    
    # # (3) ours_gen
    # df['ours_gen'] = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/ours/" + df['index'] + ".mp4"
    
    # # (4) ours_concat
    # df['ours_concat'] = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/ours_concat/" + df['index'] + ".mp4"
    
    # # (5) camclone_gen
    # df['camclone_gen'] = "/m2v_intern/mengzijie/m2v_camclone_v2/test_dir/new_camclone_100_v4/generated/" + df['index'] + ".mp4"
    
    # # (6) camclone_concat
    # df['camclone_concat'] = "/m2v_intern/mengzijie/m2v_camclone_v2/test_dir/new_camclone_100_v4/concat/" + df['index'] + ".mp4"

    df['sd2_gen'] = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/sd2.0/" + df['index'] + ".mp4"
    df['sd2_concat'] = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/sd2.0_concat/" + df['index'] + ".mp4"

    # 保存新的 CSV 文件
    df.to_csv(output_path, index=False)
    print(f"处理完成！新文件已保存至: {output_path}")

if __name__ == "__main__":
    process_csv()