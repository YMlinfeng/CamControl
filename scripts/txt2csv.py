import csv
from tqdm import tqdm


def convert_txt_to_csv_with_header(input_filename, output_filename, delimiter="\t", header=None):
    count = 0

    proc_data = []
    with open(input_filename, "r", encoding="utf-8") as infile:
        for line in tqdm(infile):
            # 分割每一行并写入csv文件
            data = line.strip().split(delimiter)
            data = [data[0]] + [",".join([i for i in data[1:4] if len(i) > 0])] + [data[4]] +data[5:7]
            if data[0].startswith("/video/hd_vg_130m/"):
                data[0] = data[0].replace("/video/hd_vg_130m/", "/ytech_m2v_hdd/hd_vg_130m/")

            width, height = int(data[-2]), int(data[-1])
            if width / height - 16 / 9 > 0.1:
                continue
            # csv_writer.writerow(data)
            if data[2] != 'NULL' and data[1]:
                proc_data.append((float(data[2]), data))
            count += 1
            # if count >= 1000:
            #     break

    proc_data.sort(key=lambda x: x[0])
    proc_data = proc_data[int(len(proc_data) * 0.25) :]
    num_item_per_bucket = len(proc_data) / 512

    with open(output_filename, "w", encoding="utf-8", newline="") as outfile:
        # 初始化csv写入器
        csv_writer = csv.writer(outfile)

        # 如果提供了表头，则写入表头
        if header:
            csv_writer.writerow(header)

        for idx, line in enumerate(proc_data):
            csv_writer.writerow(line[1][:-2] + [int(idx // num_item_per_bucket)])

        # 逐行读取输入文件并写入输出文件
    print(f"count = {count}")


# 调用函数，这里需要提供表头列表，例如['Column1', 'Column2', 'Column3']
# header = ['id', 'video_ceph_path', 'ori_caption_en', 'gen_mplug_caption_en']
# header = ['id',
# 'video_ceph_path',
# 'ori_caption_cn',
# 'gen_mplug_caption_cn',
# 'mplug_cogvlm_internlm_cn',
# 'ori_caption_en',
# 'gen_mplug_caption_en',
# 'mplug_cogvlm_internlm_en',
# 'width',
# 'height']
# video_path,llama2_caption,FlowScore,motion_bucket
header = [
    "video_path",
    "llama2_caption",
    "FlowScore",
    "motion_bucket",
]
convert_txt_to_csv_with_header("./threshold/2024-02-04_all3_new.txt", "/video/yht/data_0204_ms_top75.csv", delimiter="\t", header=header)
