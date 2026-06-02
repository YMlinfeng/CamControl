import pandas as pd
import json
import os
import subprocess
import cv2
from tqdm import tqdm

def get_image_info(image_path):
    """获取图像的宽、高和长宽比"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    return w, h, w / h

def run_ffmpeg_task(ref_v, gen_v, output_p, target_w, target_h):
    """
    FFmpeg 核心逻辑：强制高度为偶数，截取前77帧，15fps，左右拼接。
    """
    # --- 核心修复：确保高度是偶数 ---
    if target_h % 2 != 0:
        target_h -= 1
    
    target_ratio = target_w / target_h
    
    ref_filter = (
        f"fps=15,select='lt(n,77)',setpts=N/FRAME_RATE/TB,"
        f"crop='if(gt(iw/ih,{target_ratio}),ih*{target_ratio},iw)':'if(gt(iw/ih,{target_ratio}),ih,iw/{target_ratio})',"
        f"scale=-2:{target_h}"
    )
    
    gen_filter = (
        f"fps=15,select='lt(n,77)',setpts=N/FRAME_RATE/TB,"
        f"scale=-2:{target_h}"
    )
    
    filter_complex = (
        f"[0:v]{ref_filter}[ref];"
        f"[1:v]{gen_filter}[gen];"
        f"[ref][gen]hstack=inputs=2[out]"
    )
    
    cmd = [
        'ffmpeg', '-y',
        '-i', ref_v,
        '-i', gen_v,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:v', 'libx264',
        '-crf', '20',
        '-preset', 'veryfast',
        '-pix_fmt', 'yuv420p',
        output_p
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False

def process_upgrade():
    # 路径配置
    csv_path = "/m2v_intern/mengzijie/m2v_camclone_v2/output/o1.csv"
    base_out_dir = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200"
    
    if not os.path.exists(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        return

    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 定义要处理的类型
    types = ['sd2.0'] # 这里改了，stats 也会跟着变
    
    # --- 核心修改：动态初始化统计字典，防止 KeyError ---
    stats = {t: {'already_exists': 0, 'new_success': 0, 'fail': 0, 'failed_indices': []} for t in types}
    
    for t in types:
        out_dir = os.path.join(base_out_dir, f"{t}_concat")
        os.makedirs(out_dir, exist_ok=True)
        
        # 检查 CSV 中是否存在对应的列，防止列名写错报错
        target_gen_col = f"{t}_gen"
        if target_gen_col not in df.columns:
            print(f"⚠️ 警告: CSV 中找不到列 '{target_gen_col}'，请检查表头。")
            continue
        
        print(f"\n🚀 开始处理 {t.upper()} 类型的任务 (跳过已存在文件)...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{t} 进度"):
            video_index = row.get('index', f'row_{idx}')
            output_video_path = os.path.join(out_dir, f"{video_index}.mp4")

            # --- 断点续传逻辑：如果文件已存在，直接跳过 ---
            if os.path.exists(output_video_path):
                stats[t]['already_exists'] += 1
                continue

            try:
                # 1. 解析路径
                ref_v_data = json.loads(row['ref_videos']) if isinstance(row['ref_videos'], str) else row['ref_videos']
                ref_i_data = json.loads(row['ref_images']) if isinstance(row['ref_images'], str) else row['ref_images']
                
                ref_video_path = ref_v_data[0]['value']
                ref_image_path = ref_i_data[0]['value']
                gen_video_path = row[target_gen_col]
                
                # 2. 检查输入文件 (确保磁盘已挂载)
                if not os.path.exists(ref_video_path) or pd.isna(gen_video_path) or not os.path.exists(str(gen_video_path)):
                    stats[t]['fail'] += 1
                    missing = ref_video_path if not os.path.exists(ref_video_path) else gen_video_path
                    stats[t]['failed_indices'].append(f"{video_index} (文件缺失: {missing})")
                    continue

                # 3. 获取图片尺寸
                img_w, img_h, _ = get_image_info(ref_image_path)
                if img_w is None:
                    stats[t]['fail'] += 1
                    stats[t]['failed_indices'].append(f"{video_index} (图片读取失败)")
                    continue
                
                # 4. 执行拼接
                success = run_ffmpeg_task(ref_video_path, gen_video_path, output_video_path, img_w, img_h)
                
                if success:
                    stats[t]['new_success'] += 1
                else:
                    stats[t]['fail'] += 1
                    stats[t]['failed_indices'].append(f"{video_index} (FFmpeg报错)")
                
            except Exception as e:
                stats[t]['fail'] += 1
                stats[t]['failed_indices'].append(f"{video_index} (程序异常: {str(e)})")
                continue

    # --- 最终统计报告 ---
    print("\n" + "="*60)
    print("📊 [任务处理最终统计报告]")
    print("="*60)
    
    total_expected = len(df)
    
    for t in types:
        if t not in stats: continue
        s = stats[t]
        total_finished = s['already_exists'] + s['new_success']
        
        print(f"\n🔹 类型: {t.upper()}")
        print(f"   ⏭️  原本已存在 (跳过): {s['already_exists']}")
        print(f"   ✨  本次新成功: {s['new_success']}")
        print(f"   ❌  处理失败: {s['fail']}")
        print(f"   📈  当前总计成功 (已存在+新成功): {total_finished} / {total_expected}")
        
        if total_finished == total_expected:
            print(f"   ✅ 恭喜！{t.upper()} 所有视频 (共{total_expected}个) 已全部就绪。")
        else:
            print(f"   ⚠️  注意！{t.upper()} 还有 {total_expected - total_finished} 个视频未完成。")

        if s['fail'] > 0:
            print(f"   具体失败原因及 index:")
            for i in range(0, len(s['failed_indices']), 2):
                print("      " + " | ".join(s['failed_indices'][i:i+2]))
    
    print("\n" + "="*60)
    print("✅ 处理程序运行结束")

if __name__ == "__main__":
    process_upgrade()