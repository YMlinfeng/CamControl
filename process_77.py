#!/usr/bin/env python3
import csv, os, subprocess, json, glob

WORK_DIR = '/ytech_m2v4_hdd/mengzijie/recam'
REF_77_DIR = os.path.join(WORK_DIR, 'ref_77')
os.makedirs(REF_77_DIR, exist_ok=True)

def get_video_info(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-count_frames', '-show_entries', 'stream=nb_read_frames,r_frame_rate',
           '-of', 'json', path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    info = json.loads(r.stdout)
    s = info['streams'][0]
    nb = int(s.get('nb_read_frames', 0))
    rf = s.get('r_frame_rate', '0/1')
    num, den = rf.split('/')
    fps = float(num) / float(den)
    return nb, fps

def uniform_sample(input_path, output_path, nframes, fps_in):
    duration = nframes / fps_in
    fps_out = 77.0 / duration
    cmd = ['ffmpeg', '-y', '-i', input_path,
           '-vf', f'fps={fps_out:.6f}',
           '-vframes', '77',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-an',
           output_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f'  ERROR: {r.stderr[:200]}')
        return False
    return True

for csv_file in ['16.csv', '160.csv']:
    csv_path = os.path.join(WORK_DIR, csv_file)
    rows = []
    fieldnames = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    idx_121 = fieldnames.index('ref_videos_121')
    fieldnames.insert(idx_121 + 1, 'ref_videos_77')

    total = len(rows)
    for i, row in enumerate(rows):
        ref_121_path = row.get('ref_videos_121', '').strip()
        id_val = row['id']
        index_val = row['index']
        new_name = f'{id_val}_{index_val}.mp4'
        dst = os.path.join(REF_77_DIR, new_name)

        if ref_121_path and os.path.exists(ref_121_path):
            nf, fps = get_video_info(ref_121_path)
            ok = uniform_sample(ref_121_path, dst, nf, fps)
            if ok:
                row['ref_videos_77'] = dst
                print(f'[{csv_file}] {i+1}/{total} OK: {new_name} ({nf}f@{fps:.2f}fps -> 77f)')
            else:
                row['ref_videos_77'] = ''
                print(f'[{csv_file}] {i+1}/{total} FAIL: {new_name}')
        else:
            row['ref_videos_77'] = ''
            print(f'[{csv_file}] {i+1}/{total} MISSING: {ref_121_path}')

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'{csv_file} done.')

files = glob.glob(os.path.join(REF_77_DIR, '*.mp4'))
print(f'\nref_77/ total files: {len(files)}')
