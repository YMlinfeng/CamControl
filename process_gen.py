#!/usr/bin/env python3
import os, subprocess, json, glob, sys

BASE_DIR = 'test_dir/gen'
SRC_DIR = BASE_DIR
REF_DIR = os.path.join(BASE_DIR, 'ref')
GEN_DIR = os.path.join(BASE_DIR, 'gen')
CONTENT_DIR = os.path.join(BASE_DIR, 'content')
REF121_DIR = os.path.join(BASE_DIR, 'ref121')
GEN121_DIR = os.path.join(BASE_DIR, 'gen121')
CONCAT121_DIR = os.path.join(BASE_DIR, 'concat121')

for d in [REF_DIR, GEN_DIR, CONTENT_DIR, REF121_DIR, GEN121_DIR, CONCAT121_DIR]:
    os.makedirs(d, exist_ok=True)

def get_video_info(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-count_frames', '-show_entries', 'stream=nb_read_frames,r_frame_rate,width,height',
           '-of', 'json', path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    info = json.loads(r.stdout)
    s = info['streams'][0]
    nb = int(s.get('nb_read_frames', 0))
    rf = s.get('r_frame_rate', '0/1')
    num, den = rf.split('/')
    fps = float(num) / float(den)
    w = int(s.get('width', 0))
    h = int(s.get('height', 0))
    return nb, fps, w, h

def split_video(input_path, ref_out, gen_out, content_out):
    """Split a horizontally-concatenated video into 3 equal vertical strips."""
    cmd = ['ffmpeg', '-y', '-i', input_path,
           '-filter_complex',
           f'[0:v]crop=iw/3:ih:0:0[ref];[0:v]crop=iw/3:ih:iw/3:0[gen];[0:v]crop=iw/3:ih:2*iw/3:0[content]',
           '-map', '[ref]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-an', ref_out,
           '-map', '[gen]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-an', gen_out,
           '-map', '[content]', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-an', content_out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f'  SPLIT ERROR: {r.stderr[:300]}')
        return False
    return True

def uniform_sample_121(input_path, output_path, nframes, fps_in):
    """Uniformly sample video to 121 frames, following process_77.py logic."""
    duration = nframes / fps_in
    fps_out = 121.0 / duration
    cmd = ['ffmpeg', '-y', '-i', input_path,
           '-vf', f'fps={fps_out:.6f}',
           '-vframes', '121',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-an',
           output_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f'  SAMPLE ERROR: {r.stderr[:300]}')
        return False
    return True

def concat_side_by_side(ref_path, gen_path, output_path):
    """Concatenate two videos side by side (left=ref, right=gen)."""
    cmd = ['ffmpeg', '-y', '-i', ref_path, '-i', gen_path,
           '-filter_complex', '[0:v][1:v]hstack=inputs=2',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-an',
           output_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f'  CONCAT ERROR: {r.stderr[:300]}')
        return False
    return True

# ---- Step 1: Split videos into 3 parts ----
mp4_files = sorted(glob.glob(os.path.join(SRC_DIR, '*.mp4')))
total = len(mp4_files)
print(f'=== Step 1: Splitting {total} videos into 3 parts ===')

split_fail = 0
for i, f in enumerate(mp4_files):
    name = os.path.basename(f)
    ref_out = os.path.join(REF_DIR, name)
    gen_out = os.path.join(GEN_DIR, name)
    content_out = os.path.join(CONTENT_DIR, name)
    
    ok = split_video(f, ref_out, gen_out, content_out)
    if ok:
        print(f'  [{i+1}/{total}] Split OK: {name}')
    else:
        split_fail += 1
        print(f'  [{i+1}/{total}] Split FAIL: {name}')

print(f'Step 1 done. {total - split_fail}/{total} succeeded.\n')

# ---- Step 2: Uniformly sample ref and gen to 121 frames ----
ref_files = sorted(glob.glob(os.path.join(REF_DIR, '*.mp4')))
gen_files = sorted(glob.glob(os.path.join(GEN_DIR, '*.mp4')))
print(f'=== Step 2: Sampling {len(ref_files)} ref and {len(gen_files)} gen videos to 121 frames ===')

sample_fail = 0
all_pairs = []

for i, ref_f in enumerate(ref_files):
    name = os.path.basename(ref_f)
    gen_f = os.path.join(GEN_DIR, name)
    ref121_out = os.path.join(REF121_DIR, name)
    gen121_out = os.path.join(GEN121_DIR, name)
    
    nf_ref, fps_ref, _, _ = get_video_info(ref_f)
    nf_gen, fps_gen, _, _ = get_video_info(gen_f)
    
    ok_ref = uniform_sample_121(ref_f, ref121_out, nf_ref, fps_ref)
    ok_gen = uniform_sample_121(gen_f, gen121_out, nf_gen, fps_gen)
    
    if ok_ref and ok_gen:
        all_pairs.append((ref121_out, gen121_out, name))
        print(f'  [{i+1}/{len(ref_files)}] Sample OK: {name} (ref:{nf_ref}f@{fps_ref:.2f} -> 121f, gen:{nf_gen}f@{fps_gen:.2f} -> 121f)')
    else:
        sample_fail += 1
        print(f'  [{i+1}/{len(ref_files)}] Sample FAIL: {name} (ref_ok={ok_ref}, gen_ok={ok_gen})')

print(f'Step 2 done. {len(all_pairs)}/{len(ref_files)} succeeded.\n')

# ---- Step 3: Concatenate ref121 and gen121 side by side ----
print(f'=== Step 3: Concatenating {len(all_pairs)} pairs into concat121 ===')

concat_fail = 0
for i, (ref121, gen121, name) in enumerate(all_pairs):
    concat_out = os.path.join(CONCAT121_DIR, name)
    ok = concat_side_by_side(ref121, gen121, concat_out)
    if ok:
        print(f'  [{i+1}/{len(all_pairs)}] Concat OK: {name}')
    else:
        concat_fail += 1
        print(f'  [{i+1}/{len(all_pairs)}] Concat FAIL: {name}')

print(f'Step 3 done. {len(all_pairs) - concat_fail}/{len(all_pairs)} succeeded.\n')

# ---- Summary ----
print('=== Summary ===')
print(f'ref:      {len(glob.glob(os.path.join(REF_DIR, "*.mp4")))} files')
print(f'gen:      {len(glob.glob(os.path.join(GEN_DIR, "*.mp4")))} files')
print(f'content:  {len(glob.glob(os.path.join(CONTENT_DIR, "*.mp4")))} files')
print(f'ref121:   {len(glob.glob(os.path.join(REF121_DIR, "*.mp4")))} files')
print(f'gen121:   {len(glob.glob(os.path.join(GEN121_DIR, "*.mp4")))} files')
print(f'concat121:{len(glob.glob(os.path.join(CONCAT121_DIR, "*.mp4")))} files')
print('All done!')
