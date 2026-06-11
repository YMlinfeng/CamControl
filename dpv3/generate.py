"""
================================================================
合成相机参数 NPZ 生成器
================================================================
生成两类用于可视化测试的 NPZ：
  1) 鱼眼畸变      (camera_model='fisheye')
  2) 希区柯克变焦  (effect='hitchcock')

🔧 所有可调参数都在下方 CONFIG 区，方便随时改！
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R


# ================================================================
# 🔧 CONFIG  ——  所有可调参数都在这里
# ================================================================

OUTPUT_DIR = "./synthetic_npz"   # ✏️ NPZ 输出目录

# ----------------- 通用 -----------------
N_FRAMES = 125          # ✏️ 每个 npz 的帧数
IMG_W    = 512          # ✏️ 假定视频宽度（影响内参主点）
IMG_H    = 288          # ✏️ 假定视频高度

# ----------------- 鱼眼 -----------------
N_FISHEYE = 10                # ✏️ 生成多少个鱼眼 npz

# 鱼眼焦距 = IMG_W * 此值。越小 FOV 越大，弯曲效果越明显 (推荐 0.4 ~ 0.7)
FISHEYE_FOCAL_RATIO = 0.5

# 鱼眼畸变系数 [k1, k2, k3, k4] (等距模型) —— 想要更怪的镜头可在此加新档位
FISHEYE_DIST_COEFS = {
    'weak':   np.array([-0.10,  0.01,  0.000,  0.0000]),
    'medium': np.array([-0.25,  0.05, -0.005,  0.0005]),
    'strong': np.array([-0.40,  0.15, -0.020,  0.0020]),
}

# 鱼眼运动类型 —— pan: 左右摇头, forward: 前进, tilt: 上下点头
FISHEYE_MOTIONS = ['pan', 'forward', 'tilt']

# ----------------- 希区柯克 -----------------
N_HITCHCOCK = 10                # ✏️ 生成多少个希区柯克 npz

# 初始焦距 = IMG_W * 此值 (推荐 0.5 ~ 0.9)
#   太大 (>1.2)  -> FOV 太窄, dolly_out 后地板/天花板会跑出画面
#   太小 (<0.4)  -> FOV 太广, 不像电影长焦质感
HITCHCOCK_FX_INITIAL_RATIO = 0.7

# Dolly in 推进比例: 相机从 subject_distance 推到 subject_distance * 此值
#   越小越夸张 (0.3~0.5)，太小相机会几乎贴到主体
HITCHCOCK_DOLLY_IN_RATIO  = 0.5

# Dolly out 拉远比例: 相机从 subject_distance 拉到 subject_distance * 此值
#   越大越夸张 (1.5~2.0)，太大焦距会变得很长 FOV 太窄
HITCHCOCK_DOLLY_OUT_RATIO = 1.7

# 主体初始距离的候选 (不同 npz 用不同值, 循环取)
HITCHCOCK_SUBJECT_DISTANCES = [8.0, 10.0, 12.0, 14.0, 16.0]


# ================================================================
# Helper
# ================================================================

def _build_w2c(rot_c2w, cam_pos_world):
    """根据 c2w 旋转和相机世界坐标，构造 3x4 的 w2c 矩阵"""
    R_w2c = rot_c2w.T
    t_w2c = -R_w2c @ cam_pos_world
    return np.hstack([R_w2c, t_w2c.reshape(-1, 1)])


# ================================================================
# 鱼眼 npz 生成
# ================================================================

def generate_fisheye_npz(output_path, motion='pan',
                          distortion_strength='medium',
                          n_frames=N_FRAMES, w=IMG_W, h=IMG_H):
    extrinsics = np.zeros((n_frames, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((n_frames, 3, 3), dtype=np.float32)

    dist_coef = FISHEYE_DIST_COEFS[distortion_strength]
    distortion = np.tile(dist_coef, (n_frames, 1)).astype(np.float64)

    fx = fy = w * FISHEYE_FOCAL_RATIO
    K = np.array([[fx, 0, w/2.0],
                  [0, fy, h/2.0],
                  [0,  0,    1.0]], dtype=np.float32)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        if motion == 'pan':
            rot = R.from_euler('y', (t - 0.5) * np.pi / 2).as_matrix()  # ±45°
            pos = np.array([0.0, 0.0, t * 2.0])
        elif motion == 'forward':
            rot = np.eye(3)
            pos = np.array([0.0, 0.0, t * 8.0])
        elif motion == 'tilt':
            rot = R.from_euler('x', (t - 0.5) * np.pi / 3).as_matrix()  # ±30°
            pos = np.array([0.0, 0.0, t * 2.0])
        else:
            rot, pos = np.eye(3), np.zeros(3)
        extrinsics[i] = _build_w2c(rot, pos)
        intrinsics[i] = K

    np.savez(output_path,
             extrinsics=extrinsics,
             intrinsics=intrinsics,
             distortion=distortion,
             camera_model='fisheye')


# ================================================================
# 希区柯克 npz 生成
# ================================================================

def generate_hitchcock_npz(output_path, direction='dolly_in',
                            subject_distance=10.0,
                            n_frames=N_FRAMES, w=IMG_W, h=IMG_H):
    """
    希区柯克变焦原理：相机推/拉，同时焦距按 距离/初始距离 同步变化，
    使主体在画面上投影大小恒定，背景剧烈伸缩。
    """
    extrinsics = np.zeros((n_frames, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((n_frames, 3, 3), dtype=np.float32)

    subject_world_pos = np.array([0.0, 0.0, subject_distance])
    fx_initial = w * HITCHCOCK_FX_INITIAL_RATIO
    cx, cy = w / 2.0, h / 2.0

    if direction == 'dolly_in':
        end_distance = subject_distance * HITCHCOCK_DOLLY_IN_RATIO
    else:
        end_distance = subject_distance * HITCHCOCK_DOLLY_OUT_RATIO

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        cur_d   = subject_distance + t * (end_distance - subject_distance)
        cam_pos = np.array([0.0, 0.0, subject_distance - cur_d])

        # 🔑 关键: fx ∝ cur_d, 这样 fx*size/d 恒定
        scale  = cur_d / subject_distance
        fx_cur = fx_initial * scale

        extrinsics[i] = _build_w2c(np.eye(3), cam_pos)
        intrinsics[i] = np.array([[fx_cur,   0,  cx],
                                  [0,    fx_cur, cy],
                                  [0,    0,      1.0]], dtype=np.float32)

    np.savez(output_path,
             extrinsics=extrinsics,
             intrinsics=intrinsics,
             camera_model='pinhole',
             effect='hitchcock',
             subject_world_pos=subject_world_pos,
             fx_reference=np.float32(fx_initial))   # 渲染时用来算 fx/fx_ref 比例


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print(f"📦  Generating {N_FISHEYE} fisheye + {N_HITCHCOCK} hitchcock NPZs")
    print(f"    -> {OUTPUT_DIR}")
    print("=" * 60)

    # 鱼眼
    strengths = list(FISHEYE_DIST_COEFS.keys())
    for i in range(N_FISHEYE):
        motion   = FISHEYE_MOTIONS[i % len(FISHEYE_MOTIONS)]
        strength = strengths[(i // len(FISHEYE_MOTIONS)) % len(strengths)]
        p = os.path.join(OUTPUT_DIR, f"fisheye_{i:02d}_{motion}_{strength}.npz")
        generate_fisheye_npz(p, motion=motion, distortion_strength=strength)
        print(f"  💾  [fisheye]    {os.path.basename(p)}")

    # 希区柯克
    for i in range(N_HITCHCOCK):
        direction = 'dolly_in' if i % 2 == 0 else 'dolly_out'
        dist = HITCHCOCK_SUBJECT_DISTANCES[i % len(HITCHCOCK_SUBJECT_DISTANCES)]
        p = os.path.join(OUTPUT_DIR, f"hitchcock_{i:02d}_{direction}_d{dist:.0f}.npz")
        generate_hitchcock_npz(p, direction=direction, subject_distance=dist)
        print(f"  💾  [hitchcock]  {os.path.basename(p)}")

    print(f"\n✅  Done. {N_FISHEYE + N_HITCHCOCK} npz saved.")