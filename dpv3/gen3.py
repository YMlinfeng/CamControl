"""
================================================================
合成相机参数 NPZ 生成器 (改进版：三段式希区柯克)
================================================================
生成两类用于可视化测试的 NPZ：
  1) 鱼眼畸变      (camera_model='fisheye')
  2) 希区柯克变焦  (effect='hitchcock')
       └─ 三段式：随机运镜 → Hitchcock(带扰动) → 随机运镜
                 全长 ≥ 20s，三段位姿与焦距首尾顺次衔接

🔧 所有可调参数都在下方 CONFIG 区。
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R


# ================================================================
# 🔧 CONFIG  ——  所有可调参数都在这里
# ================================================================

OUTPUT_DIR = "./synthetic_npz"   # ✏️ NPZ 输出目录

# ----------------- 通用 -----------------
IMG_W = 512                # ✏️ 假定视频宽度
IMG_H = 288                # ✏️ 假定视频高度
FPS   = 25                 # ✏️ 假定帧率（用于把秒数换算成帧数）

# ----------------- 鱼眼 -----------------
N_FRAMES_FISHEYE = 125
N_FISHEYE = 10

FISHEYE_FOCAL_RATIO = 0.5
FISHEYE_DIST_COEFS = {
    'weak':   np.array([-0.10,  0.01,  0.000,  0.0000]),
    'medium': np.array([-0.25,  0.05, -0.005,  0.0005]),
    'strong': np.array([-0.40,  0.15, -0.020,  0.0020]),
}
FISHEYE_MOTIONS = ['pan', 'forward', 'tilt']

# ----------------- 希区柯克（三段式） -----------------
N_HITCHCOCK = 10

# 总时长（秒）—— 必须 ≥ 20s
HITCHCOCK_TOTAL_DURATION_SEC = 22.0

# 三段时长占比（求和 = 1）：随机 / 希区柯克 / 随机
HITCHCOCK_SEG_RATIOS = (0.35, 0.30, 0.35)

# Hitchcock 初始焦距 = IMG_W * 此值；同时也是段 1 的恒定焦距
HITCHCOCK_FX_INITIAL_RATIO = 0.7

# ✨ 加大 dolly 尺度：推得近、拉得远，节奏快
HITCHCOCK_DOLLY_IN_RATIO  = 0.30    # 推到主体 30% 距离处
HITCHCOCK_DOLLY_OUT_RATIO = 2.50    # 拉到主体 2.5 倍距离

# Hitchcock 段的扰动幅度（保证“不四平八稳”）
HITCHCOCK_PERTURB_POS_RATIO = 0.04  # 平移扰动峰值 ≈ 4% * subject_distance
HITCHCOCK_PERTURB_ROT_DEG   = 2.5   # 旋转扰动峰值（度），各轴独立
HITCHCOCK_PERTURB_N_FREQS   = 3     # 扰动叠加的正弦频率分量数

HITCHCOCK_SUBJECT_DISTANCES = [8.0, 10.0, 12.0, 14.0, 16.0]

# 段 1 / 段 3 随机运镜的幅度
RND_ROT_DEG_RANGE = (45.0, 90.0)    # 旋转角度范围（度）
RND_TRANS_RANGE   = (3.0, 6.0)      # 平移距离范围（世界单位）

RND_MOTION_TYPES = [
    'pan_large', 'tilt_large', 'roll_large',
    'truck_x',  'pedestal_y',  'dolly_z',
    'pan_with_truck', 'tilt_with_pedestal', 'orbit_y',
]


# ================================================================
# Helpers
# ================================================================

def _build_w2c(rot_c2w, cam_pos_world):
    """根据 c2w 旋转和相机世界坐标，构造 3x4 的 w2c 矩阵"""
    R_w2c = rot_c2w.T
    t_w2c = -R_w2c @ cam_pos_world
    return np.hstack([R_w2c, t_w2c.reshape(-1, 1)])


def _smoothstep(t):
    """Hermite smoothstep: 起止速度 = 0，过渡柔和。"""
    return t * t * (3.0 - 2.0 * t)


def _smooth_perturbation(n_frames, amplitude, rng,
                         n_freqs=HITCHCOCK_PERTURB_N_FREQS):
    """
    生成 1D 平滑随机扰动信号，长度 = n_frames。
    端点取值 = 0（保证段间无缝衔接），中段呈柔和波动。
    """
    if n_frames <= 1:
        return np.zeros(n_frames)
    t = np.linspace(0.0, 1.0, n_frames)
    envelope = np.sin(np.pi * t)              # 端点为 0
    signal = np.zeros(n_frames)
    for _ in range(n_freqs):
        freq  = rng.uniform(1.0, 3.5)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp   = rng.uniform(0.6, 1.0)
        signal += amp * np.sin(2.0 * np.pi * freq * t + phase)
    max_abs = max(np.max(np.abs(signal)), 1e-6)
    signal = signal / max_abs                 # 归一到 [-1, 1]
    return amplitude * envelope * signal


# ================================================================
# 鱼眼 npz 生成（保持原样）
# ================================================================

def generate_fisheye_npz(output_path, motion='pan',
                          distortion_strength='medium',
                          n_frames=N_FRAMES_FISHEYE, w=IMG_W, h=IMG_H):
    extrinsics = np.zeros((n_frames, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((n_frames, 3, 3), dtype=np.float32)

    dist_coef = FISHEYE_DIST_COEFS[distortion_strength]
    distortion = np.tile(dist_coef, (n_frames, 1)).astype(np.float64)

    fx = fy = w * FISHEYE_FOCAL_RATIO
    K = np.array([[fx, 0, w / 2.0],
                  [0, fy, h / 2.0],
                  [0,  0,    1.0]], dtype=np.float32)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        if motion == 'pan':
            rot = R.from_euler('y', (t - 0.5) * np.pi / 2).as_matrix()
            pos = np.array([0.0, 0.0, t * 2.0])
        elif motion == 'forward':
            rot = np.eye(3)
            pos = np.array([0.0, 0.0, t * 8.0])
        elif motion == 'tilt':
            rot = R.from_euler('x', (t - 0.5) * np.pi / 3).as_matrix()
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
# 段 1 / 段 3：随机大幅运镜
# ================================================================

def _generate_random_segment(start_pos, start_rot, n_frames, motion_type, rng):
    """
    从 (start_pos, start_rot) 出发的随机运镜段。
    用 smoothstep 缓动 -> 起始/结束速度都为 0，方便与其它段拼接。
    返回 list of (pos(3,), rot(3,3))。
    """
    poses = []
    sign = float(rng.choice([-1.0, 1.0]))

    # 摄像机本地轴在世界系下的方向（CV 约定：+X 右、+Y 下、+Z 前）
    right   = start_rot @ np.array([1.0, 0.0, 0.0])
    down    = start_rot @ np.array([0.0, 1.0, 0.0])
    forward = start_rot @ np.array([0.0, 0.0, 1.0])

    if motion_type in ('pan_large', 'tilt_large', 'roll_large'):
        axis = {'pan_large': 'y', 'tilt_large': 'x', 'roll_large': 'z'}[motion_type]
        angle_max = np.deg2rad(rng.uniform(*RND_ROT_DEG_RANGE)) * sign
        for i in range(n_frames):
            t = _smoothstep(i / max(n_frames - 1, 1))
            delta = R.from_euler(axis, t * angle_max).as_matrix()
            poses.append((start_pos.copy(), start_rot @ delta))

    elif motion_type in ('truck_x', 'pedestal_y', 'dolly_z'):
        axis_world = {'truck_x': right,
                      'pedestal_y': down,
                      'dolly_z': forward}[motion_type]
        dist = rng.uniform(*RND_TRANS_RANGE) * sign
        for i in range(n_frames):
            t = _smoothstep(i / max(n_frames - 1, 1))
            poses.append((start_pos + axis_world * (t * dist), start_rot.copy()))

    elif motion_type == 'pan_with_truck':
        angle_max = np.deg2rad(rng.uniform(*RND_ROT_DEG_RANGE)) * sign
        dist = rng.uniform(*RND_TRANS_RANGE) * float(rng.choice([-1.0, 1.0]))
        for i in range(n_frames):
            t = _smoothstep(i / max(n_frames - 1, 1))
            delta = R.from_euler('y', t * angle_max).as_matrix()
            poses.append((start_pos + right * (t * dist), start_rot @ delta))

    elif motion_type == 'tilt_with_pedestal':
        angle_max = np.deg2rad(rng.uniform(*RND_ROT_DEG_RANGE)) * sign
        dist = rng.uniform(*RND_TRANS_RANGE) * float(rng.choice([-1.0, 1.0]))
        for i in range(n_frames):
            t = _smoothstep(i / max(n_frames - 1, 1))
            delta = R.from_euler('x', t * angle_max).as_matrix()
            poses.append((start_pos + down * (t * dist), start_rot @ delta))

    elif motion_type == 'orbit_y':
        # 绕前方某点公转，镜头也跟着转，营造“绕物体环绕”观感
        radius = rng.uniform(4.0, 8.0)
        center = start_pos + forward * radius
        angle_max = np.deg2rad(rng.uniform(*RND_ROT_DEG_RANGE)) * sign
        for i in range(n_frames):
            t = _smoothstep(i / max(n_frames - 1, 1))
            Ry = R.from_euler('y', t * angle_max).as_matrix()
            poses.append((center + Ry @ (start_pos - center), Ry @ start_rot))

    else:
        for _ in range(n_frames):
            poses.append((start_pos.copy(), start_rot.copy()))

    return poses


# ================================================================
# 段 2：希区柯克段（dolly + zoom + 小扰动）
# ================================================================

def _generate_hitchcock_segment(start_pos, start_rot, fx_initial,
                                n_frames, direction, subject_distance,
                                rng):
    """
    沿初始光轴方向推/拉，同步调焦使主体在画面上大小恒定。
    叠加小幅平移 + 旋转扰动，让画面看起来不“四平八稳”。

    返回:
        out: list of (cam_pos(3,), cam_rot(3,3), fx_cur(float))
        subject_world_pos(3,)
    """
    # 主体放在起始相机正前方 subject_distance 处
    forward0 = start_rot @ np.array([0.0, 0.0, 1.0])
    subject_world_pos = start_pos + forward0 * subject_distance

    if direction == 'dolly_in':
        end_distance = subject_distance * HITCHCOCK_DOLLY_IN_RATIO
    else:
        end_distance = subject_distance * HITCHCOCK_DOLLY_OUT_RATIO

    # 三路独立的平移扰动（世界系 X/Y/Z）
    pert_amp_pos = HITCHCOCK_PERTURB_POS_RATIO * subject_distance
    pert_x = _smooth_perturbation(n_frames, pert_amp_pos,         rng)
    pert_y = _smooth_perturbation(n_frames, pert_amp_pos * 0.6,   rng)
    pert_z = _smooth_perturbation(n_frames, pert_amp_pos * 0.4,   rng)

    # 三路独立的旋转扰动（相机本地 pitch=x / yaw=y / roll=z，单位度）
    pert_amp_rot = HITCHCOCK_PERTURB_ROT_DEG
    pert_rx = _smooth_perturbation(n_frames, pert_amp_rot,        rng)
    pert_ry = _smooth_perturbation(n_frames, pert_amp_rot,        rng)
    pert_rz = _smooth_perturbation(n_frames, pert_amp_rot * 0.5,  rng)

    out = []
    for i in range(n_frames):
        t_lin = i / max(n_frames - 1, 1)
        # smoothstep 让 dolly 起止平滑、中段更快 -> 视觉冲击大
        t = _smoothstep(t_lin)
        cur_d = subject_distance + t * (end_distance - subject_distance)

        # 沿初始光轴的位置（主轴运动）
        cam_pos_axis = subject_world_pos - forward0 * cur_d

        # 叠加平移扰动（世界系）
        cam_pos = cam_pos_axis + np.array([pert_x[i], pert_y[i], pert_z[i]])

        # 叠加旋转扰动（相机本地系，在 start_rot 基础上微扰）
        delta_rot = R.from_euler(
            'xyz',
            [pert_rx[i], pert_ry[i], pert_rz[i]],
            degrees=True,
        ).as_matrix()
        cam_rot = start_rot @ delta_rot

        # 🔑 关键：用相机当前实际光轴方向计算到主体的“有效深度”
        # 这样即便有扰动，主体在像平面上的投影大小依然几乎恒定
        cur_forward = cam_rot @ np.array([0.0, 0.0, 1.0])
        eff_d = float(np.dot(subject_world_pos - cam_pos, cur_forward))
        eff_d = max(eff_d, 1e-3)              # 数值保护

        # Hitchcock 不变量: fx 与 eff_d 成正比
        fx_cur = fx_initial * (eff_d / subject_distance)

        out.append((cam_pos, cam_rot, fx_cur))

    return out, subject_world_pos


# ================================================================
# 完整 “随机 → 希区柯克 → 随机” NPZ
# ================================================================

def generate_hitchcock_full_npz(output_path,
                                direction='dolly_in',
                                subject_distance=10.0,
                                total_duration_sec=HITCHCOCK_TOTAL_DURATION_SEC,
                                fps=FPS,
                                seg_ratios=HITCHCOCK_SEG_RATIOS,
                                w=IMG_W, h=IMG_H,
                                seed=0):
    rng = np.random.default_rng(seed)

    n_total = int(round(total_duration_sec * fps))
    r1, r2, r3 = seg_ratios
    n1 = int(round(n_total * r1))
    n2 = int(round(n_total * r2))
    n3 = n_total - n1 - n2
    assert n1 > 1 and n2 > 1 and n3 > 1, "段太短，请增加 total_duration_sec"

    extrinsics = np.zeros((n_total, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((n_total, 3, 3), dtype=np.float32)

    cx, cy = w / 2.0, h / 2.0
    fx_seg1 = w * HITCHCOCK_FX_INITIAL_RATIO     # 段 1 焦距 = Hitchcock 起始焦距

    # ----- 段 1：随机运镜 -----
    pos0, rot0 = np.zeros(3), np.eye(3)
    motion1 = str(rng.choice(RND_MOTION_TYPES))
    seg1 = _generate_random_segment(pos0, rot0, n1, motion1, rng)
    K_seg1 = np.array([[fx_seg1, 0, cx],
                       [0, fx_seg1, cy],
                       [0, 0, 1.0]], dtype=np.float32)
    for i, (pos, rot) in enumerate(seg1):
        extrinsics[i] = _build_w2c(rot, pos)
        intrinsics[i] = K_seg1

    # ----- 段 2：希区柯克 + 扰动 -----
    pos_h_start, rot_h_start = seg1[-1]          # 段 1 末态 → 段 2 初态
    seg2, subject_world_pos = _generate_hitchcock_segment(
        pos_h_start, rot_h_start, fx_seg1,
        n2, direction, subject_distance, rng,
    )
    for j, (pos, rot, fx_cur) in enumerate(seg2):
        idx = n1 + j
        extrinsics[idx] = _build_w2c(rot, pos)
        intrinsics[idx] = np.array([[fx_cur, 0, cx],
                                    [0, fx_cur, cy],
                                    [0, 0, 1.0]], dtype=np.float32)

    # ----- 段 3：随机运镜 -----
    pos_h_end, rot_h_end, fx_h_end = seg2[-1]    # 段 2 末态 → 段 3 初态
    motion3 = str(rng.choice(RND_MOTION_TYPES))
    seg3 = _generate_random_segment(pos_h_end, rot_h_end, n3, motion3, rng)
    K_seg3 = np.array([[fx_h_end, 0, cx],
                       [0, fx_h_end, cy],
                       [0, 0, 1.0]], dtype=np.float32)
    for k, (pos, rot) in enumerate(seg3):
        idx = n1 + n2 + k
        extrinsics[idx] = _build_w2c(rot, pos)
        intrinsics[idx] = K_seg3

    # ----- 元数据 -----
    segment_starts  = np.array([0, n1, n1 + n2], dtype=np.int32)
    segment_lengths = np.array([n1, n2, n3], dtype=np.int32)
    segment_names   = np.array(['random_pre', 'hitchcock', 'random_post'])

    np.savez(output_path,
             extrinsics=extrinsics,
             intrinsics=intrinsics,
             camera_model='pinhole',
             effect='hitchcock',
             subject_world_pos=subject_world_pos,
             fx_reference=np.float32(fx_seg1),
             hitchcock_direction=direction,
             hitchcock_subject_distance=np.float32(subject_distance),
             hitchcock_dolly_in_ratio=np.float32(HITCHCOCK_DOLLY_IN_RATIO),
             hitchcock_dolly_out_ratio=np.float32(HITCHCOCK_DOLLY_OUT_RATIO),
             segment_starts=segment_starts,
             segment_lengths=segment_lengths,
             segment_names=segment_names,
             segment_random_motions=np.array([motion1, motion3]),
             fps=np.int32(fps),
             total_frames=np.int32(n_total))


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_total = int(round(HITCHCOCK_TOTAL_DURATION_SEC * FPS))
    print("=" * 64)
    print(f"📦  Generating {N_FISHEYE} fisheye + {N_HITCHCOCK} hitchcock NPZs")
    print(f"    Hitchcock clip = {n_total} frames "
          f"(~{HITCHCOCK_TOTAL_DURATION_SEC:.1f}s @ {FPS}fps), "
          f"segments {HITCHCOCK_SEG_RATIOS}")
    print(f"    Output -> {OUTPUT_DIR}")
    print("=" * 64)

    # 鱼眼
    strengths = list(FISHEYE_DIST_COEFS.keys())
    for i in range(N_FISHEYE):
        motion   = FISHEYE_MOTIONS[i % len(FISHEYE_MOTIONS)]
        strength = strengths[(i // len(FISHEYE_MOTIONS)) % len(strengths)]
        p = os.path.join(OUTPUT_DIR, f"fisheye_{i:02d}_{motion}_{strength}.npz")
        generate_fisheye_npz(p, motion=motion, distortion_strength=strength)
        print(f"  💾  [fisheye]    {os.path.basename(p)}")

    # 希区柯克（三段式）
    for i in range(N_HITCHCOCK):
        direction = 'dolly_in' if i % 2 == 0 else 'dolly_out'
        dist = HITCHCOCK_SUBJECT_DISTANCES[i % len(HITCHCOCK_SUBJECT_DISTANCES)]
        p = os.path.join(
            OUTPUT_DIR,
            f"hitchcock_{i:02d}_{direction}_d{dist:.0f}.npz",
        )
        generate_hitchcock_full_npz(
            p, direction=direction, subject_distance=dist, seed=i,
        )
        print(f"  💾  [hitchcock]  {os.path.basename(p)}")

    print(f"\n✅  Done. {N_FISHEYE + N_HITCHCOCK} npz saved.")