"""
================================================================
generate_hitchcock_npz.py
  生成"三段式"希区柯克变焦 npz：
       Segment 1 (随机推拉摇移)
    -> Segment 2 (Hitchcock + 微抖动)
    -> Segment 3 (再次随机推拉摇移)

  整段视频 ≥ 10 秒 (252 帧 @ 24fps = 10.5s)。

  Hitchcock 物理公式：
    要让 subject 在屏幕上的角大小保持不变，需要
        f(t) / d(t) = const
    即 f(t) = f_ref * d(t) / d_ref

  立方体大小恒定 (在 viz 里观察到) = npz 数学正确的视觉证明
================================================================
"""
import os
import numpy as np

# ================================================================
# 🔧 CONFIG  (所有可调参数集中在这里)
# ================================================================

OUTPUT_DIR  = "./hitchcock_npz"
NUM_VIDEOS  = 5      # 📦 一次生成几个 npz (不同 seed)；调大就批量化
FPS         = 24
TOTAL_FRAMES = 252   # ⏱️ 24 × 10.5 = 252 帧, 即 10.5 秒
                     # 调大 -> 视频更长；保持是 24 的倍数比较好
SEG1_FRAMES = 84     # 第一段长度 (3.5s)
SEG2_FRAMES = 84     # 第二段长度 (3.5s) ← Hitchcock 在这里
# Seg3 自动 = TOTAL_FRAMES - SEG1 - SEG2 = 84

# ---- 渲染分辨率 / 基础内参 ----
W, H = 512, 288
FX_BASE = 400.0      # 🎥 段 1/3 用的恒定焦距 (像素)
                     # 调大 -> 整体更长焦/更窄视场；调小 -> 更广角
CX, CY = W/2.0, H/2.0

# ---- 立方体 (即 subject) ----
SUBJECT_POS    = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 立方体世界中心
CUBE_HALF_SIZE = 0.6     # 📦 立方体半边长 (世界单位)，边长=1.2
                         # 调大 -> 立方体更醒目，但太大会贴脸
                         # 推荐 0.4 ~ 1.0
CUBE_YAW_DEG   = 25.0    # 立方体绕世界 Y 轴预转 (让我们看到两个侧面)
CUBE_PITCH_DEG = 15.0    # 立方体绕世界 X 轴预转 (让我们看到顶面)
                         # 设成 0,0 -> 立方体正对相机，只看到一面 (不立体)
                         # 推荐 (15~40, 10~25)

# ---- 段 1 / 段 3 (随机运镜) 参数 ----
SEG_RND_DIST_RANGE = (6.0, 10.0)
                         # 🎬 相机到 subject 的距离范围
                         # 调小区间 -> 距离变化少，画面更稳
                         # 注意下限要 > CUBE_HALF_SIZE * 2，否则相机进立方体
SEG_RND_AZ_AMP = 0.6     # 方位角振幅 (rad)，0.6 ≈ ±34°，越大摇得越广
SEG_RND_EL_AMP = 0.25    # 仰角振幅 (rad)，0.25 ≈ ±14°
SEG_RND_FREQ   = 1.5     # 随机信号最高频率 (周期/段)
                         # 调大 -> 摆得快/抖；调小 -> 慢、电影感

# ---- 段 2 (Hitchcock) 参数 ----
SEG2_DOLLY_IN_RATIO  = 0.28   # 📍 推到 28% 起始距离 (推得非常近)
                              # 调小 -> 推得更近，效果越戏剧但风险大
                              # 推荐 0.2 ~ 0.4
SEG2_DOLLY_OUT_RATIO = 1.85   # 📍 拉远到 185% 起始距离
                              # 调大 -> 拉得越远；推荐 1.5 ~ 2.5

SEG2_PHASE_IN_FRAC   = 0.32   # ⚡ 推近占段 2 的比例 (32%)
                              # 调小 -> 推得更"急"，更"快"，符合用户要求
SEG2_PHASE_HOLD_FRAC = 0.04   # 最近处停留比例 (戏剧停顿)
                              # 设 0 = 不停留；推荐 0.02 ~ 0.08

SEG2_LATERAL_PERT_FRAC = 0.04 # 🎲 侧向位置扰动 (起始距离的比例)
                              # 4% × d_start ≈ 几十厘米的横向抖
                              # 调大 -> 立方体在屏幕上左右摆得明显
                              # 推荐 0.02 ~ 0.08
SEG2_LATERAL_PERT_FREQ = 3.8  # 侧向扰动频率，调大 -> 抖得更快
SEG2_LOOKAT_JITTER     = 0.10 # 🎯 视线目标点偏移幅度 (世界单位)
                              # 让相机视线轻微"漂"，使立方体在屏幕里轻微转动
                              # 调大 -> 漂得多；调到 0 -> 立方体严格居中
SEG2_LOOKAT_FREQ       = 3.0


# ================================================================
# 工具函数
# ================================================================

def look_at_w2c(cam_pos, target, scene_up=None):
    """
    World: Y-axis 向下 (CV 约定)。 Camera: OpenCV (X 右, Y 下, Z 前)。
    返回 PROPER rotation (det = +1) 的 w2c，scipy Slerp 才能用。
    """
    if scene_up is None:
        scene_up = np.array([0., -1., 0.], dtype=np.float32)  # Y-down 世界里"场景上方"= -Y
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    target  = np.asarray(target,  dtype=np.float32)

    forward = target - cam_pos
    fn = np.linalg.norm(forward)
    forward = forward / fn if fn > 1e-8 else np.array([0,0,1], dtype=np.float32)

    # right = forward × scene_up  （这个叉乘顺序保证 +X 是相机右侧）
    right = np.cross(forward, scene_up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # forward 与 scene_up 平行的退化情况（相机正上/正下方）
        for alt in [np.array([1.,0.,0.], np.float32),
                    np.array([0.,0.,1.], np.float32)]:
            right = np.cross(forward, alt)
            rn = np.linalg.norm(right)
            if rn > 1e-6: break
    right = right / rn

    # ★ 关键修复：用 forward × right (不是 right × forward)
    #   这样 (right, down, forward) 是右手系 → det(R_c2w) = +1
    down = np.cross(forward, right)

    R_c2w = np.column_stack([right, down, forward]).astype(np.float32)

    # 防御性断言（可选；正常应当 ≈ 1.0）
    assert abs(np.linalg.det(R_c2w) - 1.0) < 1e-4, \
        f"look_at_w2c produced improper rotation, det={np.linalg.det(R_c2w):.4f}"

    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ cam_pos
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R_w2c
    M[:3,  3] = t_w2c
    return M


def smooth_random_signal(n_frames, n_components=3, max_freq=1.5, rng=None):
    """随机正弦叠加合成的平滑信号 ∈ [-1, 1]。"""
    rng = rng if rng is not None else np.random
    t = np.linspace(0, 1, n_frames)
    sig = np.zeros(n_frames)
    for k in range(n_components):
        f = (k+1) * (max_freq / n_components) * (0.7 + 0.6*rng.rand())
        phi = rng.rand() * 2*np.pi
        sig += (1.0 / (k+1)) * np.sin(2*np.pi*f*t + phi)
    sig /= (np.max(np.abs(sig)) + 1e-6)
    return sig.astype(np.float32)


def make_K(fx, fy=None):
    fy = fx if fy is None else fy
    return np.array([[fx, 0, CX], [0, fy, CY], [0, 0, 1]], dtype=np.float32)


def euler_to_R(yaw_deg, pitch_deg, roll_deg=0.0):
    """绕 Y、X、Z 的欧拉角合成成旋转矩阵 (世界系)。"""
    y, p, r = map(np.deg2rad, [yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]], np.float32)
    Rx = np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]], np.float32)
    Rz = np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]], np.float32)
    return (Ry @ Rx @ Rz).astype(np.float32)


# ================================================================
# 段生成器
# ================================================================

def gen_segment_random(n_frames, subject_pos, start_pos=None,
                       dist_range=SEG_RND_DIST_RANGE, rng=None):
    """
    随机推拉摇移片段 (球坐标平滑随机)。
    相机始终朝向 subject —— 这是保证立方体不跑出画面的关键设计。
    """
    rng = rng if rng is not None else np.random
    d_min, d_max = dist_range

    # 起始位置：要么随机，要么接续上一段的终点
    if start_pos is None:
        d0  = rng.uniform(d_min, d_max)
        az0 = rng.uniform(-np.pi, np.pi)
        el0 = rng.uniform(-0.2, 0.2)
    else:
        offset = start_pos - subject_pos
        d0  = float(np.linalg.norm(offset))
        az0 = float(np.arctan2(offset[0], offset[2]))
        el0 = float(np.arcsin(np.clip(offset[1] / max(d0, 1e-6), -1, 1)))

    # 三个轴各一个平滑随机偏移；减去 [0] 让起点严格等于 start_pos
    s_d  = smooth_random_signal(n_frames, 3, SEG_RND_FREQ, rng)
    s_az = smooth_random_signal(n_frames, 3, SEG_RND_FREQ, rng)
    s_el = smooth_random_signal(n_frames, 3, SEG_RND_FREQ, rng)
    s_d  -= s_d[0];  s_az -= s_az[0];  s_el -= s_el[0]

    d_amp = (d_max - d_min) * 0.35
    d_t  = np.clip(d0 + d_amp * s_d, d_min * 0.6, d_max * 1.2)
    az_t = az0 + SEG_RND_AZ_AMP * s_az
    el_t = np.clip(el0 + SEG_RND_EL_AMP * s_el, -0.5, 0.5)

    pos = np.zeros((n_frames, 3), dtype=np.float32)
    pos[:, 0] = d_t * np.cos(el_t) * np.sin(az_t)
    pos[:, 1] = d_t * np.sin(el_t)
    pos[:, 2] = d_t * np.cos(el_t) * np.cos(az_t)
    return pos + subject_pos


def gen_segment_hitchcock(n_frames, subject_pos, start_pos, fx_ref, rng=None):
    """
    希区柯克片段：
      - 主要运动：沿"相机-subject 连线"方向 推近 -> 短暂停留 -> 拉远
      - fx 严格按 d/d_ref 比例联动 (这就是 Hitchcock zoom)
      - 加侧向 (相机位置) + look-at (视线方向) 两层小扰动
    """
    rng = rng if rng is not None else np.random

    offset  = start_pos - subject_pos
    d_start = float(np.linalg.norm(offset))
    dir_vec = (offset / d_start).astype(np.float32) if d_start > 1e-6 \
              else np.array([0,0,1], dtype=np.float32)
    d_ref = d_start

    # ---- 距离曲线：推进段 + 停留 + 拉远段 ----
    d_in  = d_start * SEG2_DOLLY_IN_RATIO    # 最近距离
    d_out = d_start * SEG2_DOLLY_OUT_RATIO   # 最远距离

    n_in   = int(n_frames * SEG2_PHASE_IN_FRAC)
    n_hold = int(n_frames * SEG2_PHASE_HOLD_FRAC)
    n_out  = n_frames - n_in - n_hold

    profile = np.zeros(n_frames, dtype=np.float32)
    # cosine ease 平滑插值
    t_in = np.linspace(0, 1, n_in)
    profile[:n_in] = d_start + (d_in - d_start) * (0.5 - 0.5*np.cos(t_in*np.pi))
    profile[n_in:n_in+n_hold] = d_in
    t_out = np.linspace(0, 1, n_out)
    profile[n_in+n_hold:] = d_in + (d_out - d_in) * (0.5 - 0.5*np.cos(t_out*np.pi))

    # ---- 与 dir_vec 正交的两个轴 (用来加侧向扰动) ----
    world_up = np.array([0.,1.,0.], dtype=np.float32)
    right = np.cross(world_up, dir_vec)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([1.,0,0], dtype=np.float32), dir_vec)
    right = right / np.linalg.norm(right)
    up_p  = np.cross(dir_vec, right);  up_p = up_p / np.linalg.norm(up_p)

    # ---- 侧向位置扰动 ----
    pa = SEG2_LATERAL_PERT_FRAC * d_start
    px = pa * smooth_random_signal(n_frames, 4, SEG2_LATERAL_PERT_FREQ, rng)
    py = pa * smooth_random_signal(n_frames, 4, SEG2_LATERAL_PERT_FREQ, rng)

    positions = np.zeros((n_frames, 3), dtype=np.float32)
    for i in range(n_frames):
        positions[i] = (subject_pos + profile[i] * dir_vec
                        + px[i] * right + py[i] * up_p)

    # ---- look-at 目标也加抖动，使视线轻微"漂" ----
    la = SEG2_LOOKAT_JITTER
    lt_x = la * smooth_random_signal(n_frames, 3, SEG2_LOOKAT_FREQ, rng)
    lt_y = la * smooth_random_signal(n_frames, 3, SEG2_LOOKAT_FREQ, rng)
    lt_z = la * smooth_random_signal(n_frames, 3, SEG2_LOOKAT_FREQ, rng)
    look_targets = subject_pos[None] + np.column_stack([lt_x, lt_y, lt_z])

    # ---- 计算每帧距离 & Hitchcock 公式给出焦距 ----
    distances = np.linalg.norm(positions - subject_pos, axis=1)
    focals = fx_ref * distances / d_ref     # 关键公式
    return positions, focals.astype(np.float32), look_targets.astype(np.float32)


# ================================================================
# 主生成
# ================================================================

def generate_one_video(output_path, seed=0):
    rng = np.random.RandomState(seed)
    seg3_frames = TOTAL_FRAMES - SEG1_FRAMES - SEG2_FRAMES

    # 段 1：随机
    pos1 = gen_segment_random(SEG1_FRAMES, SUBJECT_POS, rng=rng)
    # 段 2：希区柯克 (接上一段终点)
    pos2, focals2, lt2 = gen_segment_hitchcock(
        SEG2_FRAMES, SUBJECT_POS, pos1[-1], FX_BASE, rng=rng)
    # 段 3：随机 (接段 2 终点)
    pos3 = gen_segment_random(seg3_frames, SUBJECT_POS,
                               start_pos=pos2[-1], rng=rng)

    # 拼接
    all_pos = np.vstack([pos1, pos2, pos3]).astype(np.float32)
    all_fx  = np.concatenate([
        np.full(SEG1_FRAMES, FX_BASE, dtype=np.float32),
        focals2,
        np.full(seg3_frames, FX_BASE, dtype=np.float32),
    ])
    all_targets = np.zeros((TOTAL_FRAMES, 3), dtype=np.float32)
    all_targets[:SEG1_FRAMES] = SUBJECT_POS
    all_targets[SEG1_FRAMES:SEG1_FRAMES+SEG2_FRAMES] = lt2
    all_targets[SEG1_FRAMES+SEG2_FRAMES:] = SUBJECT_POS

    # 生成每帧矩阵
    extrinsics = np.zeros((TOTAL_FRAMES, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((TOTAL_FRAMES, 3, 3), dtype=np.float32)
    for i in range(TOTAL_FRAMES):
        extrinsics[i] = look_at_w2c(all_pos[i], all_targets[i])
        intrinsics[i] = make_K(all_fx[i])

    R_cube = euler_to_R(CUBE_YAW_DEG, CUBE_PITCH_DEG, 0)
    seg_bounds = np.array([SEG1_FRAMES, SEG1_FRAMES + SEG2_FRAMES],
                           dtype=np.int32)

    np.savez(
        output_path,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        effect='hitchcock',
        subject_world_pos=SUBJECT_POS,
        fx_reference=np.float32(FX_BASE),
        cube_half_size=np.float32(CUBE_HALF_SIZE),
        cube_orientation=R_cube,
        segment_boundaries=seg_bounds,
        camera_model='pinhole',
    )

    d_seg2 = np.linalg.norm(pos2 - SUBJECT_POS, axis=1)
    print(f"  ✅ {os.path.basename(output_path)}: "
          f"seg2 d∈[{d_seg2.min():.2f},{d_seg2.max():.2f}]  "
          f"fx∈[{focals2.min():.0f},{focals2.max():.0f}]")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print(f"🎬 Generating {NUM_VIDEOS} Hitchcock NPZ files")
    print(f"   {TOTAL_FRAMES} frames @ {FPS} fps = {TOTAL_FRAMES/FPS:.1f}s each")
    print(f"   seg1={SEG1_FRAMES} | seg2={SEG2_FRAMES} | "
          f"seg3={TOTAL_FRAMES-SEG1_FRAMES-SEG2_FRAMES}")
    print("=" * 60)
    for i in range(NUM_VIDEOS):
        out = os.path.join(OUTPUT_DIR, f"hitchcock_{i:03d}.npz")
        generate_one_video(out, seed=42 + i)
    print(f"\n✅ Done -> {OUTPUT_DIR}")