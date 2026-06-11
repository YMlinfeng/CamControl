"""
增强版相机位姿可视化工具
-----------------------
新增能力：
  1. 鱼眼畸变 (equidistant fisheye)：网格线被正确画弯
  2. 希区柯克变焦 (dolly zoom)：主体大小不变 + 背景剧烈伸缩
  3. NPZ 生成器：批量产出测试数据
向后兼容：旧 npz (只有 extrinsics 或 + intrinsics) 行为不变
"""
import cv2
import numpy as np
import os
import subprocess
import sys, json
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import cKDTree


# ==================================================================
# Part 1: NPZ 生成器  —  构造鱼眼 / 希区柯克的合成相机参数
# ==================================================================

def _build_w2c(rot_c2w, cam_pos_world):
    """根据 c2w 的旋转和相机世界坐标，构造 w2c (3x4)."""
    R_w2c = rot_c2w.T
    t_w2c = -R_w2c @ cam_pos_world
    return np.hstack([R_w2c, t_w2c.reshape(-1, 1)])


def generate_fisheye_npz(output_path, n_frames=125, w=512, h=288,
                         motion='pan', distortion_strength='medium'):
    """生成鱼眼畸变 npz。motion ∈ {pan, forward, tilt}."""
    extrinsics = np.zeros((n_frames, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((n_frames, 3, 3), dtype=np.float32)

    # 鱼眼畸变系数 (equidistant model: k1,k2,k3,k4)
    coef_map = {
        'weak':   np.array([-0.10,  0.01,  0.000,  0.0000]),
        'medium': np.array([-0.25,  0.05, -0.005,  0.0005]),
        'strong': np.array([-0.40,  0.15, -0.020,  0.0020]),
    }
    dist_coef = coef_map[distortion_strength]
    distortion = np.tile(dist_coef, (n_frames, 1)).astype(np.float64)

    # 鱼眼镜头通常焦距很小（FOV 极大）
    fx = fy = w * 0.5
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        if motion == 'pan':
            rot = R.from_euler('y', (t - 0.5) * np.pi / 2).as_matrix()  # ±45°
            pos = np.array([0, 0, t * 2.0])
        elif motion == 'forward':
            rot = np.eye(3)
            pos = np.array([0, 0, t * 8.0])
        elif motion == 'tilt':
            rot = R.from_euler('x', (t - 0.5) * np.pi / 3).as_matrix()
            pos = np.array([0, 0, t * 2.0])
        else:
            rot, pos = np.eye(3), np.zeros(3)

        extrinsics[i] = _build_w2c(rot, pos)
        intrinsics[i] = K

    np.savez(output_path,
             extrinsics=extrinsics,
             intrinsics=intrinsics,
             distortion=distortion,
             camera_model='fisheye')


def generate_hitchcock_npz(output_path, n_frames=125, w=512, h=288,
                           direction='dolly_in', subject_distance=10.0,
                           subject_size=1.5):
    """
    生成希区柯克变焦 (dolly zoom) npz。
    原理：相机沿 z 推进时，焦距按距离比例反向变化，
          这样主体投影像素大小保持恒定，背景被剧烈拉扯。
    """
    extrinsics = np.zeros((n_frames, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((n_frames, 3, 3), dtype=np.float32)

    # 主体放在初始相机正前方 subject_distance 处
    subject_world_pos = np.array([0.0, 0.0, subject_distance])

    fx_initial = w * 1.5     # 初始焦距 (较长焦)
    cx, cy = w / 2.0, h / 2.0

    if direction == 'dolly_in':
        end_distance = subject_distance * 0.30   # 推到 30%
    else:
        end_distance = subject_distance * 2.50   # 拉到 250%

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        cur_d = subject_distance + t * (end_distance - subject_distance)
        cam_pos = np.array([0.0, 0.0, subject_distance - cur_d])

        # 关键公式：fx * size / dist = const  =>  fx ∝ dist
        scale = cur_d / subject_distance
        fx_cur = fx_initial * scale

        extrinsics[i] = _build_w2c(np.eye(3), cam_pos)
        intrinsics[i] = np.array([[fx_cur, 0, cx],
                                  [0, fx_cur, cy],
                                  [0, 0, 1]], dtype=np.float32)

    np.savez(output_path,
             extrinsics=extrinsics,
             intrinsics=intrinsics,
             camera_model='pinhole',
             effect='hitchcock',
             subject_world_pos=subject_world_pos,
             subject_size=subject_size)


def generate_all_test_npz(output_dir, n_each=10):
    os.makedirs(output_dir, exist_ok=True)
    motions = ['pan', 'forward', 'tilt']
    strengths = ['weak', 'medium', 'strong']
    for i in range(n_each):
        m = motions[i % 3]
        s = strengths[(i // 3) % 3]
        p = os.path.join(output_dir, f'fisheye_{i:02d}_{m}_{s}.npz')
        generate_fisheye_npz(p, motion=m, distortion_strength=s)
        print(f"  💾 [fisheye]  {p}")

    for i in range(n_each):
        d = 'dolly_in' if i % 2 == 0 else 'dolly_out'
        dist = 8.0 + (i % 5) * 2.0
        p = os.path.join(output_dir, f'hitchcock_{i:02d}_{d}_d{dist:.0f}.npz')
        generate_hitchcock_npz(p, direction=d, subject_distance=dist)
        print(f"  💾 [hitchcock] {p}")
    print(f"\n✅ 共生成 {2*n_each} 个测试 npz -> {output_dir}\n")


# ==================================================================
# Part 2: 立方体辅助 (希区柯克主体标识)
# ==================================================================

def make_cube_lines(center, size, with_diagonals=True):
    cx, cy, cz = center; s = size / 2.0
    v = np.array([
        [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s],
        [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s],
        [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    if with_diagonals:
        edges += [(0,6),(1,7),(2,4),(3,5)]   # 空间对角线，增强视觉醒目度
    p1 = np.array([v[e[0]] for e in edges], dtype=np.float32)
    p2 = np.array([v[e[1]] for e in edges], dtype=np.float32)
    return p1, p2


# ==================================================================
# Part 3: 投影函数 (支持 pinhole / fisheye)
# ==================================================================

def project_points_vec(pts_3d, K, distortion=None, model='pinhole'):
    """向量化 3D->2D 投影，支持等距鱼眼模型。"""
    Z = np.maximum(pts_3d[:, 2], 1e-6)
    x_n = pts_3d[:, 0] / Z
    y_n = pts_3d[:, 1] / Z

    if model == 'fisheye' and distortion is not None:
        # equidistant fisheye: theta_d = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
        r = np.sqrt(x_n * x_n + y_n * y_n)
        theta = np.arctan(r)
        theta = np.clip(theta, 0, np.pi / 2 * 0.98)   # 防止极端边缘溢出
        k1, k2, k3, k4 = distortion[0], distortion[1], distortion[2], distortion[3]
        t2 = theta * theta
        theta_d = theta * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)
        scale = np.where(r > 1e-8, theta_d / np.maximum(r, 1e-8), 1.0)
        u = K[0, 0] * (x_n * scale) + K[0, 2]
        v = K[1, 1] * (y_n * scale) + K[1, 2]
    else:
        u = K[0, 0] * x_n + K[0, 2]
        v = K[1, 1] * y_n + K[1, 2]
    return np.column_stack((u, v))


# ==================================================================
# Part 4: 裁剪 + 投影 (统一支持直线/曲线)
# ==================================================================

def clip_and_project_lines(p1_c, p2_c, colors, K, max_dist,
                            distortion=None, model='pinhole', n_samples=1):
    """
    z_near 裁剪后投影。
      n_samples=1  -> 返回 (proj_p1(N,2), proj_p2(N,2), alpha, colors)，画直线
      n_samples>1  -> 返回 (polylines(N,S,2), None, alpha, colors)，画折线 (鱼眼)
    """
    z1, z2 = p1_c[:, 2], p2_c[:, 2]
    z_near = 0.1
    both_front  = (z1 >= z_near) & (z2 >= z_near)
    both_behind = (z1 <  z_near) & (z2 <  z_near)
    intersect   = ~(both_front | both_behind)

    vp1_l, vp2_l, vc_l = [], [], []
    if np.any(both_front):
        vp1_l.append(p1_c[both_front]); vp2_l.append(p2_c[both_front])
        vc_l.append(colors[both_front])

    if np.any(intersect):
        p1i, p2i = p1_c[intersect], p2_c[intersect]
        z1i, z2i = p1i[:, 2], p2i[:, 2]
        t = (z_near - z1i) / (z2i - z1i + 1e-6)
        p_clip = p1i + t[:, None] * (p2i - p1i)
        new_p1 = np.where((z1i < z_near)[:, None], p_clip, p1i)
        new_p2 = np.where((z2i < z_near)[:, None], p_clip, p2i)
        vp1_l.append(new_p1); vp2_l.append(new_p2)
        vc_l.append(colors[intersect])

    if not vp1_l:
        return None, None, None, None

    vp1 = np.vstack(vp1_l).astype(np.float32)
    vp2 = np.vstack(vp2_l).astype(np.float32)
    vc  = np.vstack(vc_l)

    mid = (vp1 + vp2) / 2.0
    alpha = np.clip(1.0 - np.linalg.norm(mid, axis=1) / max_dist, 0.0, 1.0)

    if n_samples <= 1:
        pts = np.vstack((vp1, vp2))
        pts_2d = project_points_vec(pts, K, distortion, model)
        half = len(vp1)
        return pts_2d[:half], pts_2d[half:], alpha, vc
    else:
        # 把每条线段在 3D 内均匀采样 n_samples 个点，逐点畸变投影
        ts = np.linspace(0, 1, n_samples).reshape(1, n_samples, 1)
        sampled = vp1[:, None, :] * (1 - ts) + vp2[:, None, :] * ts   # (N, S, 3)
        flat = sampled.reshape(-1, 3)
        proj = project_points_vec(flat, K, distortion, model)
        return proj.reshape(len(vp1), n_samples, 2), None, alpha, vc


# ==================================================================
# Part 5: 主渲染函数 (完全向后兼容 + 新能力)
# ==================================================================

def render_camera_trajectory_to_numpy(
    pose_path: str, n: int, w: int, h: int, stride: int,
    target_indices: list = None
):
    data = np.load(pose_path, allow_pickle=True)
    keys = set(data.files) if hasattr(data, 'files') else set(data.keys())

    # ---- 加载 extrinsics (原版逻辑) ----
    poses_raw = None
    for k in ["poses","data","c2w","w2c","cam_poses","camera_poses",
              "extrinsic","cams","extrinsics"]:
        if k in keys: poses_raw = data[k]; break
    if poses_raw is None:
        for k in keys:
            if np.asarray(data[k]).ndim >= 2:
                poses_raw = data[k]; break

    poses = np.asarray(poses_raw, dtype=np.float32)
    if poses.ndim == 2:
        if poses.shape[1] == 16: poses = poses.reshape(-1, 4, 4)
        elif poses.shape[1] == 12: poses = poses.reshape(-1, 3, 4)
    if poses.ndim == 3 and poses.shape[1:] == (3, 4):
        bottom = np.broadcast_to(np.array([0,0,0,1], dtype=np.float32),
                                 (poses.shape[0], 1, 4))
        poses = np.concatenate([poses, bottom], axis=1)
    poses = np.linalg.inv(poses)

    # ---- 【新增】加载 intrinsics / distortion / camera_model / subject ----
    intrinsics_raw = data['intrinsics'] if 'intrinsics' in keys else None
    distortion_raw = data['distortion'] if 'distortion' in keys else None
    camera_model   = str(data['camera_model']) if 'camera_model' in keys else 'pinhole'
    effect         = str(data['effect']) if 'effect' in keys else None
    subj_pos  = np.asarray(data['subject_world_pos']) if 'subject_world_pos' in keys else None
    subj_size = float(data['subject_size']) if 'subject_size' in keys else 1.5

    num_poses = len(poses)
    max_valid = (num_poses - 1) * stride
    n_render = min(n, max_valid + 1)
    if n_render <= 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8), 0
    render_list = target_indices if target_indices is not None else list(range(n_render))
    out_video = np.zeros((len(render_list), h, w, 3), dtype=np.uint8)

    # ---- 轨迹插值 ----
    t_orig = np.arange(num_poses) * stride
    t_tgt  = np.arange(n_render)
    if num_poses == 1:
        poses_dense = np.repeat(poses, n_render, axis=0)
    else:
        t_interp = interp1d(t_orig, poses[:, :3, 3], axis=0)(t_tgt)
        r_interp = Slerp(t_orig, R.from_matrix(poses[:, :3, :3]))(t_tgt).as_matrix()
        poses_dense = np.zeros((n_render, 4, 4), dtype=np.float32)
        poses_dense[:, 3, 3] = 1.0
        poses_dense[:, :3, :3] = r_interp
        poses_dense[:, :3, 3]  = t_interp

    # ---- 【新增】per-frame 内参 / 畸变插值 ----
    if intrinsics_raw is not None:
        ir = np.asarray(intrinsics_raw, dtype=np.float32)
        if ir.ndim == 2: ir = ir[None]
        if num_poses == 1 or ir.shape[0] == 1:
            K_dense = np.repeat(ir, n_render, axis=0)
        else:
            K_flat = ir.reshape(num_poses, 9)
            K_dense = interp1d(t_orig, K_flat, axis=0)(t_tgt).reshape(n_render, 3, 3).astype(np.float32)
    else:
        # 完全保持原版默认 K
        f = max(w, h) * 0.8
        K_def = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
        K_dense = np.tile(K_def[None], (n_render, 1, 1))

    if distortion_raw is not None:
        dr = np.asarray(distortion_raw, dtype=np.float32)
        if dr.ndim == 1:
            distortion_dense = np.tile(dr[None], (n_render, 1))
        elif num_poses == 1 or dr.shape[0] == 1:
            distortion_dense = np.repeat(dr, n_render, axis=0)
        else:
            distortion_dense = interp1d(t_orig, dr, axis=0)(t_tgt).astype(np.float32)
    else:
        distortion_dense = None

    # ---- 3D 网格生成 (原版逻辑) ----
    trans = poses_dense[:, :3, 3]
    diffs = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
    valid_diffs = diffs[diffs > 1e-5]
    grid_step = max(np.median(valid_diffs) * 10.0, 0.5) if len(valid_diffs) > 0 else 0.5
    max_view_dist = max(grid_step * 15.0, 15.0)

    # 【新增】扩展可视范围以包含 subject
    if subj_pos is not None:
        d_sub = np.linalg.norm(trans - subj_pos[None], axis=1)
        max_view_dist = max(max_view_dist, float(np.max(d_sub)) * 1.4)

    anchor = np.vstack([trans, subj_pos.reshape(1, 3)]) if subj_pos is not None else trans
    min_xyz, max_xyz = np.min(anchor, axis=0), np.max(anchor, axis=0)
    margin = max(max_view_dist, 10.0)
    x_coords = np.arange(min_xyz[0] - margin, max_xyz[0] + margin, grid_step)
    z_coords = np.arange(min_xyz[2] - margin, max_xyz[2] + margin, grid_step)
    avg_y = np.median(trans[:, 1])
    floor_y = avg_y + max(grid_step * 3.0, 2.0)
    ceil_y  = avg_y - max(grid_step * 3.0, 2.0)

    traj_xz_list = [trans[::5, [0, 2]]] if len(trans) > 0 else [np.array([[0.0, 0.0]])]
    if subj_pos is not None:
        traj_xz_list.append(subj_pos[[0, 2]].reshape(1, 2))
    traj_xz = np.vstack(traj_xz_list)
    tree = cKDTree(traj_xz)
    xx, zz = np.meshgrid(x_coords, z_coords)
    pts_xz = np.c_[xx.ravel(), zz.ravel()]
    dists, _ = tree.query(pts_xz)
    valid_mask_2d = (dists < max(grid_step * 8.0, 8.0)).reshape(len(z_coords), len(x_coords))

    color_x, color_z, color_y = (255, 255, 0), (255, 0, 255), (0, 255, 255)
    g_p1, g_p2, g_c = [], [], []

    # X 方向线
    mask_x = valid_mask_2d[:, :-1] | valid_mask_2d[:, 1:]
    zi, xi = np.where(mask_x)
    if len(zi) > 0:
        x1, x2, z = x_coords[xi], x_coords[xi+1], z_coords[zi]
        p1 = np.column_stack((x1, np.full_like(x1, floor_y), z))
        p2 = np.column_stack((x2, np.full_like(x2, floor_y), z))
        p3 = np.column_stack((x1, np.full_like(x1, ceil_y), z))
        p4 = np.column_stack((x2, np.full_like(x2, ceil_y), z))
        g_p1.extend([p1, p3]); g_p2.extend([p2, p4])
        g_c.append(np.tile(color_x, (len(p1)*2, 1)))

    # Z 方向线
    mask_z = valid_mask_2d[:-1, :] | valid_mask_2d[1:, :]
    zi, xi = np.where(mask_z)
    if len(zi) > 0:
        x, z1, z2 = x_coords[xi], z_coords[zi], z_coords[zi+1]
        p1 = np.column_stack((x, np.full_like(x, floor_y), z1))
        p2 = np.column_stack((x, np.full_like(x, floor_y), z2))
        p3 = np.column_stack((x, np.full_like(x, ceil_y), z1))
        p4 = np.column_stack((x, np.full_like(x, ceil_y), z2))
        g_p1.extend([p1, p3]); g_p2.extend([p2, p4])
        g_c.append(np.tile(color_z, (len(p1)*2, 1)))

    # Y 方向墙
    tunnel_r = max(grid_step * 3.5, 4.0)
    wall_mask = (dists > tunnel_r) & (dists < tunnel_r + max(grid_step * 1.2, 1.0))
    wp = pts_xz[wall_mask]
    if len(wp) > 0:
        p1 = np.column_stack((wp[:, 0], np.full(len(wp), ceil_y), wp[:, 1]))
        p2 = np.column_stack((wp[:, 0], np.full(len(wp), floor_y), wp[:, 1]))
        g_p1.append(p1); g_p2.append(p2)
        g_c.append(np.tile(color_y, (len(p1), 1)))

    # ---- 【新增】Hitchcock 主体立方体 ----
    if effect == 'hitchcock' and subj_pos is not None:
        c1, c2 = make_cube_lines(subj_pos, subj_size, with_diagonals=True)
        g_p1.append(c1); g_p2.append(c2)
        g_c.append(np.tile((80, 255, 80), (len(c1), 1)))   # 鲜艳绿色

    if not g_p1:
        return out_video, n_render

    grid_p1 = np.vstack(g_p1).astype(np.float32)
    grid_p2 = np.vstack(g_p2).astype(np.float32)
    grid_colors = np.vstack(g_c).astype(np.float32)
    grid_p1_h = np.hstack((grid_p1, np.ones((len(grid_p1), 1), dtype=np.float32)))
    grid_p2_h = np.hstack((grid_p2, np.ones((len(grid_p2), 1), dtype=np.float32)))

    # ---- 渲染循环 ----
    base_thick = max(4, int(w / 320))
    w2cs_all = np.linalg.inv(poses_dense)
    # 【关键】鱼眼用 12 段折线，pinhole 保持直线
    n_samples = 12 if camera_model == 'fisheye' else 1

    for out_idx, frame_idx in enumerate(render_list):
        if frame_idx >= len(w2cs_all): break
        w2c = w2cs_all[frame_idx]
        K_f = K_dense[frame_idx]
        d_f = distortion_dense[frame_idx] if distortion_dense is not None else None

        gp1_c = (grid_p1_h @ w2c.T)[:, :3]
        gp2_c = (grid_p2_h @ w2c.T)[:, :3]
        mid_c = (gp1_c + gp2_c) / 2.0
        valid = (mid_c[:, 2] > -max(grid_step, 1.0)) & \
                (np.linalg.norm(mid_c, axis=1) < max_view_dist)
        if not np.any(valid): continue

        res = clip_and_project_lines(
            gp1_c[valid], gp2_c[valid], grid_colors[valid],
            K_f, max_view_dist, distortion=d_f, model=camera_model,
            n_samples=n_samples)
        if res[0] is None: continue

        frame = out_video[out_idx]

        if n_samples <= 1:
            # ===== Pinhole: 直线绘制 (原版逻辑) =====
            p1s, p2s, alphas, cols = res
            mask_a = alphas >= 0.05
            if not np.any(mask_a): continue
            p1s = p1s[mask_a].astype(np.int32)
            p2s = p2s[mask_a].astype(np.int32)
            alphas = alphas[mask_a]
            cols   = cols[mask_a]
            draw_c = (cols * alphas[:, None]).astype(np.int32)
            draw_t = np.maximum(2, (base_thick * alphas).astype(np.int32))
            np.clip(p1s, -w*5, w*5, out=p1s)
            np.clip(p2s, -w*5, w*5, out=p2s)
            for p1, p2, c, t in zip(p1s, p2s, draw_c, draw_t):
                cv2.line(frame, tuple(p1), tuple(p2), c.tolist(), int(t), cv2.LINE_AA)
        else:
            # ===== Fisheye: 折线绘制 (新逻辑) =====
            polylines, _, alphas, cols = res
            mask_a = alphas >= 0.05
            if not np.any(mask_a): continue
            polys = polylines[mask_a].astype(np.int32)
            alphas = alphas[mask_a]
            cols   = cols[mask_a]
            draw_c = (cols * alphas[:, None]).astype(np.int32)
            draw_t = np.maximum(2, (base_thick * alphas).astype(np.int32))
            np.clip(polys, -w*5, w*5, out=polys)
            for poly, c, t in zip(polys, draw_c, draw_t):
                cv2.polylines(frame, [poly], False, c.tolist(), int(t), cv2.LINE_AA)

    return out_video, n_render


# ==================================================================
# Part 6: 纯位姿可视化输出 (无原视频叠加)
# ==================================================================

def write_pose_only_video(pose_path, output_path, n_frames=125, w=512, h=288,
                           fps=24, stride=1):
    print(f"\n[Pure Viz] {os.path.basename(pose_path)}")
    arr, n_render = render_camera_trajectory_to_numpy(pose_path, n_frames, w, h, stride)
    if n_render == 0:
        print("  ⚠️ 无可渲染帧"); return
    tmp = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for i in range(len(arr)): out.write(arr[i])
    out.release()
    ffmpeg = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
    cmd = [ffmpeg, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        if os.path.exists(tmp): os.remove(tmp)
        print(f"  ✅ 已保存: {output_path}")
    except subprocess.CalledProcessError:
        print(f"  ⚠️ FFmpeg 失败，保留临时文件 {tmp}")


# ==================================================================
# Part 7: 原版的 parse / process (完全保留，向后兼容)
# ==================================================================

def parse_line(line):
    line = line.strip()
    if not line: return None, None
    a = line.find(' '); b = line.rfind(' ')
    if a == -1 or b == -1 or a == b: return None, None
    return line[a+1:b].strip(), line[b+1:].strip()


def parse_single_line(line: str):
    line = line.strip()
    if not line: return None, None, None
    parts = line.split(' ', 2)
    if len(parts) != 3:
        raise ValueError(f"行格式异常: {line}")
    try:
        frames_list = json.loads(parts[2])
    except json.JSONDecodeError:
        frames_list = []
    return parts[0], parts[1], frames_list


def process_single_video(video_path, pose_path, output_path, stride=2, cut_points=[]):
    """原版逻辑：叠加到真实视频上 (用于真实数据)"""
    print(f"\n[{video_path}]\n  -> pose: {pose_path}")
    if not os.path.exists(video_path) or not os.path.exists(pose_path):
        print("  ❌ 缺失文件，跳过"); return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  ❌ 视频打不开"); return
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        pose_arr, n_render = render_camera_trajectory_to_numpy(pose_path, n, w, h, stride)
    except Exception as e:
        print(f"  ❌ 渲染失败: {e}"); cap.release(); return

    tmp = output_path.replace(".mp4", "_temp.mp4")
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cp_i = 0
    for i in range(len(pose_arr)):
        ret, rgb = cap.read()
        if not ret: rgb = np.zeros((h, w, 3), dtype=np.uint8)
        pose_frame = pose_arr[i]
        if cut_points and cp_i < len(cut_points):
            if i >= cut_points[cp_i][1]:
                cp_i += 1
            elif i > cut_points[cp_i][1] - stride:
                pose_frame = np.zeros((h, w, 3), dtype=np.uint8)
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
        combined = cv2.add(cv2.convertScaleAbs(rgb, alpha=0.6), pose_frame) \
                   if i < n_render else np.zeros((h, w, 3), dtype=np.uint8)
        out.write(combined)
    cap.release(); out.release()
    ffmpeg = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
    cmd = [ffmpeg, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        if os.path.exists(tmp): os.remove(tmp)
        print(f"  🎉 保存: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ FFmpeg 失败: {e}")


# ==================================================================
# Part 8: 主入口
# ==================================================================

if __name__ == "__main__":
    NPZ_DIR = "./synthetic_npz"
    VIZ_DIR = "./viz_output"
    os.makedirs(NPZ_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    print("=" * 60)
    print("🧪 Step 1: 生成 10 鱼眼 + 10 希区柯克 测试 NPZ")
    print("=" * 60)
    # generate_all_test_npz(NPZ_DIR, n_each=10)

    print("=" * 60)
    print("🎨 Step 2: 渲染纯位姿可视化")
    print("=" * 60)
    for fname in sorted(os.listdir(NPZ_DIR)):
        if not fname.endswith('.npz'): continue
        write_pose_only_video(
            os.path.join(NPZ_DIR, fname),
            os.path.join(VIZ_DIR, fname.replace('.npz', '_viz.mp4')),
            n_frames=125, w=512, h=288, fps=24, stride=1
        )
    print(f"\n✅ 全部完成！请到 {VIZ_DIR} 查看 20 个可视化视频。")