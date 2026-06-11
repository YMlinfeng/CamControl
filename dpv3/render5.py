"""
================================================================
visualize.py  v2  —— 真 3D 立方体版
================================================================
本版相对上一版的核心升级：
  ❌ 旧：用 cv2.rectangle 硬画一个固定像素大小的绿色方框
        (任何 fx/distance 都画相同大小 → 不能算"渲染"，只是"宣称")
  ✅ 新：用 8 个世界顶点 → w2c → K 投影 → cv2.line 画 12 条边
        立方体在屏幕上的大小完全是"投影几何"决定的：
            apparent ≈ fx * cube_size / distance
        如果 Hitchcock 算对了 (f ∝ d)，则 apparent ≡ const
        否则就会变 → 失败时一眼能看到，不会被"硬画"骗过去

附加安全：
  - 立方体若被推拉摇移甩出画外，会在画面边缘画一个绿色箭头指示位置
  - 左上角实时显示 当前段名 / fx / ratio / 距离 / 立方体测得屏幕边长
  - 段 2 时段名标红，配合"立方体大小"的恒定即可直接做视觉验收

向后兼容：
  - 无 cube_half_size / cube_orientation 的旧 npz 用默认值
  - pinhole / fisheye 全支持
================================================================
"""

import os, sys, cv2, subprocess
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import cKDTree


# ================================================================
# 🔧 CONFIG
# ================================================================
NPZ_DIR    = "./hitchcock_npz"
NPZ_DIR    = "/m2v_intern/mengzijie/depthanythingv3/synthetic_npz"
OUTPUT_DIR = "./viz_output"

# ---- 视频基本参数 ----
RENDER_W      = 512
RENDER_H      = 288
RENDER_FPS    = 24
RENDER_N      = 9999     # 不设上限；npz 多少帧就渲多少
RENDER_STRIDE = 1

# ---- 鱼眼采样精度 ----
FISHEYE_LINE_SAMPLES = 12

# ---- Hitchcock 立方体外观 ----
HITCHCOCK_CUBE_COLOR        = (80, 255, 80)   # BGR 鲜绿
                                              # 调成 (0,200,0) 暗绿等都可以
HITCHCOCK_CUBE_THICKNESS    = 2               # 普通边的粗细
HITCHCOCK_CUBE_ACCENT_COLOR = (200, 255, 200) # "前面"四边的强调色
HITCHCOCK_CUBE_ACCENT_THICK = 3               # 前面四边粗一点 → 3D 感更明显
HITCHCOCK_USE_ACCENT_EDGES  = True            # 关掉 → 12 条边同色同粗

# ---- Hitchcock UI ----
HITCHCOCK_SHOW_INFO            = True   # 左上角文字信息
HITCHCOCK_SHOW_OFFSCREEN_ARROW = True   # 立方体出画时画箭头
HITCHCOCK_SEGMENT_NAMES = ["Random Move", "HITCHCOCK ZOOM", "Random Move"]

# ---- 旧 npz 缺字段时的默认值 ----
HITCHCOCK_DEFAULT_HALF_SIZE = 0.6
HITCHCOCK_DEFAULT_YAW_DEG   = 25.0
HITCHCOCK_DEFAULT_PITCH_DEG = 15.0

# ---- 背景隧道网格 (沿用上一版参数；注释见上一版) ----
GRID_STEP_FACTOR        = 10.0
GRID_MIN_STEP           = 0.5
GRID_MAX_VIEW_FACTOR    = 15.0
GRID_FLOOR_STEP_FACTOR  = 3.0
GRID_FLOOR_MIN_HEIGHT   = 2.0
GRID_CEIL_STEP_FACTOR   = 3.0
GRID_CEIL_MIN_HEIGHT    = 2.0
GRID_TUNNEL_R_FACTOR    = 3.5
GRID_ALPHA_THRESH       = 0.05
COLOR_X = (255, 255, 0)
COLOR_Z = (255, 0, 255)
COLOR_Y = (0, 255, 255)


# ================================================================
# Part A : 投影 (pinhole / fisheye)
# ================================================================

def project_points_vec(pts_3d, K, distortion=None, model='pinhole'):
    Z = np.maximum(pts_3d[:, 2], 1e-6)
    x_n = pts_3d[:, 0] / Z
    y_n = pts_3d[:, 1] / Z
    if model == 'fisheye' and distortion is not None:
        r = np.sqrt(x_n*x_n + y_n*y_n)
        theta = np.clip(np.arctan(r), 0, np.pi/2 * 0.98)
        k1, k2, k3, k4 = distortion[:4]
        t2 = theta*theta
        theta_d = theta * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)
        scale = np.where(r > 1e-8, theta_d / np.maximum(r, 1e-8), 1.0)
        u = K[0,0] * (x_n * scale) + K[0,2]
        v = K[1,1] * (y_n * scale) + K[1,2]
    else:
        u = K[0,0] * x_n + K[0,2]
        v = K[1,1] * y_n + K[1,2]
    return np.column_stack((u, v))


def clip_and_project_lines(p1_c, p2_c, colors, K, max_dist,
                            distortion=None, model='pinhole', n_samples=1):
    """通用线段近平面裁剪 + 投影。立方体边和网格线都走这里。"""
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
        proj = project_points_vec(pts, K, distortion, model)
        half = len(vp1)
        return proj[:half], proj[half:], alpha, vc
    else:
        ts = np.linspace(0, 1, n_samples).reshape(1, n_samples, 1)
        sampled = vp1[:, None, :] * (1 - ts) + vp2[:, None, :] * ts
        flat = sampled.reshape(-1, 3)
        proj = project_points_vec(flat, K, distortion, model)
        return proj.reshape(len(vp1), n_samples, 2), None, alpha, vc


# ================================================================
# Part B : 真 3D 立方体渲染  ← 本版重点
# ================================================================

def _default_cube_R():
    y = np.deg2rad(HITCHCOCK_DEFAULT_YAW_DEG)
    p = np.deg2rad(HITCHCOCK_DEFAULT_PITCH_DEG)
    Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]], np.float32)
    Rx = np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]], np.float32)
    return (Ry @ Rx).astype(np.float32)


def draw_hitchcock_cube_3d(frame, w2c, K, distortion, model,
                            cube_center, cube_half_size, cube_rotation):
    """
    真正 3D 投影一个立方体的 12 条边。
    ⚠️ 注意：这里完全没有任何"强制大小"的代码——
        屏幕大小完全由 K · (w2c · vertex) 这个数学过程决定。
    Hitchcock 算对了 → 投影大小恒定。算错了 → 大小立刻变化。

    Returns: 立方体在屏幕上的包围盒最大边长 (像素)，供 UI 显示
    """
    s = float(cube_half_size)

    # ---- 1. 立方体 local 坐标系的 8 顶点 ----
    #   索引约定: 0..3 = z=-s 后面, 4..7 = z=+s 前面
    verts_local = np.array([
        [-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],
        [-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s],
    ], dtype=np.float32)

    # ---- 2. local → world (姿态 + 平移) ----
    R_cube = np.asarray(cube_rotation, dtype=np.float32) \
             if cube_rotation is not None else np.eye(3, dtype=np.float32)
    verts_world = verts_local @ R_cube.T + np.asarray(cube_center, dtype=np.float32)[None]

    # ---- 3. 12 条边 (分三组方便上色) ----
    edges_back  = [(0,1),(1,2),(2,3),(3,0)]   # local "back" 面
    edges_front = [(4,5),(5,6),(6,7),(7,4)]   # local "front" 面
    edges_side  = [(0,4),(1,5),(2,6),(3,7)]   # 连接前后

    # ---- 4. world → camera ----
    verts_h = np.column_stack([verts_world, np.ones(8, dtype=np.float32)])
    verts_cam = (np.asarray(w2c, dtype=np.float32) @ verts_h.T).T[:, :3]

    h_img, w_img = frame.shape[:2]
    bound = max(w_img, h_img) * 8     # 投影点限幅，防止离群点击穿
    n_samples = FISHEYE_LINE_SAMPLES if model == 'fisheye' else 1

    def _draw(edge_list, color, thick):
        """裁剪 + 投影 + 画一组边 (pinhole 画直线，fisheye 画折线)。"""
        idx = np.array(edge_list, dtype=np.int32)
        p1 = verts_cam[idx[:, 0]]
        p2 = verts_cam[idx[:, 1]]
        cols = np.tile(np.array(color, dtype=np.float32), (len(edge_list), 1))
        res = clip_and_project_lines(p1, p2, cols, K, max_dist=1e10,
                                      distortion=distortion, model=model,
                                      n_samples=n_samples)
        if res[0] is None:
            return
        if n_samples <= 1:
            p1s, p2s, _, _ = res
            p1s = np.clip(p1s, -bound, bound).astype(np.int32)
            p2s = np.clip(p2s, -bound, bound).astype(np.int32)
            for pp1, pp2 in zip(p1s, p2s):
                cv2.line(frame, tuple(pp1), tuple(pp2), color, thick, cv2.LINE_AA)
        else:
            polys, _, _, _ = res
            polys = np.clip(polys, -bound, bound).astype(np.int32)
            for poly in polys:
                cv2.polylines(frame, [poly], False, color, thick, cv2.LINE_AA)

    # ---- 5. 画 12 条边 (后/侧→前 的顺序，前面边盖在上层) ----
    if HITCHCOCK_USE_ACCENT_EDGES:
        _draw(edges_back,  HITCHCOCK_CUBE_COLOR,        HITCHCOCK_CUBE_THICKNESS)
        _draw(edges_side,  HITCHCOCK_CUBE_COLOR,        HITCHCOCK_CUBE_THICKNESS)
        _draw(edges_front, HITCHCOCK_CUBE_ACCENT_COLOR, HITCHCOCK_CUBE_ACCENT_THICK)
    else:
        _draw(edges_back + edges_side + edges_front,
              HITCHCOCK_CUBE_COLOR, HITCHCOCK_CUBE_THICKNESS)

    # ---- 6. 顺便测量"立方体投影包围盒"返回给 UI ----
    in_front = verts_cam[:, 2] > 0.1
    if np.any(in_front):
        proj = project_points_vec(verts_cam[in_front].astype(np.float32),
                                   K, distortion, model)
        bw = float(proj[:, 0].max() - proj[:, 0].min())
        bh = float(proj[:, 1].max() - proj[:, 1].min())
        return max(bw, bh)
    return None


# ================================================================
# Part C : 出画指示箭头 (保险机制)
# ================================================================

def draw_offscreen_arrow_if_needed(frame, w2c, K, distortion, model, target_world):
    """
    若 target_world 不在画面内 (出框 或 跑到相机背后) 就在画面边缘
    画一个绿色箭头指向它。这样即便立方体被甩出画外，使用者也能
    立刻判断它的方位，而不是以为视频出 bug 了。
    """
    h_img, w_img = frame.shape[:2]
    th = np.append(np.asarray(target_world, dtype=np.float32), 1.0)
    tc = (np.asarray(w2c, dtype=np.float32) @ th)[:3]  # target in cam coord

    if tc[2] < 0.1:
        # 相机后方：用 camera-space x,y 的反方向当指示
        dx, dy = float(tc[0]), float(tc[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            dx, dy = 0.0, -1.0
        dir_2d = -np.array([dx, dy])
    else:
        proj = project_points_vec(tc[None].astype(np.float32), K, distortion, model)[0]
        margin = 30
        if (margin <= proj[0] < w_img - margin and
            margin <= proj[1] < h_img - margin):
            return  # 已经在画面里，啥也不画
        dir_2d = np.array([proj[0] - w_img/2, proj[1] - h_img/2])

    n = np.linalg.norm(dir_2d)
    if n < 1e-6: return
    dir_2d = dir_2d / n

    cx_im, cy_im = w_img/2, h_img/2
    em = 35  # edge margin
    tx = (w_img/2 - em) / abs(dir_2d[0]) if abs(dir_2d[0]) > 1e-6 else 1e9
    ty = (h_img/2 - em) / abs(dir_2d[1]) if abs(dir_2d[1]) > 1e-6 else 1e9
    t = min(tx, ty)
    tip  = np.array([cx_im + dir_2d[0]*t, cy_im + dir_2d[1]*t])
    base = tip - dir_2d * 45
    cv2.arrowedLine(frame, tuple(base.astype(int)), tuple(tip.astype(int)),
                    HITCHCOCK_CUBE_COLOR, 3, cv2.LINE_AA, tipLength=0.5)
    cv2.putText(frame, "CUBE",
                tuple((base - dir_2d*18 - np.array([15,-5])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                HITCHCOCK_CUBE_COLOR, 1, cv2.LINE_AA)


# ================================================================
# Part D : UI 文字
# ================================================================

def _draw_hitchcock_info(frame, frame_idx, K_f, w2c, subj_pos,
                          fx_reference, cube_pixel_size, seg_bounds):
    h_img, w_img = frame.shape[:2]

    # 判断当前段
    seg_name = "?"
    if seg_bounds is not None and len(seg_bounds) == 2:
        if   frame_idx <  seg_bounds[0]: seg_name = HITCHCOCK_SEGMENT_NAMES[0]
        elif frame_idx <  seg_bounds[1]: seg_name = HITCHCOCK_SEGMENT_NAMES[1]
        else:                            seg_name = HITCHCOCK_SEGMENT_NAMES[2]

    fx = float(K_f[0, 0])
    ratio = fx / max(fx_reference if fx_reference else fx, 1e-6)
    subj_h = np.append(subj_pos.astype(np.float32), 1.0)
    subj_cam = (w2c @ subj_h)[:3]
    dist = float(np.linalg.norm(subj_cam))

    seg_color = (60, 60, 255) if "HITCHCOCK" in seg_name else (255, 255, 255)

    cv2.putText(frame, f"[{seg_name}]", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, seg_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"frame {frame_idx}",    (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"fx = {fx:.1f}  (x{ratio:.2f})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"dist = {dist:.2f}",    (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1, cv2.LINE_AA)
    if cube_pixel_size is not None:
        cv2.putText(frame, f"cube on-screen = {cube_pixel_size:.0f} px",
                    (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    HITCHCOCK_CUBE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame,
                "Hitchcock OK <=> cube size ~ constant during HITCHCOCK ZOOM",
                (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200,200,200), 1, cv2.LINE_AA)


# ================================================================
# Part E : 主渲染函数
# ================================================================

def render_camera_trajectory_to_numpy(pose_path, n, w, h, stride,
                                       target_indices=None):
    data = np.load(pose_path, allow_pickle=True)
    keys = set(data.files)

    # ---------- extrinsics 解析 ----------
    poses_raw = None
    for k in ["poses","data","c2w","w2c","cam_poses","camera_poses",
              "extrinsic","cams","extrinsics"]:
        if k in keys:
            poses_raw = data[k]; break
    if poses_raw is None:
        for k in keys:
            if np.asarray(data[k]).ndim >= 2:
                poses_raw = data[k]; break
    poses = np.asarray(poses_raw, dtype=np.float32)
    if poses.ndim == 2:
        if   poses.shape[1] == 16: poses = poses.reshape(-1, 4, 4)
        elif poses.shape[1] == 12: poses = poses.reshape(-1, 3, 4)
    if poses.ndim == 3 and poses.shape[1:] == (3, 4):
        bottom = np.broadcast_to(np.array([0,0,0,1], dtype=np.float32),
                                 (poses.shape[0], 1, 4))
        poses = np.concatenate([poses, bottom], axis=1)
    poses = np.linalg.inv(poses)   # w2c -> c2w
    # ---- 防御：把所有 c2w 的旋转部分投影到最近的 proper rotation (det=+1) ----
    # 既能修复轻微数值误差，也能在遇到 reflection 矩阵 (det=-1) 时给出明确警告，
    # 而不是让 scipy Slerp 直接抛 ValueError。
    R_part = poses[:, :3, :3]
    dets = np.linalg.det(R_part)
    bad_mask = dets < 0.0
    if np.any(bad_mask):
        print(f"  ⚠️  {int(bad_mask.sum())}/{len(R_part)} rotations have det<0 "
            f"(improper / reflection). 自动用 SVD 修正为最近的 proper rotation。"
            f" 建议用最新的 generate_hitchcock_npz.py 重新生成。")
    U, _, Vt = np.linalg.svd(R_part)
    # 让 det(U @ Vt) = +1：对 det 为负的样本翻转 Vt 的最后一行
    det_uv = np.linalg.det(U) * np.linalg.det(Vt)
    flip = det_uv < 0
    if np.any(flip):
        Vt[flip, -1, :] *= -1
    poses[:, :3, :3] = (U @ Vt).astype(np.float32)

    # ---------- 其他字段 ----------
    intrinsics_raw = data['intrinsics'] if 'intrinsics' in keys else None
    distortion_raw = data['distortion'] if 'distortion' in keys else None
    camera_model   = str(data['camera_model']) if 'camera_model' in keys else 'pinhole'
    effect         = str(data['effect']) if 'effect' in keys else None
    subj_pos       = np.asarray(data['subject_world_pos'], dtype=np.float32) \
                      if 'subject_world_pos' in keys else None
    fx_reference   = float(data['fx_reference']) if 'fx_reference' in keys else None
    cube_half_size = float(data['cube_half_size']) if 'cube_half_size' in keys \
                      else HITCHCOCK_DEFAULT_HALF_SIZE
    cube_rotation  = np.asarray(data['cube_orientation'], dtype=np.float32) \
                      if 'cube_orientation' in keys else _default_cube_R()
    seg_bounds     = np.asarray(data['segment_boundaries'], dtype=np.int32) \
                      if 'segment_boundaries' in keys else None

    # ---------- 时间轴 ----------
    num_poses = len(poses)
    max_valid = (num_poses - 1) * stride
    n_render = min(n, max_valid + 1)
    if n_render <= 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8), 0
    render_list = target_indices if target_indices is not None else list(range(n_render))
    out_video = np.zeros((len(render_list), h, w, 3), dtype=np.uint8)

    # ---------- 插值 (姿态用 Slerp，平移和内参用线性) ----------
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

    if intrinsics_raw is not None:
        ir = np.asarray(intrinsics_raw, dtype=np.float32)
        if ir.ndim == 2: ir = ir[None]
        if num_poses == 1 or ir.shape[0] == 1:
            K_dense = np.repeat(ir, n_render, axis=0)
        else:
            K_dense = interp1d(t_orig, ir.reshape(num_poses, 9), axis=0)(t_tgt) \
                       .reshape(n_render, 3, 3).astype(np.float32)
    else:
        f = max(w, h) * 0.8
        K_def = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
        K_dense = np.tile(K_def[None], (n_render, 1, 1))
    if fx_reference is None and intrinsics_raw is not None:
        fx_reference = float(K_dense[0, 0, 0])

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

    # ---------- 背景隧道网格 (同前) ----------
    trans = poses_dense[:, :3, 3]
    diffs = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
    valid_diffs = diffs[diffs > 1e-5]
    grid_step = max(np.median(valid_diffs) * GRID_STEP_FACTOR, GRID_MIN_STEP) \
                if len(valid_diffs) > 0 else GRID_MIN_STEP
    max_view_dist = max(grid_step * GRID_MAX_VIEW_FACTOR, 15.0)
    if subj_pos is not None:
        d_sub = np.linalg.norm(trans - subj_pos[None], axis=1)
        max_view_dist = max(max_view_dist, float(np.max(d_sub)) * 1.4)
    anchor = np.vstack([trans, subj_pos.reshape(1, 3)]) if subj_pos is not None else trans
    min_xyz, max_xyz = np.min(anchor, axis=0), np.max(anchor, axis=0)
    margin = max(max_view_dist, 10.0)
    x_coords = np.arange(min_xyz[0] - margin, max_xyz[0] + margin, grid_step)
    z_coords = np.arange(min_xyz[2] - margin, max_xyz[2] + margin, grid_step)
    avg_y = np.median(trans[:, 1])
    floor_y = avg_y + max(grid_step * GRID_FLOOR_STEP_FACTOR, GRID_FLOOR_MIN_HEIGHT)
    ceil_y  = avg_y - max(grid_step * GRID_CEIL_STEP_FACTOR,  GRID_CEIL_MIN_HEIGHT)
    traj_xz_list = [trans[::5, [0, 2]]] if len(trans) > 0 else [np.array([[0., 0.]])]
    if subj_pos is not None:
        traj_xz_list.append(subj_pos[[0, 2]].reshape(1, 2))
    traj_xz = np.vstack(traj_xz_list)
    tree = cKDTree(traj_xz)
    xx, zz = np.meshgrid(x_coords, z_coords)
    pts_xz = np.c_[xx.ravel(), zz.ravel()]
    dists, _ = tree.query(pts_xz)
    valid_mask_2d = (dists < max(grid_step * 8.0, 8.0)).reshape(len(z_coords), len(x_coords))

    g_p1, g_p2, g_c = [], [], []
    mask_x = valid_mask_2d[:, :-1] | valid_mask_2d[:, 1:]
    zi, xi = np.where(mask_x)
    if len(zi) > 0:
        x1, x2, z = x_coords[xi], x_coords[xi+1], z_coords[zi]
        p1 = np.column_stack((x1, np.full_like(x1, floor_y), z))
        p2 = np.column_stack((x2, np.full_like(x2, floor_y), z))
        p3 = np.column_stack((x1, np.full_like(x1, ceil_y),  z))
        p4 = np.column_stack((x2, np.full_like(x2, ceil_y),  z))
        g_p1.extend([p1, p3]); g_p2.extend([p2, p4])
        g_c.append(np.tile(COLOR_X, (len(p1)*2, 1)))
    mask_z = valid_mask_2d[:-1, :] | valid_mask_2d[1:, :]
    zi, xi = np.where(mask_z)
    if len(zi) > 0:
        x, z1, z2 = x_coords[xi], z_coords[zi], z_coords[zi+1]
        p1 = np.column_stack((x, np.full_like(x, floor_y), z1))
        p2 = np.column_stack((x, np.full_like(x, floor_y), z2))
        p3 = np.column_stack((x, np.full_like(x, ceil_y),  z1))
        p4 = np.column_stack((x, np.full_like(x, ceil_y),  z2))
        g_p1.extend([p1, p3]); g_p2.extend([p2, p4])
        g_c.append(np.tile(COLOR_Z, (len(p1)*2, 1)))
    tunnel_r = max(grid_step * GRID_TUNNEL_R_FACTOR, 4.0)
    wall_mask = (dists > tunnel_r) & (dists < tunnel_r + max(grid_step * 1.2, 1.0))
    wp = pts_xz[wall_mask]
    if len(wp) > 0:
        p1 = np.column_stack((wp[:, 0], np.full(len(wp), ceil_y),  wp[:, 1]))
        p2 = np.column_stack((wp[:, 0], np.full(len(wp), floor_y), wp[:, 1]))
        g_p1.append(p1); g_p2.append(p2)
        g_c.append(np.tile(COLOR_Y, (len(p1), 1)))

    has_grid = bool(g_p1)
    if has_grid:
        grid_p1 = np.vstack(g_p1).astype(np.float32)
        grid_p2 = np.vstack(g_p2).astype(np.float32)
        grid_colors = np.vstack(g_c).astype(np.float32)
        grid_p1_h = np.hstack((grid_p1, np.ones((len(grid_p1), 1), dtype=np.float32)))
        grid_p2_h = np.hstack((grid_p2, np.ones((len(grid_p2), 1), dtype=np.float32)))

    # ---------- 渲染循环 ----------
    base_thick = max(4, int(w / 320))
    w2cs_all = np.linalg.inv(poses_dense)
    n_samples_grid = FISHEYE_LINE_SAMPLES if camera_model == 'fisheye' else 1

    for out_idx, frame_idx in enumerate(render_list):
        if frame_idx >= len(w2cs_all): break
        w2c = w2cs_all[frame_idx]
        K_f = K_dense[frame_idx]
        d_f = distortion_dense[frame_idx] if distortion_dense is not None else None
        frame = out_video[out_idx]

        # ----- 网格 -----
        if has_grid:
            gp1_c = (grid_p1_h @ w2c.T)[:, :3]
            gp2_c = (grid_p2_h @ w2c.T)[:, :3]
            mid_c = (gp1_c + gp2_c) / 2.0
            valid = (mid_c[:, 2] > -max(grid_step, 1.0)) & \
                    (np.linalg.norm(mid_c, axis=1) < max_view_dist)
            if np.any(valid):
                res = clip_and_project_lines(
                    gp1_c[valid], gp2_c[valid], grid_colors[valid],
                    K_f, max_view_dist, distortion=d_f, model=camera_model,
                    n_samples=n_samples_grid)
                if res[0] is not None:
                    if n_samples_grid <= 1:
                        p1s, p2s, alphas, cols = res
                        m = alphas >= GRID_ALPHA_THRESH
                        if np.any(m):
                            p1s = p1s[m].astype(np.int32); p2s = p2s[m].astype(np.int32)
                            alphas = alphas[m]; cols = cols[m]
                            dc = (cols * alphas[:, None]).astype(np.int32)
                            dt = np.maximum(2, (base_thick * alphas).astype(np.int32))
                            np.clip(p1s, -w*5, w*5, out=p1s)
                            np.clip(p2s, -w*5, w*5, out=p2s)
                            for p1, p2, c, tt in zip(p1s, p2s, dc, dt):
                                cv2.line(frame, tuple(p1), tuple(p2),
                                         c.tolist(), int(tt), cv2.LINE_AA)
                    else:
                        polys, _, alphas, cols = res
                        m = alphas >= GRID_ALPHA_THRESH
                        if np.any(m):
                            polys = polys[m].astype(np.int32)
                            alphas = alphas[m]; cols = cols[m]
                            dc = (cols * alphas[:, None]).astype(np.int32)
                            dt = np.maximum(2, (base_thick * alphas).astype(np.int32))
                            np.clip(polys, -w*5, w*5, out=polys)
                            for poly, c, tt in zip(polys, dc, dt):
                                cv2.polylines(frame, [poly], False,
                                              c.tolist(), int(tt), cv2.LINE_AA)

        # ----- Hitchcock 立方体 + UI -----
        cube_px = None
        if effect == 'hitchcock' and subj_pos is not None:
            cube_px = draw_hitchcock_cube_3d(frame, w2c, K_f, d_f, camera_model,
                                              subj_pos, cube_half_size, cube_rotation)
            if HITCHCOCK_SHOW_OFFSCREEN_ARROW:
                draw_offscreen_arrow_if_needed(frame, w2c, K_f, d_f, camera_model, subj_pos)
            if HITCHCOCK_SHOW_INFO:
                _draw_hitchcock_info(frame, frame_idx, K_f, w2c, subj_pos,
                                      fx_reference, cube_px, seg_bounds)

    return out_video, n_render


# ================================================================
# Part F : 写视频
# ================================================================

def write_pose_only_video(pose_path, output_path,
                           n_frames=RENDER_N, w=RENDER_W, h=RENDER_H,
                           fps=RENDER_FPS, stride=RENDER_STRIDE):
    print(f"[Viz] {os.path.basename(pose_path)}")
    arr, n_render = render_camera_trajectory_to_numpy(pose_path, n_frames, w, h, stride)
    if n_render == 0:
        print("  ⚠️  no valid frame"); return
    tmp = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for i in range(len(arr)):
        out.write(arr[i])
    out.release()
    ffmpeg = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
    cmd = [ffmpeg, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        if os.path.exists(tmp): os.remove(tmp)
        print(f"  ✅  {output_path}")
    except subprocess.CalledProcessError:
        print(f"  ⚠️  ffmpeg failed, kept tmp: {tmp}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if len(sys.argv) > 1:
        npz_files = sys.argv[1:]
    else:
        npz_files = sorted([os.path.join(NPZ_DIR, f) for f in os.listdir(NPZ_DIR)
                            if f.endswith('.npz')])
    print("=" * 60)
    print(f"🎨 Rendering {len(npz_files)} npz files")
    print("=" * 60)
    for npz in npz_files:
        out = os.path.join(OUTPUT_DIR,
                           os.path.basename(npz).replace('.npz', '_viz.mp4'))
        write_pose_only_video(npz, out)
    print(f"\n✅ Done -> {OUTPUT_DIR}")


    