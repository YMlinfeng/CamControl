"""
================================================================
增强版相机轨迹可视化  (v2: 三段式 Hitchcock + 3D 立方体 + PIP 追踪视图)
================================================================
向后兼容：
  - 旧 npz (只 extrinsics, 或 + intrinsics) 完全按原逻辑渲染
  - 鱼眼畸变 (camera_model='fisheye') 网格弯曲不变

新增 / 升级 (Hitchcock 部分):
  - 3D 立方体: 在 subject_world_pos 处放一个真实 3D 立方体, 通过投影
              渲染. 在 Hitchcock 段 (fx ∝ d) 投影尺寸天然恒定;
              在随机段尺寸随距离自然变化. 立方体角点跟随相机扰动出现
              轻微旋转/平移视差 -> 让微扰效果肉眼可见.
  - 外框 (2D): 仍按 fx/fx_ref 比例缩放, 配合数字说明焦距变化.
  - 画外箭头: 立方体中心移出画面时, 在画面边缘画箭头指向它.
  - PIP 追踪视图: 右下角嵌入一个小窗口, 使用一个"追踪相机"
                 (与主相机同位置, 朝向主体, 同焦距), 保证立方体
                 永远居中可见, 同时保留 Hitchcock 缩放感.
================================================================
"""

import os
import sys
import cv2
import subprocess
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import cKDTree


# ================================================================
# 🔧 CONFIG  ——  所有可调参数
# ================================================================

# ----------------------------------------------------------------
# 【I/O】
# ----------------------------------------------------------------
NPZ_DIR    = "hitchcock"
NPZ_DIR    = "hitchcock_npz"
NPZ_DIR    = "./synthetic_npz"
OUTPUT_DIR = "./viz_output_lisj"

# ----------------------------------------------------------------
# 【视频渲染基本参数】
# ----------------------------------------------------------------
RENDER_W      = 512
RENDER_H      = 288
RENDER_FPS    = 24
RENDER_N      = 600   # ⚠️ 三段式 Hitchcock 单段 ~22s * 25fps = 550 帧, 给 600 兜底
RENDER_STRIDE = 1

# ----------------------------------------------------------------
# 【鱼眼渲染】
# ----------------------------------------------------------------
FISHEYE_LINE_SAMPLES = 12

# ----------------------------------------------------------------
# 【Hitchcock - 3D 立方体】 ⭐ 新
# ----------------------------------------------------------------
HITCHCOCK_CUBE_EDGE_RATIO = 0.15
# ✏️ 立方体边长 (世界单位) = 主体初始距离 * 此值
# ✏️ 调大 -> 立方体更大. 推荐 0.10 ~ 0.25
# ✏️ 实际投影大小 ≈ fx * 边长 / 距离, 在 Hitchcock 段恒定

HITCHCOCK_CUBE_COLOR_FRONT     = (80, 255, 80)    # 🎨 BGR  靠近相机一侧的边
HITCHCOCK_CUBE_COLOR_BACK      = (40, 150, 40)    # 🎨 BGR  远离相机一侧的边 (略暗)
HITCHCOCK_CUBE_COLOR_VERT      = (80, 240, 200)   # 🎨 BGR  连接前后的 4 条 "进深边"
HITCHCOCK_CUBE_THICKNESS       = 2
HITCHCOCK_CUBE_DRAW_VERTICES   = True             # 是否在 8 个角点画小圆点
HITCHCOCK_CUBE_VERTEX_RADIUS   = 3

# ----------------------------------------------------------------
# 【Hitchcock - 2D 外框 (保留, 直观展示 fx 变化)】
# ----------------------------------------------------------------
HITCHCOCK_DRAW_OUTER_BOX     = True
HITCHCOCK_OUTER_BASE_HALF_PX = 60
HITCHCOCK_OUTER_COLOR        = (80, 200, 255)
HITCHCOCK_OUTER_THICKNESS    = 2
HITCHCOCK_SHOW_INFO_TEXT     = True

# ----------------------------------------------------------------
# 【Hitchcock - 画外箭头指示器】 ⭐ 新
# ----------------------------------------------------------------
HITCHCOCK_SHOW_OFFSCREEN_ARROW = True
HITCHCOCK_ARROW_SIZE   = 22
HITCHCOCK_ARROW_MARGIN = 30
HITCHCOCK_ARROW_COLOR  = (80, 255, 80)

# ----------------------------------------------------------------
# 【Hitchcock - PIP 追踪画中画】 ⭐ 新
# ----------------------------------------------------------------
HITCHCOCK_SHOW_PIP      = True
HITCHCOCK_PIP_W         = 192
HITCHCOCK_PIP_H         = 108
HITCHCOCK_PIP_MARGIN    = 12
HITCHCOCK_PIP_BORDER    = 2
HITCHCOCK_PIP_POSITION  = 'bottom-right'   # 'top-right' / 'bottom-right' / 'top-left' / 'bottom-left'
HITCHCOCK_PIP_BG_COLOR  = (18, 18, 22)
HITCHCOCK_PIP_BORDER_COLOR = (220, 220, 220)
HITCHCOCK_PIP_LABEL     = "TRACKING (cube-centered, fx-synced)"

# ----------------------------------------------------------------
# 【网格几何 / 颜色】(原样保留)
# ----------------------------------------------------------------
GRID_STEP_FACTOR        = 10.0
GRID_MIN_STEP           = 0.5
GRID_MAX_VIEW_FACTOR    = 15.0
GRID_FLOOR_STEP_FACTOR  = 3.0
GRID_FLOOR_MIN_HEIGHT   = 2.0
GRID_CEIL_STEP_FACTOR   = 3.0
GRID_CEIL_MIN_HEIGHT    = 2.0
GRID_TUNNEL_R_FACTOR    = 5.5
GRID_ALPHA_THRESH       = 0.05

COLOR_X = (255, 255, 0)
COLOR_Z = (255, 0, 255)
COLOR_Y = (0, 255, 255)


# ================================================================
# Part A:  投影函数 (pinhole / fisheye)  —— 原样保留
# ================================================================

def project_points_vec(pts_3d, K, distortion=None, model='pinhole'):
    Z = np.maximum(pts_3d[:, 2], 1e-6)
    x_n = pts_3d[:, 0] / Z
    y_n = pts_3d[:, 1] / Z

    if model == 'fisheye' and distortion is not None:
        r = np.sqrt(x_n*x_n + y_n*y_n)
        theta = np.arctan(r)
        theta = np.clip(theta, 0, np.pi/2 * 0.98)
        k1, k2, k3, k4 = distortion[:4]
        t2 = theta * theta
        theta_d = theta * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)
        scale = np.where(r > 1e-8, theta_d / np.maximum(r, 1e-8), 1.0)
        u = K[0, 0] * (x_n * scale) + K[0, 2]
        v = K[1, 1] * (y_n * scale) + K[1, 2]
    else:
        u = K[0, 0] * x_n + K[0, 2]
        v = K[1, 1] * y_n + K[1, 2]
    return np.column_stack((u, v))


# ================================================================
# Part B:  3D 线段裁剪 + 投影  —— 原样保留
# ================================================================

def clip_and_project_lines(p1_c, p2_c, colors, K, max_dist,
                            distortion=None, model='pinhole', n_samples=1):
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
    alpha = np.clip(1.0 - np.linalg.norm(mid, axis=1) / max_dist, 0.4, 1.0)

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
# Part C:  ⭐ 网格绘制 (抽取为函数, 便于 PIP 复用)
# ================================================================

def _render_grid_pass(frame, w2c, K_f, d_f, model,
                      grid_p1_h, grid_p2_h, grid_colors,
                      max_view_dist, grid_step, base_thick, n_samples,
                      w_img, h_img):
    """对单帧执行网格投影+绘制. 调用前网格几何应已构造好."""
    gp1_c = (grid_p1_h @ w2c.T)[:, :3]
    gp2_c = (grid_p2_h @ w2c.T)[:, :3]

    mid_c = (gp1_c + gp2_c) / 2.0
    valid = (mid_c[:, 2] > -max(grid_step, 1.0)) & \
            (np.linalg.norm(mid_c, axis=1) < max_view_dist)

    if not np.any(valid):
        return

    res = clip_and_project_lines(
        gp1_c[valid], gp2_c[valid], grid_colors[valid],
        K_f, max_view_dist, distortion=d_f, model=model,
        n_samples=n_samples)
    if res[0] is None:
        return

    if n_samples <= 1:
        p1s, p2s, alphas, cols = res
        mask_a = alphas >= GRID_ALPHA_THRESH
        if np.any(mask_a):
            p1s = p1s[mask_a].astype(np.int32)
            p2s = p2s[mask_a].astype(np.int32)
            alphas = alphas[mask_a]; cols = cols[mask_a]
            draw_c = np.maximum(10, (cols * alphas[:, None]).astype(np.int32))
            draw_t = np.maximum(1, (base_thick * alphas).astype(np.int32))
            np.clip(p1s, -w_img*5, w_img*5, out=p1s)
            np.clip(p2s, -w_img*5, w_img*5, out=p2s)
            for p1, p2, c, t in zip(p1s, p2s, draw_c, draw_t):
                cv2.line(frame, tuple(p1), tuple(p2), c.tolist(),
                         int(t), cv2.LINE_AA)
    else:
        polylines, _, alphas, cols = res
        mask_a = alphas >= GRID_ALPHA_THRESH
        if np.any(mask_a):
            polys = polylines[mask_a].astype(np.int32)
            alphas = alphas[mask_a]; cols = cols[mask_a]
            draw_c = (cols * alphas[:, None]).astype(np.int32)
            draw_t = np.maximum(1, (base_thick * alphas).astype(np.int32))
            np.clip(polys, -w_img*5, w_img*5, out=polys)
            for poly, c, t in zip(polys, draw_c, draw_t):
                cv2.polylines(frame, [poly], False, c.tolist(),
                              int(t), cv2.LINE_AA)


# ================================================================
# Part D:  ⭐ 3D 立方体渲染 (代替原 2D 绿框)
# ----------------------------------------------------------------
# 立方体放在 subj_pos 处, 角点是真实 3D 点, 用与场景同样的
# 投影管线 (pinhole/fisheye) 投影. 因此:
#   - Hitchcock 段 (fx ∝ d) 投影大小天然恒定
#   - 随机段 (fx 常量) 投影大小随距离自然变化
#   - 相机的微小旋转/平移扰动 -> 立方体可见的视差/旋转
# ================================================================

# 8 个角点的局部坐标 (单位立方体, 之后 ×半边长)
_CUBE_LOCAL = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],  # 0-3: -Z 面
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],  # 4-7: +Z 面
], dtype=np.float32)

# 12 条边: (端点 a, 端点 b, 边类型)
#   'face_near' / 'face_far' / 'vert' (深度连接边)
_CUBE_EDGES = [
    (0, 1, 'face'), (1, 2, 'face'), (2, 3, 'face'), (3, 0, 'face'),  # -Z 面
    (4, 5, 'face'), (5, 6, 'face'), (6, 7, 'face'), (7, 4, 'face'),  # +Z 面
    (0, 4, 'vert'), (1, 5, 'vert'), (2, 6, 'vert'), (3, 7, 'vert'),  # 4 条进深边
]


def _project_clipped_segment(p1_c, p2_c, K, d, model,
                              w_img, h_img, z_near=0.1, n_samples_fisheye=8):
    """对单条 3D 线段做近平面裁剪并投影. 返回 list of np.int32 polylines (每条 ≥2 点)."""
    p1, p2 = p1_c.copy(), p2_c.copy()
    if p1[2] < z_near and p2[2] < z_near:
        return []
    if p1[2] < z_near:
        t = (z_near - p1[2]) / (p2[2] - p1[2])
        p1 = p1 + t * (p2 - p1)
    elif p2[2] < z_near:
        t = (z_near - p2[2]) / (p1[2] - p2[2])
        p2 = p2 + t * (p1 - p2)

    if model == 'fisheye':
        ts = np.linspace(0, 1, n_samples_fisheye).reshape(-1, 1)
        sampled = p1[None] * (1 - ts) + p2[None] * ts
        proj = project_points_vec(sampled.astype(np.float32), K, d, model)
    else:
        proj = project_points_vec(np.stack([p1, p2]).astype(np.float32), K, d, model)
    np.clip(proj, -w_img * 5, h_img * 5 + w_img * 5, out=proj)  # 防止 int 溢出
    return [proj.astype(np.int32)]


def render_3d_cube(frame, w2c, K_f, d_f, model,
                    center_world, edge_len,
                    color_front=HITCHCOCK_CUBE_COLOR_FRONT,
                    color_back=HITCHCOCK_CUBE_COLOR_BACK,
                    color_vert=HITCHCOCK_CUBE_COLOR_VERT,
                    thickness=HITCHCOCK_CUBE_THICKNESS,
                    draw_vertices=HITCHCOCK_CUBE_DRAW_VERTICES,
                    vertex_radius=HITCHCOCK_CUBE_VERTEX_RADIUS):
    """把世界坐标 center_world 处的 3D 立方体投影到 frame 上 (线框)."""
    h_img, w_img = frame.shape[:2]
    s = edge_len * 0.5

    corners_w = _CUBE_LOCAL * s + center_world[None, :].astype(np.float32)
    corners_w_h = np.hstack([corners_w, np.ones((8, 1), dtype=np.float32)])
    corners_c = (corners_w_h @ w2c.T)[:, :3]

    # 用相机系 Z 的中位数 -> 区分近端 4 角 / 远端 4 角
    median_z = float(np.median(corners_c[:, 2]))
    is_near = corners_c[:, 2] < median_z   # 这 4 个是离相机近的角

    # ---- 画 12 条边 ----
    for a, b, etype in _CUBE_EDGES:
        if etype == 'vert':
            col = color_vert
        else:
            # 面边: 两端都是近端 -> front 色; 两端都是远端 -> back 色;
            #       否则混合用 front (基本不会出现, face 边两端属于同一 Z 面)
            both_near = bool(is_near[a]) and bool(is_near[b])
            both_far  = (not is_near[a]) and (not is_near[b])
            col = color_front if both_near else (color_back if both_far else color_front)

        polylines = _project_clipped_segment(corners_c[a], corners_c[b],
                                              K_f, d_f, model, w_img, h_img)
        for poly in polylines:
            if model == 'fisheye':
                cv2.polylines(frame, [poly], False, col, thickness, cv2.LINE_AA)
            else:
                cv2.line(frame, tuple(poly[0]), tuple(poly[-1]),
                         col, thickness, cv2.LINE_AA)

    # ---- 8 个角点的小圆点 ----
    if draw_vertices:
        for i in range(8):
            if corners_c[i, 2] <= 0.1:
                continue
            p2d = project_points_vec(corners_c[i:i+1].astype(np.float32),
                                      K_f, d_f, model)[0]
            if not np.all(np.isfinite(p2d)):
                continue
            px, py = int(p2d[0]), int(p2d[1])
            if -w_img < px < 2 * w_img and -h_img < py < 2 * h_img:
                col = color_front if is_near[i] else color_back
                cv2.circle(frame, (px, py), vertex_radius, col, -1, cv2.LINE_AA)


# ================================================================
# Part E:  ⭐ Hitchcock 复合 overlay  (立方体 + 外框 + 文本 + 箭头)
# ================================================================

def _project_center(w2c, K_f, d_f, model, subj_pos):
    subj_h = np.append(subj_pos.astype(np.float32), 1.0)
    subj_cam = (w2c @ subj_h)[:3]
    if subj_cam[2] <= 0.1:
        return None, subj_cam
    subj_2d = project_points_vec(subj_cam[None].astype(np.float32),
                                  K_f, d_f, model)[0]
    return subj_2d, subj_cam


def draw_offscreen_arrow(frame, target_2d, color,
                          arrow_size=HITCHCOCK_ARROW_SIZE,
                          margin=HITCHCOCK_ARROW_MARGIN):
    """目标点在画面外时, 在画面边缘画一个朝向它的箭头. 返回 True 表示画了."""
    h, w = frame.shape[:2]
    cx, cy = float(target_2d[0]), float(target_2d[1])
    if margin <= cx < w - margin and margin <= cy < h - margin:
        return False  # 在画面内, 不画

    fx_c, fy_c = w * 0.5, h * 0.5
    dx, dy = cx - fx_c, cy - fy_c
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return False

    # 把方向缩放到画面边缘内 margin 处
    sx = (w * 0.5 - margin) / max(abs(dx), 1e-6)
    sy = (h * 0.5 - margin) / max(abs(dy), 1e-6)
    s = min(sx, sy)
    ax = int(fx_c + dx * s)
    ay = int(fy_c + dy * s)

    angle = float(np.arctan2(dy, dx))
    tip   = (ax + int(np.cos(angle) * arrow_size * 0.65),
             ay + int(np.sin(angle) * arrow_size * 0.65))
    base  = (ax - int(np.cos(angle) * arrow_size * 0.35),
             ay - int(np.sin(angle) * arrow_size * 0.35))
    left  = (base[0] + int(np.cos(angle + 2.5) * arrow_size * 0.55),
             base[1] + int(np.sin(angle + 2.5) * arrow_size * 0.55))
    right = (base[0] + int(np.cos(angle - 2.5) * arrow_size * 0.55),
             base[1] + int(np.sin(angle - 2.5) * arrow_size * 0.55))
    pts = np.array([tip, left, right], dtype=np.int32)
    cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, "SUBJECT",
                (max(5, ax - 30), max(15, ay - arrow_size - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    return True


def draw_hitchcock_overlay_3d(frame, w2c, K_f, d_f, model,
                               subj_pos, fx_reference, cube_edge_len):
    """3D 立方体 + 2D 外框 + 文本 + 画外箭头. (代替原版 2D 双框)"""
    h_img, w_img = frame.shape[:2]

    # 1) 3D 立方体 (永远尝试渲染, 角点会自动裁剪)
    render_3d_cube(frame, w2c, K_f, d_f, model, subj_pos, cube_edge_len)

    # 2) 2D 外框 (按 fx/fx_ref 比例缩放, 表达焦距变化)
    subj_2d, subj_cam = _project_center(w2c, K_f, d_f, model, subj_pos)
    focal_ratio = float(K_f[0, 0]) / max(fx_reference, 1e-6)

    if subj_2d is not None and HITCHCOCK_DRAW_OUTER_BOX:
        cx_px, cy_px = int(subj_2d[0]), int(subj_2d[1])
        if -w_img < cx_px < 2 * w_img and -h_img < cy_px < 2 * h_img:
            outer = int(HITCHCOCK_OUTER_BASE_HALF_PX * focal_ratio)
            cv2.rectangle(frame,
                          (cx_px - outer, cy_px - outer),
                          (cx_px + outer, cy_px + outer),
                          HITCHCOCK_OUTER_COLOR,
                          HITCHCOCK_OUTER_THICKNESS, cv2.LINE_AA)

    # 3) 数字标签
    if HITCHCOCK_SHOW_INFO_TEXT:
        info1 = f"f={K_f[0, 0]:.0f}px  (x{focal_ratio:.2f})"
        d_cam = float(subj_cam[2]) if subj_cam[2] > 0 else \
                float(np.linalg.norm(subj_cam))
        info2 = f"d={d_cam:.2f}"
        cv2.putText(frame, info1, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, info2, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    "CUBE=3D subject  OUTER=focal (scales w/ fx)",
                    (10, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # 4) 画外箭头
    if HITCHCOCK_SHOW_OFFSCREEN_ARROW and subj_2d is not None:
        draw_offscreen_arrow(frame, subj_2d, HITCHCOCK_ARROW_COLOR)
    elif HITCHCOCK_SHOW_OFFSCREEN_ARROW and subj_cam[2] <= 0.1:
        # 主体在相机背后: 用 "屏幕反向" 当箭头方向
        back_2d = np.array([w_img - subj_2d[0] if subj_2d is not None else w_img / 2,
                            h_img - subj_2d[1] if subj_2d is not None else h_img / 2])
        # 简化: 直接朝下指示 "主体在背后"
        cv2.putText(frame, "SUBJECT BEHIND CAMERA",
                    (w_img // 2 - 110, h_img // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, HITCHCOCK_ARROW_COLOR,
                    1, cv2.LINE_AA)


# ================================================================
# Part F:  ⭐ PIP 追踪视图
# ----------------------------------------------------------------
# 一个"虚拟追踪相机":
#   - 位置 = 主相机位置 (世界坐标)
#   - 朝向 = 永远指向 subj_pos
#   - fx   = 主相机 fx     (同步焦距 → 保留 Hitchcock 效果!)
#   - 主点 = PIP 画面中心  (立方体居中显示)
# ================================================================

def _build_lookat_w2c(cam_pos, target_pos, world_up=None):
    """构造一个 4x4 的 w2c, 让相机站在 cam_pos 看向 target_pos.
       CV 约定: +X 右, +Y 下, +Z 前. world_up 默认 -Y."""
    if world_up is None:
        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    forward = target_pos.astype(np.float32) - cam_pos.astype(np.float32)
    fn = float(np.linalg.norm(forward))
    if fn < 1e-6:
        return None
    forward = forward / fn

    right = np.cross(forward, world_up)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        # forward 与 world_up 平行 → 选一个备用 right
        right = np.cross(forward, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        rn = float(np.linalg.norm(right))
        if rn < 1e-6:
            right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        right = right / float(np.linalg.norm(right))
    else:
        right = right / rn

    down = np.cross(forward, right)
    R_c2w = np.column_stack([right, down, forward]).astype(np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_c2w.T
    w2c[:3, 3]  = -R_c2w.T @ cam_pos
    return w2c


def render_pip_view(grid_p1_h, grid_p2_h, grid_colors,
                     w2c_main, K_main, d_main, model,
                     subj_pos, cube_edge_len,
                     max_view_dist, grid_step,
                     pip_w=HITCHCOCK_PIP_W, pip_h=HITCHCOCK_PIP_H):
    """渲染 PIP 子帧, 返回 (pip_h, pip_w, 3) 的 uint8 数组."""
    # 主相机的世界位置
    R_w2c = w2c_main[:3, :3]
    t_w2c = w2c_main[:3, 3]
    cam_pos = (-R_w2c.T @ t_w2c).astype(np.float32)

    w2c_pip = _build_lookat_w2c(cam_pos, subj_pos.astype(np.float32))
    if w2c_pip is None:
        pip = np.full((pip_h, pip_w, 3), HITCHCOCK_PIP_BG_COLOR, dtype=np.uint8)
        cv2.putText(pip, "subject at cam", (10, pip_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return pip

    # 内参: 用主相机 fx (保留 Hitchcock 缩放), 主点在 PIP 中心
    fx_cur = float(K_main[0, 0])
    K_pip = np.array([[fx_cur, 0, pip_w * 0.5],
                       [0, fx_cur, pip_h * 0.5],
                       [0, 0, 1.0]], dtype=np.float32)

    pip = np.full((pip_h, pip_w, 3), HITCHCOCK_PIP_BG_COLOR, dtype=np.uint8)

    base_thick_pip = max(1, int(pip_w / 320))
    n_samples_pip = FISHEYE_LINE_SAMPLES if model == 'fisheye' else 1

    # PIP 也画网格 (这样能看到"背景剧烈伸缩 vs 立方体不变"的对比!)
    _render_grid_pass(pip, w2c_pip, K_pip, d_main, model,
                      grid_p1_h, grid_p2_h, grid_colors,
                      max_view_dist, grid_step, base_thick_pip, n_samples_pip,
                      pip_w, pip_h)

    # PIP 中渲染立方体
    render_3d_cube(pip, w2c_pip, K_pip, d_main, model,
                    subj_pos, cube_edge_len,
                    thickness=max(1, HITCHCOCK_CUBE_THICKNESS - 1),
                    vertex_radius=max(2, HITCHCOCK_CUBE_VERTEX_RADIUS - 1))

    # 中心十字 (帮助看出立方体是否真的"钉在中心")
    cx_p, cy_p = pip_w // 2, pip_h // 2
    cross = 6
    cv2.line(pip, (cx_p - cross, cy_p), (cx_p + cross, cy_p),
             (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(pip, (cx_p, cy_p - cross), (cx_p, cy_p + cross),
             (180, 180, 180), 1, cv2.LINE_AA)

    return pip


def composite_pip(main_frame, pip_frame,
                   position=HITCHCOCK_PIP_POSITION,
                   margin=HITCHCOCK_PIP_MARGIN,
                   border=HITCHCOCK_PIP_BORDER,
                   border_color=HITCHCOCK_PIP_BORDER_COLOR,
                   label=HITCHCOCK_PIP_LABEL):
    """把 PIP 贴到主画面对应角落."""
    h, w = main_frame.shape[:2]
    ph, pw = pip_frame.shape[:2]
    if   position == 'top-right':    x0, y0 = w - pw - margin, margin
    elif position == 'bottom-right': x0, y0 = w - pw - margin, h - ph - margin
    elif position == 'top-left':     x0, y0 = margin, margin
    elif position == 'bottom-left':  x0, y0 = margin, h - ph - margin
    else:                            x0, y0 = w - pw - margin, h - ph - margin

    # 安全裁剪 (PIP 太大时)
    x0 = max(border + 2, min(x0, w - pw - border - 2))
    y0 = max(border + 14, min(y0, h - ph - border - 2))

    cv2.rectangle(main_frame,
                  (x0 - border, y0 - border),
                  (x0 + pw + border, y0 + ph + border),
                  border_color, border, cv2.LINE_AA)
    main_frame[y0:y0 + ph, x0:x0 + pw] = pip_frame
    if label:
        cv2.putText(main_frame, label, (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)


# ================================================================
# Part G:  主渲染函数
# ================================================================

def render_camera_trajectory_to_numpy(pose_path, n, w, h, stride,
                                       target_indices=None):
    data = np.load(pose_path, allow_pickle=True)
    keys = set(data.files)

    # ---------- 1. extrinsics 解析 ----------
    poses_raw = None
    for k in ["poses", "data", "c2w", "w2c", "cam_poses", "camera_poses",
              "extrinsic", "cams", "extrinsics"]:
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
        bottom = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32),
                                 (poses.shape[0], 1, 4))
        poses = np.concatenate([poses, bottom], axis=1)

    poses = np.linalg.inv(poses)  # w2c -> c2w

    # ---------- 2. 新字段解析 ----------
    intrinsics_raw = data['intrinsics'] if 'intrinsics' in keys else None
    distortion_raw = data['distortion'] if 'distortion' in keys else None
    camera_model   = str(data['camera_model']) if 'camera_model' in keys else 'pinhole'
    effect         = str(data['effect']) if 'effect' in keys else None
    subj_pos       = np.asarray(data['subject_world_pos'], dtype=np.float32) \
                        if 'subject_world_pos' in keys else None
    fx_reference   = float(data['fx_reference']) if 'fx_reference' in keys else None
    subj_distance  = float(data['hitchcock_subject_distance']) \
                        if 'hitchcock_subject_distance' in keys else None

    # ---------- 3. 时间轴 ----------
    num_poses = len(poses)
    max_valid = (num_poses - 1) * stride
    n_render = min(n, max_valid + 1) if n is not None and n > 0 else max_valid + 1
    n_render = max(n_render, 1)
    n_render = min(n_render, num_poses)  # 不外推
    if n_render <= 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8), 0
    render_list = target_indices if target_indices is not None else list(range(n_render))
    out_video = np.zeros((len(render_list), h, w, 3), dtype=np.uint8)

    # ---------- 4. 轨迹插值 ----------
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

    # ---------- 5. 内参插值 ----------
    if intrinsics_raw is not None:
        ir = np.asarray(intrinsics_raw, dtype=np.float32)
        if ir.ndim == 2: ir = ir[None]
        if num_poses == 1 or ir.shape[0] == 1:
            K_dense = np.repeat(ir, n_render, axis=0)
        else:
            K_flat = ir.reshape(num_poses, 9)
            K_dense = interp1d(t_orig, K_flat, axis=0)(t_tgt) \
                        .reshape(n_render, 3, 3).astype(np.float32)
    else:
        f = max(w, h) * 0.8
        K_def = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                         dtype=np.float32)
        K_dense = np.tile(K_def[None], (n_render, 1, 1))

    if fx_reference is None and intrinsics_raw is not None:
        fx_reference = float(K_dense[0, 0, 0])

    # ---------- 6. 畸变插值 ----------
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

    # ---------- 7. 3D 隧道网格 ----------
    trans = poses_dense[:, :3, 3]
    diffs = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
    valid_diffs = diffs[diffs > 1e-5]

    grid_step = max(np.median(valid_diffs) * GRID_STEP_FACTOR, GRID_MIN_STEP) \
                if len(valid_diffs) > 0 else GRID_MIN_STEP
    max_view_dist = max(grid_step * GRID_MAX_VIEW_FACTOR, 40.0)

    if subj_pos is not None:
        d_sub = np.linalg.norm(trans - subj_pos[None], axis=1)
        max_view_dist = max(max_view_dist, float(np.max(d_sub)) * 1.4)

    anchor = np.vstack([trans, subj_pos.reshape(1, 3)]) \
                if subj_pos is not None else trans
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
    valid_mask_2d = (dists < max(grid_step * 8.0, 8.0)) \
                        .reshape(len(z_coords), len(x_coords))

    g_p1, g_p2, g_c = [], [], []

    # X 方向线
    mask_x = valid_mask_2d[:, :-1] | valid_mask_2d[:, 1:]
    zi, xi = np.where(mask_x)
    if len(zi) > 0:
        x1, x2, z = x_coords[xi], x_coords[xi + 1], z_coords[zi]
        p1 = np.column_stack((x1, np.full_like(x1, floor_y), z))
        p2 = np.column_stack((x2, np.full_like(x2, floor_y), z))
        p3 = np.column_stack((x1, np.full_like(x1, ceil_y),  z))
        p4 = np.column_stack((x2, np.full_like(x2, ceil_y),  z))
        g_p1.extend([p1, p3]); g_p2.extend([p2, p4])
        g_c.append(np.tile(COLOR_X, (len(p1) * 2, 1)))

    # Z 方向线
    mask_z = valid_mask_2d[:-1, :] | valid_mask_2d[1:, :]
    zi, xi = np.where(mask_z)
    if len(zi) > 0:
        x, z1, z2 = x_coords[xi], z_coords[zi], z_coords[zi + 1]
        p1 = np.column_stack((x, np.full_like(x, floor_y), z1))
        p2 = np.column_stack((x, np.full_like(x, floor_y), z2))
        p3 = np.column_stack((x, np.full_like(x, ceil_y),  z1))
        p4 = np.column_stack((x, np.full_like(x, ceil_y),  z2))
        g_p1.extend([p1, p3]); g_p2.extend([p2, p4])
        g_c.append(np.tile(COLOR_Z, (len(p1) * 2, 1)))

    # Y 方向线
    tunnel_r = max(grid_step * GRID_TUNNEL_R_FACTOR, 4.0)
    wall_mask = (dists > tunnel_r) & (dists < tunnel_r + max(grid_step * 1.2, 1.0))
    wp = pts_xz[wall_mask]
    if len(wp) > 0:
        p1 = np.column_stack((wp[:, 0], np.full(len(wp), ceil_y),  wp[:, 1]))
        p2 = np.column_stack((wp[:, 0], np.full(len(wp), floor_y), wp[:, 1]))
        g_p1.append(p1); g_p2.append(p2)
        g_c.append(np.tile(COLOR_Y, (len(p1), 1)))

    grid_ready = bool(g_p1)
    if grid_ready:
        grid_p1 = np.vstack(g_p1).astype(np.float32)
        grid_p2 = np.vstack(g_p2).astype(np.float32)
        grid_colors = np.vstack(g_c).astype(np.float32)
        grid_p1_h = np.hstack((grid_p1, np.ones((len(grid_p1), 1), dtype=np.float32)))
        grid_p2_h = np.hstack((grid_p2, np.ones((len(grid_p2), 1), dtype=np.float32)))
    else:
        grid_p1_h = grid_p2_h = grid_colors = None

    # ---------- 8. Hitchcock 立方体边长 ----------
    cube_edge_len = None
    if effect == 'hitchcock' and subj_pos is not None:
        if subj_distance is not None and subj_distance > 1e-3:
            ref_d = subj_distance
        else:
            ref_d = float(np.linalg.norm(trans[0] - subj_pos))
        cube_edge_len = max(HITCHCOCK_CUBE_EDGE_RATIO * ref_d, 0.1)

    # ---------- 9. 渲染循环 ----------
    base_thick = max(4, int(w / 320))
    w2cs_all = np.linalg.inv(poses_dense)
    n_samples = FISHEYE_LINE_SAMPLES if camera_model == 'fisheye' else 1

    for out_idx, frame_idx in enumerate(render_list):
        if frame_idx >= len(w2cs_all): break
        w2c = w2cs_all[frame_idx]
        K_f = K_dense[frame_idx]
        d_f = distortion_dense[frame_idx] if distortion_dense is not None else None
        frame = out_video[out_idx]

        # ---- 主视图网格 ----
        if grid_ready:
            _render_grid_pass(frame, w2c, K_f, d_f, camera_model,
                              grid_p1_h, grid_p2_h, grid_colors,
                              max_view_dist, grid_step,
                              base_thick, n_samples, w, h)

        # ---- Hitchcock: 主视图 3D 立方体 + 外框 + 文本 + 箭头 ----
        if effect == 'hitchcock' and subj_pos is not None \
                and fx_reference is not None and cube_edge_len is not None:
            draw_hitchcock_overlay_3d(frame, w2c, K_f, d_f, camera_model,
                                       subj_pos, fx_reference, cube_edge_len)

            # ---- PIP 追踪视图 ----
            if HITCHCOCK_SHOW_PIP and grid_ready:
                pip_frame = render_pip_view(
                    grid_p1_h, grid_p2_h, grid_colors,
                    w2c, K_f, d_f, camera_model,
                    subj_pos, cube_edge_len,
                    max_view_dist, grid_step,
                )
                composite_pip(frame, pip_frame)

    return out_video, n_render


# ================================================================
# Part H:  写视频
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
    cmd = [ffmpeg, "-y", "-i", tmp,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=True)
        if os.path.exists(tmp): os.remove(tmp)
        print(f"  ✅  {output_path}")
    except subprocess.CalledProcessError:
        print(f"  ⚠️  ffmpeg failed, kept tmp: {tmp}")


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if len(sys.argv) > 1:
        npz_files = sys.argv[1:]
    else:
        npz_files = sorted([os.path.join(NPZ_DIR, f) for f in os.listdir(NPZ_DIR)
                            if f.endswith('.npz')])

    print("=" * 60)
    print(f"🎨  Rendering {len(npz_files)} npz")
    print("=" * 60)
    for npz in npz_files:
        out = os.path.join(OUTPUT_DIR,
                           os.path.basename(npz).replace('.npz', '_viz.mp4'))
        write_pose_only_video(npz, out)
    print(f"\n✅  Done -> {OUTPUT_DIR}")