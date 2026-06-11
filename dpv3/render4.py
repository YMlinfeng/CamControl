"""
================================================================
增强版相机轨迹可视化  (超详细注释版)
================================================================
向后兼容：
  - 旧 npz (只 extrinsics, 或 + intrinsics) 完全按原逻辑渲染
新增能力：
  - 鱼眼畸变 (camera_model='fisheye') -> 网格线变弯曲
  - 希区柯克变焦 (effect='hitchcock') -> 双框叠加
        内框: 像素大小恒定          = 表示主体大小
        外框: 大小 ∝ 当前 fx       = 表示焦距变化
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
# 🔧 CONFIG 区  ——  所有可调参数都集中在这里，方便快速调试
# ================================================================

# ----------------------------------------------------------------
# 【I/O 输入输出路径】
# ----------------------------------------------------------------
NPZ_DIR    = "./synthetic_npz"   # 📁 输入 npz 文件所在文件夹
NPZ_DIR    = "hitchcock"   # 📁 输入 npz 文件所在文件夹
OUTPUT_DIR = "./viz_output_lisj"      # 📁 输出 mp4 视频文件夹 (不存在会自动创建)

# ----------------------------------------------------------------
# 【视频渲染基本参数】
# ----------------------------------------------------------------
RENDER_W      = 512   # 🎞️ 输出视频宽 (像素)
                      #    调大 -> 画面更清晰，但渲染更慢、文件更大
                      #    调小 -> 渲染快，适合快速预览，推荐 256~1024
RENDER_H      = 288   # 🎞️ 输出视频高 (像素)，配合 RENDER_W 决定画面比例
                      #    512x288 ≈ 16:9，常见的视频比例
RENDER_FPS    = 24    # 🎞️ 输出视频帧率
                      #    24=电影感, 30=常规, 60=丝滑但帧多文件大
RENDER_N      = 125   # 🎞️ 最多渲染多少帧 (npz 帧数超过时会按这个值截断)
                      #    npz 帧数不足时按 npz 实际帧数渲染
RENDER_STRIDE = 1     # 🎞️ pose 抽样步长
                      #    =1 用所有 pose；=2 跳一帧用一帧 (加快、运动更跳)
                      #    一般保持 1，除非 npz 帧太密想加速

# ----------------------------------------------------------------
# 【鱼眼渲染：曲线采样精度】
# ----------------------------------------------------------------
# 每条 3D 直线在投影前会被切成 N 个采样点，再连成折线模拟弯曲。
# 这是鱼眼模式下唯一需要调的参数。
FISHEYE_LINE_SAMPLES = 12
# ✏️ 越大 -> 曲线越平滑 (尤其画面边缘的强弯曲处)，但渲染越慢
# ✏️ 越小 -> 曲线变成折线，转角生硬，但快
# 推荐 8~20。强畸变 (strong) 时建议 16+；弱畸变 (weak) 时 8 够用

# ----------------------------------------------------------------
# 【希区柯克双框叠加】 —— 这是这次重点修复的部分
# ----------------------------------------------------------------
# 双框含义：
#   内框 (绿)：屏幕上像素大小恒定，代表"主体在画面里大小不变"
#   外框 (橙)：屏幕上像素大小 ∝ 当前 fx，代表"焦距正在变化"
# 直觉：
#   dolly_in (推 + 变广角) -> fx 变小 -> 外框收缩向内框靠
#   dolly_out (拉 + 变长焦) -> fx 变大 -> 外框扩张远离内框

HITCHCOCK_INNER_HALF_PX = 25
# ✏️ 内框的"半边长" (像素)。整框边长 = 2 * 该值 (这里就是 50x50)
# ✏️ 调大 -> 内框更大，但永远不变(代表主体大小恒定)
# ✏️ 调小 -> 内框更小，需要靠近相机才看得清
# 推荐 15~40，太大会盖住背景网格

HITCHCOCK_OUTER_BASE_HALF_PX = 60
# ✏️ 外框的"基础半边长" (像素)，即 当前fx 等于 参考fx 时外框的大小
# ✏️ 实际外框 = 此值 * (当前fx / 参考fx)
# ✏️ 调大 -> 外框整体更大，焦距变化更醒目，但可能跑出画面
# ✏️ 调小 -> 外框小，可能与内框重叠分不清
# 推荐 50~100。一般保证 此值 > 内框值 + 20 才有视觉差

HITCHCOCK_INNER_COLOR = (80, 255, 80)   # 🎨 内框颜色 (BGR! 不是 RGB)
                                         # 默认鲜绿，对应"主体"
HITCHCOCK_OUTER_COLOR = (80, 200, 255)  # 🎨 外框颜色 (BGR)
                                         # 默认橙黄色，对应"焦距"
                                         # 想换颜色直接改 3 个数 0~255

HITCHCOCK_INNER_THICKNESS = 2  # ✏️ 内框线粗 (像素)。调大更醒目，太大会糊
HITCHCOCK_OUTER_THICKNESS = 2  # ✏️ 外框线粗 (像素)

HITCHCOCK_DRAW_CONNECTORS = True
# ✏️ 是否画 4 条连接"内框角 -> 外框角"的辅助线
# ✏️ True  -> 像一个"梯形漏斗"，能非常清楚看出外框相对内框收/扩
# ✏️ False -> 只显示两个独立方框，画面更干净

HITCHCOCK_SHOW_INFO_TEXT = True
# ✏️ 是否在画面左上角显示数字信息 (当前焦距、焦距倍率、主体距离)
# ✏️ True  -> 调试时强烈建议开，能看到精确数值
# ✏️ False -> 出最终展示视频时可关掉，画面更干净

# ----------------------------------------------------------------
# 【背景 3D 网格几何】 —— 控制"地板/天花板/侧墙"的形状
# ----------------------------------------------------------------
# 网格基本逻辑：用相机移动的中位步长 * 倍数 = 一格的大小
# 然后在地板/天花板/侧墙各画网格线，让画面有"隧道感"

GRID_STEP_FACTOR = 10.0
# ✏️ 网格一格的大小 = (相机帧间平均位移) * 此值
# ✏️ 调大 -> 网格更稀疏，每格更大，画面更"开阔"
# ✏️ 调小 -> 网格更密，画面更"局促"，但太密会糊成一片
# 推荐 5~20

GRID_MIN_STEP = 0.5
# ✏️ 网格一格的最小绝对值。当相机几乎静止时 (位移很小)
#    防止网格变得无穷密集
# ✏️ 调大 -> 静止场景网格更稀疏
# 推荐 0.3~1.0

GRID_MAX_VIEW_FACTOR = 15.0
# ✏️ 网格可见的最远距离 = grid_step * 此值
# ✏️ 调大 -> 能看到更远的网格，远处线很密
# ✏️ 调小 -> 远处直接被剔除，画面更"近视"
# 推荐 10~25

GRID_FLOOR_STEP_FACTOR = 3.0
# ✏️ 地板与相机的距离 = max(grid_step * 此值, GRID_FLOOR_MIN_HEIGHT)
# ✏️ 调大 -> 地板离相机更远，画面下半部分网格更小
# 推荐 2~5

GRID_FLOOR_MIN_HEIGHT = 2.0
# ✏️ 地板距相机的最小高度 (绝对值)
# ✏️ 调大 -> 地板被强制压低，永远看得到
# ✏️ 调小 -> 地板可能"贴脸"，画面下半都是地板线
# 推荐 1.5~4.0

GRID_CEIL_STEP_FACTOR = 3.0
# ✏️ 天花板与相机的距离倍数，逻辑同地板
# 推荐 2~5

GRID_CEIL_MIN_HEIGHT = 2.0
# ✏️ 天花板距相机的最小高度 (绝对值)
# ✏️ 调大 -> 天花板更高，画面上方更空旷
# 推荐 1.5~4.0

GRID_TUNNEL_R_FACTOR = 3.5
GRID_TUNNEL_R_FACTOR = 5.5
# ✏️ 侧墙(左右黄色墙)距相机轨迹的水平距离 = grid_step * 此值
# ✏️ 调大 -> 隧道更宽，侧墙更远
# ✏️ 调小 -> 隧道更窄，像挤在巷子里
# 推荐 2~6

GRID_ALPHA_THRESH = 0.05
# ✏️ 透明度低于这个阈值的网格线直接不画 (优化性能 + 避免边缘抖动)
# ✏️ 调大 -> 远处线提前消失，画面"雾感"更重
# ✏️ 调小 -> 远处线坚持画到底，可能出现一闪一闪的远景线
# 推荐 0.02~0.1

# ----------------------------------------------------------------
# 【网格线颜色】 —— BGR 格式！不是 RGB！
# ----------------------------------------------------------------
COLOR_X = (255, 255, 0)   # 🎨 X 方向(水平横向)的地板/天花板线: 青色
                          #    调成 (0,0,255) 就是红色, (0,255,0) 就是绿色
COLOR_Z = (255, 0, 255)   # 🎨 Z 方向(纵深方向)的地板/天花板线: 品红
COLOR_Y = (0, 255, 255)   # 🎨 Y 方向(垂直)的侧墙线: 黄色


# ================================================================
# Part A:  投影函数 (支持 pinhole / fisheye)
# ----------------------------------------------------------------
# 这部分一般不用改。如果你要换鱼眼模型 (比如 OpenCV equidistant
# 改成 ATAN 模型)，可以在 model == 'fisheye' 分支里改公式。
# ================================================================

def project_points_vec(pts_3d, K, distortion=None, model='pinhole'):
    """
    向量化 3D->2D 投影，支持等距 (equidistant) 鱼眼模型。
    输入 pts_3d 必须已经在相机坐标系下 (z 朝前为正)。
    """
    # 防止除以 0：把 z 截到 >=1e-6
    Z = np.maximum(pts_3d[:, 2], 1e-6)
    x_n = pts_3d[:, 0] / Z  # 归一化平面坐标
    y_n = pts_3d[:, 1] / Z

    if model == 'fisheye' and distortion is not None:
        # OpenCV 等距鱼眼模型：
        #   r = sqrt(x_n^2 + y_n^2)
        #   theta = atan(r)
        #   theta_d = theta * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)
        #   u = fx * (x_n * theta_d / r) + cx
        r = np.sqrt(x_n*x_n + y_n*y_n)
        theta = np.arctan(r)
        # ✏️ 这里 0.98 是为了防止 theta 太接近 90° (画面边缘)
        # 时投影爆掉。如果要画更极端的鱼眼可以调到 0.99
        theta = np.clip(theta, 0, np.pi/2 * 0.98)
        k1, k2, k3, k4 = distortion[:4]
        t2 = theta * theta
        theta_d = theta * (1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)
        scale = np.where(r > 1e-8, theta_d / np.maximum(r, 1e-8), 1.0)
        u = K[0,0] * (x_n * scale) + K[0,2]
        v = K[1,1] * (y_n * scale) + K[1,2]
    else:
        # 标准小孔成像
        u = K[0,0] * x_n + K[0,2]
        v = K[1,1] * y_n + K[1,2]
    return np.column_stack((u, v))


# ================================================================
# Part B:  3D 线段裁剪 + 投影
# ----------------------------------------------------------------
# 这部分负责把 3D 线段先做 z_near 裁剪 (剔除相机后方部分)，
# 再投影到 2D。鱼眼模式下还会把每条线再细分成多个采样点。
# 一般不用改，除非要改"近平面"位置。
# ================================================================

def clip_and_project_lines(p1_c, p2_c, colors, K, max_dist,
                            distortion=None, model='pinhole', n_samples=1):
    """
    对一批 3D 线段做 z_near 裁剪后投影到 2D。
      n_samples=1  -> 返回 (proj_p1, proj_p2, alpha, cols) 用于画直线
      n_samples>1  -> 返回 (polylines(N,S,2), None, alpha, cols) 用于画折线
    """
    z1, z2 = p1_c[:, 2], p2_c[:, 2]

    # ✏️ z_near = 近平面距离。<= 这个 z 的点会被认为在相机后面
    # ✏️ 调大 (比如 0.5) -> 相机附近的网格会被剪掉，画面"远离"
    # ✏️ 调小 (比如 0.01) -> 几乎不剪，但贴脸的线投影会爆炸变形
    z_near = 0.1

    both_front  = (z1 >= z_near) & (z2 >= z_near)  # 两端都在前方
    both_behind = (z1 <  z_near) & (z2 <  z_near)  # 两端都在后方 -> 整条丢弃
    intersect   = ~(both_front | both_behind)       # 一前一后 -> 在近平面处截断

    vp1_l, vp2_l, vc_l = [], [], []
    if np.any(both_front):
        vp1_l.append(p1_c[both_front]); vp2_l.append(p2_c[both_front])
        vc_l.append(colors[both_front])
    if np.any(intersect):
        p1i, p2i = p1_c[intersect], p2_c[intersect]
        z1i, z2i = p1i[:, 2], p2i[:, 2]
        # 用线性插值找近平面与线段的交点
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

    # 用线段中点到相机的距离算 alpha (远了变淡)
    mid = (vp1 + vp2) / 2.0
    alpha = np.clip(1.0 - np.linalg.norm(mid, axis=1) / max_dist, 0.0, 1.0)

    if n_samples <= 1:
        # 小孔成像：投影两个端点画直线
        pts = np.vstack((vp1, vp2))
        proj = project_points_vec(pts, K, distortion, model)
        half = len(vp1)
        return proj[:half], proj[half:], alpha, vc
    else:
        # 鱼眼：把每条线分成 n_samples 个点，分别投影，再连折线
        ts = np.linspace(0, 1, n_samples).reshape(1, n_samples, 1)
        sampled = vp1[:, None, :] * (1 - ts) + vp2[:, None, :] * ts
        flat = sampled.reshape(-1, 3)
        proj = project_points_vec(flat, K, distortion, model)
        return proj.reshape(len(vp1), n_samples, 2), None, alpha, vc


# ================================================================
# Part C:  希区柯克双框绘制
# ----------------------------------------------------------------
# 这个函数完全在 2D 屏幕空间画方框，绕开了之前 3D 立方体
# 透视带来的"内框也变大"的问题。
# 如果你想自定义双框样式，主要改这里。
# ================================================================

def draw_hitchcock_overlay(frame, w2c, K_f, d_f, model,
                            subj_pos, fx_reference):
    """
    在主体投影位置画两层方框 (内框=主体，外框=焦距)。

    参数:
      frame         : 当前帧 numpy 数组 (HxWx3, BGR), 直接在上面画
      w2c           : 世界->相机 4x4 矩阵
      K_f           : 当前帧 3x3 内参
      d_f           : 当前帧畸变系数 (鱼眼时非空)
      model         : 'pinhole' 或 'fisheye'
      subj_pos      : 主体世界坐标 (3,)
      fx_reference  : 参考焦距 (npz 里存的初始 fx)，用来算缩放比
    """
    # ---- 把主体的世界坐标变换到相机坐标系 ----
    subj_h = np.append(subj_pos.astype(np.float32), 1.0)  # 补 1 变齐次
    subj_cam = (w2c @ subj_h)[:3]

    # 主体跑到相机背后 -> 不画 (避免投影爆掉)
    # ✏️ 0.1 = 近平面距离，和上面 z_near 保持一致
    # if subj_cam[2] <= 0.1:
    #     return

    # ---- 投影主体到 2D ----
    subj_2d = project_points_vec(subj_cam[None].astype(np.float32),
                                  K_f, d_f, model)[0]
    h_img, w_img = frame.shape[:2]
    cx_px, cy_px = int(subj_2d[0]), int(subj_2d[1])

    # 主体投影离画面太远 (超出 1 倍画面外) -> 不画
    # ✏️ 这个范围调大可以容忍主体偏出画面更多，但意义不大
    if not (-w_img < cx_px < 2 * w_img and -h_img < cy_px < 2 * h_img):
        return

    # ---- 计算两框大小 ----
    inner = HITCHCOCK_INNER_HALF_PX
    # 焦距比 = 当前fx / 参考fx
    #   dolly_in 时 fx 缩小 -> ratio < 1 -> 外框缩小
    #   dolly_out 时 fx 放大 -> ratio > 1 -> 外框放大
    focal_ratio = float(K_f[0, 0]) / max(fx_reference, 1e-6)

    # 外框：随 fx 线性缩放
    # ✏️ +4 是保底：保证外框比内框至少大 4 像素，不会被内框吞掉
    outer = int(HITCHCOCK_OUTER_BASE_HALF_PX * focal_ratio)
    # outer = max(int(HITCHCOCK_OUTER_BASE_HALF_PX * focal_ratio), inner + 4)

    # ---- 先画外框 (焦距) ----
    cv2.rectangle(frame,
                  (cx_px - outer, cy_px - outer),
                  (cx_px + outer, cy_px + outer),
                  HITCHCOCK_OUTER_COLOR,
                  HITCHCOCK_OUTER_THICKNESS, cv2.LINE_AA)

    # ---- 再画内框 (主体) —— 后画在上层，保证可见 ----
    cv2.rectangle(frame,
                  (cx_px - inner, cy_px - inner),
                  (cx_px + inner, cy_px + inner),
                  HITCHCOCK_INNER_COLOR,
                  HITCHCOCK_INNER_THICKNESS, cv2.LINE_AA)

    # ---- 内外四角连线 (辅助线，让"外框相对内框收/扩"更直观) ----
    if HITCHCOCK_DRAW_CONNECTORS:
        # 遍历 4 个角的方向 (左上、右上、左下、右下)
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            cv2.line(frame,
                     (cx_px + sx*inner, cy_px + sy*inner),   # 内框角
                     (cx_px + sx*outer, cy_px + sy*outer),   # 外框角
                     HITCHCOCK_OUTER_COLOR, 1, cv2.LINE_AA)
            # ✏️ 想让连接线更粗：把上面的 1 调成 2 或 3
            # ✏️ 想让连接线另外颜色：把 HITCHCOCK_OUTER_COLOR 换成自定义元组

    # ---- 数字标签 ----
    if HITCHCOCK_SHOW_INFO_TEXT:
        info1 = f"f={K_f[0,0]:.0f}px  (x{focal_ratio:.2f})"  # 当前焦距和倍率
        info2 = f"d={subj_cam[2]:.2f}"                        # 主体距相机距离
        # ✏️ putText 参数: (img, text, 位置, 字体, 字号, 颜色, 线粗, 抗锯齿)
        # ✏️ 0.55 = 字号缩放，调大字会变大。0.4~0.8 都合理
        cv2.putText(frame, info1, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, info2, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        # 底部图例
        cv2.putText(frame, "INNER=subject (const)  OUTER=focal (scales w/ fx)",
                    (10, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)


# ================================================================
# Part D:  主渲染函数
# ----------------------------------------------------------------
# 整体流程：
#   1. 读 npz 解析所有字段 (extrinsics/intrinsics/distortion/effect/...)
#   2. 在时间轴上做插值 (姿态用 Slerp + 平移线性，内参逐元素线性)
#   3. 用相机轨迹决定网格的范围、间距、地板高度等
#   4. 逐帧投影网格线 -> 画到画布
#   5. 如果是 hitchcock 效果，再额外画双框
# ================================================================

def render_camera_trajectory_to_numpy(pose_path, n, w, h, stride,
                                       target_indices=None):
    data = np.load(pose_path, allow_pickle=True)
    keys = set(data.files)

    # ---------- 1. 解析 extrinsics ----------
    # 兼容多种 key 名 (poses, c2w, w2c, extrinsics, ...)
    poses_raw = None
    for k in ["poses","data","c2w","w2c","cam_poses","camera_poses",
              "extrinsic","cams","extrinsics"]:
        if k in keys:
            poses_raw = data[k]; break
    if poses_raw is None:
        # 实在找不到就找第一个看起来像矩阵的 key
        for k in keys:
            if np.asarray(data[k]).ndim >= 2:
                poses_raw = data[k]; break
    poses = np.asarray(poses_raw, dtype=np.float32)

    # 兼容 (N, 16) / (N, 12) 等扁平格式
    if poses.ndim == 2:
        if   poses.shape[1] == 16: poses = poses.reshape(-1, 4, 4)
        elif poses.shape[1] == 12: poses = poses.reshape(-1, 3, 4)

    # 兼容 3x4 (补一行 [0,0,0,1] 变 4x4)
    if poses.ndim == 3 and poses.shape[1:] == (3, 4):
        bottom = np.broadcast_to(np.array([0,0,0,1], dtype=np.float32),
                                 (poses.shape[0], 1, 4))
        poses = np.concatenate([poses, bottom], axis=1)

    # 我们存的是 w2c，渲染时需要 c2w (相机在世界中的位置)
    poses = np.linalg.inv(poses)

    # ---------- 2. 解析新字段 ----------
    intrinsics_raw = data['intrinsics'] if 'intrinsics' in keys else None
    distortion_raw = data['distortion'] if 'distortion' in keys else None
    camera_model   = str(data['camera_model']) if 'camera_model' in keys else 'pinhole'
    effect         = str(data['effect']) if 'effect' in keys else None
    subj_pos     = np.asarray(data['subject_world_pos']) if 'subject_world_pos' in keys else None
    fx_reference = float(data['fx_reference']) if 'fx_reference' in keys else None

    # ---------- 3. 时间轴设置 ----------
    num_poses = len(poses)
    max_valid = (num_poses - 1) * stride
    n_render = min(n, max_valid + 1)
    if n_render <= 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8), 0
    render_list = target_indices if target_indices is not None else list(range(n_render))
    out_video = np.zeros((len(render_list), h, w, 3), dtype=np.uint8)

    # ---------- 4. 轨迹插值 ----------
    # 旋转用 Slerp (球面线性插值，比线性插值平滑得多)
    # 平移用 interp1d 线性插值
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
        # 没有内参就给个默认 (焦距 = 长边 * 0.8)
        # ✏️ 想改默认 FOV：把 0.8 调小 = 更广角；调大 = 更长焦
        f = max(w, h) * 0.8
        K_def = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
        K_dense = np.tile(K_def[None], (n_render, 1, 1))

    # npz 里没存 fx_reference 时用第一帧的 fx 当参考
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

    # ---------- 7. 构造 3D 隧道网格 ----------
    trans = poses_dense[:, :3, 3]  # 相机位置序列
    diffs = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
    valid_diffs = diffs[diffs > 1e-5]

    # grid_step = 中位步长 * GRID_STEP_FACTOR (保底 GRID_MIN_STEP)
    grid_step = max(np.median(valid_diffs) * GRID_STEP_FACTOR, GRID_MIN_STEP) \
                if len(valid_diffs) > 0 else GRID_MIN_STEP
    max_view_dist = max(grid_step * GRID_MAX_VIEW_FACTOR, 15.0)

    # 如果有主体位置，保证视野能涵盖到主体
    if subj_pos is not None:
        d_sub = np.linalg.norm(trans - subj_pos[None], axis=1)
        max_view_dist = max(max_view_dist, float(np.max(d_sub)) * 1.4)
        # ✏️ 1.4 是"看到主体外再多看 40%"的余量，调大可避免主体附近被截

    # 决定网格分布的 XZ 范围
    anchor = np.vstack([trans, subj_pos.reshape(1, 3)]) if subj_pos is not None else trans
    min_xyz, max_xyz = np.min(anchor, axis=0), np.max(anchor, axis=0)
    margin = max(max_view_dist, 10.0)
    x_coords = np.arange(min_xyz[0] - margin, max_xyz[0] + margin, grid_step)
    z_coords = np.arange(min_xyz[2] - margin, max_xyz[2] + margin, grid_step)

    # 地板/天花板的 Y 坐标 (用相机 Y 中位值 ± 偏移)
    avg_y = np.median(trans[:, 1])
    floor_y = avg_y + max(grid_step * GRID_FLOOR_STEP_FACTOR, GRID_FLOOR_MIN_HEIGHT)
    ceil_y  = avg_y - max(grid_step * GRID_CEIL_STEP_FACTOR,  GRID_CEIL_MIN_HEIGHT)

    # 用 KD 树过滤掉远离相机轨迹的网格点 (优化性能)
    traj_xz_list = [trans[::5, [0, 2]]] if len(trans) > 0 else [np.array([[0., 0.]])]
    if subj_pos is not None:
        traj_xz_list.append(subj_pos[[0, 2]].reshape(1, 2))
    traj_xz = np.vstack(traj_xz_list)
    tree = cKDTree(traj_xz)
    xx, zz = np.meshgrid(x_coords, z_coords)
    pts_xz = np.c_[xx.ravel(), zz.ravel()]
    dists, _ = tree.query(pts_xz)
    # ✏️ 8.0 = 离轨迹超过 grid_step*8 的网格点直接丢
    # ✏️ 调大 -> 保留更多远处网格；调小 -> 画面更聚焦相机周围
    valid_mask_2d = (dists < max(grid_step * 8.0, 8.0)).reshape(len(z_coords), len(x_coords))

    g_p1, g_p2, g_c = [], [], []

    # ---- X 方向线 (横向，对应地板和天花板) ----
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

    # ---- Z 方向线 (纵深，对应地板和天花板的"沿路"方向) ----
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

    # ---- Y 方向线 (垂直，左右两侧"墙壁") ----
    tunnel_r = max(grid_step * GRID_TUNNEL_R_FACTOR, 4.0)
    # ✏️ 这里的 1.2 决定墙的"厚度" (画几条墙线)
    # ✏️ 调大 -> 墙线更多更厚；调小 -> 几乎只有一圈线
    wall_mask = (dists > tunnel_r) & (dists < tunnel_r + max(grid_step * 1.2, 1.0))
    wp = pts_xz[wall_mask]
    if len(wp) > 0:
        p1 = np.column_stack((wp[:, 0], np.full(len(wp), ceil_y),  wp[:, 1]))
        p2 = np.column_stack((wp[:, 0], np.full(len(wp), floor_y), wp[:, 1]))
        g_p1.append(p1); g_p2.append(p2)
        g_c.append(np.tile(COLOR_Y, (len(p1), 1)))

    if not g_p1:
        return out_video, n_render

    grid_p1 = np.vstack(g_p1).astype(np.float32)
    grid_p2 = np.vstack(g_p2).astype(np.float32)
    grid_colors = np.vstack(g_c).astype(np.float32)
    grid_p1_h = np.hstack((grid_p1, np.ones((len(grid_p1), 1), dtype=np.float32)))
    grid_p2_h = np.hstack((grid_p2, np.ones((len(grid_p2), 1), dtype=np.float32)))

    # ---------- 8. 渲染循环 ----------
    # ✏️ base_thick = 网格线基础粗细 (像素)，随分辨率自适应
    # ✏️ 4 = 最小粗细；320 = 分辨率归一化基数
    # ✏️ 想线更细：把 4 改成 2；想线更粗：把 4 改成 6
    base_thick = max(4, int(w / 320))
    w2cs_all = np.linalg.inv(poses_dense)
    n_samples = FISHEYE_LINE_SAMPLES if camera_model == 'fisheye' else 1

    for out_idx, frame_idx in enumerate(render_list):
        if frame_idx >= len(w2cs_all): break
        w2c = w2cs_all[frame_idx]
        K_f = K_dense[frame_idx]
        d_f = distortion_dense[frame_idx] if distortion_dense is not None else None

        # 把所有 3D 网格点变到相机坐标系
        gp1_c = (grid_p1_h @ w2c.T)[:, :3]
        gp2_c = (grid_p2_h @ w2c.T)[:, :3]

        # 粗筛：把太靠后或太远的线段去掉
        mid_c = (gp1_c + gp2_c) / 2.0
        valid = (mid_c[:, 2] > -max(grid_step, 1.0)) & \
                (np.linalg.norm(mid_c, axis=1) < max_view_dist)

        frame = out_video[out_idx]

        if np.any(valid):
            res = clip_and_project_lines(
                gp1_c[valid], gp2_c[valid], grid_colors[valid],
                K_f, max_view_dist, distortion=d_f, model=camera_model,
                n_samples=n_samples)

            if res[0] is not None:
                if n_samples <= 1:
                    # ---- Pinhole 模式: 画直线 ----
                    p1s, p2s, alphas, cols = res
                    mask_a = alphas >= GRID_ALPHA_THRESH
                    if np.any(mask_a):
                        p1s = p1s[mask_a].astype(np.int32)
                        p2s = p2s[mask_a].astype(np.int32)
                        alphas = alphas[mask_a]; cols = cols[mask_a]
                        # 颜色按 alpha 衰减 (越远越淡)
                        draw_c = (cols * alphas[:, None]).astype(np.int32)
                        # 线粗也按 alpha 衰减 (越远越细)
                        draw_t = np.maximum(2, (base_thick * alphas).astype(np.int32))
                        # ✏️ np.maximum(2, ...) 中的 2 = 最小线粗
                        # ✏️ 调大 -> 远处线也粗；调小到 1 -> 远处线更细
                        np.clip(p1s, -w*5, w*5, out=p1s)
                        np.clip(p2s, -w*5, w*5, out=p2s)
                        for p1, p2, c, t in zip(p1s, p2s, draw_c, draw_t):
                            cv2.line(frame, tuple(p1), tuple(p2), c.tolist(),
                                     int(t), cv2.LINE_AA)
                else:
                    # ---- Fisheye 模式: 画折线 ----
                    polylines, _, alphas, cols = res
                    mask_a = alphas >= GRID_ALPHA_THRESH
                    if np.any(mask_a):
                        polys = polylines[mask_a].astype(np.int32)
                        alphas = alphas[mask_a]; cols = cols[mask_a]
                        draw_c = (cols * alphas[:, None]).astype(np.int32)
                        draw_t = np.maximum(2, (base_thick * alphas).astype(np.int32))
                        np.clip(polys, -w*5, w*5, out=polys)
                        for poly, c, t in zip(polys, draw_c, draw_t):
                            cv2.polylines(frame, [poly], False, c.tolist(),
                                          int(t), cv2.LINE_AA)

        # ---- 希区柯克双框叠加 (画在网格之上) ----
        if effect == 'hitchcock' and subj_pos is not None and fx_reference is not None:
            draw_hitchcock_overlay(frame, w2c, K_f, d_f, camera_model,
                                    subj_pos, fx_reference)

    return out_video, n_render


# ================================================================
# Part E:  写视频文件 (mp4)
# ----------------------------------------------------------------
# 先用 OpenCV 写一个 mp4v 临时文件，再用 ffmpeg 转 h264 兼容性好的 mp4
# 如果没装 ffmpeg 就保留临时文件 (浏览器可能播不了，但能用 vlc)
# ================================================================

def write_pose_only_video(pose_path, output_path,
                           n_frames=RENDER_N, w=RENDER_W, h=RENDER_H,
                           fps=RENDER_FPS, stride=RENDER_STRIDE):
    print(f"[Viz] {os.path.basename(pose_path)}")
    arr, n_render = render_camera_trajectory_to_numpy(pose_path, n_frames, w, h, stride)
    if n_render == 0:
        print("  ⚠️  no valid frame"); return
    tmp = output_path.replace('.mp4', '_temp.mp4')
    # ✏️ 'mp4v' = OpenCV 内置编码器，兼容性最好但文件大
    # ✏️ 想换成 'avc1' (h264) 需要系统有对应编码器
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for i in range(len(arr)):
        out.write(arr[i])
    out.release()

    # 用 ffmpeg 再转一次 h264，让浏览器能播
    ffmpeg = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
    cmd = [ffmpeg, "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        if os.path.exists(tmp): os.remove(tmp)
        print(f"  ✅  {output_path}")
    except subprocess.CalledProcessError:
        print(f"  ⚠️  ffmpeg failed, kept tmp: {tmp}")


# ================================================================
# Main 入口
# ================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 命令行可以指定单独的 npz：python visualize.py path/to/x.npz
    # 否则渲染 NPZ_DIR 下所有 npz
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