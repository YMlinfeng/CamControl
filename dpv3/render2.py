import cv2
import numpy as np
import os
import subprocess
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial import cKDTree
import sys, json
import cv2
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial import cKDTree

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import cKDTree


# ============================================================
# 投影模型：pinhole(可带径向切向畸变) / fisheye(Kannala-Brandt)
# ============================================================
def _project_pinhole(pts_3d, K, dist=None):
    """
    pts_3d: (N, 3) 相机坐标系
    K: (3, 3)
    dist: None 或 (k1, k2, p1, p2[, k3])  Brown-Conrady
    return: (N, 2) 像素, (N,) valid_mask (z > 0)
    """
    Z = pts_3d[:, 2]
    valid = Z > 1e-6
    Z_safe = np.where(valid, Z, 1.0)
    x = pts_3d[:, 0] / Z_safe
    y = pts_3d[:, 1] / Z_safe

    if dist is not None and np.any(np.asarray(dist) != 0):
        d = np.zeros(5, dtype=np.float64)
        d[:len(dist)] = dist[:5] if len(dist) >= 5 else dist
        k1, k2, p1, p2, k3 = d
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = x_d, y_d

    u = x * K[0, 0] + K[0, 2]
    v = y * K[1, 1] + K[1, 2]
    return np.column_stack((u, v)), valid


def _project_fisheye(pts_3d, K, dist=None, max_theta=np.deg2rad(110)):
    """
    Kannala-Brandt 鱼眼模型 (OpenCV fisheye)
    dist: None 或 (k1, k2, k3, k4)
    max_theta: 单像入射角阈值，超过则视为不可见
    """
    X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
    r = np.sqrt(X * X + Y * Y)
    # 注意：用 arctan2(r, Z) 可以让 Z<0 时 theta>pi/2，自然处理大 FOV
    theta = np.arctan2(r, Z)
    valid = theta < max_theta

    if dist is not None and np.any(np.asarray(dist) != 0):
        d = np.zeros(4, dtype=np.float64)
        d[:len(dist)] = dist[:4]
        k1, k2, k3, k4 = d
        t2 = theta * theta
        theta_d = theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    else:
        theta_d = theta

    r_safe = np.where(r > 1e-8, r, 1e-8)
    scale = theta_d / r_safe
    x_p = X * scale
    y_p = Y * scale
    u = x_p * K[0, 0] + K[0, 2]
    v = y_p * K[1, 1] + K[1, 2]
    return np.column_stack((u, v)), valid


def project_points(pts_3d, K, dist=None, model="pinhole"):
    if model == "fisheye":
        return _project_fisheye(pts_3d, K, dist)
    return _project_pinhole(pts_3d, K, dist)


# ============================================================
# 3D 线段细分：在畸变投影下，直线 -> 曲线
# ============================================================
def subdivide_segments(p1, p2, n_seg=12):
    """
    p1, p2: (N, 3)
    return: pts (N, n_seg+1, 3)  每条线被切成 n_seg 段
    """
    t = np.linspace(0.0, 1.0, n_seg + 1, dtype=np.float32)
    return p1[:, None, :] * (1 - t[None, :, None]) + p2[:, None, :] * t[None, :, None]


def clip_and_project_polyline(p1_c, p2_c, colors, K, dist, model,
                              max_dist, n_seg=12, z_near=0.1):
    """
    替代原 clip_and_project_lines。
    1) 先做 z_near 裁剪
    2) 细分成 n_seg 段
    3) 用 project_points 批量投影 (支持畸变 / fisheye)
    4) 返回每条折线的点序列
    """
    z1, z2 = p1_c[:, 2], p2_c[:, 2]
    both_front = (z1 >= z_near) & (z2 >= z_near)
    both_behind = (z1 < z_near) & (z2 < z_near)
    intersect = ~(both_front | both_behind)

    parts_p1, parts_p2, parts_c = [], [], []
    if np.any(both_front):
        parts_p1.append(p1_c[both_front]); parts_p2.append(p2_c[both_front])
        parts_c.append(colors[both_front])
    if np.any(intersect):
        p1i, p2i = p1_c[intersect], p2_c[intersect]
        t = (z_near - p1i[:, 2]) / (p2i[:, 2] - p1i[:, 2] + 1e-6)
        p_clip = p1i + t[:, None] * (p2i - p1i)
        m1 = (p1i[:, 2] < z_near)[:, None]
        m2 = (p2i[:, 2] < z_near)[:, None]
        parts_p1.append(np.where(m1, p_clip, p1i))
        parts_p2.append(np.where(m2, p_clip, p2i))
        parts_c.append(colors[intersect])

    # 鱼眼/广角下还可能有 z<0 但仍可见，model=="fisheye" 时放宽 z_near
    if model == "fisheye":
        behind_visible = both_behind & (np.linalg.norm((p1_c + p2_c) / 2, axis=1) < max_dist)
        if np.any(behind_visible):
            parts_p1.append(p1_c[behind_visible])
            parts_p2.append(p2_c[behind_visible])
            parts_c.append(colors[behind_visible])

    if not parts_p1:
        return None, None, None

    vp1 = np.vstack(parts_p1).astype(np.float32)
    vp2 = np.vstack(parts_p2).astype(np.float32)
    vc = np.vstack(parts_c).astype(np.float32)

    pts3d = subdivide_segments(vp1, vp2, n_seg=n_seg)        # (N, S+1, 3)
    N, S1, _ = pts3d.shape
    flat = pts3d.reshape(-1, 3)
    pts2d, valid = project_points(flat, K, dist, model=model)
    pts2d = pts2d.reshape(N, S1, 2)
    valid = valid.reshape(N, S1)

    mid = (vp1 + vp2) * 0.5
    alpha = np.clip(1.0 - np.linalg.norm(mid, axis=1) / max_dist, 0.0, 1.0)

    return pts2d, valid, (alpha, vc)


# ============================================================
# 主渲染
# ============================================================
def render_camera_trajectory_to_numpy(
    pose_path, n, w, h, stride,
    target_indices=None,
    camera_model=None,        # "pinhole" / "fisheye" / None(自动)
    use_npz_intrinsics=True,  # 若 False 退回原硬编码 K（兼容旧行为）
    n_seg=12,                 # 线段细分数
):
    data = np.load(pose_path, allow_pickle=True)

    # ---- extrinsics ----
    extr = None
    for k in ["extrinsics", "poses", "data", "c2w", "w2c",
              "cam_poses", "camera_poses", "extrinsic", "cams"]:
        if k in data.files:
            extr = data[k]; break
    poses = np.asarray(extr, dtype=np.float32)
    if poses.ndim == 3 and poses.shape[1:] == (3, 4):
        bottom = np.broadcast_to(np.array([0, 0, 0, 1], np.float32),
                                 (poses.shape[0], 1, 4))
        poses = np.concatenate([poses, bottom], axis=1)
    poses_c2w = np.linalg.inv(poses)  # npz 里是 w2c
    num_poses = len(poses_c2w)

    # ---- intrinsics ----
    intr_all = None
    if use_npz_intrinsics and "intrinsics" in data.files:
        intr_all = np.asarray(data["intrinsics"], dtype=np.float32)  # (N,3,3)

    # ---- distortion ----
    dist_all = None
    if "dist_coeffs" in data.files:
        dist_all = np.asarray(data["dist_coeffs"], dtype=np.float32)

    # ---- camera model 自动判断 ----
    if camera_model is None:
        if "camera_model" in data.files:
            camera_model = str(data["camera_model"])
        else:
            camera_model = "fisheye" if dist_all is not None and \
                           dist_all.shape[-1] == 4 else "pinhole"

    max_valid = (num_poses - 1) * stride
    n_render = min(n, max_valid + 1)
    if n_render <= 0:
        return np.zeros((0, h, w, 3), np.uint8), 0
    render_list = target_indices if target_indices is not None else list(range(n_render))
    out_video = np.zeros((len(render_list), h, w, 3), np.uint8)

    # ---- 插值 c2w & K & dist ----
    t_orig = np.arange(num_poses) * stride
    t_tgt = np.arange(n_render)
    trans = poses_c2w[:, :3, 3]
    rots = poses_c2w[:, :3, :3]
    if num_poses == 1:
        poses_dense = np.repeat(poses_c2w, n_render, axis=0)
        K_dense = np.repeat(intr_all, n_render, axis=0) if intr_all is not None else None
        dist_dense = np.repeat(dist_all, n_render, axis=0) if dist_all is not None else None
    else:
        t_interp = interp1d(t_orig, trans, axis=0)(t_tgt)
        r_interp = Slerp(t_orig, R.from_matrix(rots))(t_tgt).as_matrix()
        poses_dense = np.zeros((n_render, 4, 4), np.float32)
        poses_dense[:, 3, 3] = 1
        poses_dense[:, :3, :3] = r_interp
        poses_dense[:, :3, 3] = t_interp
        K_dense = (interp1d(t_orig, intr_all, axis=0)(t_tgt).astype(np.float32)
                   if intr_all is not None else None)
        dist_dense = (interp1d(t_orig, dist_all, axis=0)(t_tgt).astype(np.float32)
                      if dist_all is not None else None)

    # ---- 生成 3D 网格（保留你原逻辑） ----
    diffs = np.linalg.norm(trans[1:] - trans[:-1], axis=1) if num_poses > 1 else np.array([0.0])
    valid_diffs = diffs[diffs > 1e-5]
    grid_step = max(np.median(valid_diffs) * 10.0, 0.5) if len(valid_diffs) > 0 else 0.5
    max_view_dist = max(grid_step * 15.0, 15.0)
    min_xyz, max_xyz = np.min(trans, axis=0), np.max(trans, axis=0)
    margin = max(max_view_dist, 10.0)
    x_coords = np.arange(min_xyz[0] - margin, max_xyz[0] + margin, grid_step)
    z_coords = np.arange(min_xyz[2] - margin, max_xyz[2] + margin, grid_step)
    avg_y = np.median(trans[:, 1])
    floor_y = avg_y + max(grid_step * 3.0, 2.0)
    ceil_y  = avg_y - max(grid_step * 3.0, 2.0)
    traj_xz = trans[::5, [0, 2]] if len(trans) > 0 else np.array([[0., 0.]])
    tree = cKDTree(traj_xz)
    xx, zz = np.meshgrid(x_coords, z_coords)
    pts_xz = np.c_[xx.ravel(), zz.ravel()]
    dists, _ = tree.query(pts_xz)
    valid_mask_2d = (dists < max(grid_step * 8.0, 8.0)).reshape(len(z_coords), len(x_coords))

    color_x, color_z, color_y = (255, 255, 0), (255, 0, 255), (0, 255, 255)
    gp1, gp2, gc = [], [], []

    mask_x = valid_mask_2d[:, :-1] | valid_mask_2d[:, 1:]
    zi, xi = np.where(mask_x)
    if len(zi):
        x1, x2, z = x_coords[xi], x_coords[xi + 1], z_coords[zi]
        for y in (floor_y, ceil_y):
            gp1.append(np.column_stack((x1, np.full_like(x1, y), z)))
            gp2.append(np.column_stack((x2, np.full_like(x2, y), z)))
        gc.append(np.tile(color_x, (len(zi) * 2, 1)))

    mask_z = valid_mask_2d[:-1, :] | valid_mask_2d[1:, :]
    zi, xi = np.where(mask_z)
    if len(zi):
        x, z1, z2 = x_coords[xi], z_coords[zi], z_coords[zi + 1]
        for y in (floor_y, ceil_y):
            gp1.append(np.column_stack((x, np.full_like(x, y), z1)))
            gp2.append(np.column_stack((x, np.full_like(x, y), z2)))
        gc.append(np.tile(color_z, (len(zi) * 2, 1)))

    tunnel_r = max(grid_step * 3.5, 4.0)
    wall_mask = (dists > tunnel_r) & (dists < tunnel_r + max(grid_step * 1.2, 1.0))
    wpts = pts_xz[wall_mask]
    if len(wpts):
        wx, wz = wpts[:, 0], wpts[:, 1]
        gp1.append(np.column_stack((wx, np.full_like(wx, ceil_y),  wz)))
        gp2.append(np.column_stack((wx, np.full_like(wx, floor_y), wz)))
        gc.append(np.tile(color_y, (len(wpts), 1)))

    if not gp1:
        return out_video, n_render

    grid_p1 = np.vstack(gp1).astype(np.float32)
    grid_p2 = np.vstack(gp2).astype(np.float32)
    grid_colors = np.vstack(gc).astype(np.float32)
    grid_p1_h = np.hstack((grid_p1, np.ones((len(grid_p1), 1), np.float32)))
    grid_p2_h = np.hstack((grid_p2, np.ones((len(grid_p2), 1), np.float32)))

    base_thick = max(2, int(w / 480))
    w2cs_all = np.linalg.inv(poses_dense)

    # ---- 渲染循环 ----
    for out_idx, frame_idx in enumerate(render_list):
        if frame_idx >= len(w2cs_all):
            break
        w2c = w2cs_all[frame_idx]

        # 每帧选 K / dist / model
        if K_dense is not None:
            K = K_dense[frame_idx]
        else:
            f = max(w, h) * 0.8
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
        dist = dist_dense[frame_idx] if dist_dense is not None else None

        gp1_c = (grid_p1_h @ w2c.T)[:, :3]
        gp2_c = (grid_p2_h @ w2c.T)[:, :3]
        mid_c = (gp1_c + gp2_c) / 2.0
        dist_c = np.linalg.norm(mid_c, axis=1)

        # 简单视锥剔除 (鱼眼放宽)
        if camera_model == "fisheye":
            visible = dist_c < max_view_dist
        else:
            visible = (mid_c[:, 2] > -max(grid_step, 1.0)) & (dist_c < max_view_dist)
        if not np.any(visible):
            continue

        pts2d, valid_mask, meta = clip_and_project_polyline(
            gp1_c[visible], gp2_c[visible], grid_colors[visible],
            K, dist, camera_model, max_view_dist, n_seg=n_seg
        )
        if pts2d is None:
            continue
        alphas, cols = meta
        keep = alphas >= 0.05
        if not np.any(keep):
            continue

        frame = out_video[out_idx]
        pts2d, valid_mask = pts2d[keep], valid_mask[keep]
        alphas, cols = alphas[keep], cols[keep]
        draw_cols = (cols * alphas[:, None]).astype(np.int32)
        draw_thicks = np.maximum(1, (base_thick * alphas).astype(np.int32))

        # 画折线 (一条线 = 一组细分点)
        for poly, vmask, c, t in zip(pts2d, valid_mask, draw_cols, draw_thicks):
            poly_v = poly[vmask]
            if len(poly_v) < 2:
                continue
            # 过滤掉跳跃到屏外的极端点，防止鱼眼边缘画飞线
            diffs = np.linalg.norm(np.diff(poly_v, axis=0), axis=1)
            if len(diffs) and diffs.max() > max(w, h):
                # 在突变处断开
                pieces, start = [], 0
                for j, d in enumerate(diffs):
                    if d > max(w, h):
                        if j + 1 - start >= 2:
                            pieces.append(poly_v[start:j + 1])
                        start = j + 1
                if len(poly_v) - start >= 2:
                    pieces.append(poly_v[start:])
            else:
                pieces = [poly_v]

            for pc in pieces:
                cv2.polylines(frame, [pc.astype(np.int32)], False,
                              c.tolist(), int(t), cv2.LINE_AA)

    return out_video, n_render
# 批量处理相关函数 (保持不变)
# ==========================================
def parse_line(line):
    line = line.strip()
    if not line: return None, None
    first_space_idx = line.find(' ')
    last_space_idx = line.rfind(' ')
    if first_space_idx == -1 or last_space_idx == -1 or first_space_idx == last_space_idx:
        return None, None
    video_path = line[first_space_idx + 1 : last_space_idx].strip()
    pose_path = line[last_space_idx + 1 :].strip()
    # pose_path = line[last_space_idx + 1 :].strip()
    return video_path, pose_path


def parse_single_line(line: str):
    """
    解析 txt 文件中的单行数据
    
    参数:
        line (str): txt 文件中的一行文本
        
    返回:
        tuple: (video_id (str), video_path (str), frames_list (list))
    """
    # 去除首尾的空白字符（包括换行符 \n）
    line = line.strip()
    
    # 如果是空行，直接返回 None
    if not line:
        return None, None, None
        
    # 按空格切分，最大切分次数设为 2，确保严格分为 3 个部分
    parts = line.split(' ', 2)
    
    # 校验格式是否正确
    if len(parts) != 3:
        raise ValueError(f"行格式异常，无法解析为3个字段: {line}")
        
    video_id = parts[0]
    video_path = parts[1]
    frames_str = parts[2]
    
    # 将字符串格式的列表还原为真正的 Python 列表
    try:
        frames_list = json.loads(frames_str)
    except json.JSONDecodeError:
        # 如果解析失败（极少发生），容错处理为空列表
        frames_list = []
        
    return video_id, video_path, frames_list

def process_single_video(video_path, pose_path, output_path, stride=2, cut_points=[]):
    print(f"\n[{video_path}]")
    print(f"  -> 相机参数: {pose_path}")
    if not os.path.exists(video_path):
        print(f"  ❌ 视频文件不存在，跳过")
        return
    if not os.path.exists(pose_path):
        print(f"  ❌ 相机参数文件不存在，跳过")
        return

    temp_output_path = output_path.replace(".mp4", "_temp.mp4")

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  ❌ 无法打开视频文件，跳过")
        return
        
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  ✅ 视频信息: {n}帧, 分辨率 {w}x{h}, 帧率 {fps} FPS")

    print(f"  🚀 开始在内存中渲染 3D 网格 (stride={stride})...")
    try:
        pose_video_array, n_render = render_camera_trajectory_to_numpy(pose_path, n, w, h, stride)
    except Exception as e:
        print(f"  ❌ 渲染失败: {e}")
        cap.release()
        return
    
    print(f"  🎥 正在将位姿叠加到原始视频上...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cut_point_i = 0

    for i in range(len(pose_video_array)):
        ret, rgb_frame = cap.read()
        if not ret: 
            rgb_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        pose_frame = pose_video_array[i]
        if cut_points and cut_point_i < len(cut_points):
            if i >= cut_points[cut_point_i][1]:
                cut_point_i += 1
            elif i > cut_points[cut_point_i][1] - stride:
                pose_frame = np.zeros((h, w, 3), dtype=np.uint8)
                rgb_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        if i < n_render:
            # 如果原视频不是纯黑，这里会将其压暗(alpha=0.6)以凸显网格
            dimmed_rgb = cv2.convertScaleAbs(rgb_frame, alpha=0.6, beta=0)
            combined_frame = cv2.add(dimmed_rgb, pose_frame)
        else:
            combined_frame = np.zeros((h, w, 3), dtype=np.uint8)
            
        out.write(combined_frame)

    cap.release()
    out.release()

    if not os.path.exists(temp_output_path):
        print("  ⚠️ 临时视频未生成，跳过 FFmpeg 转换。")
        return 
        
    print("  🔄 正在使用 FFmpeg 转换为网页兼容的 H.264 格式...")
    ffmpeg_exec = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
    
    ffmpeg_cmd = [
        ffmpeg_exec, "-y", "-i", temp_output_path, 
        "-c:v", "libx264", "-pix_fmt", "yuv420p", 
        output_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        print(f"  🎉 转换成功！已保存至: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ FFmpeg 转换失败。保留了原始文件: {temp_output_path}")
        print(f"  FFmpeg 真实报错信息:\n{e.stderr}")

if __name__ == "__main__":
    # p_path = "/m2v_intern/mengzijie/depthanythingv3/006_运镜参考_067_video_fisheye.npz"
    # v_path = "/ytech_milm_disk2/lishujuan/motion-test/Depth-Anything-3/data/Omni_运镜参考_shared/运镜参考_067/video.mp4"
    # out_file = "/m2v_intern/mengzijie/depthanythingv3/output_fish/test2.mp4"
    # process_single_video(v_path, p_path, out_file)

    p_path = "/m2v_intern/mengzijie/depthanythingv3/npz/006_运镜参考_067_video_hitchcock.npz"
    v_path = "/ytech_milm_disk2/lishujuan/motion-test/Depth-Anything-3/data/Omni_运镜参考_shared/运镜参考_067/video.mp4"
    out_file = "/m2v_intern/mengzijie/depthanythingv3/output_hitchcock/test2.mp4"
    process_single_video(v_path, p_path, out_file)
    
    # # list_file = sys.argv[1]
    # list_file = "complete_testset.txt"
    # output_dir = "/ytech_milm_disk2/lishujuan/motion-test/Depth-Anything-3/output"
    # stride = 2
    # os.makedirs(output_dir, exist_ok=True)
    
    # if not os.path.exists(list_file):
    #     print(f"❌ 找不到列表文件: {list_file}")
    # else:
    #     with open(list_file, 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
            
    #     print(f"📂 共找到 {len(lines)} 个任务，开始批量处理...")
        
    #     for i, line in enumerate(lines):
    #         if line.strip() == "":
    #             continue
            
    #         id, video_path, cut_points = parse_single_line(line)
    #         # video_path, pose_path = parse_line(line)
        
    #         pose_path = os.path.join(output_dir, video_path.lstrip('/').replace(video_path[-4:], ".npz"))
        
    #         if video_path and pose_path:
    #             base_name = os.path.basename(pose_path).replace('.npz', '_cam.mp4')
    #             final_output_path = os.path.join("/m2v_intern/mengzijie/depthanythingv3/output", f"{i:03d}_{base_name}")
    #             process_single_video(video_path, pose_path, final_output_path, stride=stride, cut_points=cut_points)
    #         else:
    #             print(f"⚠️ 格式解析失败，跳过此行: {line.strip()}")
    #         # exit() # the first case
    #     print("\n✅ 所有任务处理完毕！")