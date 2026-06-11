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

def clip_and_project_lines(p1_c, p2_c, colors, K, max_dist):
    """
    高效的向量化 3D 线段裁剪与投影函数。
    """
    z1 = p1_c[:, 2]
    z2 = p2_c[:, 2]
    
    z_near = 0.1
    both_front = (z1 >= z_near) & (z2 >= z_near)
    both_behind = (z1 < z_near) & (z2 < z_near)
    intersect = ~(both_front | both_behind)
    
    valid_p1, valid_p2, valid_colors = [], [], []
    
    if np.any(both_front):
        valid_p1.append(p1_c[both_front])
        valid_p2.append(p2_c[both_front])
        valid_colors.append(colors[both_front])
        
    if np.any(intersect):
        p1_int = p1_c[intersect]
        p2_int = p2_c[intersect]
        z1_int = p1_int[:, 2]
        z2_int = p2_int[:, 2]
        
        t = (z_near - z1_int) / (z2_int - z1_int + 1e-6)
        p_clip = p1_int + t[:, None] * (p2_int - p1_int)
        
        m1_behind = z1_int < z_near
        m2_behind = z2_int < z_near
        
        new_p1 = np.where(m1_behind[:, None], p_clip, p1_int)
        new_p2 = np.where(m2_behind[:, None], p_clip, p2_int)
        
        valid_p1.append(new_p1)
        valid_p2.append(new_p2)
        valid_colors.append(colors[intersect])
        
    if not valid_p1:
        return None, None, None, None
        
    vp1 = np.vstack(valid_p1)
    vp2 = np.vstack(valid_p2)
    vc = np.vstack(valid_colors)
    
    pts_3d = np.vstack((vp1, vp2))
    pts_2d, _ = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), K, None)
    pts_2d = pts_2d.reshape(-1, 2)
    
    half = len(vp1)
    proj_p1 = pts_2d[:half]
    proj_p2 = pts_2d[half:]
    
    # 计算线段中点距离，用于平滑的 Fog 渐隐效果
    mid_pts = (vp1 + vp2) / 2.0
    dist = np.linalg.norm(mid_pts, axis=1)
    alpha = np.clip(1.0 - (dist / max_dist), 0.0, 1.0)
    
    return proj_p1, proj_p2, alpha, vc


def clip_and_project_lines(p1_c, p2_c, colors, K, max_dist):
    """
    极致优化的 3D 线段裁剪与投影函数。
    使用纯 Numpy 手写针孔投影，替代较慢的 cv2.projectPoints。
    """
    z1 = p1_c[:, 2]
    z2 = p2_c[:, 2]
    
    z_near = 0.1
    both_front = (z1 >= z_near) & (z2 >= z_near)
    both_behind = (z1 < z_near) & (z2 < z_near)
    intersect = ~(both_front | both_behind)
    
    valid_p1, valid_p2, valid_colors = [], [], []
    
    if np.any(both_front):
        valid_p1.append(p1_c[both_front])
        valid_p2.append(p2_c[both_front])
        valid_colors.append(colors[both_front])
        
    if np.any(intersect):
        p1_int = p1_c[intersect]
        p2_int = p2_c[intersect]
        z1_int = p1_int[:, 2]
        z2_int = p2_int[:, 2]
        
        t = (z_near - z1_int) / (z2_int - z1_int + 1e-6)
        p_clip = p1_int + t[:, None] * (p2_int - p1_int)
        
        new_p1 = np.where((z1_int < z_near)[:, None], p_clip, p1_int)
        new_p2 = np.where((z2_int < z_near)[:, None], p_clip, p2_int)
        
        valid_p1.append(new_p1)
        valid_p2.append(new_p2)
        valid_colors.append(colors[intersect])
        
    if not valid_p1:
        return None, None, None, None
        
    vp1 = np.vstack(valid_p1)
    vp2 = np.vstack(valid_p2)
    vc = np.vstack(valid_colors)
    
    pts_3d = np.vstack((vp1, vp2))
    
    # 【优化点】手写纯 Numpy 向量化针孔投影，比 cv2.projectPoints 快得多
    Z = np.maximum(pts_3d[:, 2], 1e-6) # 防止除以0
    u = (pts_3d[:, 0] * K[0, 0] / Z) + K[0, 2]
    v = (pts_3d[:, 1] * K[1, 1] / Z) + K[1, 2]
    pts_2d = np.column_stack((u, v))
    
    half = len(vp1)
    proj_p1 = pts_2d[:half]
    proj_p2 = pts_2d[half:]
    
    # 计算线段中点距离，用于平滑的 Fog 渐隐效果
    mid_pts = (vp1 + vp2) / 2.0
    dist = np.linalg.norm(mid_pts, axis=1)
    alpha = np.clip(1.0 - (dist / max_dist), 0.0, 1.0)
    
    return proj_p1, proj_p2, alpha, vc


def render_camera_trajectory_to_numpy(
    pose_path: str, 
    n: int, 
    w: int, 
    h: int, 
    stride: int,
    target_indices: list = None  # 【新增参数】专为 Dataset 设计，按需渲染指定帧
):
    """
    高度优化的渲染函数，适合集成在 PyTorch Dataset 中。
    """
    # ==========================================
    # 1. 加载并格式化相机位姿 (保持不变)
    # ==========================================
    data = np.load(pose_path, allow_pickle=True)
    poses_raw = None
    for k in ["poses", "data", "c2w", "w2c", "cam_poses", "camera_poses", "extrinsic", "cams", 'extrinsics']:
        if k in data:
            poses_raw = data[k]
            break
        
    if poses_raw is None:
        for k in (data.files if hasattr(data, "files") else data.keys()):
            if np.asarray(data[k]).ndim >= 2:
                poses_raw = data[k]
                break

    poses = np.asarray(poses_raw, dtype=np.float32)

    if poses.ndim == 2:
        if poses.shape[1] == 16: poses = poses.reshape(-1, 4, 4)
        elif poses.shape[1] == 12: poses = poses.reshape(-1, 3, 4)
    if poses.ndim == 3 and poses.shape[1:] == (3, 4):
        bottom = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32), (poses.shape[0], 1, 4))
        poses = np.concatenate([poses, bottom], axis=1)

    poses = np.linalg.inv(poses)
    num_poses = len(poses)
    
    # 如果没有指定 target_indices，则默认渲染全部有效帧
    max_valid_frame_idx = (num_poses - 1) * stride
    n_render = min(n, max_valid_frame_idx + 1)
    if n_render <= 0: return np.zeros((0, h, w, 3), dtype=np.uint8), 0
    
    render_list = target_indices if target_indices is not None else list(range(n_render))
    out_video = np.zeros((len(render_list), h, w, 3), dtype=np.uint8)

    # ==========================================
    # 2. 轨迹插值
    # ==========================================
    t_orig = np.arange(num_poses) * stride
    t_target = np.arange(n_render)

    if num_poses == 1:
        poses_dense = np.repeat(poses, n_render, axis=0)
    else:
        translations = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        interp_t = interp1d(t_orig, translations, axis=0)
        t_interp = interp_t(t_target)
        slerp = Slerp(t_orig, R.from_matrix(rotations))
        r_interp = slerp(t_target).as_matrix()

        poses_dense = np.zeros((n_render, 4, 4), dtype=np.float32)
        poses_dense[:, 3, 3] = 1.0
        poses_dense[:, :3, :3] = r_interp
        poses_dense[:, :3, 3] = t_interp

    # ==========================================
    # 3. 极致向量化的 3D 空间网格生成 (消除 Python for 循环)
    # ==========================================
    trans = poses_dense[:, :3, 3]
    diffs = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
    valid_diffs = diffs[diffs > 1e-5]
    grid_step = max(np.median(valid_diffs) * 10.0, 0.5) if len(valid_diffs) > 0 else 0.5
    max_view_dist = max(grid_step * 15.0, 15.0) 
    
    min_xyz, max_xyz = np.min(trans, axis=0), np.max(trans, axis=0)
    margin = max(max_view_dist, 10.0)
    x_coords = np.arange(min_xyz[0] - margin, max_xyz[0] + margin, grid_step)
    z_coords = np.arange(min_xyz[2] - margin, max_xyz[2] + margin, grid_step)
    
    avg_y = np.median(trans[:, 1])
    floor_y, ceil_y = avg_y + max(grid_step * 3.0, 2.0), avg_y - max(grid_step * 3.0, 2.0)
    
    traj_xz = trans[::5, [0, 2]] if len(trans) > 0 else np.array([[0.0, 0.0]])
    tree = cKDTree(traj_xz)
    xx, zz = np.meshgrid(x_coords, z_coords)
    pts_xz = np.c_[xx.ravel(), zz.ravel()]
    dists, _ = tree.query(pts_xz)
    valid_mask_2d = (dists < max(grid_step * 8.0, 8.0)).reshape(len(z_coords), len(x_coords))
    
    color_x, color_z, color_y = (255, 255, 0), (255, 0, 255), (0, 255, 255)
    grid_p1_list, grid_p2_list, grid_colors_list = [], [], []

    # 【优化点】完全向量化生成 X 轴线
    mask_x = valid_mask_2d[:, :-1] | valid_mask_2d[:, 1:]
    zi, xi = np.where(mask_x)
    if len(zi) > 0:
        x1, x2, z = x_coords[xi], x_coords[xi+1], z_coords[zi]
        p1_x = np.column_stack((x1, np.full_like(x1, floor_y), z))
        p2_x = np.column_stack((x2, np.full_like(x2, floor_y), z))
        p3_x = np.column_stack((x1, np.full_like(x1, ceil_y), z))
        p4_x = np.column_stack((x2, np.full_like(x2, ceil_y), z))
        grid_p1_list.extend([p1_x, p3_x])
        grid_p2_list.extend([p2_x, p4_x])
        grid_colors_list.append(np.tile(color_x, (len(p1_x)*2, 1)))

    # 【优化点】完全向量化生成 Z 轴线
    mask_z = valid_mask_2d[:-1, :] | valid_mask_2d[1:, :]
    zi, xi = np.where(mask_z)
    if len(zi) > 0:
        x, z1, z2 = x_coords[xi], z_coords[zi], z_coords[zi+1]
        p1_z = np.column_stack((x, np.full_like(x, floor_y), z1))
        p2_z = np.column_stack((x, np.full_like(x, floor_y), z2))
        p3_z = np.column_stack((x, np.full_like(x, ceil_y), z1))
        p4_z = np.column_stack((x, np.full_like(x, ceil_y), z2))
        grid_p1_list.extend([p1_z, p3_z])
        grid_p2_list.extend([p2_z, p4_z])
        grid_colors_list.append(np.tile(color_z, (len(p1_z)*2, 1)))

    # 【优化点】完全向量化生成 Y 轴线
    tunnel_radius = max(grid_step * 3.5, 4.0)
    wall_mask = (dists > tunnel_radius) & (dists < tunnel_radius + max(grid_step * 1.2, 1.0))
    wall_pts = pts_xz[wall_mask]
    if len(wall_pts) > 0:
        wx, wz = wall_pts[:, 0], wall_pts[:, 1]
        p1_y = np.column_stack((wx, np.full_like(wx, ceil_y), wz))
        p2_y = np.column_stack((wx, np.full_like(wx, floor_y), wz))
        grid_p1_list.append(p1_y)
        grid_p2_list.append(p2_y)
        grid_colors_list.append(np.tile(color_y, (len(p1_y), 1)))

    if not grid_p1_list:
        return out_video, n_render

    grid_p1 = np.vstack(grid_p1_list).astype(np.float32)
    grid_p2 = np.vstack(grid_p2_list).astype(np.float32)
    grid_colors = np.vstack(grid_colors_list).astype(np.float32)

    grid_p1_h = np.hstack((grid_p1, np.ones((len(grid_p1), 1), dtype=np.float32)))
    grid_p2_h = np.hstack((grid_p2, np.ones((len(grid_p2), 1), dtype=np.float32)))

    # ==========================================
    # 4. 渲染循环 (极速版)
    # ==========================================
    focal_length = max(w, h) * 0.8
    K = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype=np.float32)
    base_thick = max(4, int(w / 320)) 

    # 【优化点】批量计算所有相机的 w2c 矩阵
    w2cs_all = np.linalg.inv(poses_dense)

    for out_idx, frame_idx in enumerate(render_list):
        if frame_idx >= len(w2cs_all): break
        w2c = w2cs_all[frame_idx]
        
        # 【优化点】更快的矩阵乘法 (N, 4) @ (4, 4).T -> (N, 4)
        gp1_c = (grid_p1_h @ w2c.T)[:, :3]
        gp2_c = (grid_p2_h @ w2c.T)[:, :3]
        
        mid_c = (gp1_c + gp2_c) / 2.0
        z_c = mid_c[:, 2]
        dist_c = np.linalg.norm(mid_c, axis=1)
        
        valid_render = (z_c > -max(grid_step, 1.0)) & (dist_c < max_view_dist)
        if not np.any(valid_render): continue
        
        g_proj1, g_proj2, g_alpha, g_colors = clip_and_project_lines(
            gp1_c[valid_render], gp2_c[valid_render], grid_colors[valid_render], K, max_view_dist
        )
        
        if g_proj1 is not None:
            # 【优化点】向量化预计算所有线段的颜色和粗细，避免在 for 循环中做乘法
            valid_alpha = g_alpha >= 0.05
            if not np.any(valid_alpha): continue
            
            p1s = g_proj1[valid_alpha].astype(np.int32)
            p2s = g_proj2[valid_alpha].astype(np.int32)
            alphas = g_alpha[valid_alpha]
            cols = g_colors[valid_alpha]
            
            draw_colors = (cols * alphas[:, None]).astype(np.int32)
            draw_thicks = np.maximum(2, (base_thick * alphas).astype(np.int32))
            
            frame = out_video[out_idx]
            # 仅保留最纯粹的 OpenCV 绘制逻辑
            for p1, p2, c, t in zip(p1s, p2s, draw_colors, draw_thicks):
                cv2.line(frame, tuple(p1), tuple(p2), c.tolist(), t, cv2.LINE_AA)

    return out_video, n_render

# ==========================================
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
    # list_file = sys.argv[1]
    list_file = "complete_testset.txt"
    output_dir = "/ytech_milm_disk2/lishujuan/motion-test/Depth-Anything-3/output"
    stride = 2
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(list_file):
        print(f"❌ 找不到列表文件: {list_file}")
    else:
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"📂 共找到 {len(lines)} 个任务，开始批量处理...")
        
        for i, line in enumerate(lines):
            if line.strip() == "":
                continue
            
            id, video_path, cut_points = parse_single_line(line)
            # video_path, pose_path = parse_line(line)
        
            pose_path = os.path.join(output_dir, video_path.lstrip('/').replace(video_path[-4:], ".npz"))
        
            if video_path and pose_path:
                base_name = os.path.basename(pose_path).replace('.npz', '_cam.mp4')
                final_output_path = os.path.join("/m2v_intern/mengzijie/depthanythingv3/output", f"{i:03d}_{base_name}")
                process_single_video(video_path, pose_path, final_output_path, stride=stride, cut_points=cut_points)
            else:
                print(f"⚠️ 格式解析失败，跳过此行: {line.strip()}")
            # exit() # the first case
        print("\n✅ 所有任务处理完毕！")