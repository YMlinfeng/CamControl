import numpy as np
import os


def create_simulated_npz(input_path, output_type="hitchcock",
                         d0=5.0, total_move=3.0,
                         fisheye_fov_deg=180.0,
                         fisheye_k=(-0.05, 0.01, 0.0, 0.0)):
    """
    output_type:
        - "hitchcock":  dolly zoom, 沿相机 forward 方向位移 total_move,
                        焦距按 d_new/d0 比例同步变化, 保证 d0 处主体不变。
        - "fisheye":    设 KB 鱼眼模型, FOV ≈ fisheye_fov_deg
    """
    data = np.load(input_path, allow_pickle=True)
    extr = data['extrinsics'].copy()     # (N, 3, 4)  w2c
    intr = data['intrinsics'].copy()     # (N, 3, 3)
    n = len(extr)

    save_kwargs = {}

    if output_type == "hitchcock":
        # extrinsics 是 w2c, 相机中心 C = -R^T @ t
        # 我们想让相机沿 自身 forward (= +Z_camera) 平移 d_i
        # 对应 w2c 的更新: t_new = t_old - [0,0,d_i]^T (因为 p_c = R P + t, 减小 z 等价相机前移)
        for i in range(n):
            d_i = (i / max(n - 1, 1)) * total_move          # 0 -> total_move
            extr[i, 2, 3] -= d_i                            # 相机前移
            # 焦距同步缩放: 把 d0 处投影大小撑回去
            scale = (d0 + d_i) / d0                         # 远了, f 要变大 (zoom in)
            intr[i, 0, 0] *= scale                          # fx
            intr[i, 1, 1] *= scale                          # fy
        out = input_path.replace(".npz", "_hitchcock.npz")
        np.savez(out, extrinsics=extr, intrinsics=intr,
                 camera_model=np.array("pinhole"))
        print(f"[Hitchcock] move={total_move}m, ref_d={d0}m  ->  {out}")

    elif output_type == "fisheye":
        # 给一个 ~fisheye_fov_deg 的鱼眼: f = (短边/2) / tan(FOV/4) 这种粗略估计
        # 更直接: 设 r_max = min(cx, cy), theta_max = FOV/2
        # KB: r_pix = f * theta_d, 取 theta_d≈theta, f = r_max / theta_max
        theta_max = np.deg2rad(fisheye_fov_deg / 2.0)
        for i in range(n):
            cx, cy = intr[i, 0, 2], intr[i, 1, 2]
            r_max = min(cx, cy)
            f_new = r_max / theta_max
            intr[i, 0, 0] = f_new
            intr[i, 1, 1] = f_new

        dist = np.tile(np.asarray(fisheye_k, dtype=np.float32), (n, 1))
        out = input_path.replace(".npz", "_fisheye.npz")
        np.savez(out, extrinsics=extr, intrinsics=intr,
                 dist_coeffs=dist,
                 camera_model=np.array("fisheye"))
        print(f"[Fisheye] FOV={fisheye_fov_deg}°, k={fisheye_k}  ->  {out}")

    else:
        raise ValueError(output_type)


if __name__ == "__main__":
    src = "/m2v_intern/mengzijie/depthanythingv3/npz/006_运镜参考_067_video.npz"
    create_simulated_npz(src, "hitchcock", d0=5.0, total_move=10.0) # 3.0
    create_simulated_npz(src, "fisheye", fisheye_fov_deg=170.0,
                         fisheye_k=(-0.02, 0.0, 0.0, 0.0))