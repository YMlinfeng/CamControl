import os
import glob
import subprocess

SRC_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/eval_dataset_200/sd2.0"
DST_DIR = "/m2v_intern/mengzijie/m2v_camclone_v2/concat_25/sd2.0"
TARGET_FPS = 25

# 需要处理的视频后缀
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v")


def convert_to_fps(src_path, dst_path, fps):
    """用 ffmpeg 把视频重编码到指定 fps"""
    cmd = [
        "ffmpeg",
        "-y",                      # 覆盖已存在文件
        "-i", src_path,
        "-r", str(fps),            # 输出帧率
        "-c:v", "libx264",
        "-crf", "12",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",            # 音频直接拷贝（无音频会自动忽略）
        dst_path,
    ]
    # 用 run 捕获错误；音频拷贝失败时回退为重编码音频
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        # 回退：音频改成 aac 重编码
        cmd_fallback = [
            "ffmpeg", "-y", "-i", src_path,
            "-r", str(fps),
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            dst_path,
        ]
        result = subprocess.run(
            cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # 收集所有视频文件
    files = []
    for ext in VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(SRC_DIR, f"*{ext}")))
        files.extend(glob.glob(os.path.join(SRC_DIR, f"*{ext.upper()}")))
    files = sorted(set(files))

    if not files:
        print(f"[warn] 在 {SRC_DIR} 下没有找到视频文件")
        return

    print(f"共找到 {len(files)} 个视频，开始转换到 {TARGET_FPS}fps ...")

    ok, fail = 0, 0
    for i, src_path in enumerate(files):
        name = os.path.basename(src_path)          # 命名方式不变
        dst_path = os.path.join(DST_DIR, name)
        try:
            convert_to_fps(src_path, dst_path, TARGET_FPS)
            ok += 1
            print(f"[{i + 1}/{len(files)}] OK  {name} -> {dst_path}")
        except Exception as e:
            fail += 1
            print(f"[{i + 1}/{len(files)}] FAIL {name}: {e}")

    print(f"\n完成：成功 {ok} 个，失败 {fail} 个，输出目录: {DST_DIR}")


if __name__ == "__main__":
    main()