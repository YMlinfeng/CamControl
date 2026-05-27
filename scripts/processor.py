import cv2
import numpy as np


class ImgProcessor:
    def __init__(self, batch, w, h, det_list=[], cat_ori=False, use_tensorrt=False, num_frames=16, do_end_padding=True):
        self.batch = batch
        self.w, self.h = w, h
        self.num_frames = num_frames
        self.do_end_padding = do_end_padding
        self.padding_length = None

        self.use_tensorrt = use_tensorrt
        if use_tensorrt:
            self.w, self.h = (self.w + 32) // 64 * 64, (self.h + 32) // 64 * 64

        self.cat_ori = cat_ori
        self.det_list = det_list

    def img_process(self, path):
        img = imread_resize(path, self.h, self.w)
        controls = []
        for each_det in self.det_list:
            controls.append(each_det(img, detect_resolution=min(img.shape[:2]), image_resolution=min(img.shape[:2]), output_type="np"))
        controls = np.concatenate(controls, -1)
        return np.concatenate([img, controls], -1) / 255

    def pre_process(self, path_chunk):
        res = []
        for p in path_chunk:
            res.append(self.img_process(p))
        res = np.stack(res)
        res = np.split(res, res.shape[-1] // 3, -1)
        img, control = res[0], list(res[1:])
        res = [path_chunk, img, control]
        return res

    def imgs2chunk(self, paths):
        chunk_length = self.batch * self.num_frames
        for i in range(0, len(paths), chunk_length):
            path_chunk = paths[i : i + chunk_length]
            if len(path_chunk) < chunk_length and self.do_end_padding:
                self.padding_length = chunk_length - len(path_chunk)
                path_chunk = path_chunk + [path_chunk[-1]] * self.padding_length
            yield self.pre_process(path_chunk)

    def post_process(self, video_in, video_out, video_control, output_type):
        res = video_out
        if "input" in output_type:
            res = np.concatenate([video_in, res], 2)
        if "control" in output_type:
            video_control = np.concatenate(video_control, 2)
            res = np.concatenate([res, video_control], 2)
        if self.do_end_padding:
            res = res[: self.padding_length]
        return res


def imread_resize(path, h_max, w_max):
    img = cv2.imread(path)[..., ::-1]
    img = resize_image(img, min(h_max, w_max))
    return img


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H))  # , interpolation=cv2.INTER_LANCZOS4)
    return img
