import ipdb
from PIL import Image
import os.path as osp
from tqdm import tqdm
from functools import partial
import pandas as pd
from controlnet_aux import DWposeDetector, OpenposeDetector, CannyDetector
import torch
import imageio.v3 as iio
import tempfile
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import BlipForConditionalGeneration, BlipProcessor

from src.utils import resize_and_crop_image, get_path, mkdir, open_file


class Preprocessor:
    def __init__(self, mode="v2v", first_n_second=None, output_root=None, width=None, height=None, do_crop=False):
        self.mode = mode
        self.first_n_second = first_n_second
        self.width, self.height = width, height
        self.do_crop = do_crop

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).cuda()
        self.ops = {
            "openpose": OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to("cuda"),
            "canny": CannyDetector(),
        }

        self.output_root = tempfile.mkdtemp() if output_root is None else output_root


        if self.mode == 'v2v':
            self.data_root = osp.join(self.output_root, 'data')
            mkdir(self.data_root, exist_ok=True)

            self.openpose_root = osp.join(self.output_root, 'openpose')
            self.canny_root = osp.join(self.output_root, 'canny')
            mkdir(self.openpose_root, exist_ok=True)
            mkdir(self.canny_root, exist_ok=True)
        elif self.mode == 'i2v':
            mkdir(self.output_root, exist_ok=True)

        self.img_type = ["png", "jpg", "bmp", "jpeg"]
        self.video_type = ['mp4', 'avi', 'mkv', 'wmv', 'webm']

    @torch.inference_mode()
    def __call__(self, input_data):
        data_type = self.video_type if self.mode == "v2v" else self.img_type

        if input_data.endswith(".txt"):
            data_paths = open_file(input_data)
        elif osp.isdir(input_data):
            data_paths = get_path(input_data, '|'.join(data_type))
        elif osp.splitext(input_data)[-1].lower()[1:] in data_type:
            data_paths = [input_data]

        return getattr(self, self.mode)(data_paths)

    def generate_caption(self, images):
        text = "a photograph of"
        inputs = self.processor(images, text, return_tensors="pt").to(device="cuda", dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        caption = caption.replace("a photograph of ", "")

        return caption

    def i2v(self, img_paths):
        data = {
                "image_path": [],
                "caption": []
                }
        for img_path in img_paths:
            data["image_path"].append(img_path)
            img = Image.open(img_path)
            prompt = self.generate_caption(img)
            data["caption"].append(prompt)

        output_path = osp.join(self.output_root, "data.csv")
        pd.DataFrame(data).to_csv(output_path)
        return output_path

    def v2v(self, video_paths):
        data = {
            "video_path": [],
            "caption": [],
        }

        for video_path in tqdm(video_paths):
            video_name = osp.basename(video_path)
            imgs = iio.imread(video_path)
            fps = iio.immeta(video_path)["fps"]

            # clip
            if self.first_n_second is not None:
                imgs = imgs[:int(np.round(self.first_n_second*fps))]

            # caption
            prompt = self.generate_caption(imgs[0][..., ::-1])
            data["caption"].append(prompt)

            # resize
            if self.width is not None:
                imgs = np.stack([resize_and_crop_image(img, self.height, self.width, self.do_crop) for img in imgs])
            output_path = osp.join(self.data_root, video_name)
            iio.imwrite(output_path, imgs, fps=fps)
            data['video_path'].append(output_path)


            res = []
            new_imgs = []
            for key, op in self.ops.items():
                data.setdefault(key+"_path", [])
                res = []
                for img in imgs:
                    out = op(img, image_resolution=min(img.shape[:2]), output_type='np')
                    res.append(out)
                res = np.stack(res)
                output_path = osp.join(getattr(self, key+"_root"), video_name)
                iio.imwrite(output_path, res, fps=fps)
                data[key+"_path"].append(output_path)

        output_path = osp.join(self.output_root, "data.csv")
        pd.DataFrame(data).to_csv(output_path)
        return output_path


if __name__ == "__main__":
    # for v2v:
    # python scripts/control_preprocess.py --input_data {video_list_txt | video_file_path | video_file_dir} --output_root {output_root} --first_n_second 10 --width 360 --height 640 --mode v2v
    # for i2v:
    # python scripts/control_preprocess.py --input_data {img_list_txt | img_file_path | img_file_dir} --output_root {output_root} --mode i2v
    import argparse

    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--first_n_second", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--do_crop", action="store_true")
    parser.add_argument("--mode", choices=["v2v", "i2v"])

    args = parser.parse_args()

    preprocessor = Preprocessor(mode=args.mode, output_root=args.output_root, first_n_second=args.first_n_second, width=args.width, height=args.height, do_crop=args.do_crop)

    output_path = preprocessor(args.input_data)
    print(output_path)
