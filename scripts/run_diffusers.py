import torch

import imageio.v3 as iio

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained("/group/ckpt/diffusers/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
iio.imwrite("/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/results/svd_diffusers/generated.mp4", frames, fps=7)
