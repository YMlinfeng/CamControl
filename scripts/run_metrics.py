import torch
from tqdm import tqdm
from einops import rearrange
import argparse
from mpi4py import MPI 

from stable_diffusion_application.kaimm.accelerate_utils import prepare

import deepspeed
import deepspeed.utils.zero_to_fp32

from src.models.autoencoders.visual_tokenizer import VisualTokenizerConfig
from src.data.base_data import DataConfig
from src.engine.metrics import MetricConfig


@torch.autocast(device_type="cuda")
def main():
    deepspeed.init_distributed()
    metric = MetricConfig(type_name=tuple(args.type_name.split(','))).setup()

    vaeConfig = VisualTokenizerConfig()
    vae = vaeConfig.from_pretrained(ckpt_path=args.ckpt_path)

    dataConfig = DataConfig(
        path=args.file,
        height=128,
        width=128,
        image_path_column="image_path",
        caption_column=None,
        video_path_column=None,
        batch_size=args.batch_size,
        num_processes=args.num_processes,
        num_frames=1,
    )
    data = dataConfig.setup()

    metric, vae, data.dataloader = prepare(
        metric, vae, data.dataloader
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    metric = metric.to("cuda:{}".format(rank))
    vae = vae.to("cuda:{}".format(rank))

    total_SSIM = 0
    total_PSNR = 0
    total_LPIPS = 0
    total_batch = 0
    for batch in tqdm(data.dataloader):
        video = batch['images'].to("cuda:{}".format(rank))
        videos = rearrange(video, "b c ... -> b c 1 ...")
        output, _ = vae(videos)
        output = torch.clamp(output, -1, 1)
        output = rearrange(output, "b c 1 h w -> b c h w")
        metric.reset()
        metric(output,video)
        res = metric.compute()
        torch.cuda.empty_cache()

        if rank == 0:
            total_batch += 1
            total_SSIM += res['SSIM'][0]
            total_PSNR += res['PSNR'][0]
            total_LPIPS += res['LPIPS'][0]
            print(res)
            print("SSIM: ", total_SSIM)
    
    comm.Barrier()
    
    if rank == 0:
        print("\nAverage SSIM: ", total_SSIM/total_batch)
        print("Average PSNR: ", total_PSNR/total_batch)
        print("Average LPIPS: ", total_LPIPS/total_batch)    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Quality Assessment')

    parser.add_argument('-f', '--file', type=str, required=True, help="Path to data csv file, the first column is 'video_path', the second colunmn is 'caption'")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch_size, default: 1')
    parser.add_argument('-p', '--num_processes', type=int, default=1, help='num_processes, default: 1')
    parser.add_argument('-t', '--type_name', type=str, default="DOVER,CLIPScore,SDScore,CLIPTemp,FlowScore", help="metric type_name, default: 'DOVER,CLIPScore,SDScore,CLIPTemp,FlowScore'")
    parser.add_argument('-c', '--ckpt_path', type=str, default=None, help="Path to the model checkpoint")

    args = parser.parse_args()

    main()