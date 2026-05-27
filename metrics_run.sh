#!/bin/bash
# 检测 NVIDIA 的 `nvidia-smi` 命令是否存在
# if ! command -v nvidia-smi &> /dev/null
# then
#     echo "nvidia-smi 命令不存在，请确保安装了 NVIDIA 驱动"
#     exit 1
# fi

# 使用 nvidia-smi 获取 GPU 数量
np=$(nvidia-smi -L | wc -l)
ip_addr="127.0.0.1"
echo "GPU number: $np"

temp_dir=$(mktemp -d)
hostfile="$temp_dir/hostfile"
# 将 GPU 数量信息写入文件
echo "127.0.0.1 slots=$np" > "$hostfile"
echo "Temp hostfile: $hostfile"

Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)


metrics_script="scripts/run_metrics.py"


file="/group/dengyufan/repo/magvit2-pytorch/val.csv"

# type="DOVER,CLIPScore,SDScore,CLIPTemp,FlowScore,WarpingError"
type="SSIM,PSNR,LPIPS"

batch_size=1
num_processes=4
ckpt="/group/dengyufan/m2v-diffusers/exps/vtoken0226_05_lpips_kle-6_mean/checkpoints/checkpoint-3584000/visual_tokenizer/pytorch_model.ckpt"

mpirun --allow-run-as-root -np $np \
    -mca plm_rsh_args "-p ${Port}"  \
        -hostfile $hostfile \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x MPI_THREAD_SINGLE=1 \
		-bind-to none  -map-by slot \
        --mca btl tcp,self \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_MIN_NCHANNELS=16 \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_DEBUG=WARN \
		python "$metrics_script" -f "$file" -b "$batch_size" -p "$num_processes" -t "$type" -c "$ckpt"
