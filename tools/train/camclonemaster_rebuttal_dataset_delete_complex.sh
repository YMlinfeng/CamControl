#!/bin/bash
export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# source /m2v_intern/luoyawen/Miniconda/miniconda/bin/activate m2v0524
cd /m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/

hostfile=/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/tools/hostfile/re_1_H200_3.txt
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
echo $np
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 1. save_root
save_root="/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps"
folders=$(find $save_root -maxdepth 1 -type d)

# 初始化最大数字为 -1
max_number=10#-1

# 遍历所有子文件夹
for folder in $folders; do
    # 获取文件夹的名称（不包括路径）
    folder_name=$(basename "$folder")
    # 使用正则表达式检查文件夹名称是否以四位数字开头
    if [[ $folder_name =~ ^([0-9]{4}) ]]; then
        # 获取匹配的数字
        number=10#${BASH_REMATCH[1]}
        echo $number"  "$max_number
        # 检查数字是否大于当前最大数字
        if [[ $number -gt $max_number ]]; then
            # 更新最大数字
            max_number=$number
        fi
    fi
done
# 输出最大数字
max_number=$((max_number + 1))
padded_number=$(printf "%04d" "$max_number")
echo "当前idx: $padded_number"

# 2. method && 3. name
method=1b_camclonemaster_rebuttal_dataset_delete_complex
name=1b_camclonemaster_rebuttal_dataset_delete_complex

METHOD_NAME="$method"
CONFIG_EXP=$padded_number"--""$name"

# 4. 实验目录为exps
exp_root=log
mkdir -p "$exp_root"/"$CONFIG_EXP"

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
        -x NCCL_IB_TIMEOUT=22 \
        -x NCCL_DEBUG=WARN \
		python main.py "$METHOD_NAME" --experiment-name "$CONFIG_EXP" "${@:3}" \
        2>&1 | tee "$exp_root"/"$CONFIG_EXP"/$(date +%Y.%m.%d_%H:%M:%S).log

# python main.py "1b_base" --experiment-name "train_debug"