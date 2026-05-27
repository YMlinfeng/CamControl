#!/bin/bash
bash /share/zhangshenglong/rocm_rccl_debug/rm-smi-1.sh
bash /share/zhangshenglong/rocm_rccl_debug/test_torch_allreduce_slot1.sh

hostfile=/etc/mpi/hostfile
gpus=$(head -n 1 /etc/mpi/hostfile | grep -Eo '[0-9]+$')
slots=8
sed -i "s/slots=[0-9]*/slots=$slots/" $hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
echo $np
echo $Port

export PYTHONPATH=/share/houliang/rocm_packages/diffusers/src:$PYTHONPATH
mpirun --allow-run-as-root \
    -hostfile /etc/mpi/hostfile \
    -mca btl self,tcp \
    -mca btl_tcp_if_include eth01 \
    -mca oob_tcp_if_include eth01 \
    -mca btl_openib_allow_ib false \
    -mca pml ob1 \
    -x PYTHONPATH \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_SOCKET_IFNAME=eth01 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 -x MY_NODE_IP=$LOCAL_IP \
    -x NCCL_DEBUG=WARN \
    -x PATH \
    "${@:1}"