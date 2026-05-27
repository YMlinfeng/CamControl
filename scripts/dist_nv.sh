#!/bin/bash
hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
echo $np


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
		"${@:1}" \


