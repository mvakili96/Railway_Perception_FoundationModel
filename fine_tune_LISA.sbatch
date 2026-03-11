#!/bin/bash
#SBATCH -J LISA
#SBATCH -p gen-part
#SBATCH --qos=gpu1-32h
#SBATCH --account=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 32:00:00
#SBATCH -o logs_LISA/%x-%j.out

set -euo pipefail

IMG="/scratch/mohammadjavad/LISA.sif"

PROC_DATA="/scratch/mohammadjavad/GSVA/"
PROC_CKPT="/scratch/mohammadjavad/LISA/"

mkdir -p "$HOME/logs_LISA"

# ======= Comms / threading =======
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# ======= Run inside the container =======
apptainer exec --nv \
  --bind "$HOME/LISA:/workspace/home" \
  --bind "$PROC_DATA:/workspace/datasets" \
  --bind "$PROC_CKPT:/workspace/checkpoints" \
  "$IMG" bash -lc '
    set -euo pipefail
    cd /workspace/home

    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg
    export OPEN3D_DISABLE_GUI=1

    export MASTER_ADDR=127.0.0.1
    export GLOO_SOCKET_IFNAME=lo
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_IB_DISABLE=1
    export OMP_NUM_THREADS=1
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export NCCL_DEBUG=INFO

    NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
    echo "Using $NUM_GPUS GPU(s)"

    deepspeed --num_gpus=$NUM_GPUS --master_port=24999 train_ds.py \
    --version="/workspace/checkpoints/LISA-7B-v1" \
    --dataset_dir="/workspace/datasets/demo_dataset" \
    --precision="bf16" \
    --dataset="reason_seg_rail" \
    --sample_rates="1" \
    --exp_name="lisa-7b-merged" \
    --hf_merged_model \
    --explanatory=-1

  '
