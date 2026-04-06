#!/bin/bash
#SBATCH -J LISA
#SBATCH -p gen-part
#SBATCH --qos=gpu8-4h
#SBATCH --account=research
#SBATCH --nodes=2
#SBATCH --exclude=lse-hpcnode10,lse-hpcnode[5-7]
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH -t 4:00:00
#SBATCH -o logs_LISA/%x-%j.out

set -euo pipefail

IMG="/scratch/mohammadjavad/LISA.sif"
PROC_DATA="/scratch/mohammadjavad/GSVA"
PROC_CKPT="/scratch/mohammadjavad/LISA"
PROC_CODE="/scratch/mohammadjavad/LISA_codebase"

mkdir -p "$HOME/logs_LISA"

export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

if [ -z "$MASTER_ADDR" ]; then
  echo "Failed to resolve IPv4 address for $MASTER_HOST"
  exit 1
fi

echo "MASTER_HOST=$MASTER_HOST"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$SLURM_NTASKS"

srun \
  --label \
  --kill-on-bad-exit=1 \
  --export=ALL,MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT" \
  apptainer exec --nv \
    --bind "$PROC_CODE:/workspace/home" \
    --bind "$PROC_DATA:/workspace/datasets" \
    --bind "$PROC_CKPT:/workspace/checkpoints" \
    "$IMG" bash -lc '
  set -euo pipefail
  cd /workspace/home

  export PYTHONUNBUFFERED=1
  export MPLBACKEND=Agg
  export OPEN3D_DISABLE_GUI=1

  export WORLD_SIZE=$SLURM_NTASKS
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  
  export HF_HOME=/workspace/checkpoints/hf_cache
  export TRANSFORMERS_CACHE=$HF_HOME/transformers
  export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
  export HF_DATASETS_CACHE=$HF_HOME/datasets
  export HF_HUB_DISABLE_TELEMETRY=1

  mkdir -p $HF_HOME
  mkdir -p $TRANSFORMERS_CACHE
  mkdir -p $HUGGINGFACE_HUB_CACHE
  mkdir -p $HF_DATASETS_CACHE

  echo HOST=$(hostname) RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
  
  python train_ds.py \
    --local_rank=$LOCAL_RANK \
    --version=/workspace/checkpoints/LISA-7B-v1 \
    --dataset_dir=/workspace/datasets/demo_dataset \
    --precision=bf16 \
    --batch_size=2 \
    --epochs=10 \
    --grad_accumulation_steps=10 \
    --dice_loss_weight=1.0 \
    --dataset="sem_seg||reason_seg_rail" \
    --sem_seg_data="railsem" \
    --sample_rates="45,1" \
    --exp_name=lisa-7b-merged \
    --hf_merged_model \
    --explanatory=0.2
'
