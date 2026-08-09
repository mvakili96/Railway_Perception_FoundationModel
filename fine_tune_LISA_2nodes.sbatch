#!/bin/bash
#SBATCH -J LISA
#SBATCH -p gen-part
#SBATCH --qos=gpu8-4h
#SBATCH --account=research
#SBATCH --nodes=2
#SBATCH --exclude=lse-hpcnode14
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
export NCCL_SOCKET_IFNAME=ens3f0np0
export GLOO_SOCKET_IFNAME=ens3f0np0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

MASTER_ADDR=$(
  srun \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="$MASTER_HOST" \
    bash -lc "ip -o -4 addr show dev ens3f0np0 | awk '{print \$4}' | cut -d/ -f1"
)

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

if [ -z "$MASTER_ADDR" ]; then
  echo "Failed to find ens3f0np0 IPv4 address on $MASTER_HOST"
  exit 1
fi

echo "MASTER_HOST=$MASTER_HOST"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$SLURM_NTASKS"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

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

  export PYTHONPATH=/workspace/checkpoints/wandb_python:${PYTHONPATH:-}

  export WANDB_DIR=/workspace/checkpoints/wandb
  export WANDB_CACHE_DIR=/workspace/checkpoints/wandb_cache
  export WANDB_CONFIG_DIR=/workspace/checkpoints/wandb_config
  export WANDB_PROJECT=lisa-rail
  export WANDB_MODE=online

  mkdir -p $WANDB_DIR
  mkdir -p $WANDB_CACHE_DIR
  mkdir -p $WANDB_CONFIG_DIR

  python -c "import wandb; print(wandb.__version__)"

  python train_ds.py \
    --local_rank=$LOCAL_RANK \
    --version=/workspace/checkpoints/LISA-7B-v1 \
    --dataset_dir=/workspace/datasets/demo_dataset \
    --reason_seg_weight_map_dir_name=weight_maps \
    --reason_seg_weight_map_weight=1.0 \
    --precision=bf16 \
    --batch_size=2 \
    --epochs=20 \
    --steps_per_epoch=50 \
    --grad_accumulation_steps=1 \
    --dice_loss_weight=1.0 \
    --bce_loss_weight=2.0 \
    --ce_loss_weight=1 \
    --rail_ego_side_loss_weight=3.0 \
    --use_rail_reasoning_weighted_ce \
    --lr 0.0001 \
    --boundary_bce_band_width=0 \
    --boundary_bce_weight=1.0 \
    --dataset="sem_seg||reason_seg_rail" \
    --sem_seg_data="railsem" \
    --sample_rates="1,4" \
    --exp_name=lisa-7b-merged \
    --hf_merged_model \
    --train_mm_projector \
    --train_sam_prompt_encoder \
    --train_sam_last_blocks=16 \
    --train_clip_last_blocks=8 \
    --clip_lr=0.00001 \
    --rail_counterfactual_flip_prob=0.5 \
    --explanatory=0.5 \
    --epoch_reasoning_inference \
    --epoch_reasoning_json=/workspace/datasets/demo_dataset/reason_seg/ReasonSegRail/explanatory/train.json \
    --use_wandb \
    --wandb_project=lisa-rail \
    --wandb_name=lisa-7b-${SLURM_JOB_ID} \
    --wandb_dir=/workspace/checkpoints/wandb \
    --wandb_mode=online
'