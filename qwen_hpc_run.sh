#!/bin/bash
# Shared rank-local container runner. Called by srun, never by a login-node Python.
set -euo pipefail
export QWEN_IMG="${QWEN_IMG:-/scratch/mohammadjavad/Qwen3VL-v1.sif}"
: "${QWEN_CODE:?Set QWEN_CODE}"
: "${QWEN_DATA:?Set QWEN_DATA}"
: "${QWEN_CHECKPOINTS:?Set QWEN_CHECKPOINTS}"
: "${QWEN_MODEL:?Set QWEN_MODEL}"
script=$1
shift
exec apptainer exec --cleanenv --nv \
  --bind "$QWEN_CODE:/workspace/home" \
  --bind "$QWEN_DATA:/workspace/datasets:ro" \
  --bind "$QWEN_CHECKPOINTS:/workspace/checkpoints:ro" \
  --bind "$QWEN_MODEL:/workspace/qwen_model:ro" \
  --env "RANK=${SLURM_PROCID:-0},LOCAL_RANK=${SLURM_LOCALID:-0},WORLD_SIZE=${SLURM_NTASKS:-1}" \
  --env "MASTER_ADDR=${MASTER_ADDR:-127.0.0.1},MASTER_PORT=${MASTER_PORT:-29500}" \
  --env "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens3f0np0},GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens3f0np0}" \
  --env NCCL_IB_DISABLE=1 --env NCCL_SOCKET_FAMILY=AF_INET \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONUNBUFFERED=1 --env OMP_NUM_THREADS=1 \
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
  --env "TORCH_EXTENSIONS_DIR=/workspace/home/runs/qwen_extensions" \
  "$QWEN_IMG" /opt/conda/bin/python "/workspace/home/$script" "$@"
