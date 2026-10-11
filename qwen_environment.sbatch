#!/bin/bash
# Native Qwen-only gate: one GPU, no SAM or distributed training.
#SBATCH -J Qwen-env
#SBATCH -p gen-part
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=lse-hpcnode5,lse-hpcnode14
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o qwen-env-%j.out
# No gpu8-4h QoS: this job requests only one GPU. Set a site-approved QoS
# at submission if the cluster requires one (sbatch --qos=... ...).

set -euo pipefail

export QWEN_IMG="${QWEN_IMG:-/scratch/mohammadjavad/Qwen3VL-v1.sif}"
export QWEN_CODE="${QWEN_CODE:-/scratch/mohammadjavad/LISA_codebase}"
export QWEN_DATA="${QWEN_DATA:-/scratch/mohammadjavad/GSVA}"
export QWEN_MODEL="${QWEN_MODEL:-/scratch/mohammadjavad/LISA/Qwen3-VL-8B-Instruct}"
export QWEN_SMOKE_IMAGE="${QWEN_SMOKE_IMAGE:-/workspace/datasets/demo_dataset/reason_seg/ReasonSeg/val/rs08006.jpg}"

test -f "$QWEN_IMG"
test -f "$QWEN_CODE/qwen_environment.py"
test -d "$QWEN_DATA"
test -f "$QWEN_MODEL/qwen_snapshot_manifest.json"
mkdir -p "$QWEN_CODE/runs/qwen_environment"

srun --ntasks=1 --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  apptainer exec --cleanenv --nv \
    --bind "$QWEN_CODE:/workspace/home" \
    --bind "$QWEN_DATA:/workspace/datasets:ro" \
    --bind "$QWEN_MODEL:/workspace/qwen_model:ro" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm did not assign a GPU}" \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env OMP_NUM_THREADS=1 \
    "$QWEN_IMG" /opt/conda/bin/python /workspace/home/qwen_environment.py smoke \
      --model-dir=/workspace/qwen_model \
      --image="$QWEN_SMOKE_IMAGE" \
      --report="/workspace/home/runs/qwen_environment/smoke_${SLURM_JOB_ID}.json"
'
