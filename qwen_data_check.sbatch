#!/bin/bash
# CPU processor/data gate, no model weights or training. Use confirmed one-GPU QoS.
# The allocated GPU is not exercised; Stage 1 already verified CUDA and Qwen.
#SBATCH -J Qwen-data
#SBATCH -p gen-part
#SBATCH --qos=gpu32-1h
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=lse-hpcnode5,lse-hpcnode14
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -o qwen-data-%j.out

set -euo pipefail
export QWEN_IMG="${QWEN_IMG:-/scratch/mohammadjavad/Qwen3VL-v1.sif}"
export QWEN_CODE="${QWEN_CODE:-/scratch/mohammadjavad/LISA_codebase}"
export QWEN_DATA="${QWEN_DATA:-/scratch/mohammadjavad/GSVA}"
export QWEN_MODEL="${QWEN_MODEL:-/scratch/mohammadjavad/LISA/Qwen3-VL-8B-Instruct}"

test -f "$QWEN_IMG"
test -f "$QWEN_CODE/qwen_data_check.py"
test -d "$QWEN_DATA"
test -f "$QWEN_MODEL/qwen_snapshot_manifest.json"
mkdir -p "$QWEN_CODE/runs/qwen_processing"

srun --ntasks=1 --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  apptainer exec --cleanenv --nv \
    --bind "$QWEN_CODE:/workspace/home" \
    --bind "$QWEN_DATA:/workspace/datasets:ro" \
    --bind "$QWEN_MODEL:/workspace/qwen_model:ro" \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env OMP_NUM_THREADS=1 \
    "$QWEN_IMG" /opt/conda/bin/python /workspace/home/qwen_data_check.py \
      --model-dir=/workspace/qwen_model \
      --dataset-root=/workspace/datasets/demo_dataset \
      --report="/workspace/home/runs/qwen_processing/check_${SLURM_JOB_ID}.json"
'
