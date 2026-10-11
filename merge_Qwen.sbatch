#!/bin/bash
# Full-state consolidation uses CPU RAM; model probes run sequentially on one L40.
#SBATCH -J Qwen-merge
#SBATCH -p gen-part
#SBATCH --qos=gpu32-1h
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --exclude=lse-hpcnode5,lse-hpcnode14
#SBATCH -t 01:00:00
#SBATCH -o qwen-merge-%j.out

set -euo pipefail
export QWEN_IMG="${QWEN_IMG:-/scratch/mohammadjavad/Qwen3VL-v1.sif}"
export QWEN_CODE="${QWEN_CODE:-/scratch/mohammadjavad/LISA_codebase}"
export QWEN_DATA="${QWEN_DATA:-/scratch/mohammadjavad/GSVA}"
export QWEN_CHECKPOINTS="${QWEN_CHECKPOINTS:-/scratch/mohammadjavad/LISA}"
export QWEN_MODEL="${QWEN_MODEL:-$QWEN_CHECKPOINTS/Qwen3-VL-8B-Instruct}"
: "${QWEN_RUN_DIR:?Set container training run directory}"
: "${QWEN_EXPORT_DIR:?Set separate container export directory}"
: "${QWEN_VERIFY_IMAGE:?Set one container image path for mandatory equivalence check}"
srun --kill-on-bad-exit=1 bash "$QWEN_CODE/qwen_hpc_run.sh" merge_qwen_sam.py \
  "--run_dir=$QWEN_RUN_DIR" "--output_dir=$QWEN_EXPORT_DIR" \
  "--verify_image=$QWEN_VERIFY_IMAGE" "$@"
