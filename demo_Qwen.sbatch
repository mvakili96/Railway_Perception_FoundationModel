#!/bin/bash
#SBATCH -J Qwen-test
#SBATCH -p gen-part
#SBATCH --qos=gpu32-1h
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=lse-hpcnode5,lse-hpcnode14
#SBATCH -t 01:00:00
#SBATCH -o qwen-test-%j.out

set -euo pipefail
export QWEN_IMG="${QWEN_IMG:-/scratch/mohammadjavad/Qwen3VL-v1.sif}"
export QWEN_CODE="${QWEN_CODE:-/scratch/mohammadjavad/LISA_codebase}"
export QWEN_DATA="${QWEN_DATA:-/scratch/mohammadjavad/GSVA}"
export QWEN_CHECKPOINTS="${QWEN_CHECKPOINTS:-/scratch/mohammadjavad/LISA}"
export QWEN_MODEL="${QWEN_MODEL:-$QWEN_CHECKPOINTS/Qwen3-VL-8B-Instruct}"
: "${QWEN_EXPORT_DIR:?Set container export directory}"
: "${QWEN_RESULTS_DIR:?Set empty separate container results directory}"
# Supply either --image_path or --manifest as sbatch script arguments.
srun --kill-on-bad-exit=1 bash "$QWEN_CODE/qwen_hpc_run.sh" chat_batch_qwen.py \
  "--model_path=$QWEN_EXPORT_DIR" "--output_dir=$QWEN_RESULTS_DIR" "$@"
