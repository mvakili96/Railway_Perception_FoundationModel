#!/bin/bash
#SBATCH -J Qwen-test
#SBATCH -p gen-part
#SBATCH --qos=scavenger
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --exclude=lse-hpcnode5,lse-hpcnode7,lse-hpcnode14
#SBATCH -t 04:00:00
#SBATCH -o qwen-test-%j.out

set -euo pipefail
export QWEN_IMG="${QWEN_IMG:-/scratch/mohammadjavad/Qwen3VL-v1.sif}"
export QWEN_CODE="${QWEN_CODE:-/scratch/mohammadjavad/QWEN-LISA_Codebase}"
export QWEN_DATA="${QWEN_DATA:-/scratch/mohammadjavad/GSVA}"
export QWEN_CHECKPOINTS="${QWEN_CHECKPOINTS:-/scratch/mohammadjavad/LISA}"
export QWEN_MODEL="${QWEN_MODEL:-$QWEN_CHECKPOINTS/Qwen3-VL-8B-Instruct}"
# Edit these defaults for another experiment. The export must match merge_Qwen.
# PNGs/answer records live in RESULTS; JPEGs/overlays match demo_LISA outputs.
export QWEN_EXPORT_DIR="${QWEN_EXPORT_DIR:-/workspace/home/exports/qwen_159894}"
export QWEN_RESULTS_DIR="${QWEN_RESULTS_DIR:-/workspace/home/test_outputs/qwen_159894}"
export QWEN_TEST_IMAGES="${QWEN_TEST_IMAGES:-/workspace/checkpoints/rs19_crop_1024}"
export QWEN_MASK_SAVE_PATH="${QWEN_MASK_SAVE_PATH:-/workspace/home/mask_output}"
export QWEN_VIS_SAVE_PATH="${QWEN_VIS_SAVE_PATH:-/workspace/home/vis_output}"
export QWEN_PROMPT="${QWEN_PROMPT:-Based on the blade positions in this switch, which route corresponds to the route the train takes? Please respond with segmentation mask and explain why.}"
export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-512}"
# An explicit --manifest or --image_path overrides the default folder source.
source_args=("--image_path=$QWEN_TEST_IMAGES")
for argument in "$@"; do
  case "$argument" in
    --manifest|--manifest=*|--image_path|--image_path=*) source_args=(); break ;;
  esac
done
srun --kill-on-bad-exit=1 bash "$QWEN_CODE/qwen_hpc_run.sh" chat_batch_qwen.py \
  "--model_path=$QWEN_EXPORT_DIR" "--output_dir=$QWEN_RESULTS_DIR" \
  "--mask_save_path=$QWEN_MASK_SAVE_PATH" "--vis_save_path=$QWEN_VIS_SAVE_PATH" \
  "--prompt=$QWEN_PROMPT" "--max_new_tokens=$QWEN_MAX_NEW_TOKENS" \
  ${source_args[@]+"${source_args[@]}"} "$@"
