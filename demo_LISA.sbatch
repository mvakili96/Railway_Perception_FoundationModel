#!/bin/bash
#SBATCH -J LISA_DEMO
#SBATCH -p gen-part
#SBATCH --qos=scavenger
#SBATCH --account=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /home/mvakili/logs_LISA_demo/demo-%j.out

set -euo pipefail

IMG="${IMG:-/scratch/mohammadjavad/LISA.sif}"
PROC_CKPT="${PROC_CKPT:-/scratch/mohammadjavad/LISA}"
PROC_CODE="${PROC_CODE:-/scratch/mohammadjavad/LISA_codebase}"
PROC_DATA="${PROC_DATA:-/scratch/mohammadjavad/LISA_data}"
PROC_OUTPUT="${PROC_OUTPUT:-/scratch/mohammadjavad/LISA_outputs}"

MODEL_NAME="${MODEL_NAME:-LISA-7B-debug-clip-147016}"
TEST_IMAGE_SUBDIR="${TEST_IMAGE_SUBDIR:-test/images}"
RUN_NAME="${RUN_NAME:-demo}"
LISA_PROMPT="${LISA_PROMPT:-Based on the blade positions in this switch, which route corresponds to the route the train takes? Please respond with segmentation mask and explain why.}"

mkdir -p "$PROC_OUTPUT/$RUN_NAME"

apptainer exec --nv \
  --bind "$PROC_CODE:/workspace/home" \
  --bind "$PROC_CKPT:/workspace/checkpoints:ro" \
  --bind "$PROC_DATA:/workspace/data:ro" \
  --bind "$PROC_OUTPUT:/workspace/output" \
  --env MODEL_NAME="$MODEL_NAME" \
  --env TEST_IMAGE_SUBDIR="$TEST_IMAGE_SUBDIR" \
  --env RUN_NAME="$RUN_NAME" \
  --env LISA_PROMPT="$LISA_PROMPT" \
  "$IMG" bash -lc '
    set -euo pipefail
    cd /workspace/home

    export PYTHONUNBUFFERED=1

    CUDA_VISIBLE_DEVICES=0 \
    python chat_batch.py \
      --version "/workspace/checkpoints/${MODEL_NAME}" \
      --vision-tower "/workspace/checkpoints/${MODEL_NAME}/vision_tower" \
      --precision bf16 \
      --model_max_length 1024 \
      --prompt "$LISA_PROMPT" \
      --image_path "/workspace/data/${TEST_IMAGE_SUBDIR}" \
      --vis_save_path "/workspace/output/${RUN_NAME}/overlays" \
      --mask_save_path "/workspace/output/${RUN_NAME}/masks"
  '
