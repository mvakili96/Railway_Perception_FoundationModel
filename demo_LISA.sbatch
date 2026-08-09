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

IMG="/scratch/mohammadjavad/LISA.sif"
PROC_CKPT="/scratch/mohammadjavad/LISA"
PROC_CODE="/scratch/mohammadjavad/LISA_codebase"

apptainer exec --nv \
  --bind "$PROC_CODE:/workspace/home" \
  --bind "$PROC_CKPT:/workspace/checkpoints" \
  "$IMG" bash -lc '
    set -euo pipefail
    cd /workspace/home

    export PYTHONUNBUFFERED=1

    CUDA_VISIBLE_DEVICES=0 \
    python chat_batch.py \
      --version "/workspace/checkpoints/LISA-7B-debug-clip-147016" \
      --vision-tower "/workspace/checkpoints/LISA-7B-debug-clip-147016/vision_tower" \
      --precision bf16 \
      --model_max_length 1024 \
      --prompt "Based on the blade positions in this switch, which route corresponds to the route the train takes? Please respond with segmentation mask and explain why." \
      --image_path "/workspace/checkpoints/rs19_crop_1024" \
      --vis_save_path "/workspace/home/vis_output" \
      --mask_save_path "/workspace/home/mask_output"
  '