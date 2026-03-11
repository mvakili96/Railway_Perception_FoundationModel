#!/bin/bash
#SBATCH -J LISA_MERGE
#SBATCH -p gen-part
#SBATCH --qos=gpu1-32h
#SBATCH --account=research
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs_LISA_merge/merge-%j.out

set -euo pipefail

IMG="/scratch/mohammadjavad/LISA.sif"

PROC_CKPT="/scratch/mohammadjavad/LISA"

mkdir -p "$HOME/logs_LISA_merge"

apptainer exec \
  --bind "$HOME/LISA:/workspace/home" \
  --bind "$PROC_CKPT:/workspace/checkpoints" \
  "$IMG" bash -lc '

    set -euo pipefail
    cd /workspace/home/runs/lisa-7b-merged/ckpt_model

    echo "===== Step 1: Convert DeepSpeed checkpoint to fp32 ====="

    python zero_to_fp32.py . ../fp32_model --max_shard_size "100GB"
   
    echo "===== Step 2: Merge LoRA weights ====="
    cd /workspace/home
   
    CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
    --version="/workspace/checkpoints/LISA-7B-v1" \
    --weight="/workspace/home/runs/lisa-7b-merged/fp32_model/pytorch_model.bin" \
    --save_path="/workspace/checkpoints/LISA-7B-demo"


    echo "===== DONE ====="
'
